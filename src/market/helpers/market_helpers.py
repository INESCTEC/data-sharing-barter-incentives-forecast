import numpy as np
import pandas as pd

from time import time
from loguru import logger
from functools import partial
from joblib import Parallel, delayed

from src.market.util.decorators import timeit
from src.market.helpers.model_helpers import (
    calculate_gain,
    calculate_noise_and_gain
)


# #############################################################################
# Main functions to calculate buyer payment / seller revenue
# #############################################################################

# @timeit
def calc_buyer_payment(
        features,
        targets,
        bid_price,
        gain_func,
        market_price,
        n_hours,
        b_min,
        b_max,
        epsilon,
        buyer_features_idx,
        sellers_features_idx
):
    if bid_price == 0:
        # if bid price == 0, buyer wont pay anything.
        return pd.DataFrame(), 0.0, 0.0

    # Partial, leave "bid_price" to be changed below on f call:
    f = partial(
        calculate_noise_and_gain,
        features=features,
        targets=targets,
        gain_func=gain_func,
        n_hours=n_hours,
        market_price=market_price,
        b_max=b_max,
        buyer_features_idx=buyer_features_idx,
        sellers_features_idx=sellers_features_idx
    )

    noisy_features, gain = f(bid_price)
    xaxis = np.arange(b_min, bid_price + epsilon, epsilon)
    if bid_price <= b_min:
        payment = gain * bid_price
    elif len(xaxis) == 1:
        payment = max(0, gain * bid_price)
    else:
        # Rectangular Integration
        # >> approximates the integral of a function with a rectangle
        # 'xaxis' contains a range of prices where gain will be evaluated at
        # I_ = sum([f(v)[1] for v in xaxis]) * (xaxis[1] - xaxis[0])
        # payment = max(0, bid_price * gain - I_)
        payment = max(0, bid_price * gain)

    return noisy_features, gain, payment


# @timeit
def calc_sellers_revenue(
        noisy_features,
        targets,
        gain_func,
        buyer_resource_id: str,
        buyer_resource_payment: float,
        buyer_market_fee: float,
        sellers_id_list: list,
        sellers_features_name: list,
        buyer_features_idx: list,
        user_features_list: list,
        K,
        lambd,
        n_hours: int):

    logger.debug("-" * 70)
    logger.debug(f"Distributing revenue for buyer resource_id {buyer_resource_id}...")
    # Set payment to distribute by sellers
    # Equal to actual payment - market fee
    payment = buyer_resource_payment - buyer_market_fee

    # -- Calculate percentage revenue (% of buyer payment)
    pct_revenue_split, shapley_value = shapley_robust(
        buyer_features_idx=buyer_features_idx,
        Y=targets,
        X=noisy_features,
        K=K,
        lambd=lambd,
        n_hours=n_hours,
        gain_func=gain_func,
    )
    # sum of pct should be 1 to assure buyer payment is correctly split
    # print(sum(pct_revenue_split))

    if len(pct_revenue_split) != len(sellers_features_name):
        raise Exception("Mismatch between dimensions of "
                        "pct_revenue_split array and sellers_features_name "
                        "arrays (should have equal lengths)")

    # todo: assess and solve possible precision problems here
    if round(sum(pct_revenue_split), 9) != 1.0:
        raise Exception(f"Sum of revenue split different of one, "
                        f"for buyer resource ID {buyer_resource_id}."
                        f"\nSum value: {sum(pct_revenue_split)}"
                        f"\nPct values: {str(pct_revenue_split)}"
                        )
    # -- Check valid features for revenue:
    # Note: These are all the features EXCEPT current agent features
    # can be features created by the market automatic feature engineering
    # process (which will contain the 'buyer_resource_id' on its
    # name. Or features created by the user and shared with the market
    # which will contain one of the id's on the 'user_features_list'
    ignored_features = [buyer_resource_id] + user_features_list
    valid_features_idx = [idx for idx, x in enumerate(sellers_features_name)
                          if (x.startswith("seller"))
                          and (x.split('__')[1] not in ignored_features)]
    # -- assign revenue to sellers:
    revenue_split = dict([
        (seller_id, {"pct_revenue": 0, "abs_revenue": 0, "shapley_value": 0})
        for seller_id in sellers_id_list
    ])
    for j, idx in enumerate(valid_features_idx):
        feat = sellers_features_name[idx]
        seller_resource_id = feat.split('__')[1]
        logger.debug(f"seller resource {seller_resource_id} has to receive "
                     f"{pct_revenue_split[j] * buyer_resource_payment}")
        revenue_split[seller_resource_id]["pct_revenue"] += pct_revenue_split[j]
        revenue_split[seller_resource_id]["abs_revenue"] += pct_revenue_split[j] * payment  # noqa
        revenue_split[seller_resource_id]["shapley_value"] += shapley_value[j]
    return revenue_split


# #############################################################################
# Functions used to distribute market revenue (buyers payment) by sellers
# #############################################################################

def square_rooted(x):
    return np.round(np.sqrt(sum([a * a for a in x])), 3)


def cos_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return np.round(np.abs(numerator) / float(denominator), 3)


# 7. PAYMENT DIVISION - PAPER'S ALGORITHM 1
# @timeit
def aux_shap_aprox(m, M, K, X, Y, n_hours, gain_func, buyer_features_idx):
    phi_ = 0
    for k in np.arange(0, K):
        # np.random.seed(k)   # disable in production
        sg = np.random.permutation(M)
        i = 1
        while sg[0] == m:
            # np.random.seed(i)   # disable in production
            sg = np.random.permutation(M)
            i += 1
        pos = sg[np.arange(0, np.where(sg == m)[0][0])]

        G = calculate_gain(
            features=X,
            targets=Y,
            buyer_feature_pos=buyer_features_idx,
            market_features_pos=pos,
            n_hours=n_hours,
            gain_func=gain_func,
        )
        pos = sg[np.arange(0, np.where(sg == m)[0][0] + 1)]
        Gplus = calculate_gain(
            features=X,
            targets=Y,
            buyer_feature_pos=buyer_features_idx,
            market_features_pos=pos,
            n_hours=n_hours,
            gain_func=gain_func,
        )
        phi_ += max(0, Gplus - G)
    return m, phi_


# @timeit
def shapley_aprox(buyer_features_idx, Y, X, K, n_hours, gain_func):
    M = X.shape[1] - len(buyer_features_idx)
    res = []
    # t0 = time()
    for m in np.arange(0, M):
        res.append(aux_shap_aprox(m, M, K, X, Y, n_hours, gain_func, buyer_features_idx)[1])
    # print(f"Finished for cycle approx in {time() - t0:.2f}s")
    # t0 = time()
    phi = np.array([r for r in res]).transpose()
    # print(f"Finished transpose approx in {time() - t0:.2f}s")
    return phi / K


# @timeit
def shapley_aprox_parallel(Y, X, K, n_hours, gain_func):
    M = X.shape[1] - 1
    f = partial(
        aux_shap_aprox,
        M=M,
        K=K,
        X=X,
        Y=Y,
        n_hours=n_hours,
        gain_func=gain_func
    )
    _r = np.arange(0, M)
    # t0 = time()
    res = Parallel(n_jobs=-1)(delayed(f)(m) for m in _r)
    # print(f"Finished parallel approx in {time() - t0:.2f}s")
    # t0 = time()
    res = sorted(res, key=lambda x: x[0])
    # print(f"Finished sort approx in {time() - t0:.2f}s")
    # t0 = time()
    phi = np.array([x[1] for x in res]).transpose()
    # print(f"Finished transpose approx in {time() - t0:.2f}s")
    return phi / K


# @timeit
def shapley_robust(buyer_features_idx, Y, X, K, lambd, n_hours, gain_func):
    M = X.shape[1] - len(buyer_features_idx)

    if M < 1:
        raise ValueError("Number of market features must be at least 1.")
    elif M == 1:
        # if there is only 1 feature beside buyers features, send all revenue
        # to that seller feature
        return [1]
    else:
        # Create one coefficient for each feature
        phi_ = np.repeat(0.0, M)
        # Calculate shapley approx for each feature
        phi = shapley_aprox(buyer_features_idx, Y, X, K, n_hours, gain_func)

        if phi.sum() > 0:  # means that at least one feature has value > 0
            # Penalize redundant features:
            for m in np.arange(0, M):
                if phi[m] == 0:
                    phi_[m] = 0
                else:
                    s = 0  # will increase based on similarity between features
                    for k in np.arange(0, M):
                        if k != m:
                            s += cos_similarity(X[:, m], X[:, k])
                    # penalize phi for feature M based on similarity with
                    # other features
                    phi_[m] = phi[m] * np.exp(-lambd * s)

            # Create new phi (after penalizations)
            phi = phi_ / phi_.sum()

        return phi, phi_


# #############################################################################
# Functions to update market price
# #############################################################################

# 5. REVENUE - PAPER'S EQUATION (19)
# @timeit
def revenue(features, targets, gain_func, market_price, bid_price, Bmin, Bmax,
            epsilon, n_hours, buyer_features_idx, sellers_features_idx):
    # Function that computes the final value to be paid by buyer
    reps = 5
    expected_revenue = np.repeat(0.0, reps)
    for i in range(reps):
        np.random.seed(i)
        _, _, expected_revenue[i] = calc_buyer_payment(
            features=features,
            targets=targets,
            bid_price=bid_price,
            gain_func=gain_func,
            market_price=market_price,
            b_min=Bmin,
            b_max=Bmax,
            epsilon=epsilon,
            n_hours=n_hours,
            buyer_features_idx=buyer_features_idx,
            sellers_features_idx=sellers_features_idx
        )
    # print("Expected revenue:")
    # print(expected_revenue)
    return expected_revenue.mean()


# 6. PRICE UPDATE - PAPER'S ALGORITHM 2
# @timeit
def aux_price(idx,
              market_price,
              w_last,
              features,
              targets,
              gain_func,
              bid_price,
              Bmax,
              delta,
              Bmin,
              epsilon,
              n_hours,
              buyer_features_idx,
              sellers_features_idx):
    g = revenue(
        features=features,
        targets=targets,
        gain_func=gain_func,
        market_price=market_price,
        bid_price=bid_price,
        Bmin=Bmin,
        Bmax=Bmax,
        epsilon=epsilon,
        n_hours=n_hours,
        buyer_features_idx=buyer_features_idx,
        sellers_features_idx=sellers_features_idx
    ) / Bmax
    # todo: @ricardo - desacoplar calculo de ganho de atualização de w
    w = w_last * (1 + delta * g)
    return idx, w


# @timeit
def market_price_update_parallel(features,
                                 targets,
                                 gain_func,
                                 bid_price,
                                 Bmin,
                                 Bmax,
                                 epsilon,
                                 delta,
                                 n_hours,
                                 possible_p,
                                 w,
                                 buyer_features_idx,
                                 sellers_features_idx,
                                 n_jobs,
                                 **kwargs):
    f = partial(
        aux_price,
        features=features,
        targets=targets,
        gain_func=gain_func,
        bid_price=bid_price,
        Bmax=Bmax,
        delta=delta,
        Bmin=Bmin,
        epsilon=epsilon,
        n_hours=n_hours,
        buyer_features_idx=buyer_features_idx,
        sellers_features_idx=sellers_features_idx
    )

    res = Parallel(n_jobs=n_jobs)(delayed(f)(i, mp, w[i])
                                  for i, mp in enumerate(possible_p))
    res = sorted(res, key=lambda x: x[0])
    w = np.array([x[1] for x in res]).transpose()
    Wn = np.sum(w)  # this line was missing
    probs = w / Wn
    return probs, w


# @timeit
def market_price_update(bid_price,
                        buyer_x,
                        market_x,
                        buyer_y,
                        Bmin,
                        Bmax,
                        epsilon,
                        delta,
                        n_hours,
                        w):
    # update price weights
    # - N number of buyers
    # - w last weights
    # - b buyer bid
    # - Y - buyer target timeseries
    # - X - buyer feature timeseries
    market_price_arr = np.arange(Bmin, Bmax + epsilon, epsilon)
    res = []
    # print("updating market price ...")
    for j, mp in enumerate(market_price_arr):
        # print(f"iteration {j} - market price {mp}")
        res.append(
            aux_price(
                idx=j,
                market_price=mp,
                w_last=w[j],
                bid_price=bid_price,
                buyer_y=buyer_y,
                buyer_x=buyer_x,
                market_x=market_x,
                Bmax=Bmax,
                delta=delta,
                Bmin=Bmin,
                epsilon=epsilon,
                n_hours=n_hours
            )[1]
        )
    w = np.array(res)
    w = w.transpose()
    Wn = np.sum(w)  # this line was missing
    probs = w / Wn
    return probs, w
