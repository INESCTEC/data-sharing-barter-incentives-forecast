from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import pingouin as pg
import time
import sys


class StandardFilter(ABC):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def feature_ranking(self):
        ...

    # Get Feature Ranking Results
    def get_scores(self, results):
        assert isinstance(results, dict)
        y_name = self.y_name[0]

        if "lst_pvalues" in list(results[y_name].keys()):
            values = zip(
                results[y_name]["lst_feat"],
                results[y_name]["lst_scores"],
                results[y_name]["lst_pvalues"],
            )
            names1 = [y_name for i in range(len(results[y_name]))]
            names2 = ["features", "scores", "pvalues"]
            return pd.DataFrame(values, columns=[names1, names2])
        else:
            values = zip(
                results[y_name]["lst_feat"],
                results[y_name]["lst_scores"],
            )
            names1 = [y_name for i in range(len(results[y_name]))]
            names2 = ["features", "scores"]
            return pd.DataFrame(values, columns=[names1, names2])

    def get_runtime(self, results):
        assert isinstance(results, dict)
        return results["duration"]

    # Apply Feature Selection Method based on Threshold or Topbest
    def feature_selection(
        self, scores, type_selection, thresh, percentile, significance_level
    ):
        y_name = self.y_name[0]
        ops = {
            "thresh": self.apply_threshold,
            "top_best": self.apply_topbest,
        }
        chosen_selection_function = ops.get(type_selection, self.invalid_op)
        if type_selection == "thresh":
            select_feat_mask, count_feat = chosen_selection_function(
                scores, thresh, significance_level
            )
        elif type_selection == "top_best":
            select_feat_mask, count_feat = chosen_selection_function(
                scores, percentile, significance_level
            )
        else:
            print("no valid selection method")
            sys.exit(0)
        feature_all_names = scores[y_name]["features"].values
        feature_selected_names = list(
            feature_all_names[np.where(select_feat_mask == 1)]
        )
        return feature_selected_names, count_feat

    def apply_threshold(self, scores, thresh, significance_level):
        y_name = self.y_name[0]
        if "pvalues" not in list(scores[y_name].columns):
            sel_feat = (scores[y_name]["scores"] >= thresh).values.astype(int).T
        else:
            sel_feat = (
                (
                    (scores[y_name]["scores"] >= thresh)
                    & (scores[y_name]["pvalues"] < significance_level)
                )
                .values.astype(int)
                .T
            )
        count_feat = np.sum(sel_feat)
        return sel_feat, count_feat

    def apply_topbest(self, scores, percentile_level, significance_level):
        y_name = self.y_name[0]
        percentile_value = np.percentile(
            scores[y_name]["scores"].values, percentile_level
        )
        if "pvalues" not in list(scores[y_name].columns):
            sel_feat = (
                (scores[y_name]["scores"] >= percentile_value).values.astype(int).T
            )
        else:
            sel_feat = (
                (
                    (scores[y_name]["scores"] >= percentile_value)
                    & (scores[y_name]["pvalues"] < significance_level)
                )
                .values.astype(int)
                .T
            )
        count_feat = np.sum(sel_feat)
        return sel_feat, count_feat

    def invalid_op(self):
        raise Exception("Invalid operation")


class MarginalFilter(StandardFilter):
    def __init__(
        self,
        method_name,
        X,
        y,
        feat_seller_names,
        y_name
    ):

        self.method_name = method_name
        self.X = X
        self.y = y
        self.feat_seller_names = feat_seller_names
        self.y_name = y_name

    # Filter Methods
    def Pearson(self, feature, target):
        results = stats.pearsonr(feature, target)
        statistic, pvalue = results.statistic, results.pvalue
        return abs(statistic), pvalue

    def Spearman(self, feature, target):
        results = stats.spearmanr(feature, target)
        statistic, pvalue = results.statistic, results.pvalue
        return abs(statistic), pvalue

    def Mut_Inf(self, feature, target):
        from sklearn.feature_selection import mutual_info_regression

        neighbour_list = [10]
        MIscore_list = []
        for points in neighbour_list:
            score = mutual_info_regression(
                feature.reshape(-1, 1), target, n_neighbors=points
            )[0]
            MIscore_list.append(score)
        max_score = np.median(MIscore_list)
        return max_score

    # Apply Feature Ranking Method
    def feature_ranking(self):

        tot_set = defaultdict(dict)
        start = time.time()

        # loop for the number of targets
        for j, targ_name in enumerate(self.y_name):
            # loop for the number of X and collect the score
            list_scores = []
            pvalue_collect = False
            if self.method_name != "MI":
                pvalue_collect = True
                list_pvalues = []

            for i, _ in enumerate(self.feat_seller_names):

                # Pearson's correlation
                if self.method_name == "Pearson":
                    score, pvalue = self.Pearson(self.X[:, i], self.y[:, j])
                    list_scores.append(score)
                    list_pvalues.append(pvalue)

                # Spearman's correlation
                if self.method_name == "Spearman":
                    score, pvalue = self.Spearman(self.X[:, i], self.y[:, j])
                    list_scores.append(score)
                    list_pvalues.append(pvalue)

                # Mutual Information
                if self.method_name == "MI":
                    score = self.Mut_Inf(self.X[:, i], self.y[:, j])
                    list_scores.append(score)

            list_feat_names = self.feat_seller_names
            array_scores = np.array(list_scores)
            array_scores = abs(array_scores) / np.sum(abs(array_scores))
            list_scores = list(array_scores)

            tot_set[targ_name]["lst_feat"] = list_feat_names
            tot_set[targ_name]["lst_scores"] = list_scores
            if pvalue_collect:
                tot_set[targ_name]["lst_pvalues"] = list_pvalues

        end = time.time()
        tot_set["duration"] = end - start
        return tot_set


class PartialFilter(StandardFilter):
    def __init__(
        self,
        method_name,
        dfX,
        dfy,
        feat_names,
        y_name,
        feat_buyer_names
    ):
        self.method_name = method_name
        self.dfX = dfX
        self.dfy = dfy
        self.feat_names = feat_names
        self.y_name = y_name
        self.feat_buyer_names = feat_buyer_names
        self.feat_seller_names = list(set(self.feat_names) - set(self.feat_buyer_names))

    def to_number(self, name):
        if "lag" in name:
            number = int(name.split("_")[-1].replace("lag", ""))
        elif "(" in name:
            number = int(name.split("_")[-1].split("(")[0])
        elif "__" in name:
            number = int(name.split("__")[1])
        else:
            number = int(name.split("_")[-1])
        return number

    def correlation_init(self, method, df, target):
        features = [feat for feat in list(df.columns) if feat != target]
        columns = [features, [target]]
        if self.feat_buyer_names is None:
            corr = pg.pairwise_corr(df, columns, method=method)
        else:
            corr = pg.pairwise_corr(
                df, columns, covar=self.feat_buyer_names, method=method
            )
        corr.columns = [
            "pvalue"
            if (col == "p-unc") or (col == "p-corr") or (col == "p-adjust")
            else col
            for col in corr.columns
        ]
        feature_selected = corr.iloc[abs(corr.r).argmax()].X
        score = abs(corr.iloc[abs(corr.r).argmax()].r)
        pvalue = corr.iloc[abs(corr.r).argmax()].pvalue
        return feature_selected, score, pvalue

    def correlation_partial(self, method, df, feat_not_selected, feat_selected, target):
        feat_names_subset = [feat_not_selected, feat_selected, target]
        if self.feat_buyer_names is None:
            part_corr = pg.partial_corr(
                df[feat_names_subset],
                x=feat_not_selected,
                y=target,
                covar=[feat_selected],
                method=method,
            )
        else:
            part_corr = pg.partial_corr(
                df[feat_names_subset],
                x=feat_not_selected,
                y=target,
                covar=[feat_selected].extend(self.feat_buyer_names),
                method=method,
            )
        part_corr.columns = [
            "pvalue"
            if (col == "p-unc")
            or (col == "p-corr")
            or (col == "p-adjust")
            or (col == "p-val")
            else col
            for col in part_corr.columns
        ]
        return part_corr

    def sort_by_name(self, scores):
        y_name = self.y_name[0]
        scores[y_name, "reorder"] = scores[y_name]["features"].map(self.to_number)
        scores = scores.sort_values((y_name, "reorder"))
        return scores.reset_index().drop([(y_name, "reorder"), "index"], axis=1)

    # Apply Feature Ranking Method
    def feature_ranking(self):
        tot_set = defaultdict(dict)
        start = time.time()
        pvalue_collect = True
        for j, y_name in enumerate(self.y_name):
            dfy = self.dfy[[y_name]]
            lst_scores = []
            lst_pvalues = []
            lst_selected = []
            df = pd.concat([self.dfX, dfy], axis=1)
            if self.method_name == "Partial-Pearson":
                feature, score, pvalue = self.correlation_init("pearson", df, y_name)
            elif self.method_name == "Partial-Spearman":
                feature, score, pvalue = self.correlation_init("spearman", df, y_name)
            else:
                sys.exit(1)
            if self.feat_buyer_names is None:
                lst_not_selected = list(df.columns)[:-1]
                features_set = len(self.feat_names) - 1
            else:
                lst_not_selected = [
                    not_selected
                    for not_selected in list(df.columns)[:-1]
                    if not_selected not in self.feat_buyer_names
                ]
                features_set = len(self.feat_names) - len(self.feat_buyer_names) - 1
            lst_selected.append(feature)
            lst_scores.append(score)
            lst_pvalues.append(pvalue)

            for run in range(features_set):
                lst_feature_partial = []
                lst_scores_partial = []
                lst_pvalue_partial = []
                lst_not_selected.remove(lst_selected[-1])
                for i, feat_not_selected in enumerate(lst_not_selected):
                    for j, feat_selected in enumerate(lst_selected):
                        if self.method_name == "Partial-Pearson":
                            part_corr = self.correlation_partial(
                                "pearson", df,
                                feat_not_selected,
                                feat_selected, y_name
                            )
                        elif self.method_name == "Partial-Spearman":
                            part_corr = self.correlation_partial(
                                "spearman", df,
                                feat_not_selected,
                                feat_selected, y_name
                            )
                        else:
                            sys.exit(1)
                        ps = abs(part_corr.r[0])
                        if j == 0:
                            score = ps
                            pvalue = part_corr.pvalue[0]
                            feature = feat_not_selected
                        else:
                            if ps < score:
                                score = ps
                                pvalue = part_corr.pvalue[0]
                                feature = feat_not_selected
                    lst_feature_partial.append(feature)
                    lst_scores_partial.append(score)
                    lst_pvalue_partial.append(pvalue)
                lst_selected.append(
                    lst_feature_partial[np.array(lst_scores_partial).argmax()]
                )
                lst_pvalues.append(
                    lst_pvalue_partial[np.array(lst_scores_partial).argmax()]
                )
                lst_scores.append(max(lst_scores_partial))

            lst_feat_names = lst_selected
            array_scores = np.array(lst_scores)
            array_scores = abs(array_scores) / np.sum(abs(array_scores))
            lst_scores = list(array_scores)

            tot_set[y_name]["lst_feat"] = lst_feat_names
            tot_set[y_name]["lst_scores"] = lst_scores
            if pvalue_collect:
                tot_set[y_name]["lst_pvalues"] = lst_pvalues
        end = time.time()
        tot_set["duration"] = end - start
        return tot_set


class mRMR(StandardFilter):
    def __init__(
        self,
        seed,
        nr_neighbors,
        dfX,
        dfy,
        feat_names,
        y_name,
        feat_buyer_names
    ):
        self.seed = seed
        self.nr_neighbors = nr_neighbors
        self.dfX = dfX
        self.dfy = dfy
        self.feat_names = feat_names
        self.y_name = y_name
        self.feat_buyer_names = feat_buyer_names
        self.feat_seller_names = list(set(self.feat_names) - set(self.feat_buyer_names))

    def mutual_info(self, dfX, dfy, nr_neighbors, seed):
        from sklearn.feature_selection import mutual_info_regression

        score = pd.Series(
            mutual_info_regression(
                dfX, dfy, n_neighbors=nr_neighbors, random_state=seed
            ),
            index=dfX.columns,
        )
        return score

    def feature_ranking(self):
        import time

        tot_set = defaultdict(dict)
        start = time.time()
        y_name = self.y_name[0]

        # compute the overall relevant and redundant mutual information
        dict_mi = defaultdict(dict)
        mi_relevant = self.mutual_info(self.dfX, self.dfy, self.nr_neighbors, self.seed)
        for i, name in enumerate(self.feat_names):
            series = self.mutual_info(
                self.dfX, self.dfX[name], self.nr_neighbors, self.seed
            )
            dict_mi[name] = series
        mi_redundant = pd.DataFrame(dict_mi).clip(0.00001)

        # compute feature ranking
        selected = []
        not_selected = self.feat_names
        scores_lst = []
        for i in range(len(self.feat_names)):
            score = mi_relevant.loc[not_selected] / mi_redundant.loc[
                not_selected, selected
            ].mean(axis=1).fillna(0.00001)
            best = score.index[score.argmax()]
            scores_lst.append(score.max())
            selected.append(best)
            not_selected.remove(best)

        # # discard buyer features from ranking
        index_buyer_feat = [selected.index(name) for name in self.feat_buyer_names]
        seller_names = [
            name for i, name in enumerate(selected) if i not in index_buyer_feat
        ]
        seller_scores = np.delete(np.array(scores_lst), index_buyer_feat, axis=0)

        # normalize scores
        abs_scores = abs(seller_scores)
        seller_scores = abs_scores / np.sum(abs_scores)

        tot_set[y_name]["lst_feat"] = seller_names
        tot_set[y_name]["lst_scores"] = seller_scores
        end = time.time()
        tot_set["duration"] = end - start
        return tot_set
