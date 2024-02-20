import sys
from loguru import logger
from collections import defaultdict
from src.market.preprocessing.utils.utils import check_mkdir
from src.market.preprocessing.feature_selection.filter_methods import (
    MarginalFilter,
    PartialFilter
)
from src.market.preprocessing.feature_selection.filter_methods import mRMR


class FeatureProcess:
    def __init__(self, seed, method_name,
                 type_selection, percentile, threshold, significance_level,
                 nr_neighbors, path_to_save_fs, dir_fs, filename_scores,
                 filename_fs, file_format):
        self.seed = seed
        self.method_name = method_name
        self.type_selection = type_selection
        self.percentile = percentile
        self.threshold = threshold
        self.significance_level = significance_level
        self.nr_neighbors = nr_neighbors
        self.path_to_save_fs = path_to_save_fs
        self.dir_fs = dir_fs
        self.filename_scores = filename_scores
        self.filename_fs = filename_fs
        self.file_format = file_format

    @staticmethod
    def preprocess_drop_nan_records(dfx_seller, dfx_buyer, dfy_buyer=None):
        if dfy_buyer is not None:
            # join seller and buyer features and buyer target to drop nans:
            data = dfx_seller.join(dfx_buyer).join(dfy_buyer)
            # get the list of timestamps with nan values
            timestamp_nan = list(data[data.isna().any(axis=1)].index)

            if len(timestamp_nan) > 0:
                logger.debug(f"Found {len(timestamp_nan)} NaN Timestamps "
                             f"when merging seller(x) / buyer(x) / buyer(y) "
                             f"dataframes")
                # drop nan s valuess
                data = data.dropna(subset=list(dfy_buyer.columns))
                # reset df buyer and sellers
                dfx_buyer = data[list(dfx_buyer.columns)]
                dfy_buyer = data[list(dfy_buyer.columns)]
                dfx_seller = data[list(dfx_seller.columns)]
            return dfx_seller, dfx_buyer, dfy_buyer
        else:
            # join seller and buyer features and buyer target to drop nans:
            data = dfx_seller.join(dfx_buyer)
            # get the list of timestamps with nan values
            timestamp_nan = list(data[data.isna().any(axis=1)].index)

            if len(timestamp_nan) > 0:
                logger.debug(f"Found {len(timestamp_nan)} NaN Timestamps "
                             f"when merging seller(x) / buyer(x) / buyer(y) "
                             f"dataframes")
                # drop nan s valuess
                data = data.dropna(subset=list(dfx_buyer.columns))
                # reset df buyer and sellers
                dfx_buyer = data[list(dfx_buyer.columns)]
                dfx_seller = data[list(dfx_seller.columns)]
            return dfx_seller, dfx_buyer

    def feature_selection(self, dfx_seller, dfx_buyer, dfy_buyer, save=False):
        import json
        import pandas as pd
        import numpy as np

        if save:
            # make directory if path does not exist
            check_mkdir(self.path_to_save_fs)

        if not isinstance(dfx_seller, pd.DataFrame):
            raise ValueError("dfX_seller must be pandas dataframes")
        if not isinstance(dfx_buyer, pd.DataFrame):
            raise ValueError("dfX_buyer must be pandas dataframes")
        if not isinstance(dfy_buyer, pd.DataFrame):
            raise ValueError("dfY_buyer must be pandas dataframes")

        # filter out records with nan's values
        dfx_seller, dfx_buyer, dfy_buyer = self.preprocess_drop_nan_records(
            dfx_seller, dfx_buyer, dfy_buyer
        )

        # TODO: refactoring
        dfx = pd.concat([dfx_buyer, dfx_seller], axis=1)
        dfy = dfy_buyer.copy()
        feat_buyer_names = list(dfx_buyer.columns)
        y = dfy.values.reshape(-1, 1)
        X_names = list(dfx.columns)
        X_names_sellers = sorted(set(X_names).difference(set(feat_buyer_names)))
        y_name = list(dfy.columns)
        X_sellers = dfx[X_names_sellers].values
        # TODO: refactoring

        # init nested dict
        results_block = defaultdict(lambda: defaultdict(dict))

        if (
            (self.method_name == "Pearson")
            or (self.method_name == "Spearman")
            or (self.method_name == "MI")
        ):
            marginal_filter = MarginalFilter(
                self.method_name,
                X_sellers,
                y,
                X_names_sellers,
                y_name
            )
            results = marginal_filter.feature_ranking()
            scores = marginal_filter.get_scores(results)
            (
                feature_selected,
                nr_feature_selected,
            ) = marginal_filter.feature_selection(
                scores,
                self.type_selection,
                self.threshold,
                self.percentile,
                self.significance_level,
            )

        elif (self.method_name == "Partial-Pearson") or (
                self.method_name == "Partial-Spearman"
        ):
            partial_filter = PartialFilter(
                self.method_name,
                dfx,
                dfy,
                X_names,
                y_name,
                feat_buyer_names
            )
            results = partial_filter.feature_ranking()
            ordered_scores = partial_filter.get_scores(results)
            scores = partial_filter.sort_by_name(ordered_scores)
            (
                feature_selected,
                nr_feature_selected,
            ) = partial_filter.feature_selection(
                scores,
                self.type_selection,
                self.threshold,
                self.percentile,
                self.significance_level,
            )
        elif self.method_name == "mRMR":
            mrmr_filter = mRMR(
                self.seed,
                self.nr_neighbors,
                dfx,
                dfy,
                X_names,
                y_name,
                feat_buyer_names
            )
            results = mrmr_filter.feature_ranking()
            scores = mrmr_filter.get_scores(results)
            feature_selected, nr_feature_selected = mrmr_filter.feature_selection(
                scores,
                self.type_selection,
                self.threshold,
                self.percentile,
                self.significance_level,
            )
        else:
            print(self.method_name)
            sys.exit(1)
        results_block[self.method_name]["feature_selected"] = ",".join(feature_selected)
        results_block[self.method_name]["nr_feature_selected"] = str(nr_feature_selected)
        names_cols = list(scores[y_name].columns)
        for name_col in names_cols:
            values_list = scores[name_col].values.reshape(1, -1)[0].tolist()
            converted_list = [str(element) for element in values_list]
            results_block[self.method_name]["results"][name_col[1]] = ",".join(
                converted_list
            )
        if save:
            # save results as csv files
            np.savetxt(
                self.path_to_save_fs
                + self.method_name
                + "_"
                + self.filename_fs,
                feature_selected,
                delimiter=", ",
                fmt="% s",
            )
            scores.to_csv(
                self.path_to_save_fs
                + self.method_name
                + "_"
                + self.filename_scores,
                index=False,
            )

        if save:
            if self.file_format == "json":
                full_path = ".".join(
                    (self.path_to_save_fs + self.dir_fs, self.file_format)
                )
                with open(full_path, "w") as outfile:
                    json.dump(results_block, outfile)
        return results_block

    def get_feature_selected(self, dict_results):
        if not isinstance(dict_results, dict):
            raise ValueError("dict_results must be dict")
        if not isinstance(self.method_name, str):
            raise ValueError("method_name must be str")
        list_feature_selected = dict_results[self.method_name]["feature_selected"].split(",")
        nr_feature_selected = float(dict_results[self.method_name]["nr_feature_selected"])
        return list_feature_selected, int(nr_feature_selected)

    @staticmethod
    def get_feature_selection_df(
        dfx_seller, list_feature_selected, nr_feature_selected
    ):
        import pandas as pd

        if nr_feature_selected > 1:
            if not isinstance(dfx_seller, pd.DataFrame):
                raise ValueError("dfx_seller must be pandas dataframe")
            if not isinstance(list_feature_selected, list):
                raise ValueError("list_feature_selected must be list")
            if len(list_feature_selected) != nr_feature_selected:
                raise ValueError("must be equals")
            dfx_seller_reduced = dfx_seller[list_feature_selected]
            return dfx_seller_reduced
        return dfx_seller
