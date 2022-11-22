import pandas as pd
import re
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from src.tools.tools import input_value
from conf.config import data_inputs_paths as data_inputs
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from conf.config import data_inputs_paths

import sys

class PreprocessData:
    """
    Preprocessing
    """
    def __init__(self, data_inputs):
        self.path_to_save_x = data_inputs["path_prepro_x"]

    def run_preprocessing(self,
                          df_x,
                          cat_max_lvls=10,
                          abs_corr_thresh=0.90,
                          feat_to_keep="",
                          max_var_toDrop=0.01,
                          do_save=False,
                          type_dummies=np.bool
                          ):
        print("Save mode: %s" % do_save)


        # # Find FIPS column name
        # FIPS_name = df_x.filter(regex='FIPS|fips').columns[0]
        # df_x = df_x.rename(columns={FIPS_name: "FIPS_CODE"})
        #
        # df_x.columns = standard_name_cols(df_x.columns)
        # df_x[["STATE", "AREA_NAME"]] = upper_consistent(df_x[["STATE", "AREA_NAME"]])

        # fix data types
        object_columns = ["STATE", "AREA_NAME", "FIPS_CODE"]
        numeric_columns = [col for col in df_x.columns if col not in object_columns]
        df_x[numeric_columns] = df_x[numeric_columns].astype("Float64")

        # df with correct types
        df_x_types = self._split_categ_conti(df_x, cat_max_lvls)

        # handle nan
        df_x_no_nan = self.check_nan(df_x_types, True, group_cols=["STATE"])
        # if any transform categorical variables into dummies
        categ_cols = df_x_no_nan.select_dtypes("category").columns
        if len(categ_cols) > 0:
            df_x_dummies = self.categ_to_dummies(df_x_no_nan.select_dtypes("category"),
                                             type_dummies,
                                             dummy_na=False)
            df_x_conti_dummies = pd.concat(
               [df_x_dummies,
                 df_x_no_nan.select_dtypes(np.number)],
                axis=1
            )
        else:
            df_x_conti_dummies = df_x_no_nan


        df_x_preprocessed = self.select_features(
            df_x_conti_dummies,
            abs_corr_thresh,
            feat_to_keep,
            max_var_toDrop
        )

        df_x_scaled = self.scale_continuous_features(df_x_preprocessed)

        if do_save:
            df_x_scaled.to_csv(self.path_to_save_x, sep=";", index=True)
            print("saved")
        return df_x_scaled

    def _split_features_categ_conti(self, df, cat_max_lvls):

        """Tag each column name as Categorical or Continuous features.

        The following strategy is used :

        - 1) dtype_object (--> CATEGORICAL)
        - 2) dtype_number

          - 2.1) Time Series (at least min_TS_months months) (--> CONTINUOUS)
          - 2.2) No Time Series

            - 2.2.1) regex for categorical (--> CATEGORICAL)
            - 2.2.2) regex for continuous (--> CONTINUOUS)
            - 2.2.3) remaining features

              - 2.2.3.1) few distinct values (--> CATEGORICAL)
              - 2.2.3.2) large distinct values (--> CONTINUOUS)

        Parameters
        ----------
        df : pandas.DataFrame
        min_ts_months : int, default 3
            The minimum number of features "Time Series" to consider as a valid Time Series.
            Used for function find_TS()
        cat_max_lvls : int, default 70
            For remaining features, the split Categorical vs Continuous is done by a threshold.
            This threshold is first given by the maximum percentage change.
            But if this method gives a threshold larger than cat_max_lvls,
            the threshold used for the split will be cat_max_lvls.
            (with <= thresh ==> categorical
            and  > thresh ==> continuous)

        Returns
        -------
        dict
            Return a dictionary of the splits
        """
        # 1) Select dtype = "Object" (--> CATEGORICAL):
        col_object = df.dtypes[df.dtypes == object].index.tolist()

        # 2) Select dtype = numbers:
        col_numb = df.dtypes[df.dtypes != object].index.tolist()

        # 2.2.1) regex for categorical (--> CATEGORICAL):
        pattern_quali = "FIPS|STATE|AREA|CODE"
        col_numb_rgx_quali = [
            col
            for col in col_numb
            if re.search(pattern_quali, col, flags=re.IGNORECASE)
        ]

        # 2.2.2) regex for continuous (--> CONTINUOUS):
        pattern_quanti = (
            "PERCENT|POP|ESTIMATES|CENSUS|ALL|BACHELOR|SCHOOL|COLLEGE")
        col_numb_rgx_quanti = [
            col
            for col in col_numb
            if re.search(pattern_quanti, col, flags=re.IGNORECASE)
        ]

        # 2.2.3) remaining features:
        col_numb_remain = list(
            set(col_numb)
            - set(col_numb_rgx_quali)
            - set(col_numb_rgx_quanti)
        )

        if col_numb_remain:
            # Count Distinct Values and Sort Asc:
            distinct_remain = df[col_numb_remain].apply(lambda x: len(x.unique()))
            distinct_remain = distinct_remain.sort_values()

            # Find Automatic Threshold by Max Percentage Change
            # (with restriction: distinct values < 1000):
            filtered_rem = distinct_remain[distinct_remain < 300]
            print("\n Features with < 300 unique values")
            print(filtered_rem)
            print(filtered_rem.pct_change())
            feat_cutoff = filtered_rem.pct_change().idxmax(skipna=True)

            auto_threshold_after = filtered_rem[feat_cutoff]
            auto_threshold = filtered_rem[filtered_rem <= auto_threshold_after][-1]


            # Test to whether use the Automatic Threshold or cat_max_lvls:
            if auto_threshold <= cat_max_lvls:
                thresh = auto_threshold
            else:
                thresh = cat_max_lvls

            print(f"The threshold for distinct values is {thresh}")

            # 2.2.3.1) few distinct values (--> CATEGORICAL):
            col_numb_remain_few_dist = distinct_remain[
                distinct_remain <= thresh
                ].index.tolist()

            # 2.2.3.2) large distinct values (--> CONTINUOUS):
            col_numb_remain_large_dist = distinct_remain[
                distinct_remain > thresh
                ].index.tolist()

            # FINAL : Group all Categorical & Group all Continuous:
            col_categ = col_object + col_numb_rgx_quali + col_numb_remain_few_dist
            col_conti = (
                    col_numb_rgx_quanti + col_numb_remain_large_dist
            )
        else:
            # FINAL : Group all Categorical & Group all Continuous:
            col_categ = col_object + col_numb_rgx_quali
            col_conti = col_numb_rgx_quanti

        # Sanity Check : Total Number of features = sum of all Categorical + Continuous:
        # delete duplicated columns
        total_cols = col_conti + col_categ
        col_dub = set([x for x in total_cols if total_cols.count(x) > 1])
        if col_dub:
            col_categ = [item for item in col_categ if item not in col_dub]
        assert len(df.columns) == len(col_conti) + len(col_categ)

        # Fill the returned dictionary:
        splitter = {
            "col_conti": col_conti,
            "col_categ": col_categ,
        }
        return splitter

    def _split_categ_conti(self, df_x, cat_max_lvls):

        splitter = self._split_features_categ_conti(df_x, cat_max_lvls)
        col_conti = splitter["col_conti"]
        col_categ = splitter["col_categ"]

        df_x_categ = df_x[col_categ].astype("category")
        df_x_conti = df_x[col_conti]

        df_x_concat = pd.concat([df_x_categ, df_x_conti], axis=1)

        return df_x_concat

    def categ_to_dummies(self, df_x_categ_, type_dummies, dummy_na=True, ):
        df_x_dummies = pd.get_dummies(df_x_categ_,
                                      columns=list(df_x_categ_.columns),
                                      drop_first=True,
                                      # with option "dummy_na=True",
                                      # all the NaN will be encoded as a distinct category
                                      # so for all the categorical dummy features, *
                                      # no NaN will remain
                                      dummy_na=dummy_na,
                                      dtype=type_dummies
                                      )
        # Sanity Checks after dummization:
        assert not df_x_dummies.isna().values.any(), "NaN left (Something went wrong)"
        # assert all(
        #     [_dtype == np.dtype(bool) for _dtype in list(df_x_dummies.dtypes)]
        # ), "Not all the features are dtype=bool"

        return df_x_dummies



    def check_nan(self, data, do_imputation=False, group_cols=[]):
        nb_lines = data.shape[0]
        nb_nan = data.isna().sum() / nb_lines
        part_nan = nb_nan[nb_nan > 0]

        r = ""
        if not part_nan.empty:
            r += "There are features with NAN values :\n"
            r += f"Number of features with NAN = {len(part_nan.keys())}\n"
            r += f"Number of values with NAN = {sum(data.isna().sum())}\n"
            r += "=" * 88 + "\n"
            r += "[Feature] <--- has empty values ---> [percentage of empty values]\n"
            r += "=" * 88 + "\n"

            for feat, pct_f in part_nan.items():
                r += f"{feat:30s}  <--->  {pct_f:.2%}\n"
            print("\n\n%s" % r)

            if do_imputation:
                # drop columns with nan >= 0.2
                cols_to_drop = part_nan[part_nan >= 0.2]
                # print(cols_to_drop)
                drop_nan_cols = cols_to_drop.keys().tolist()
                data = data.drop(drop_nan_cols, axis=1)
                print("Dropped cols with >= 0.2 NAN values : %s" % drop_nan_cols)

                # input the rest with the median and mode
                cols_to_impute_ = part_nan[part_nan < 0.2].keys()
                data_no_nan = self._imput_missing_vals(data, cols_to_impute_, group_cols)
        else:
            r += "There are none features with NAN values\n"
            data_no_nan = data

        print("\n\n%s" % r)

        return data_no_nan

    def _imput_missing_vals(self, df, cols_to_impute, group_cols):

        r = ""
        r += "=" * 88 + "\n"
        r += "[Missing values before]  <--- [Feature] ---> [Missing values after]\n"
        r += "=" * 88 + "\n"
        r += f"Value imputation using {group_cols} median/mode values \n"

        for col in cols_to_impute:
            nan_ = df[col].isna().sum() / df.shape[0]
            df[col] = df.groupby(group_cols)[col].apply(input_value)
            nan_after = df[col].isna().sum() / df.shape[0]
            r += f"{nan_:.2%}  <--- {col} --->  {nan_after:.2%}\n"
        print("\n\n%s" % r)

        return df

    def filter_by_std_(self, df, max_var_toDrop):
        # Detect vars with low std
        constant_filter = VarianceThreshold(max_var_toDrop)
        constant_filter.fit(df)

        constant_cols = [column for column in df.columns
                         if column not in df.columns[constant_filter.get_support()]]

        print("Columns with std less than %s: %s" % (max_var_toDrop, constant_cols))

        df_no_cst = df.drop(constant_cols, axis=1)

        return df_no_cst

    def drop_correlated_feats(self, df, abs_corr_thresh, feat_to_keep=""):
        """Drop features that are too correlated
        (direct or inverse corr.) to others features.

        Parameters
        ----------
        df : pandas.DataFrame
        abs_corr_thresh : float, default 0.99999
            Only features with an absolute correlation
            greater than or equal to abs_corr_thresh are considered correlated.
            Must be a float between [0.0, 1.0] :
            - Greater than 1.0 will have no effect (df returned as given)
            - Less than or equal to 0.0 will consider all features correlated
            (df returned with only one random feature)
        feat_to_keep : str, default ""
            Make sure this feature will not be dropped,
            in case it is part of a correlated group of features.

        Returns
        -------
        pandas.DataFrame
            DataFrame with no correlated features.
        dict of {str : set of str}
            Dictionary of feature that will be kept (in keys)
            and its correlated feature(s) that will be dropped (in values).
        """
        # Compute Correlation Matrix:
        corr = np.corrcoef(
            df.values.astype(np.float), rowvar=False
        )  # 20x faster than df.corr()

        # To catch direct corr. (corr == 1) and inverse corr. (corr == -1):
        corr = np.abs(corr)

        # Find all the correlated tuples (feat1-feat2):
        features = df.columns
        corr_tuples = []
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                if i < j:
                    # as corr. matrix is symmetric,
                    # only look in the upper triangular matrix:
                    if corr[i, j] >= abs_corr_thresh:
                        corr_tuples.append((feat1, feat2))

        # Find all the ConnectedComponents
        # (groups of features that are correlated to each other):
        all_cc = []
        for feat1, feat2 in corr_tuples:
            not_in_a_single_cc = True
            for cc in all_cc:
                if feat1 in cc or feat2 in cc:
                    not_in_a_single_cc = False
                    cc.update((feat1, feat2))
                    break
            if not_in_a_single_cc:
                new_cc = {feat1, feat2}
                all_cc.append(new_cc)

        # For each ConnectedComponents,
        # map one feat. to keep --> all other correlated features:
        corr_feat = {cc.pop(): cc for cc in all_cc}

        # List and drop all the features to remove:
        if not feat_to_keep:
            feat_to_drop = {feat for feat_set in corr_feat.values()
                            for feat in feat_set}
        else:
            feat_to_drop = {feat for feat_set in corr_feat.values()
                            for feat in feat_set if feat not in feat_to_keep}

        df = df.drop(columns=feat_to_drop)

        r = ""
        if corr_feat:
            r += "There are correlated features based on threshold :\n"
            r += f"Number of features to keep = {len(corr_feat)}\n"
            r += f"Number of features to drop = {sum(map(len, corr_feat.values()))}\n"
            r += "=" * 88 + "\n"
            r += "[Feature TO KEEP] <--- is correlated with ---> [Feature(s) TO DROP]\n"
            r += "=" * 88 + "\n"
            for feat, correlated_f in corr_feat.items():
                r += f"{feat:30s}  <--->  {correlated_f}\n"
            r += "=" * 88 + "\n"
        else:
            r += "There is no correlated features\n"

        print("\n %s" % r)

        print("df_x.shape = %s", df.shape)

        return df

    def select_features(self, df, abs_corr_thresh, feat_to_keep, max_var_toDrop):
        try:
            print("... Selecting features by variance and correlation")

            df_no_cst = self.filter_by_std_(df, max_var_toDrop)
            df_no_corr = self.drop_correlated_feats(
                df_no_cst, abs_corr_thresh, feat_to_keep)

            drop_cols = [i for i in list(
                df.columns) if i not in list(df_no_corr.columns)]
            print(
                "Dropped cols after selecting by variance and correlation :\n\n %s" % drop_cols)

            return df_no_corr
        except Exception as e:
            raise e

    def _split_state_county(self, df_x, county_fips_list):
        ## county
        df_x_county = df_x[df_x["FIPS_CODE"].isin(county_fips_list)]
        # Check if all counties were found
        county_found = df_x_county["FIPS_CODE"].unique()
        ## Missing codes correspond to Alaska : Unlike other states within the United States, Alaska does not administer its presidential elections at the county-level but rather at the lower chamber legislative district, or the House District
        ## For now, I drop Alaska
        r = ""
        if len(county_found) != len(county_fips_list):
            county_n_found = [county for county in county_fips_list if county not in county_found]
            r += "=" * 88 + "\n"
            r += f"Nb of counties found: {len(county_found)} / {len(county_fips_list)}\n"
            r += f"Missing county(ies): {county_n_found}\n"
            r += "=" * 88 + "\n"
        else:
            r += "All county fips found\n"
            r += "=" * 88 + "\n"
        print("\n %s" % r)

        # drop duplicates
        df_x_county = df_x_county.drop_duplicates(["STATE", "AREA_NAME"])
        return df_x_county

        # ***********
        # * Scaling *
        # ***********

    def _scaler(self, scaling_method):

        if scaling_method not in ("robust", "minmax", "standard"):
            msg = "Wrong choosen_scaler value"
            print(msg)
            raise ValueError(msg)

        if scaling_method == "robust":
            # removes median and scales data according to quantile range:
            scaler = RobustScaler(quantile_range=(25.0, 75.0))

        elif scaling_method == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))

        elif scaling_method == "standard":
            scaler = StandardScaler(with_mean=True, with_std=True)

        print("Using scaler: {scaler}")
        return scaler

    def _fit_transform_conti(self, scaler, df, conti_feat):
        df = df.copy()
        df.loc[:, conti_feat] = scaler.transform(
            df[conti_feat].astype(np.float64))
        return df

    def scale_continuous_features(self, df_x, choosen_scaler="minmax"):
        print("Scaling using {choosen_scaler}")
        # Choose the scaler:
        scaler = self._scaler(choosen_scaler)

        # Data with input dtype float32 were all converted to float64 by sklearn scaler
        # Find features to scale (only continuous features):
        conti_feat = df_x.dtypes[df_x.dtypes != np.bool].index

        # Fit the scaler:
        scaler = scaler.fit(df_x[conti_feat].astype(np.float64))

        return self._fit_transform_conti(scaler, df_x, conti_feat)