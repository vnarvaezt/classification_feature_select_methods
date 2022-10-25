import logging
import os
import re

import numpy as np
import pandas as pd
from pycraft.utils.utils_print import log_func
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

os.environ["HADOOP_HOME"] = "/usr/hdp/current/hadoop-client"
os.environ["ARROW_LIBHDFS_DIR"] = "/usr/hdp/2.6.4.0-91/usr/lib"
os.environ["CLASSPATH"] = "`$HADOOP_HOME/bin/hdfs classpath --glob`"

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class PreprocessData:
    """
    Preprocessing datawith feature selection

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        pass

    def run_preprocessing(self,
                        df_x,
                        cols_label_encoder,
                        col_drops,
                        index_col,
                        cat_max_lvls,
                        dummy_na=False,
                        abs_corr_thresh=0.95,
                        feat_to_keep="",
                        max_var_toDrop=0.01):

        df_x = df_x.set_index(index_col)
        df_x_encode = df_x[cols_label_encoder]
        # These operations only work for numerics columns
        df_x = self.drop_unecessary_variables(
            df_x, list_var_exlues=col_drops+cols_label_encoder)
        df_x = self.correct_var_types(df_x).fillna(0)
        df_x = self.drop_constant_variables(df_x)
        df_x = self.reduce_mem_usage(df_x)
        # Add columns to encode after cleaning and reducing memory
        df_x = pd.concat([df_x, df_x_encode], axis=1)
        df_x = self.prepro_categ_cont(
            df_x, cols_label_encoder, cat_max_lvls, dummy_na)

        # select variables based on variance and correlation
        df_x = self.select_features(df_x, abs_corr_thresh,
                                    feat_to_keep,
                                    max_var_toDrop)

        return df_x

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
        pattern_quali = "^(top_|id_|code_|societaire_|sgmt_|flag_|type_|libl_|situ_|csp_)."
        col_numb_rgx_quali = [
            col
            for col in col_numb
            if re.search(pattern_quali, col, flags=re.IGNORECASE)
        ]

        # 2.2.2) regex for continuous (--> CONTINUOUS):
        pattern_quanti = (
        "^(nb_|delta_|z_|ratio_|mnt_|mtt_|mean_|TYP_|duree_|total_|age|anciennete|encours_|c\d{1,2}).")
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

        # Count Distinct Values and Sort Asc:
        distinct_remain = df[col_numb_remain].apply(lambda x: len(x.unique()))
        distinct_remain = distinct_remain.sort_values()

        # Find Automatic Threshold by Max Percentage Change
        # (with restriction: distinct values < 1000):
        filtered_rem = distinct_remain[distinct_remain < 1000]
        feat_cutoff = filtered_rem.pct_change().idxmax()

        auto_threshold_after = filtered_rem[feat_cutoff]
        auto_threshold = filtered_rem[filtered_rem < auto_threshold_after][-1]

        # Test to whether use the Automatic Threshold or cat_max_lvls:
        if auto_threshold <= cat_max_lvls:
            thresh = auto_threshold
        else:
            thresh = cat_max_lvls

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
            #col_numb_ts +
            col_numb_rgx_quanti + col_numb_remain_large_dist
        )

        # Sanity Check : Total Number of features = sum of all Categorical + Continuous:
        assert len(df.columns) == len(col_conti) + len(col_categ)

        # Fill the returned dictionary:
        splitter = {
            "col_conti": col_conti,
            "col_categ": col_categ,
        }
        return splitter

    def _split_categ_conti(self, df_x, cols_ordinal_encoder, cat_max_lvls=10):

        splitter = self._split_features_categ_conti(df_x, cat_max_lvls)
        col_conti = splitter["col_conti"]
        col_categ = splitter["col_categ"]

        df_x_categ = df_x[col_categ]
        df_x_conti = df_x[col_conti]

        cat_to_dummies = [
            categ for categ in df_x_categ.columns if categ not in cols_ordinal_encoder]

        df_x_cat_to_dummies = df_x_categ[cat_to_dummies]
        df_x_to_encode = df_x_categ[cols_ordinal_encoder]

        return(df_x_conti, df_x_cat_to_dummies, df_x_to_encode)

    ###################################
    # Preprocessing categ et continue #
    ###################################

    def _categ_to_dummies(self, df_x_categ_, dummy_na=True):
        df_x_dummies = pd.get_dummies(df_x_categ_,
                                      columns=list(df_x_categ_.columns),
                                      drop_first=True,
                                      # with option "dummy_na=True",
                                      # all the NaN will be encoded as a distinct category
                                      # so for all the categorical dummy features, *
                                      # no NaN will remain
                                      dummy_na=dummy_na,
                                      dtype=np.bool
                                      )
        # Sanity Checks after dummization:
        assert not df_x_dummies.isna().values.any(), "NaN left (Something went wrong)"
        assert all(
            [_dtype == np.dtype(bool) for _dtype in list(df_x_dummies.dtypes)]
        ), "Not all the features are dtype=bool"

        return df_x_dummies

    def _ordinal_encoder(self, df, vars_to_encode, category="auto"):
        try:
            le = OrdinalEncoder(category)
            le.fit(df[vars_to_encode])
            df.loc[:, vars_to_encode] = le.transform(df[vars_to_encode])
            log.info("Using ordinal encoder on : %s", vars_to_encode)
            return df
        except Exception as e:
            raise e

    def _reconcat_categ_conti(self, df_x_conti, df_x_dumm):
        # Re-Concat (Continuous + Categorical) features:
        # Cast continuous to np.float32
        # (np.float32 lighter than default np.float (=np.float64)):
        df_x = pd.concat([df_x_conti.astype(np.float32), df_x_dumm], axis=1)
        return df_x

    def _reconcat_ordinal_conti(self, df_x_conti, df_x_enc):
        # Re-Concat (ordinal + df_x) features:
        # Cast encoded vars to np.float32
        # (np.float32 lighter than default np.float (=np.float64)):
        df_x = pd.concat([df_x_conti, df_x_enc.astype(np.float32)], axis=1)
        return df_x

    def prepro_categ_cont(self, df_x, cols_label_encoder, cat_max_lvls, dummy_na=False):
        df_x_conti, df_x_to_dummies, df_x_ord_enc = self._split_categ_conti(
            df_x, cols_label_encoder, cat_max_lvls
        )

        df_x_dummies = self._categ_to_dummies(df_x_to_dummies, dummy_na)
        df_x_ordinal = self._ordinal_encoder(df_x_ord_enc, cols_label_encoder)

        # concat dummies and numeric variables
        df_x = self._reconcat_categ_conti(df_x_conti, df_x_dummies)

        # concat ordinal and df_x
        df_x = self._reconcat_ordinal_conti(df_x, df_x_ordinal)
        return df_x

    #################################
    # filter by std and correlation #
    #################################
    def filter_by_std_(self, df, max_var_toDrop):

        # Detect vars with low std
        constant_filter = VarianceThreshold(max_var_toDrop)
        constant_filter.fit(df)

        constant_cols = [column for column in df.columns
                         if column not in df.columns[constant_filter.get_support()]]

        log.info("Colonnes avec un std null: %s", constant_cols)

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
        if not feat_to_keep:
            # take the feat to keep at random:
            corr_feat = {cc.pop(): cc for cc in all_cc}
        else:
            # if feat_to_keep in cc --> keep it / else
            # --> keep one feat at random:
            corr_feat = dict(
                (feat_to_keep, cc - {feat_to_keep})
                if feat_to_keep in cc
                else (cc.pop(), cc)
                for cc in all_cc
            )

        # List and drop all the features to remove:
        feat_to_drop = {feat for feat_set in corr_feat.values()
                        for feat in feat_set}
        df = df.drop(columns=feat_to_drop)

        r = ""
        if corr_feat:
            r += "There are correlation features :\n"
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

        log.info("\n %s", r)

        log.info("df_x.shape = %s" , df.shape)

        return df

    def select_features(self, df, abs_corr_thresh=0.95, feat_to_keep="", max_var_toDrop=0.01):
        try:
            log.info("... Selecting features by variance and correlation")
            df_no_cst = self.filter_by_std_(df, max_var_toDrop)
            df_no_corr = self.drop_correlated_feats(
                df_no_cst, abs_corr_thresh, feat_to_keep)

            drop_cols = [i for i in list(
                df.columns) if i not in list(df_no_corr.columns)]
            log.info(
                "Dropped cols after selecting by variane and correlation : %s", drop_cols)

            return df_no_corr
        except Exception as e:
            raise e

    # ***********
    # * Scaling *
    # ***********

    def _scaler(self, scaling_method):

        if scaling_method not in ("robust", "minmax", "standard"):
            msg = "Wrong choosen_scaler value"
            log.critical(msg)
            raise ValueError(msg)

        if scaling_method == "robust":
            # removes median and scales data according to quantile range:
            scaler = RobustScaler(quantile_range=(25.0, 75.0))

        elif scaling_method == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))

        elif scaling_method == "standard":
            scaler = StandardScaler(with_mean=True, with_std=True)

        log.info("Using scaler: {scaler}")
        return scaler

    def _fit_transform_conti(self, scaler, df, conti_feat):
        df = df.copy()
        df.loc[:, conti_feat] = scaler.transform(
            df[conti_feat].astype(np.float64))
        return df

    def split_train_test(self, df, test_size, random_state):
        try:
            log.info("Splitting into train and test")
            X_train_, X_test_ = train_test_split(
                df, test_size=test_size,
                random_state=random_state, shuffle=True
            )
            return X_train_, X_test_
        except Exception as e:
            raise e

    @log_func  # sert a quoi???
    def scale_continuous_features(self, df_x_train, df_x_test, choosen_scaler="minmax"):
        log.info("Scaling using {choosen_scaler}")
        # Choose the scaler:
        scaler = self._scaler(choosen_scaler)

        # Choose the dataset to fit on:
        fit_on = pd.concat([df_x_train, df_x_test], axis=0)

        # Data with input dtype float32 were all converted to float64 by sklearn scaler
        # Find features to scale (only continuous features):
        conti_feat = fit_on.dtypes[fit_on.dtypes != np.bool].index

        # Fit the scaler:
        scaler = scaler.fit(fit_on[conti_feat].astype(np.float64))

        return [
            self._fit_transform_conti(scaler, df, conti_feat)
            for df in [df_x_train, df_x_test]
        ]

    def reduce_mem_usage(self, df):
        """
        Method tooked from Kaggle
        Usefull for reducing dataframe size

        Change variables types according
        to min and max value of the
        variable.
        """

        start_mem_usg = df.memory_usage().sum() / 1024**2
        print("Memory usage of properties dataframe is :",
              start_mem_usg, " MB")

        for c in df.columns:
            if df[c].dtype != object:

                # Print current column type
                print("******************************")
                print("Column: ", c)
                print("dtype before: ", df[c].dtype)

                # make variables for Int, max and min
                IsInt = False
                mx = df[c].max()
                mn = df[c].min()

                if not np.isfinite(df[c]).all():
                    continue

                asint = df[c].fillna(0).astype(np.int64)
                result = (df[c] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            df[c] = df[c].astype(np.uint8)
                        elif mx < 65535:
                            df[c] = df[c].astype(np.uint16)
                        elif mx < 4294967295:
                            df[c] = df[c].astype(np.uint32)
                        else:
                            df[c] = df[c].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[c] = df[c].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            df[c] = df[c].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            df[c] = df[c].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            df[c] = df[c].astype(np.int64)

                # Make float datatypes 32 bit
                else:
                    df[c] = df[c].astype(np.float32)

                # Print new column type
                print("dtype after: ", df[c].dtype)
                print("******************************")

        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024**2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ",
              round(100*mem_usg/start_mem_usg, 2),
              "% of the initial size")
        return df

    def drop_unecessary_variables(self, df, list_var_exlues=["NU_CPTE", "MAX_DT_CM",
                                                             "ID_USER_LAST_CM",
                                                             "ID_PART", "cat_isolement",
                                                             "DT_max"]):
        try:
            log.info("-------")
            log.info("Here we are dropping unecessary variables")
            log.info("-------")

            return df.drop(list_var_exlues, axis=1)
        except Exception as e:
            raise e

    def correct_var_types(self, df):
        try:
            log.info(5*"-------")
            log.info("Here we are correcting variables types")
            log.info(5*"-------")
            object_dtypes = [c for c in df.columns
                             if df[c].dtypes == "object"]

            for c in object_dtypes:
                df[c] = df[c].astype(float)
            return df
        except Exception as e:
            raise e

    def create_indicators(self, df):
        """ Unnecessary"""
        try:
            log.info("-------")
            log.info("Here we are creating indicator variables")
            log.info("-------")
            null_columns = [c for c in df.columns if df[c].isnull().sum() > 0]
            for c in null_columns:
                df[c] = df[c].apply(lambda x: int(x > 0))
            return df
        except Exception as e:
            raise e

    def drop_constant_variables(self, df):
        try:
            log.info("-------")
            log.info("Here we are dropping constant variables")
            log.info("-------")
            cols_to_drop = []
            for c in df.columns:
                cond_1 = df[c].mean() == 0
                cond_2 = df[c].max() == 0
                cond_3 = df[c].min() == 0
                if cond_1 & cond_2 & cond_3:
                    cols_to_drop.append(c)
            return df.drop(cols_to_drop, axis=1)
        except Exception as e:
            raise e
