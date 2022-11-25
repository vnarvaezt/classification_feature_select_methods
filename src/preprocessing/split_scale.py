import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class SplitandScale:
    def __init__(self, data_inputs):
        self.path_x = data_inputs["path_prepro_x"]
        self.path_y = data_inputs["path_prepro_y"]

    def run_split_scale(self, feature_scaler="robust"):

        # load x and y
        df_x = pd.read_csv(self.path_x, sep=";")
        df_y = pd.read_csv(self.path_y, sep=";")

        # join x and y
        df_x_y = pd.merge(
            df_x, df_y, left_on="FIPS_CODE", right_on="COUNTY_FIPS", how="inner"
        )

        df_x_y = df_x_y.set_index(["STATE_NAME", "COUNTY_FIPS"], drop=True)
        # FIXME : county code is repeated after join
        df_x_y = df_x_y.drop(["FIPS_CODE", "STATE"], axis=1)

        X_prepro = df_x_y.drop(["TARGET"], axis=1)
        y_prepro = df_x_y[["TARGET"]]

        Xtrain, Xtest, ytrain, ytest = self.split_x_y(X_prepro, y_prepro)

        # Scale continuous features:
        Xtrain, Xtest = self.scale_continuous_features(Xtrain, Xtest, feature_scaler)

        return Xtrain, Xtest, ytrain, ytest

    # Train / Test split:
    def split_x_y(self, X, y):
        df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
            X, y, test_size=0.2, stratify=y["TARGET"]
        )
        print(f"\n df_x_train = {df_x_train.shape} / df_x_test = {df_x_test.shape}")
        print(f"\n df_y_train = {df_y_train.shape} / df_y_test = {df_y_test.shape}")
        print(f"\n Class distribution y_train :")
        print(df_y_train["TARGET"].value_counts(normalize=True))
        print(f"\n Class distribution y_test :")
        print(df_y_test["TARGET"].value_counts(normalize=True))

        return df_x_train, df_x_test, df_y_train, df_y_test

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

        print(f"Using scaler: {scaling_method}")
        return scaler

    def _fit_transform_conti(self, scaler, df, conti_feat):
        df = df.copy()
        df.loc[:, conti_feat] = scaler.transform(df[conti_feat].astype(np.float64))
        return df

    def scale_continuous_features(self, df_x_train, df_x_test, choosen_scaler):

        # Choose the scaler:
        scaler_cf = self._scaler(choosen_scaler)

        # Data with input dtype float32 were all converted to float64 by sklearn scaler
        # Find features to scale (only continuous features):
        conti_feat = df_x_train.dtypes[df_x_train.dtypes != np.bool].index

        # Fit the scaler:
        scaler = scaler_cf.fit(df_x_train[conti_feat].astype(np.float64))

        return [
            self._fit_transform_conti(scaler_cf, df, conti_feat)
            for df in [df_x_train, df_x_test]
        ]
