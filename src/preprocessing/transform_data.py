import numpy as np
import pandas as pd

from src.tools.tools import read_excel, standard_name_cols, upper_consistent

class TransformData:
    def __init__(self):
        pass
    def transform_data(self, data_paths):
        self.load_data(data_paths)
        # preprocessed y
        county_fips = self.preprocess_y()
        # short preprocessing ofr x and join all x
        all_x = self.union_all_x(county_fips)
        return all_x


    def load_data(self, data_paths):
        self.path_population_x = data_paths["path_population_x"]
        self.path_education_x = data_paths["path_education_x"]
        self.path_poverty_x = data_paths["path_poverty_x"]
        self.path_input_y = data_paths["path_input_y"]
        self.path_unemploymnt_x = data_paths["path_unemploymnt_x"]
        self.path_save_y = data_paths["path_prepro_y"]

        # read files
        # import y
        self.df_presidential_2020 = pd.read_csv(
            self.path_input_y, converters={"county_fips": str}
        )
        # import x
        self.df_education = read_excel(self.path_education_x)
        self.df_population = read_excel(self.path_population_x)
        self.df_poverty = read_excel(self.path_poverty_x)
        self.df_unemployment = read_excel(self.path_unemploymnt_x)


    def preprocess_y(self):
        df_y = self.df_presidential_2020.copy(deep=True)
        # upper case all column names
        df_y.columns = standard_name_cols(df_y.columns)
        # upper case data for these columns
        cols_to_upper_case = ["STATE_NAME", "COUNTY_NAME"]
        df_y[cols_to_upper_case] = upper_consistent(df_y[cols_to_upper_case])
        df_y["TARGET"] = np.where(df_y["VOTES_GOP"] < df_y["VOTES_DEM"], 1, 0)
        df_target = df_y[["COUNTY_FIPS", "STATE_NAME", "TARGET"]]
        # df_target = df_target[df_target["STATE_NAME"] != "DISTRICT OF COLUMBIA"]

        df_target.to_csv(self.path_save_y, sep=";", index=False)

        # county code list
        county_fips_list = df_target["COUNTY_FIPS"].unique()

        return county_fips_list


    def union_all_x(self, county_fips_list):
        cols_to_drop_population = [
            "RURAL_URBAN_CONTINUUM_CODE_2003",
            "URBAN_INFLUENCE_CODE_2003",
            "URBAN_INFLUENCE_CODE_2013",
            "RURAL_URBAN_CONTINUUM_CODE_2013",
        ]
        cols_to_drop_educ = [
            "STATE",
            "AREA_NAME",
            "2003_RURAL_URBAN_CONTINUUM_CODE",
            "2003_URBAN_INFLUENCE_CODE",
            "2013_RURAL_URBAN_CONTINUUM_CODE",
            "2013_URBAN_INFLUENCE_CODE",
        ]
        cols_to_drop_poverty = [
            "STATE",
            "AREA_NAME",
            "RURAL_URBAN_CONTINUUM_CODE_2003",
            "URBAN_INFLUENCE_CODE_2003",
            "RURAL_URBAN_CONTINUUM_CODE_2013",
            "URBAN_INFLUENCE_CODE_2013",
        ]
        cols_to_drop_unemployment = ["STATE", "AREA_NAME"]

        df_population_county = self._preprocess_x(
            self.df_population, county_fips_list, cols_to_drop_population
        )

        self.df_education["Area name"] = self.df_education["Area name"].replace(
            "Lousiana", "LOUISIANA"
        )
        df_education_county = self._preprocess_x(
            self.df_education, county_fips_list, cols_to_drop_educ
        )
        df_poverty = self.df_poverty.rename({"Stabr": "STATE"}, axis=1)
        df_poverty_county = self._preprocess_x(
            df_poverty, county_fips_list, cols_to_drop_poverty
        )

        self.df_unemployment = self.df_unemployment.rename({"Stabr": "STATE"}, axis=1)
        df_unemployment_county = self._preprocess_x(
            self.df_unemployment, county_fips_list, cols_to_drop_unemployment
        )

        df_features = (
            df_population_county.merge(df_education_county, on=["FIPS_CODE"])
            .merge(df_poverty_county, on="FIPS_CODE")
            .merge(df_unemployment_county, on="FIPS_CODE")
        )

        r = "Join check"
        r += f"\ndf_population_county: {df_population_county.shape}\n"
        r += f"df_education_county: {df_education_county.shape}\n"
        r += f"df_poverty_county: {df_poverty_county.shape}\n"
        r += f"df_unemployment: {df_unemployment_county.shape}\n"
        r += f"df_features: {df_features.shape}\n"
        print("\n %s" % r)

        df_features = df_features.drop("AREA_NAME", axis=1)
        return df_features


    def _preprocess_x(self, df_x, county_fips_list, feats_to_drop=None):

        if feats_to_drop is None:
            feats_to_drop = []
        # Find FIPS column name
        FIPS_name = df_x.filter(regex="FIPS|fips").columns[0]
        df_x = df_x.rename(columns={FIPS_name: "FIPS_CODE"})

        df_x.columns = standard_name_cols(df_x.columns)
        data_to_upper_case = ["STATE", "AREA_NAME"]
        df_x[data_to_upper_case] = upper_consistent(df_x[data_to_upper_case])
        # select only the counties
        df_x_county = self._split_state_county(df_x, county_fips_list)

        if feats_to_drop:
            df_x_county = df_x_county.drop(feats_to_drop, axis=1)
        return df_x_county


    def _split_state_county(self, df_x, county_fips_list):
        # county
        df_x_county = df_x[df_x["FIPS_CODE"].isin(county_fips_list)]
        # Check if all counties were found
        county_found = df_x_county["FIPS_CODE"].unique()
        ## Missing codes correspond to Alaska :
        ## Unlike other states within the United States,
        ## Alaska does not administer its presidential
        ## elections at the county-level but rather at the lower chamber legislative district, or the House District
        ## For now, I drop Alaska
        r = ""
        if len(county_found) != len(county_fips_list):
            county_n_found = [
                county for county in county_fips_list if county not in county_found
            ]
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


# def transform_data(data_inputs_paths):
#     ##################### x #################
#     cols_to_drop_population = ["RURAL_URBAN_CONTINUUM_CODE_2003",
#                                "URBAN_INFLUENCE_CODE_2003",
#                                "URBAN_INFLUENCE_CODE_2013",
#                                "RURAL_URBAN_CONTINUUM_CODE_2013",
#                                ]
#     df_population = read_excel(path_population_x)
#     df_population_county = standard_x(df_population,
#                                       county_fips_list,
#                                       cols_to_drop_population)
#
#     df_education = read_excel(path_education_x)
#     df_education["Area name"] = df_education["Area name"].replace(
#         "Lousiana", "LOUISIANA"
#     )
#
#     cols_to_drop_educ = [
#         "STATE", "AREA_NAME"
#                  "2003_RURAL_URBAN_CONTINUUM_CODE",
#         "2003_URBAN_INFLUENCE_CODE",
#         "2013_RURAL_URBAN_CONTINUUM_CODE",
#         "2013_URBAN_INFLUENCE_CODE"
#     ]
#     df_education_county = standard_x(
#         df_education, county_fips_list, cols_to_drop_educ
#     )
#
#     df_poverty = read_excel(path_poverty_x)
#     df_poverty = df_poverty.rename({"Stabr": "STATE"}, axis=1)
#     # duplicated columns
#     cols_to_drop_poverty = ["STATE", "AREA_NAME",
#                             "RURAL_URBAN_CONTINUUM_CODE_2003",
#                             "URBAN_INFLUENCE_CODE_2003",
#                             "RURAL_URBAN_CONTINUUM_CODE_2013",
#                             "URBAN_INFLUENCE_CODE_2013"
#                             ]
#
#     # "RURAL_URBAN_CONTINUUM_CODE_2013", "URBAN_INFLUENCE_CODE_2013,"
#     df_poverty_county = standard_x(
#         df_poverty, county_fips_list, cols_to_drop_poverty
#     )
#
#     df_unemployment = read_excel(path_unemploymnt_x)
#     df_unemployment = df_unemployment.rename({"Stabr": "STATE"}, axis=1)
#
#     df_unemployment_county = standard_x(
#         df_unemployment, county_fips_list, ["STATE", "AREA_NAME"]
#     )
#
#     del df_population
#     del df_education
#     del df_poverty
#     del df_unemployment
#
#     df_features = (
#         df_population_county.merge(df_education_county, on=["FIPS_CODE"])
#             .merge(df_poverty_county, on="FIPS_CODE")
#             .merge(df_unemployment_county, on="FIPS_CODE")
#     )
#
#     r = "Join check"
#     r += f"\ndf_population_county: {df_population_county.shape}\n"
#     r += f"df_education_county: {df_education_county.shape}\n"
#     r += f"df_poverty_county: {df_poverty_county.shape}\n"
#     r += f"df_unemployment: {df_unemployment_county.shape}\n"
#     r += f"df_features: {df_features.shape}\n"
#     print("\n %s" % r)
#
#     return df_features
