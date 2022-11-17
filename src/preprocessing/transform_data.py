from src.tools.tools import (
    standard_name_cols,
    upper_consistent,
    read_excel,
)
import pandas as pd
import numpy as np


def transform_data(data_inputs_paths):

    path_input_y = data_inputs_paths["path_input_y"]
    path_population_x = data_inputs_paths["path_population_x"]
    path_education_x = data_inputs_paths["path_education_x"]
    path_poverty_x = data_inputs_paths["path_poverty_x"]
    path_unemploymnt_x = data_inputs_paths["path_unemploymnt_x"]
    path_save_y = data_inputs_paths["path_prepro_y"]

    # import y
    df_presidential_2020 = pd.read_csv(path_input_y,
                                       converters={"county_fips": str})
    df_presidential_2020.columns = standard_name_cols(df_presidential_2020.columns)
    # upper case data for these columns
    cols_to_upper_case = ["STATE_NAME", "COUNTY_NAME"]
    df_presidential_2020[cols_to_upper_case] = upper_consistent(
        df_presidential_2020[cols_to_upper_case]
    )
    df_presidential_2020["TARGET"] = np.where(df_presidential_2020["VOTES_GOP"] < df_presidential_2020["VOTES_DEM"], 1, 0)

    df_target = df_presidential_2020[["COUNTY_FIPS", "STATE_NAME", "TARGET"]]
    df_target = df_target[df_target["STATE_NAME"] != "DISTRICT OF COLUMBIA"]
    df_target.to_csv(path_save_y, sep=";", index=False)
    # TODO: delete ALASKA results
    # county code list
    county_fips_list = df_target["COUNTY_FIPS"].unique()


##################### x #################
    cols_to_drop_population = ["RURAL_URBAN_CONTINUUM_CODE_2003",
                               "URBAN_INFLUENCE_CODE_2003",
                               "URBAN_INFLUENCE_CODE_2013",
                               "RURAL_URBAN_CONTINUUM_CODE_2013",
                               ]
    df_population = read_excel(path_population_x)
    df_population_county = standard_x(df_population,
                                      county_fips_list,
                                      cols_to_drop_population)


    df_education = read_excel(path_education_x)
    df_education["Area name"] = df_education["Area name"].replace(
        "Lousiana", "LOUISIANA"
    )
    df_education_county = standard_x(
        df_education, county_fips_list, ["STATE", "AREA_NAME"]
    )

    df_poverty = read_excel(path_poverty_x)
    df_poverty = df_poverty.rename({"Stabr": "STATE"}, axis=1)
    # duplicated columns
    cols_to_drop = ["STATE", "AREA_NAME",
                    "RURAL_URBAN_CONTINUUM_CODE_2003",
                    "URBAN_INFLUENCE_CODE_2003",
                    ]

    #"RURAL_URBAN_CONTINUUM_CODE_2013", "URBAN_INFLUENCE_CODE_2013,"
    df_poverty_county = standard_x(
        df_poverty, county_fips_list, cols_to_drop
                                   )

    df_unemployment = read_excel(path_unemploymnt_x)
    df_unemployment = df_unemployment.rename({"Stabr": "STATE"}, axis=1)

    df_unemployment_county = standard_x(
        df_unemployment, county_fips_list, ["STATE", "AREA_NAME"]
    )

    del df_population
    del df_education
    del df_poverty
    del df_unemployment

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

    return df_features


def standard_x(df_x, county_fips_list, feats_to_drop=[]):
    """

    :rtype: dataframe
    """
    # Find FIPS column name
    FIPS_name = df_x.filter(regex="FIPS|fips").columns[0]
    df_x = df_x.rename(columns={FIPS_name: "FIPS_CODE"})

    df_x.columns = standard_name_cols(df_x.columns)
    data_to_upper_case = ["STATE", "AREA_NAME"]
    df_x[data_to_upper_case] = upper_consistent(df_x[data_to_upper_case])
    # select only the counties
    df_x_county = _split_state_county(df_x, county_fips_list)
    if feats_to_drop:
        df_x_county = df_x_county.drop(feats_to_drop, axis=1)
    return df_x_county


def _split_state_county(df_x, county_fips_list):

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
