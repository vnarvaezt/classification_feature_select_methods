import os

ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, "data/")
# os.makedirs(DATA_DIR, exist_ok=True)

data_inputs_paths = {
    "path_input_y": f"{DATA_DIR}/2020_US_County_Level_Presidential_Results.csv",
    "path_electoral_college": f"{DATA_DIR}/Electoral_College.csv",
    "path_population_x": f"{DATA_DIR}/PopulationEstimates.xls",
    "path_education_x": f"{DATA_DIR}/Education.xls",
    "path_poverty_x": f"{DATA_DIR}/PovertyEstimates.xls",
    "path_unemploymnt_x": f"{DATA_DIR}/Unemployment.xls",
    "path_prepro_x": f"{DATA_DIR}/prepro_x.csv",
    "path_prepro_y": f"{DATA_DIR}/prepro_y.csv",
    "path_results_model": f"{DATA_DIR}/models_test_predictions.csv",
}

STEPS = ["PREPROCESSING", "MODELLING"]

model_config = {
    "class_weight": "balanced",  # None
    "reduce_features": "boruta",
    "models_for_pred": [
        "SVM",
        # "RandomForestClassifier"
    ],
    "seed": 1234,
    "feats_to_keep": [
        "POP_ESTIMATE_2019",
        "DEATHS_2019",
        "DOMESTIC_MIG_2019",
        "CIVILIAN_LABOR_FORCE_2019",
        "HIGH_SCHOOL_DIPLOMA_ONLY__2015_19",
        "LESS_THAN_A_HIGH_SCHOOL_DIPLOMA__2015_19",
        "SOME_COLLEGE_OR_ASSOCIATE_S_DEGREE__2015_19",
        "PERCENT_OF_ADULTS_WITH_A_HIGH_SCHOOL_DIPLOMA_ONLY__2000",
        "CI90LB517_2019",
        "CI90LB017P_2019",
        "UNEMPLOYED_2019",
        "MED_HH_INCOME_PERCENT_OF_STATE_TOTAL_2019",
        "MED_HH_INCOME_PERCENT_OF_STATE_TOTAL_2019",
        "Metro_2013",
    ],
}
