import os
ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, "data/")
#os.makedirs(DATA_DIR, exist_ok=True)

data_inputs_paths = {
    "path_input_y": f"{DATA_DIR}/2020_US_County_Level_Presidential_Results.csv",
    "path_electoral_college": f"{DATA_DIR}/Electoral_College.csv",
    "path_population_x": f"{DATA_DIR}/PopulationEstimates.xls",
    "path_education_x": f"{DATA_DIR}/Education.xls",
    "path_poverty_x": f"{DATA_DIR}/PovertyEstimates.xls",
    "path_unemploymnt_x": f"{DATA_DIR}/Unemployment.xls",
    "path_prepro_x": f"{DATA_DIR}/prepro_x.csv",
    "path_prepro_y": f"{DATA_DIR}/prepro_y.csv"
}

model_config = {
    "class_weight": "balanced", #None

}