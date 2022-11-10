from conf.config import data_inputs_paths as data_inputs
from utils.preprocessing_x import PreprocessData
from utils.transform_data import transform_data

from utils.tools import split_train_test


x_raw = transform_data(data_inputs)
split_train_test(x_raw, test_size=0.2, random_state=42)
prepro = PreprocessData()
feats_to_keep = ["STATE", "AREA_NAME", "FIPS_CODE"]
x_preprocess = prepro.run_preprocessing(
    x_raw.copy(deep=True),
    feat_to_keep=feats_to_keep)

path_to_save_x = data_inputs["path_prepro_x"]