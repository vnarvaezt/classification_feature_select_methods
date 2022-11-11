from conf.config import data_inputs_paths as data_inputs
from utils.preprocessing_x import PreprocessData
from utils.transform_data import transform_data
from utils.tools import split_train_test

x_raw = transform_data(data_inputs)

#split_train_test(x_raw, test_size=0.2, random_state=42)
prepro = PreprocessData()
feats_to_keep = ["STATE", "AREA_NAME", "FIPS_CODE"]
x_preprocess = prepro.run_preprocessing(
    x_raw.copy(deep=True),
    feat_to_keep=feats_to_keep)

path_to_save_x = data_inputs["path_prepro_x"]

#TODO
# RURAL_URBAN_CONTINUUM_CODE_2013', 'URBAN_INFLUENCE_CODE_2013 sont en double: trouver dans quel df ils ont
# correlation entre des variables categorielles
## --> comment supprimer des variables categorielles trop correles entre elles?
## --> transfo en dummies
# standardisation
## jointure X, Y
## split train/test
## base line model: logit

