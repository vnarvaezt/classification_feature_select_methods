from conf.config import data_inputs_paths, model_config
from src.models.split_scale import SplitandScale
from src.models.modelling import ModelData
from src.preprocessing.preprocessing_x import PreprocessData
from src.preprocessing.transform_data import TransformData

if __name__ == '__main__':
    # # initial cleaning and join all x
    # td = TransformData()
    # all_x = td.transform_data(data_inputs_paths)
    # all_x = all_x.set_index(['FIPS_CODE', 'STATE'])
    #
    # # preprocessing
    # prepro = PreprocessData(data_inputs_paths)
    # X_prepro = prepro.run_preprocessing(
    #     all_x.copy(deep=True),
    #     do_save=True,
    #     feat_to_keep=["CIVILIAN_LABOR_FORCE_2019"],
    #     abs_corr_thresh=0.9
    # )

    # split train and test
    ss = SplitandScale(data_inputs_paths)
    X_train, X_test, y_train, y_test = ss.run_split_scale()
    # modellisation
    md = ModelData(X_train, X_test, y_train["TARGET"], y_test['TARGET'], model_config)
    md.run_modelling()