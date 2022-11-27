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
    #     feat_to_keep=["CIVILIAN_LABOR_FORCE_2019",
    #     'HIGH_SCHOOL_DIPLOMA_ONLY__2015_19",
    #     'LESS_THAN_A_HIGH_SCHOOL_DIPLOMA__2015_19',
    #     'SOME_COLLEGE_OR_ASSOCIATE_S_DEGREE__2015_19,
    #     'PERCENT_OF_ADULTS_WITH_A_HIGH_SCHOOL_DIPLOMA_ONLY__2000',
    #     'CI90LB517_2019', 'CI90LB017P_2019',
    #     'UNEMPLOYED_2019', 'MED_HH_INCOME_PERCENT_OF_STATE_TOTAL_2019',
    #     'MED_HH_INCOME_PERCENT_OF_STATE_TOTAL_2019',
    #     'Metro_2013'],
    #     abs_corr_thresh=0.9
    # )

    # split train and test
    ss = SplitandScale(data_inputs_paths)
    X_train, X_test, y_train, y_test = ss.run_split_scale()
    # modelling
    md = ModelData(X_train, X_test, y_train["TARGET"], y_test['TARGET'], model_config)
    md.run_modelling()