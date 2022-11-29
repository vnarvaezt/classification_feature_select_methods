import argparse

from conf.config import STEPS, data_inputs_paths, model_config
from src.models.modelling import ModelData
from src.models.split_scale import SplitandScale
from src.preprocessing.preprocessing_x import PreprocessData
from src.preprocessing.transform_data import TransformData


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        help=f"First step (included). Possible values: {', '.join(STEPS)}",
        required=False,
        default=STEPS[0],
    )
    parser.add_argument(
        "--end",
        help=f"Last step (included). Possible values: {', '.join(STEPS)}",
        required=False,
        default=STEPS[-1],
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = vars(parse_args())

    start_step = arguments["start"]
    end_step = arguments["end"]

    steps = STEPS[STEPS.index(start_step) : STEPS.index(end_step) + 1]

    if "PREPROCESSING" in steps:
        print(">>> Running preprocessing <<<")
        # initial cleaning and join all x
        td = TransformData()
        all_x = td.transform_data(data_inputs_paths)
        all_x = all_x.set_index(["FIPS_CODE", "STATE"])

        # preprocessing
        prepro = PreprocessData(data_inputs_paths)
        X_prepro = prepro.run_preprocessing(
            all_x.copy(deep=True),
            do_save=True,
            feat_to_keep=model_config["feats_to_keep"],
            abs_corr_thresh=0.9,
        )

    if "MODELLING" in steps:
        print(">>> Running modelling <<<")
        # split train and test
        ss = SplitandScale(data_inputs_paths)
        X_train, X_test, y_train, y_test = ss.run_split_scale()
        # modelling
        md = ModelData(
            X_train, X_test, y_train["TARGET"], y_test["TARGET"], model_config
        )
        md.run_modelling()
