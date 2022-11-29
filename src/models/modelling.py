import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from numpy import mean
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.models.utils import (calculate_precision_recall_curve, evaluate_model,
                              feature_selection, make_confusion_matrix)


class ModelData:
    def __init__(self, Xtrain, Xtest, ytrain, ytest, model_params):
        self.models_for_prediction = dict.fromkeys(
            model_params["models_for_pred"], "models"
        )
        self.feature_select = model_params["reduce_features"]
        self.class_weight = model_params["class_weight"]
        self.seed = model_params["seed"]
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest

    def get_class_weight(self):
        # define class weight
        if self.class_weight is not None:
            model_class_weight = compute_class_weight(
                self.class_weight, classes=np.unique(self.ytrain), y=self.ytrain
            )
        return dict(enumerate(model_class_weight))

    def learning_logit(self, X_train, y_train, X_test, y_test, weight):

        lr = LogisticRegression(
            class_weight=weight, max_iter=1000, random_state=self.seed
        )
        lr.fit(X_train, y_train)
        lr_scores = evaluate_model(lr, X_train, y_train, self.seed)
        print(
            "Logistic classification: F1 score (train with k fold) %.3f"
            % mean(lr_scores)
        )

        calculate_precision_recall_curve(
            lr, X_train, y_train, X_test, y_test, "logit_baseLine"
        )

        make_confusion_matrix(y_test, lr.predict(X_test), "testset_logit_baseline")
        return lr.predict(X_test)

    def run_modelling(self):

        cw = self.get_class_weight()
        hp = self.model_hyperparameters(cw)
        models = set(self.models_for_prediction.keys()) & set(hp.keys())

        # base line model
        y_pred_logit = self.learning_logit(
            self.Xtrain, self.ytrain, self.Xtest, self.ytest, weight=cw
        )

        # feature selection method: boruta, CVRFE, kbest, None
        estimation_method = RandomForestClassifier(
            n_jobs=-1, class_weight=cw, max_depth=3
        )
        # filter explanatory vars if specify
        explanatory_vars = feature_selection(
            self.feature_select, self.Xtrain, self.ytrain, estimator=estimation_method
        )

        results_train, results_test = list(), list()
        model_predictions = self.ytest.copy(deep=True)
        model_predictions = model_predictions.reset_index()
        # loop for each model in config
        for model_name in models:
            # fit model
            fit_m = self.fit_model(
                self.Xtrain[explanatory_vars], self.ytrain, hp[model_name]
            )
            # compute metrics
            print(f"\n ...Training model: %s" % model_name)
            scores_train = evaluate_model(
                fit_m, self.Xtrain[explanatory_vars], self.ytrain, self.seed
            )
            # predict y on test set
            y_pred_test = fit_m.predict(self.Xtest[explanatory_vars])

            # save prediction on df
            model_predictions["y_pred_test" + str(model_name)] = y_pred_test

            # compute scores
            F1_test = f1_score(self.ytest, y_pred_test)
            results_train.append(scores_train)
            results_test.append(F1_test)

            print(
                "f1_score: %.3f (train, using RepeatedStratifiedKFold), %.3f (test)"
                % (mean(scores_train), F1_test)
            )

            calculate_precision_recall_curve(
                fit_m,
                self.Xtrain[explanatory_vars],
                self.ytrain,
                self.Xtest[explanatory_vars],
                self.ytest,
                model_name,
            )
            make_confusion_matrix(self.ytest, y_pred_test, f"testset_{model_name}")

        model_predictions["y_pred_test_logit"] = y_pred_logit
        # model_predictions = model_predictions.reset_index()

        print(model_predictions)
        return model_predictions

    def model_hyperparameters(self, weight):
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=self.seed, class_weight=weight
            ),
            "SVM": SVC(
                kernel="linear",
                gamma="scale",
                class_weight=weight,
                probability=True,
                random_state=self.seed,
            ),
            "BaggingClassifier": BaggingClassifier(
                n_estimators=300, random_state=self.seed
            ),
            "BalancedBaggingClassifier": BalancedBaggingClassifier(
                random_state=self.seed
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=300,
                class_weight=weight,
                max_depth=3,
                random_state=self.seed,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=300, max_depth=3, random_state=self.seed
            ),
            "XGBClassifier": XGBClassifier(
                n_estimators=80,
                eval_metric="logloss",
                random_state=self.seed,
                learning_rate=0.1,
            ),
        }

        return models

    def fit_model(self, X, y, estimator, **kwargs):
        model = Pipeline(steps=[("estimator", estimator)])
        return model.fit(X, y, **kwargs)
