import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from conf.config import data_inputs_paths as data_inputs

from src.preprocessing.preprocessing_x import PreprocessData
from src.preprocessing.transform_data import transform_data

x_raw = transform_data(data_inputs)

prepro = PreprocessData()
feats_to_keep = ["STATE", "AREA_NAME", "FIPS_CODE"]
x_raw = x_raw.set_index(["AREA_NAME", "FIPS_CODE"])
x_preprocess = prepro.run_preprocessing(
    x_raw.copy(deep=True),
    feat_to_keep=feats_to_keep,
    do_save=True
)

############# script starts here #######

path_x = data_inputs["path_prepro_x"]
path_y = data_inputs["path_prepro_y"]
df_x = pd.read_csv(path_x, sep=";")
df_y = pd.read_csv(path_y, sep=";")
# join x and y
df_x_y = pd.merge(df_x,
                  df_y,
                  left_on="FIPS_CODE",
                  right_on="COUNTY_FIPS",
                  how="inner")
index_cols = ["STATE_NAME",
              "AREA_NAME"]
df_x_y = df_x_y.set_index(["STATE_NAME", "COUNTY_FIPS"], drop=True)
# FIXME : county code is repeated after join
df_x_y = df_x_y.drop("FIPS_CODE", axis=1)

X_prepro = df_x_y.drop(["TARGET", "AREA_NAME"], axis=1)
y_prepro = df_x_y[["TARGET"]]
# split train, test
X_train, X_test, y_train, y_test = train_test_split(
    X_prepro,
    y_prepro,
    test_size=0.2,
    random_state=42,
    stratify=y_prepro["TARGET"]  # FIXME: stratify par state?
)
print("\n\n Repartition de la target:")
print(y_train.TARGET.value_counts(normalize=True))

##################### base line model ###########
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from conf.config import model_config
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# Package xgboost
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
import time

# define random state
seed = 42

# define class weights
class_weight = model_config["class_weight"]
cw = compute_class_weight(
    class_weight,
    classes=np.unique(y_train["TARGET"]),
    y=y_train["TARGET"],
)
cw_dict = dict(enumerate(cw))


# option 1 : definir le nb de features et applique decision trees
# get a list of models to evaluate
def get_models(weights=cw_dict, n_features=50):
    models = dict()
    # lr
    rfe = RFE(estimator=LogisticRegression(class_weight=weights), n_features_to_select=n_features)
    model = DecisionTreeClassifier()
    models['lr'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # cart
    rfe = RFE(estimator=DecisionTreeClassifier(class_weight=weights), n_features_to_select=n_features)
    model = DecisionTreeClassifier()
    models['cart'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # rf
    rfe = RFE(estimator=RandomForestClassifier(class_weight=weights), n_features_to_select=n_features)
    model = DecisionTreeClassifier()
    models['rf'] = Pipeline(steps=[('s', rfe), ('m', model)])
    # gbm
    rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=n_features)
    model = DecisionTreeClassifier()
    models['gbm'] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
    scores = cross_val_score(model, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
    return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
start = time.time()
for name, model in models.items():
    scores = evaluate_model(model, X_train, y_train["TARGET"])
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
end = time.time()
delta = (end - start) / 60
print(f"modelisation took {delta:.2} minutes")

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.savefig("images/model_performance_RFE")

# option 2: compute RFECV with a random forest and then use the top vars to run a model, test several models
from sklearn.feature_selection import VarianceThreshold, RFECV, SelectKBest, chi2, mutual_info_classif
# Package pour représentation graphique
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import plotly.figure_factory as ff
import plotly.graph_objects as go


def compute_RFECV(X, y, estimator=RandomForestClassifier(class_weight=cw_dict)):
    # Instanciation de la RFECV()
    rfecv_selector = RFECV(estimator=estimator,
                           min_features_to_select=10,
                           scoring='f1_macro',
                           n_jobs=-1,
                           step=1,
                           cv=5
                           )

    # Entraîner le modèle
    rfecv_selector.fit(X, y)
    mean_test_f1_score = rfecv_selector.cv_results_["mean_test_score"]
    n_features_selected = rfecv_selector.n_features_
    min_features_to_select = rfecv_selector.min_features_to_select
    n_features_in_rfecv = rfecv_selector.n_features_in_

    fig = px.line(y=mean_test_f1_score,
                  x=range(min_features_to_select, len(mean_test_f1_score) + min_features_to_select),
                  labels={
                      "x": "Nombre de variables sélectionnées",
                      "y": "Score moyen de cross-validaion"
                  },
                  title=f"Résultats RFECV avec {rfecv_selector.cv} k-fold"
                  )
    fig.add_vline(x=n_features_selected, line_width=3, line_dash="dash", line_color="#011C5D")
    fig.add_vrect(x0=30, x1=n_features_in_rfecv, line_width=0, fillcolor="#D38F00", opacity=0.2)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/RFE_random_forest.png")

    # Récupérer la liste des variables issues de RFECV et transformer là en liste
    rfe_features = rfecv_selector.get_feature_names_out().tolist()
    return rfe_features


models = {
    "LogisticRegression":
        LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight=cw_dict
        ),
    "BaggingClassifier":
        BaggingClassifier(
            n_estimators=300,
            random_state=seed
        ),
    "RandomForestClassifier":
        RandomForestClassifier(
            n_estimators=300,
            class_weight=cw_dict,
            max_depth=3,
            random_state=seed,
        ),
    "GradientBoosting":
        GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            random_state=seed
        ),
    "XGBClassifier":
        XGBClassifier(
            use_label_encoder=False,
            n_estimators=80,
            eval_metric='logloss',
            random_state=seed,
            learning_rate=0.1
        )
}


def fit_model(X, y, estimator, **kwargs):
    model = Pipeline(
        steps=[
            ("estimator", estimator)
        ]
    )

    return model.fit(X, y, **kwargs)


# def metrics(y_true, y_pred):
#     accuracy = accuracy_score(y_true,
#                               y_pred)
#     f1_s = f1_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     r = ""
#     r += f"Accuracy {accuracy:.2%}\n"
#     r += f"Precision {precision:.2%}\n"
#     r += f"Recall {recall:.2%}\n"
#     r += f"F1 score {f1_s:.2%}\n"
#     print("\n\n%s" % r)
#     return accuracy, f1_s, recall, precision
#
#
# def model_evaluation(model, x_train, y_train, x_test, y_test):
#     print(len(x_train.columns))
#     # Prédiction sur le jeu de données test
#     y_train_pred = model.predict(x_train)
#     y_test_pred = model.predict(x_test)
#     # metriques de performance
#     accuracy_train, f1_train, recall_train, precision_train = metrics(y_train, y_train_pred)
#     accuracy_test, f1_test, recall_test, precision_test = metrics(y_test, y_test_pred)
#     return f1_train, f1_test

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
    scores = cross_val_score(model, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
    return scores

# fit random forest with RFE to select variables
rfe_RF_features = compute_RFECV(X_train, y_train["TARGET"])

results_train, results_test, names = list(), list(), list()
# Boucler sur chacun des modèles
for name, model in models.items():
    fit_m = fit_model(X_train[rfe_RF_features], y_train["TARGET"], model)
    scores_train = evaluate_model(fit_m, X_train[rfe_RF_features], y_train["TARGET"])
    y_pred_test = fit_m.predict(X_test[rfe_RF_features])
    F1_test = f1_score(y_test["TARGET"], y_pred_test)
    results_train.append(scores_train)
    results_test.append(F1_test)
    names.append(name)
    print('>%s %.3f (train) %.3f (test)' % (name, mean(scores_train), F1_test))
# TODO: add precision vs recall curves

# Définitions des hyperparamètres
params_grid = {'gamma': [0, 0.1, 0.3],
              'learning_rate': [0.01, 0.1, 0.3],
              'max_depth': [3, 5, 6],
              'n_estimators': [100, 200, 300],
              'reg_alpha': [0, 0.1, 0.4, 1],
              'reg_lambda': [0, 0.1, 0.4, 1]
             }

# Instanciation du classifieur
xgb_estimator = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=seed
)

# Recherche du meilleure hyperparamètre
cv_grid_search = GridSearchCV(
    estimator=xgb_estimator,
    param_grid=params_grid,
    scoring='f1_macro',
    n_jobs=-1,
    cv=5,
    verbose=2
)

# Entrainement et évaluation du modèle
randSearch_fit = fit_model(X_train[rfe_RF_features], y_train["TARGET"], cv_grid_search)
y_pred_train = randSearch_fit.predict(X_train[rfe_RF_features])
y_pred_test = randSearch_fit.predict(X_test[rfe_RF_features])
F1_train = f1_score(y_train["TARGET"], y_pred_train)
F1_test = f1_score(y_test["TARGET"], y_pred_test)
print('> %.3f (train) %.3f (test)' % (F1_train, F1_test))

print(cv_grid_search.best_estimator_)






def make_confusion_matrix(y_true, y_pred, alias, labels=None):
    if labels is None:
        labels = ['Republicans', 'Democrats']
    z = confusion_matrix(y_true, y_pred)
    x = labels
    y = labels

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='RdBu')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      yaxis={"title": "Real value"},

                      )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=12),
                            x=0.5,
                            y=-0.15,
                            text="Predicted value",
                            showarrow=False,
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.write_html(f"confusion_matrix{alias}.html")

#
#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.special import expit
# # and plot the result
# plt.figure(1, figsize=(4, 3))
# plt.clf()
# plt.scatter(X_train.iloc[:, 100].values, y_train, color="black", zorder=20)
# X_test = np.linspace(-5, 10, 300)
#
# loss = expit(X_test * clf.coef_ + clf.intercept_)
# plt.plot(X_test, loss, color="red", linewidth=3)
# plt.savefig('my_plot.png')
