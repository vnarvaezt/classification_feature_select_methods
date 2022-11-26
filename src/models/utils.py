import os
import numpy as np
from sklearn.metrics import make_scorer, f1_score, precision_recall_curve, confusion_matrix
from sklearn.metrics import auc
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from matplotlib import pyplot
from boruta import BorutaPy


####################
# model evaluation #
####################
# evaluate a given model using cross-validation
def evaluate_model(model, X, y, seed):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
    scores = cross_val_score(model, X, y, scoring=make_scorer(f1_score), cv=cv, n_jobs=-1)
    return scores


def calculate_precision_recall_curve(model, trainX, trainy, testX, testy):
    # metrics
    md_precision_train, md_recall_train, md_f1_train, md_auc_train = _metrics(model, trainX, trainy)
    md_precision_test, md_recall_test, md_f1_test, md_auc_test = _metrics(model, testX, testy)

    print('\n >>f1 score: %.3f (train) %.3f (test)' % (md_f1_train, md_f1_test))
    print('>>auc: %.3f (train) %.3f (test)' % (md_auc_train, md_auc_test))

    # plot the precision-recall curves
    no_skill = len(testy[testy == 1]) / len(testy)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(md_recall_train, md_precision_train, marker='.', label='train')
    pyplot.plot(md_recall_test, md_precision_test, marker='.', label='test')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    return (pyplot)


def _metrics(model, X, y):
    # predict proba
    md_proba = model.predict_proba(X)
    # keep only positive outcome
    md_proba = md_proba[:, 1]
    # predict class value
    y_hat = model.predict(X)

    # calculate precision and recall for each threshold
    md_precision, md_recall, _ = precision_recall_curve(y, md_proba)
    # calculate f1 and auc
    md_f1, md_auc = f1_score(y, y_hat), auc(md_recall, md_precision)
    metrics = md_precision, md_recall, md_f1, md_auc

    return metrics


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
    fig.write_html(f"images/confusion_matrix{alias}.html")
    fig.show()


####################
# feature selection #
####################
# SélectionneZ les k variables numériques les plus liées à la cible
def feature_selection_KBest(X, y, nb_features) -> list:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    # sur le base du critère d'information mutuelle.
    mic_selector = SelectKBest(score_func=mutual_info_classif, k=nb_features)
    mic_selector.fit(X[num_cols], y)

    # Liste des variables retenues
    mic_features = np.array(num_cols)[mic_selector.get_support()].tolist()
    print(f"%s most important features according to <<Kbest method>>" % nb_features)
    print(mic_features)

    return mic_features


def compute_RFECV(X, y, estimator, min_features):
    # Instanciation de la RFECV()
    rfecv_selector = RFECV(estimator=estimator,
                           min_features_to_select=min_features,
                           scoring=make_scorer(f1_score),
                           n_jobs=-1,
                           step=1,
                           cv=5
                           )

    # Train model
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
    fig.show()
    fig.write_image("images/RFE_random_forest.png")

    # Récupérer la liste des variables issues de RFECV et transformer là en liste
    rfe_features = rfecv_selector.get_feature_names_out().tolist()
    return rfe_features


def compute_boruta(X_train, y_train, estimator, n_features) -> list:
     # Compléter les paramètres de la méthode boruta
    set_seed = 1204
    boruta_selector = BorutaPy(estimator, n_estimators=n_features, verbose=0, random_state=set_seed)
    # Entraîner boruta
    X_boruta = X_train.values
    y_boruta = y_train.values.ravel()
    boruta_selector.fit(X_boruta, y_boruta)
    boruta_features = X_train.columns[boruta_selector.support_].tolist()
    return boruta_features


def feature_selection(fs_method, Xtrain, ytrain, estimator) -> list:
    if fs_method not in ("kbest", "rfecv", "boruta", None):
        msg = "Wrong reduce_features value"
        print(msg)
        raise ValueError(msg)

    if fs_method == "kbest":
        # removes median and scales data according to quantile range:
        features = feature_selection_KBest(Xtrain, ytrain, estimator)

    elif fs_method == "rfecv":
        features = compute_RFECV(Xtrain, ytrain, estimator=estimator, min_features=20)

    elif fs_method == "boruta":
        features = compute_boruta(Xtrain, ytrain, estimator, n_features="auto")

    elif fs_method is None:
        features = Xtrain.columns.tolist()
    print(f"Using feature selection method: {fs_method}")
    return features