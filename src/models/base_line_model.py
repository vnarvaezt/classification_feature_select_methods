from conf.config import data_inputs_paths as data_inputs
from src.preprocessing.preprocessing_x import PreprocessData
from src.preprocessing.transform_data import transform_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
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
df_y = df_y[df_y["STATE_NAME"] != "DISTRICT OF COLUMBIA"]
# join x and y
df_x_y = pd.merge(df_x,
                  df_y,
                  left_on="FIPS_CODE",
                  right_on="COUNTY_FIPS",
                  how="inner")
index_cols = ["STATE_NAME",
              "AREA_NAME"]
df_x_y = df_x_y.set_index(["STATE_NAME", "COUNTY_FIPS"], drop=True)

X_prepro = df_x_y.drop(["TARGET", "AREA_NAME"], axis=1)
y_prepro = df_x_y[["TARGET"]]
# split train, test
X_train, X_test, y_train, y_test = train_test_split(
    X_prepro,
    y_prepro,
    test_size=0.2,
    random_state=42,
    stratify=y_prepro["TARGET"] # todo: stratify par state?
    )

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

proba_train_pred = clf.predict_proba(X_train)
mean_acc_train = clf.score(X_train, y_train)

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
# and plot the result
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X_train.iloc[:, 100].values, y_train, color="black", zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_)
plt.plot(X_test, loss, color="red", linewidth=3)
plt.savefig('my_plot.png')

#TODO
# RURAL_URBAN_CONTINUUM_CODE_2013', 'URBAN_INFLUENCE_CODE_2013 sont en double: trouver dans quel df ils ont
# correlation entre des variables categorielles
## completer l'analyse explo
## --> comment supprimer des variables categorielles trop correles entre elles?
## --> transfo en dummies
# standardisation
## jointure X, Y
## split train/test
## base line model: logit

# %load ../09_utils/cramer_v.py
# Charger la fonction du calcul de V de Cramer pour un dataframe
# calcul V de Cramer pour deux variables

# Package de manipulation des tableaux et dataframe
import pandas as pd
import numpy as np
# Typing des fonctions
from typing import List

# Package pour analyse statistique
import scipy.stats as ss
def cramer_v_coeff(x: List, y: List) -> float:
    """Cette fonction permet de calculer le V de
    Cramer entre deux varaibles catégorielles.

    Args:
        x : Le vecteur de variable x.
        y : Le vecteur de variable y.

    Returns:
        float: La valeur V de cramer.
    """
    # Virer les NAs
    complete_cases = x.isna() * y.isna()
    x = x[~complete_cases]
    y = y[~complete_cases]

    # Calcul du Khi-deux max (dénomimateur du V de Cramer)
    n = len(x)
    khi2_max = n * min(len(x.value_counts()), len(y.value_counts())) - 1

    # Calcul du khi-deux (numérateur du V de Cramer)
    conf_matrix = pd.crosstab(x, y)
    khi2 = ss.chi2_contingency(observed=conf_matrix, correction=True)

    # Calcul V de Cramer et récupération p_value associée
    cramer = round(np.sqrt(khi2[0] / khi2_max), 4)
    p_value = khi2[1]

    return cramer, p_value


# calcul V de Cramer pour un dataframe
# Laisser les étudiants boucler eux mêmes
def compute_cramer_v(data: pd.DataFrame) -> pd.DataFrame:
    """Calculer le V de cramer pour un dataframe.

    Args:
        data: Jeu de données sur lequel on souhaite
        calculer le V de Cramer.

    Returns:
        DataFrame contenant les différents V de Cramer.
    """
    ncols = data.shape[1]
    cols = data.columns
    cramer_matrix = np.eye(ncols)
    for j in range(ncols - 1):
        for i in range(j + 1, ncols):
            cramer_matrix[[i, j], [j, i]] = cramer_v_coeff(
                x=data.iloc[:, j],
                y=data.iloc[:, i]
            )[0]
    cramer_matrix = pd.DataFrame(cramer_matrix, columns=cols, index=cols)
    return cramer_matrix

cramer_matrix = compute_cramer_v(x_preprocess.select_dtypes(exclude=np.number))