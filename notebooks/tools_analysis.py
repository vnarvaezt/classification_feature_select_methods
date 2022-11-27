# Package de manipulation des tableaux et dataframe
# Typing des fonctions
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
# Package pour analyse statistique
import scipy.stats as ss
import seaborn as sns


def check_nan(data):
    nb_lines = data.shape[0]
    nb_columns = data.shape[1]

    nb_nan = data.isna().sum() / nb_lines
    part_nan = nb_nan[nb_nan > 0]

    r = ""
    if not part_nan.empty:
        r += "There are features with NAN values :\n"
        r += f"Number of features with NAN = {len(part_nan.keys())}\n"
        r += f"Number of values with NAN = {sum(data.isna().sum())}\n"
        r += "=" * 88 + "\n"
        r += "[Feature] <--- has empty values ---> [percentage of empty values]\n"
        r += "=" * 88 + "\n"
        for feat, pct_f in part_nan.items():
            r += f"{feat:30s}  <--->  {pct_f:.2%}\n"
        r += "=" * 88 + "\n"
    else:
        r += "There are none features with NAN values\n"
    print("\n\n%s" % r)


def categorical_analysis(X, y, columns):
    y = y.replace({0: "REPUBLICANS", 1: "DEMOCRATS"})

    for col in columns:
        # Construction tableau de contingence
        df_crosstab = pd.crosstab(index=X[col], columns=y, normalize="index")
        df_crosstab = df_crosstab.reset_index()
        df_crosstab = df_crosstab.sort_values("DEMOCRATS", ascending=False)
        # Représentation graphique

        fig_top_countys = px.bar(
            df_crosstab,
            x=["DEMOCRATS", "REPUBLICANS"],
            y=col,
            title=f"{col} -county results. US 2020 elections",  # change title
            color_discrete_sequence=px.colors.qualitative.G10,
            labels={"value": "%", "COUNTY_NAME": "County", "variable": "Party"},
        )
        fig_top_countys.add_vline(
            x=0.5,
            line_width=3,
            line_dash="dash",
            line_color="green",
            annotation_text="50%",
            annotation_position="bottom right",
        )
        fig_top_countys.show()


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
                x=data.iloc[:, j], y=data.iloc[:, i]
            )[0]
    cramer_matrix = pd.DataFrame(cramer_matrix, columns=cols, index=cols)
    return cramer_matrix


## Linear correlation
def _compute_triangular_matrix(person_matrix, abs_corr_thresh):
    # Filtrer sur le matrice triangulaire inférieure (ou supérieure)
    mask = np.triu(np.ones_like(person_matrix))

    # Visualisation de la matrice de corrélation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(
        f"Correlation after initial preprocessing (only variables with corr()<{abs_corr_thresh:.0%})"
    )
    sns.heatmap(
        person_matrix,
        mask=mask,
        annot=False,
        linewidths=0.5,
        ax=ax,
        vmin=-abs_corr_thresh,
        vmax=abs_corr_thresh,
    )


def numeric_corr_analysis(
    df_prep, alias, y, threshold=0.3, abs_corr_thresh=0.8, width_=800, height_=800
):
    # join with Y
    df_prep_y = pd.merge(
        df_prep, y, left_on="FIPS_CODE", right_on="COUNTY_FIPS", how="inner"
    )
    df_prep_y["WINNER_PARTY"] = np.where(
        df_prep_y["WINNER_DEMOCRATS"] == 1, "DEMOCRATS", "REPUBLICANS"
    )

    person_matrix_num = df_prep_y.select_dtypes(np.number).corr()

    person_matrix_democrats = person_matrix_num[["WINNER_DEMOCRATS"]].sort_values(
        by="WINNER_DEMOCRATS", ascending=False
    )
    person_matrix_democrats = person_matrix_democrats.loc[
        person_matrix_democrats["WINNER_DEMOCRATS"] != 1
    ]

    corr_y = px.bar(
        person_matrix_democrats,
        x=person_matrix_democrats.index,
        y="WINNER_DEMOCRATS",
        title=f"Correlation between the target and <<{alias}>> variables",  # change title
        # color_discrete_sequence=px.colors.qualitative.G10,
        labels={
            "index": "",
            "WINNER_DEMOCRATS": "Correlation",
            #        "variable":"Party"
        },
        width=width_,
        height=height_,
    )

    corr_y.add_hline(
        y=threshold,
        line_width=1,
        line_dash="dash",
        line_color="black",
        annotation_text=f"{threshold:.0%}",
        annotation_position="bottom right",
    )
    corr_y.add_hline(
        y=-threshold,
        line_width=1,
        line_dash="dash",
        line_color="black",
        annotation_text=f"-{threshold:.0%}",
        annotation_position="bottom right",
    )

    corr_y.show()
    print("Most correlated variables with Y (threshold +/- %.1f)" % threshold)
    print(
        person_matrix_democrats[
            np.abs(person_matrix_democrats["WINNER_DEMOCRATS"]) >= threshold
        ]
    )

    _compute_triangular_matrix(person_matrix_num, abs_corr_thresh)

    return df_prep_y
