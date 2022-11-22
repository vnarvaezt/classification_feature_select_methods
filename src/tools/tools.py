import re
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Tools
def upper_consistent(df):
    df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    return df


# Standardise columns names
def standard_name_cols(data_columns):
    data_columns = data_columns.str.upper()
    col_names = [re.sub(r"[^a-zA-Z0-9]", "_", col) for col in data_columns]

    return col_names


def read_excel(path):

    with open(path, mode="rb") as excel_file:
        _excel_file = pd.read_excel(excel_file,
                                    # converters={'FIPS Code': str}
                                    )
    print(f"\n========== File name {excel_file}\n")
    print(_excel_file.columns[0])

    _excel_file.set_index(_excel_file.columns[0], inplace=True)

    # drop nan
    _excel_file = _excel_file.dropna(how="all")
    _excel_file = _excel_file.reset_index()

    # set header
    header_row = _excel_file.iloc[0]
    # final df
    df = pd.DataFrame(_excel_file.values[1:], columns=header_row)

    return df







def check_duplicates(data, keys):
    df_dup = data[data.duplicated(subset=keys, keep=False)]
    r = ""
    if not df_dup.empty:
        r += f"There are duplicates based on keys : {keys}\n"
        print(df_dup.head())
    else:
        r += f"There none duplicates based on keys {keys}\n"

    print("\n%s" % r)


def split_train_test(df, test_size, random_state):
    try:
        print("Splitting into train and test")
        X_train_, X_test_ = train_test_split(
            df, test_size=test_size,
            random_state=random_state, shuffle=True
        )
        return X_train_, X_test_
    except Exception as e:
        raise e


def input_value(x):

    if x.dtype == np.number:
        x_no_na = x.fillna(float(x.median()))
    else:
        x_no_na = x.fillna(x.mode()[0])
    return x_no_na