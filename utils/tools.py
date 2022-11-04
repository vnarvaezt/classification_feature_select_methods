import re
import pandas as pd
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

    df = pd.DataFrame(_excel_file.values[1:], columns=header_row)

    return df


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


def check_duplicates(data, keys):
    df_dup = data[data.duplicated(subset=keys, keep=False)]
    r = ""
    if not df_dup.empty:
        r += f"There are duplicates based on keys : {keys}\n"
        print(df_dup.head())
    else:
        r += f"There none duplicates based on keys {keys}\n"

    print("\n%s" % r)