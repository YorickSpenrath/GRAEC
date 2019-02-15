from warnings import warn

import pandas as pd
from GRAEC.Functions import Filefunctions
import locale


def fixDates(frame, date_column_string='datum'):
    if date_column_string is None:
        # parse Index
        frame.index = pd.to_datetime(frame.index, infer_datetime_format=True)
        return

    for date in [x for x in frame.columns if date_column_string in x]:
        frame[date] = pd.to_datetime(frame[date], infer_datetime_format=True)


def sort_file_on_date(fn_in, feature, fn_out=None):
    assert (isinstance(feature, str) or isinstance(feature, int)), 'Feature must be str or int: {}'.format(feature)
    Filefunctions.exists_assert(fn_in)
    df = import_df(fn_in)

    if isinstance(feature, int):
        feature = df.columns[feature]

    fixDates(frame=df, date_column_string=feature)

    df.sort_values(by=[feature], ascending=True, inplace=True)

    if fn_out is None:
        fn_out = fn_in

    export_df(df, fn_out)


def import_df(fn, skip_rows=None, dtype=str, na_values=[]):
    return pd.read_csv(fn,
                       encoding=locale.getpreferredencoding(False),
                       sep=';',
                       dtype=dtype,
                       skiprows=skip_rows,
                       na_values=na_values)


def export_df(df, fn, index=False):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(fn, str)
    df.to_csv(path_or_buf=fn,
              sep=';',
              index=index,
              encoding=locale.getpreferredencoding(False))


def drop_feature(fn, feature, fn_out=None):
    Filefunctions.exists_assert(fn)
    assert (isinstance(feature, str) or isinstance(feature, int))
    df = import_df(fn)
    if isinstance(feature, int):
        feature = df.columns[feature]

    if feature not in df.columns:
        warn('Feature is not a column')
        return

    df.drop(labels=[feature], axis=1, inplace=True)
    if fn_out is None:
        fn_out = fn
    export_df(df, fn_out)
