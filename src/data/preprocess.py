import pandas as pd
import numpy as np
import sys
import os
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import src.config as cfg

def set_idx(df : pd.DataFrame, idx_col : str) -> pd.DataFrame:
    return df.set_index(idx_col)


def drop_col(df : pd.DataFrame, colname : str) -> pd.DataFrame:
    return df.drop(colname, axis=1) if colname in df.columns else df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')

    ohe_int_cols = df[cfg.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df


def extract_target(df : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]

def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = drop_col(df, cfg.UNNECESSARY_ID)
    df = fill_sex(df)
    df = cast_types(df)
    return df