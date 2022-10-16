import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import src.config as cfg


def sleep_time(value):
    hours, minutes, seconds = value.split(":")
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    if hours >= 12:
        return -(3600*24 -(hours * 3600 + minutes * 60 + seconds))
    else:
        return hours * 3600 + minutes * 60 + seconds

    
def wake_up_time(value):
    hours, minutes, seconds = value.split(":")
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    return hours * 3600 + minutes * 60 + seconds


def add_sleep_time(df : pd.DataFrame) -> pd.DataFrame:
    wake_up = df[cfg.WAKE_UP_TIME].apply(wake_up_time).astype(np.int32)
    go_bed = df[cfg.GO_BED_TIME].apply(sleep_time).astype(np.int32)
    df["Продолжительность сна"] = np.round((wake_up - go_bed) / 3600).astype(np.int8)
    return df


def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame:
    df[f'{cfg.EDU_COL}_ord'] = df[cfg.EDU_COL].str.slice(0, 1).astype(np.int8).values
    return df

def second_hand_smoke_count(value : str) -> float:
    if value == '1-2 раза в неделю':
        return 1.5
    if value == '2-3 раза в день':
        return 17.5
    if value == '3-6 раз в неделю':
        return 4.5
    if value == '4 и более раз в день':
        return 28.0
    if value == 'не менее 1 раза в день':
        return 7.0
    return 0.0

def add_smoke_ord(df : pd.DataFrame) -> pd.DataFrame:
    df['Частота пасс кур'] = df['Частота пасс кур'].apply(second_hand_smoke_count).astype(np.float32)
    return df

def feature_gen(df : pd.DataFrame) -> pd.DataFrame:
    df = add_sleep_time(df)
    df = add_ord_edu(df)
    df = add_smoke_ord(df)
    return df