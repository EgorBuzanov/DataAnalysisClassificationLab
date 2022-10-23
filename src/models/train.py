import os
import pickle
import pandas as pd
import numpy as np
import category_encoders as ce
import train_cfg as cfg
from typing import Tuple, Union
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import fbeta_score, make_scorer


def category_as_object(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('object')
    return df


def split_data(train: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                   pd.DataFrame, pd.DataFrame]:
    
    train_data, val_data, train_target, val_target = train_test_split(train, target,
                                                                      train_size=cfg.TRAIN_SIZE,
                                                                      random_state=cfg.RS)
    return train_data, val_data, train_target, val_target


def grid(model: MultiOutputClassifier, train_data: pd.DataFrame,
         train_target: pd.DataFrame) -> MultiOutputClassifier:
    fbeta_score_recall = make_scorer(fbeta_score, beta=2.0, average='micro')
    params = {'estimator__model__n_estimators': np.arange(10, 100, 5)}
    scores = RandomizedSearchCV(model, params, scoring=fbeta_score_recall, error_score='raise')
    scores.fit(train_data, train_target)
    return scores.best_estimator_


def create_pipeline() -> Pipeline:
    real_pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
        ]
    )
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocess_pipe = ColumnTransformer(transformers=[
        ('real_cols', real_pipe, cfg.REAL_COLS),
        ('cat_cols', cat_pipe, cfg.CAT_COLS),
        ('woe_cat_cols', ce.WOEEncoder(), cfg.CAT_COLS),
        ('ohe_cols', 'passthrough', cfg.OHE_COLS)
    ]
    )
    model_pipe = Pipeline([
        ('preprocess', preprocess_pipe),
        ('model', RandomForestClassifier())
    ]
    )
    return model_pipe


def sklearn_model(train_data: pd.DataFrame, train_target: pd.DataFrame) -> MultiOutputClassifier:
    pipeline = create_pipeline()
    multiout_model_pipe = MultiOutputClassifier(pipeline, n_jobs=4)
    model = grid(multiout_model_pipe, train_data, train_target)
    return model


def catboost_model(train_data: pd.DataFrame, train_target: pd.DataFrame) -> CatBoostClassifier:
    model = CatBoostClassifier(loss_function='MultiLogloss',
                                silent=True,
                                random_seed=cfg.RS,
                                cat_features=cfg.CAT_COLS)
    params  = {'learning_rate': [0.01, 0.1],
               'depth': [4, 10],
               'l2_leaf_reg': [1, 2, 3, 4, 5],
               'early_stopping_rounds': [50, 100, 150]}
    model.randomized_search(params, train_data, train_target, 4, verbose=False)
    return model


def best_model(sklearn_model: MultiOutputClassifier,
               catboost_model: CatBoostClassifier,
               val_data: pd.DataFrame,
               val_target: pd.DataFrame) -> Tuple[Union[MultiOutputClassifier, CatBoostClassifier],
                                                  np.ndarray]:
    val_predict_sklearn = sklearn_model.predict(val_data)
    val_predict_catboost = catboost_model.predict(val_data)
    sklearn_metric = fbeta_score(val_target, val_predict_sklearn, beta=2, average='micro')
    catboost_metric = fbeta_score(val_target, val_predict_catboost, beta=2, average='micro')
    return ((sklearn_model, val_predict_sklearn) if sklearn_metric > catboost_metric
                                                 else (catboost_model, val_predict_catboost))


def save_model(model: Union[MultiOutputClassifier, CatBoostClassifier], path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(model, f)
        
        
def save_labels(true: np.ndarray, pred: np.ndarray, path: str) -> None:
    true_filepath = os.path.join(path, "y_true.pkl")
    pred_filepath = os.path.join(path, "y_pred.pkl")
    with open(true_filepath, 'wb') as f:
        pickle.dump(true, f)    
    with open(pred_filepath, 'wb') as f:
        pickle.dump(pred, f)    
        
def train_model(train: pd.DataFrame,
                target: pd.DataFrame) -> Tuple[Union[MultiOutputClassifier, CatBoostClassifier],
                                               np.ndarray, np.ndarray]:
    train = category_as_object(train)
    train_data, val_data, train_target, val_target = split_data(train, target)
    sk_model = sklearn_model(train_data, train_target)
    cat_model = catboost_model(train_data, train_target)
    model, val_predict = best_model(sk_model, cat_model, val_data, val_target)
    return model, val_target, val_predict


