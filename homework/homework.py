# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#

import os
import gzip
import json
import pickle

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)


# Paths
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"
TRAIN_PATH = "files/input/train_data.csv.zip"
TEST_PATH = "files/input/test_data.csv.zip"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV (possibly zipped) and apply required cleaning rules."""
    # pandas detecta compresión por la extensión .zip si corresponde
    df = pd.read_csv(path, compression="zip")

    # Renombrar objetivo
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})

    # Eliminar ID si existe
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Convertir columnas numéricas esperadas a numéricas (seguridad)
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # EDUCATION: valores > 4 agrupar en 4 (others)
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if pd.notna(x) and x > 4 else x)

    # Remover registros con información no disponible (NaN)
    df = df.dropna().reset_index(drop=True)

    # Asegurar target entero 0/1
    if "default" in df.columns:
        df["default"] = df["default"].astype(int)

    return df

def save_model_as_pickle_gzip(obj, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, "wb") as f:
        pickle.dump(obj, f)


def write_metrics_lines(metrics_list, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in metrics_list:
            f.write(json.dumps(entry) + "\n")


def main():
    # Carga y limpieza
    train = load_and_clean(TRAIN_PATH)
    test = load_and_clean(TEST_PATH)

    # Separar X / y
    X_train = train.drop(columns=["default"])
    y_train = train["default"]

    X_test = test.drop(columns=["default"])
    y_test = test["default"]

    # Definir features categóricas y numéricas explícitamente
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    # todas las demás columnas en X_train que no estén en categóricas (orden estable)
    numerical_features = [c for c in X_train.columns if c not in categorical_features]

    # Preprocesador:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                numerical_features,
            ),
        ],
        remainder="drop",
    )


