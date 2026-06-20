import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sac_graph5g.config import DEVICE, MAX_FEATURES


def infer_target_column(frame, candidates=None):
    if candidates is None:
        candidates = [
            "label", "Label", "LABEL", "class", "Class", "CLASS", "target", "Target", "TARGET",
            "attack", "Attack", "attack_type", "Attack Type", "type", "Type", "slice type",
            "Slice Type", "network type", "Network Type", "traffic type", "Traffic Type",
            "output", "Output", "category", "Category"
        ]
    for name in candidates:
        if name in frame.columns and 2 <= frame[name].nunique(dropna=True) <= max(50, len(frame) // 2):
            return name
    low_cardinality = [
        c for c in frame.columns
        if 2 <= frame[c].nunique(dropna=True) <= max(20, int(0.2 * len(frame)))
    ]
    return low_cardinality[-1] if low_cardinality else frame.columns[-1]


def load_and_preprocess_data(data_file, target_column=None):
    if not data_file.exists():
        raise FileNotFoundError(f"Could not find {data_file.resolve()}.")

    df = pd.read_csv(data_file)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)

    target_col = target_column or infer_target_column(df)
    df = df.dropna(subset=[target_col]).copy()

    if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique(dropna=True) > 50:
        df[target_col] = pd.qcut(df[target_col], q=4, labels=["Q1_low", "Q2_midlow", "Q3_midhigh", "Q4_high"], duplicates="drop")

    counts = df[target_col].astype(str).value_counts()
    rare = counts[counts < 3]
    if len(rare):
        df = df[~df[target_col].astype(str).isin(rare.index)].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df[target_col].astype(str))
    class_names = list(label_encoder.classes_)
    X_raw_all = df.drop(columns=[target_col])

    return X_raw_all, y_all, class_names


def make_preprocessor(raw_frame):
    numeric_cols = raw_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in raw_frame.columns if c not in numeric_cols]
    
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)
        
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot)
    ])
    
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ], sparse_threshold=0.3)


def stratified_splits(y, seed):
    idx = np.arange(len(y))
    strat = y if np.min(np.bincount(y)) >= 3 else None
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat)
    
    temp_y = y[temp_idx]
    strat_temp = temp_y if np.min(np.bincount(temp_y)) >= 2 else None
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=strat_temp)
    
    return train_idx, val_idx, test_idx


def prepare_context(seed, X_raw_all, y_all):
    train_idx, val_idx, test_idx = stratified_splits(y_all, seed)
    pre = make_preprocessor(X_raw_all)
    
    X_train_pre = pre.fit_transform(X_raw_all.iloc[train_idx])
    X_pre = pre.transform(X_raw_all)
    
    n_features = X_pre.shape[1]
    n_comp = min(MAX_FEATURES, X_train_pre.shape[0] - 1, n_features - 1)
    
    if n_features > MAX_FEATURES and n_comp >= 2:
        reducer = TruncatedSVD(n_components=n_comp, random_state=seed)
        reducer.fit(X_train_pre)
        X = reducer.transform(X_pre).astype(np.float32)
        svd_var = float(reducer.explained_variance_ratio_.sum())
    else:
        X = X_pre.toarray() if hasattr(X_pre, "toarray") else np.asarray(X_pre)
        X = X.astype(np.float32)
        svd_var = None
        
    return {
        "seed": seed,
        "X": X,
        "y": y_all.astype(np.int64),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "x_tensor": torch.tensor(X, dtype=torch.float32, device=DEVICE),
        "y_tensor": torch.tensor(y_all, dtype=torch.long, device=DEVICE),
        "train_idx_t": torch.tensor(train_idx, dtype=torch.long, device=DEVICE),
        "val_idx_t": torch.tensor(val_idx, dtype=torch.long, device=DEVICE),
        "test_idx_t": torch.tensor(test_idx, dtype=torch.long, device=DEVICE),
        "svd_explained_variance": svd_var,
    }


def build_knn_graph(features, k=10, metric="cosine"):
    n = features.shape[0]
    k = int(max(1, min(k, n - 1)))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(features)
    
    _, neigh = nbrs.kneighbors(features)
    rows = np.repeat(np.arange(n), k)
    cols = neigh[:, 1:].reshape(-1)
    
    adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)
    adj.setdiag(1.0)
    adj = adj.tocoo()
    
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_inv = np.zeros_like(deg, dtype=np.float32)
    deg_inv[deg > 0] = np.power(deg[deg > 0], -0.5)
    
    vals = adj.data.astype(np.float32) * deg_inv[adj.row] * deg_inv[adj.col]
    idx = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long, device=DEVICE)
    val = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    
    return torch.sparse_coo_tensor(idx, val, size=(n, n), device=DEVICE).coalesce()
