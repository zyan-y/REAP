
import numpy as np
import pandas as pd
import os
import random
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def load_embeddings(embeddings_folder):
    X_list = []
    y_list = []
    for file_name in sorted(os.listdir(embeddings_folder)):
        if file_name.endswith('.npz'):
            file_path = os.path.join(embeddings_folder, file_name)
            data = np.load(file_path)
            X, y = data['X'], data['y']
            X_list.append(X)
            y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def check_log_y(y):
    return np.log1p(y)
    

def process_datasets(embeddings_folder, cv_file, cv_split='fold_random_5', test_idx=0, standardize_y=False):
    X, y = load_embeddings(embeddings_folder)
    y = check_log_y(y)
    cv_indices = pd.read_csv(cv_file)[cv_split].values
    
    train_mask = cv_indices != test_idx
    test_mask = cv_indices == test_idx
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if standardize_y:
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()
        
    return X_train, y_train, X_test, y_test


def split_single_multi(embeddings_folder, cv_file, train_ratio=0.8, seed=None):
    X, y = load_embeddings(embeddings_folder)
    y = check_log_y(y)
    names = pd.read_csv(cv_file)['mutant'].values

    is_single = np.array([('-' not in name) for name in names])
    is_multi = np.array([('-' in name) for name in names])

    X_single = X[is_single]
    y_single = y[is_single]

    X_multi = X[is_multi]
    y_multi = y[is_multi]

    X_single_train, X_single_test, y_single_train, y_single_test = train_test_split(X_single, y_single, train_size=train_ratio, random_state=seed)
    
    return X_single_train, y_single_train, X_single_test, y_single_test, X_multi, y_multi


def split_few_shot(embeddings_folder, sample_size=100, seed=None):
    X, y = load_embeddings(embeddings_folder)
    y = check_log_y(y)

    N = X.shape[0]
    if sample_size * 2 > N:
        raise ValueError(f"The number of samples {N} is too small for the sample_size")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)

    idx_train = perm[:sample_size]
    idx_val   = perm[sample_size : sample_size * 2]
    idx_test  = perm[sample_size * 2 :]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    return X_train, y_train, X_val, y_val, X_test, y_test




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 