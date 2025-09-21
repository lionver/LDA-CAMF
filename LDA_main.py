import random
import torch.nn as nn
import torch.optim as optim
from utils import *
from model_LDA import CA
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    batch_size = 3000
    location="test"
    learn_rate = 0.04
    weight_decay = 1e-3
    seed = 114514
    matrix_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    RANDOM_STATE = seed
    data_ues = 1
    if data_ues == 1:
        dis_sim = pd.read_csv('./Data_test/dis_Similarity_55.csv', header=None).values
        lnc_sim = pd.read_csv('./Data_test/lnc_Similarity_55.csv', header=None).values
        labels = pd.read_csv('./Data_test/label_1.csv', header=None).values
    else:
        dis_sim = pd.read_csv('./Data_test/data2/dis_Similarity_55.csv', header=None).values
        lnc_sim = pd.read_csv('./Data_test/data2/lnc_Similarity_55.csv', header=None).values
        labels = pd.read_csv('./Data_test/data2/label_2.csv', header=None).values
    positive_indices = np.argwhere(labels == 1)
    negative_indices = np.argwhere(labels == 0)
    n_positive = len(positive_indices)
    n_negative = int(n_positive)
    np.random.seed(RANDOM_STATE)
    selected_negatives = negative_indices[np.random.choice(negative_indices.shape[0], n_negative, replace=False)]
    all_samples = np.vstack([positive_indices, selected_negatives])
    y = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    X = np.array([np.hstack((lnc_sim[i], dis_sim[j])) for i, j in all_samples])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print("this is the {} fold".format(fold))
        label_matrix = torch.tensor(labels)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = CA()
        train_X = torch.tensor(X_train)
        train_y = torch.tensor(y_train)
        test_X = torch.tensor(X_test)
        test_y = torch.tensor(y_test)
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
        def train(model):
            model.train()
            for s_data in tqdm(train_loader):
                optimizer.zero_grad()
                output, _ = model(s_data[0])
                matrix_loss = matrix_criterion(output, label_matrix)
                matrix_loss.backward()
            return matrix_loss

        def test(model):
            model.eval()
            with torch.no_grad():
                for s_data in tqdm(train_loader):
                    _, train_pair = model(s_data[0])
                    train_label = s_data[1]
                    train_pair_org = s_data[0].numpy()
                train_pair = train_pair.numpy()
                params = {
                    "n_estimators": 125,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "objective": "binary:logistic",
                    "n_jobs": -1
                }
                clf = XGBClassifier(**params)
                train_pair = np.hstack((train_pair, train_pair_org))
                clf.fit(train_pair, train_label)
                for s_data in tqdm(test_loader):
                    _, test_pair = model(s_data[0])
                    test_label = s_data[1]
                    test_pair_org = s_data[0].numpy()
                test_pair = test_pair.numpy()
                test_pair = np.hstack((test_pair, test_pair_org))
                test_pred = clf.predict_proba(test_pair)

            return test_pred, test_label
        for epoch in range(200):
            train(model, epoch)
        test_pred, test_label = test(model)
        test_pred =  test_pred[:,1]



