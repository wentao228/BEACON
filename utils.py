import numpy as np
from sklearn.utils import shuffle
import torch
from rdkit import Chem
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import random
import logging


def eval_cpi_2(y_pred, labels):
    labels = np.array(labels.detach().cpu())
    y_pred = np.array(y_pred.detach().cpu())
    y_pred_labels = np.array([0 if i < 0.5 else 1 for i in y_pred])

    acc = accuracy_score(labels, y_pred_labels)
    roc_score = roc_auc_score(labels, y_pred)
    pre_score = precision_score(labels, y_pred_labels)
    recall = recall_score(labels, y_pred_labels)
    pr, re, _ = precision_recall_curve(labels, y_pred, pos_label=1)
    aupr = auc(re, pr)

    return acc, roc_score, pre_score, recall, aupr


def stratified_split(config, dataset, train_valid_test=0.2, valid_test=0.5):
    seed = config['training']['split_seed']  # seed=2021
    classes_dict = dict()

    for sample in dataset:
        label = sample[4]
        if label not in classes_dict:
            classes_dict[label] = list()
            classes_dict[label].append(sample)
        else:
            classes_dict[label].append(sample)
    train = list()
    valid = list()
    test = list()
    for c in classes_dict:
        samples = classes_dict[c]
        c_train, c_valid = train_test_split(samples, test_size=train_valid_test, random_state=seed, shuffle=True)
        c_valid, c_test = train_test_split(c_valid, test_size=valid_test, random_state=seed, shuffle=True)
        train.extend(c_train)
        valid.extend(c_valid)
        test.extend(c_test)
    train = shuffle(train, random_state=seed)
    valid = shuffle(valid, random_state=seed)
    test = shuffle(test, random_state=seed)
    return train, valid, test


def data_iter(batch_size, features):
    num_samples = len(features)
    indices = list(range(num_samples))
    random.shuffle(indices)
    features = torch.from_numpy(np.array(features))
    for i in range(0, num_samples, batch_size):
        compounds_id = list()
        proteins_id = list()
        d_e_ids = list()
        p_e_ids = list()
        labels = list()
        j = torch.LongTensor(indices[i:min(i + batch_size, num_samples)])
        features_select = features.index_select(0, j)
        for (u_id, p_e_id, cid, d_e_id, label) in features_select:
            compounds_id.append(int(cid))
            proteins_id.append(int(u_id))
            d_e_ids.append(int(d_e_id))
            p_e_ids.append(int(p_e_id))
            labels.append([int(label)])
        yield np.array(compounds_id), np.array(proteins_id), d_e_ids, p_e_ids, np.array(labels)


def get_all_data(features):
    compounds_id = list()
    proteins_id = list()
    d_e_ids = list()
    p_e_ids = list()
    labels = list()

    for (u_id, p_e_id, cid, d_e_id, label) in features:
        compounds_id.append(int(cid))
        proteins_id.append(int(u_id))
        d_e_ids.append(int(d_e_id))
        p_e_ids.append(int(p_e_id))
        labels.append([int(label)])

    return np.array(compounds_id), np.array(proteins_id), d_e_ids, p_e_ids, np.array(labels)


def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def smiles2fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
    fp = fp.ToBitString()
    fp = torch.from_numpy(np.array(list(map(int, fp))))
    return fp


def smiles2maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    fp = fp.ToBitString()
    fp = torch.from_numpy(np.array(list(map(int, fp))))
    return fp
