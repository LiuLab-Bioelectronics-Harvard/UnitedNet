import anndata as ad
import numpy
import random
import scanpy as sc
import torch
import torch.utils.data as D

from src.constants import *
import pickle

class Dataset(D.Dataset):
    def __init__(self, modalities, labels):
        super().__init__()
        self.modalities = [
            torch.tensor(modality, dtype=torch.float) for modality in modalities
        ]
        if labels is None:
            labels = [-1 for _ in range(len(modalities[0]))]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        modalities = [modality[index] for modality in self.modalities]
        return modalities, self.labels[index]

    def __len__(self):
        return len(self.labels)


def create_dataloader_from_dataset(dataset, shuffle, batch_size):
    g = torch.Generator()

    return D.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
    )


def create_dataset(model, adatas, fit_label):
    modalities = [adata.X for adata in adatas]
    if str_label in adatas[0].obs.keys():
        if fit_label:
            labels = model.label_encoder.fit_transform(list(adatas[0].obs[str_label]))
        else:
            labels = model.label_encoder.transform(list(adatas[0].obs[str_label]))
    else:
        labels = None
    return Dataset(modalities, labels)


def create_dataloader(model, adatas, shuffle=False, batch_size=512, fit_label=False):
    dataset = create_dataset(model, adatas, fit_label)
    return create_dataloader_from_dataset(dataset, shuffle, batch_size)


def create_joint_dataloader(
        model, adatas0, adatas1, shuffle=False, batch_size=512, fit_label=False
):
    dataset = D.ConcatDataset(
        [
            create_dataset(model, adatas0, fit_label),
            create_dataset(model, adatas1, fit_label),
        ]
    )
    return create_dataloader_from_dataset(dataset, shuffle, batch_size)


import numpy as np


def anndata_from_outputs(model, outputs):
    _, predictions, fused_latents = outputs
    adata = ad.AnnData(fused_latents if type(fused_latents) == np.ndarray else fused_latents.cpu().numpy())
    if hasattr(model, 'class_weights'):
        adata.obs["predicted_label"] = model.label_encoder.inverse_transform(
            predictions.tolist() if type(predictions) == np.ndarray else predictions.cpu().tolist()
        )
    else:
        adata.obs["predicted_label"] = predictions.cpu().tolist()
    adata.obs["predicted_label"] = adata.obs["predicted_label"].astype('category')
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)
    return adata


import os


def save_obj(path_relec, filename, shap_values_X):
    if not os.path.exists(path_relec):
        os.makedirs(path_relec)
    for feature_id in range(len(shap_values_X)):
        if type(shap_values_X[feature_id]) == list:
            for mod in range(len(shap_values_X[feature_id])):
                target_file_loc = f'{path_relec}/{filename}_mod_{mod}_type_{feature_id}.dat'
                if os.path.exists(target_file_loc):
                    os.remove(target_file_loc)
                fp = np.memmap(target_file_loc, dtype='float32', mode='w+', shape=shap_values_X[feature_id][mod].shape)
                fp[:] = shap_values_X[feature_id][mod][:]
                fp.flush()
        else:
            target_file_loc = f'{path_relec}/{filename}_{feature_id}.dat'
            if os.path.exists(target_file_loc):
                os.remove(target_file_loc)
            fp = np.memmap(target_file_loc, dtype='float32', mode='w+', shape=shap_values_X[feature_id].shape)
            fp[:] = shap_values_X[feature_id][:]
            fp.flush()


def load_obj(path_relec, filename, feature_num, shape_shap, type_rele=False, mod_num=None):
    all_shap_value = []
    for feature_id in range(feature_num):
        if type_rele:
            fp_all = []
            for mod in range(mod_num):
                target_file_loc = f'{path_relec}/{filename}_mod_{mod}_type_{feature_id}.dat'
                fp = np.array(np.memmap(target_file_loc, dtype='float32', mode='r', shape=shape_shap[mod]))
                fp_all.append(fp)
            all_shap_value.append(fp_all)
        else:
            target_file_loc = f'{path_relec}/{filename}_{feature_id}.dat'
            fp = np.array(np.memmap(target_file_loc, dtype='float32', mode='r', shape=shape_shap))
            all_shap_value.append(fp)
    return all_shap_value


def type_specific_mean(adata_x, label_x):
    cluster_prototype_x = {}
    for lb in adata_x.obs[label_x].unique():
        _sub_expre = adata_x[adata_x.obs[label_x] == lb].X.mean(axis=0).toarray()
        cluster_prototype_x[lb] = _sub_expre
    cluster_prototype_x = torch.tensor(np.array(list(cluster_prototype_x.values())))
    return cluster_prototype_x


from sklearn.model_selection import StratifiedKFold


def partitions(celltype, n_partitions, seed=0):
    """
    adapted from https://github.com/AllenInstitute/coupledAE-patchseq
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Safe to ignore warning - there are celltypes with a low sample number that are not crucial for the analysis.
    with warnings.catch_warnings():
        skf = StratifiedKFold(n_splits=n_partitions, random_state=seed, shuffle=True)

    # Get all partition indices from the sklearn generator:
    ind_dict = [{'train': train_ind, 'val': val_ind} for train_ind, val_ind in
                skf.split(X=np.zeros(shape=celltype.shape), y=celltype)]
    return ind_dict


import pandas as pd


def important_relevance(shap_values_X, feature_names, output_names, relevance_save_file, target_relevance_num=100,
                        replace=False):
    if os.path.exists(relevance_save_file) and (not replace):
        feature_importance_df = pd.read_csv(relevance_save_file)
    else:
        feature_importance = []
        for ii, target_feature in enumerate(output_names):
            vals = np.abs(shap_values_X[ii]).mean(0)
            _feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                               columns=['col_name', 'feature_importance_vals'])
            _feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            feature_importance.append(_feature_importance)
        feature_importance_df = pd.concat(feature_importance, keys=output_names).reset_index().rename(
            columns={'level_0': 'target_feature'})
        feature_importance_df.to_csv(relevance_save_file)

    return feature_importance_df


import matplotlib.pyplot as plt


def save_umap(adata_all, label, test_batch, nametype, root_save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sc.pl.umap(adata_all, color=label, ax=ax, show=False)
    fig.savefig(root_save_path + f'/plot/{test_batch}_{nametype}_{label}.png', dpi=300)


def generate_adata(data, nonnan_indices, cell_type_label, cols, rows, batch):
    data = data.loc[data.index[nonnan_indices]]
    adata = ad.AnnData(X=np.array(data), obs=list(data.index))
    adata.obs['label'] = cell_type_label
    adata.obs['imagecol'] = cols
    adata.obs['imagerow'] = rows
    adata.obs['batch'] = batch
    return adata

from sklearn import preprocessing
def patch_seq_pre_ps(adata_rna_raw, adata_ephys_raw, adata_morph_raw, cv, ind_dict,split=False):
    adata_rna, adata_ephys, adata_morph = adata_rna_raw.copy(), adata_ephys_raw.copy(), adata_morph_raw.copy()
    adatas_train, adatas_test = [], []
    assert (adata_rna.X >= 0).all(), "poluted input"
    for mod in [adata_rna, adata_ephys, adata_morph]:
        mod.obs['label'] = mod.obs['cell_type_TEM']
        if split:
            m_train = mod[ind_dict[cv]['train']]
            scaler = preprocessing.StandardScaler().fit(m_train.X)
            m_train.X = scaler.transform(m_train.X)

            m_test = mod[ind_dict[cv]['val']]
            scaler = preprocessing.StandardScaler().fit(m_test.X)
            m_test.X = scaler.transform(m_test.X)
        else:
            scaler = preprocessing.StandardScaler().fit(mod.X)
            mod.X = scaler.transform(mod.X)
            m_train = mod[ind_dict[cv]['train']]
            m_test = mod[ind_dict[cv]['val']]

        adatas_train.append(m_train)
        adatas_test.append(m_test)
    adatas_all = [ad.concat([m_train, m_test]) for m_train, m_test in zip(adatas_train, adatas_test)]
    return adatas_train, adatas_test, adatas_all
