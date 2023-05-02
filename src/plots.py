import numpy as np
import matplotlib
import pylab
from sklearn.preprocessing import MinMaxScaler
import copy
from scipy.stats import rankdata
import scanpy as sc
import pandas as pd
from mne_connectivity.viz import plot_connectivity_circle

scaler = MinMaxScaler()


def color_cm(cmap, NUM_COLORS, ):
    color = []
    color_idx = 0
    cm = pylab.get_cmap(cmap)
    for i in range(NUM_COLORS):
        color.append(matplotlib.colors.to_hex(cm(1. * i / NUM_COLORS)))  # color will now be an RGBA tuple
    return color


def type_relevance_chord_plot(shap_values, p_fe, p_fe_idx, p_l_less, predict_label, colors_type, all_less_type,
                              technique, pr_ty_dict, thres=7, only_show_good=True, linewidth=1, linecolormap='Reds',
                              node_width=5, make_plot=True,fontsize_names=15,
                              potential_coloarmaps=['spring', 'summer', 'winter', 'autumn']):
    colors_mods = [color_cm(potential_coloarmaps[ii], len(c_p_fe)) for ii, c_p_fe in enumerate(p_fe)]
    scaler = MinMaxScaler()
    all_feature_num = sum([len(pp) for pp in p_fe])
    all_type_features = {}
    for ls_t in all_less_type:
        c_v = []
        node_names = []
        c_v_in = []
        c_v_out = []
        node_colors = []
        node_angles = []
        uni_sub = np.unique(predict_label[p_l_less == ls_t])
        if len(uni_sub) != 1:
            type_angle = (np.linspace(-10, 10, len(uni_sub))).astype(int)[::+1] % 360
        else:
            type_angle = [0]
        for uni_idx, tp in enumerate(uni_sub):
            cur_feature_num = 0
            start = 45
            cur_node_names = []
            all_type_features[pr_ty_dict[tp]] = {}
            for ii in range(len(p_fe)):
                # here we used all the shap feature for downstream calculation
                ft = scaler.fit_transform(
                    np.mean(np.abs(shap_values[tp][ii]), axis=0)[p_fe_idx[ii]].reshape(-1,
                                                                                                               1)).squeeze()
                c_v.append(ft)
                c_v_in.append(np.arange(cur_feature_num, cur_feature_num + len(ft)))
                c_v_out.append(np.array([all_feature_num + uni_idx] * len(ft)))
                cur_feature_num += len(ft)

                good_fe = copy.deepcopy(p_fe[ii])
                if type(thres) == float:
                    good_fe[ft < thres] = ''
                else:
                    good_fe[np.argpartition(ft, -thres)[:len(ft) - thres]] = ''
                cur_node_names.append(list(good_fe))
                if uni_idx == 0:
                    end = start + 270 / len(p_fe)
                    node_angles.append((np.linspace(start, end, len(p_fe_idx[ii]))).astype(int)[::+1] % 360)
                    start = end
                    node_colors.append(colors_mods[ii])
                all_type_features[pr_ty_dict[tp]][ii] = good_fe[good_fe != '']
            node_names.append(list(np.hstack(cur_node_names)))
            node_colors.append([colors_type[pr_ty_dict[tp]]])
            node_angles.append(type_angle[uni_idx])

        new_node_names = []
        for values in zip(*node_names):
            vl_n = ''
            for vl in values:
                vl_n = vl_n or vl
            new_node_names.append(vl_n)
        node_names = new_node_names
        for uni_idx, tp in enumerate(uni_sub):
            node_names.append(pr_ty_dict[tp])

        c_v = np.hstack(c_v)
        c_v_in = np.hstack(c_v_in)
        c_v_out = np.hstack(c_v_out)
        node_colors = np.hstack(node_colors)
        node_angles = np.hstack(node_angles)
        node_names = np.hstack(node_names)
        print(ls_t)
        print(node_names[node_names != ''])
        if only_show_good:
            good_in_index = np.where(np.in1d(c_v_in, np.where(node_names != '')[0]))[0]
        else:
            good_in_index = np.arange(len(c_v))
        c_v_in_s = rankdata(c_v_in[good_in_index], method='dense') - 1
        c_v_out_s = rankdata(c_v_out[good_in_index], method='dense') + max(c_v_in_s)
        names_s = node_names[node_names != '']
        node_angles_s = np.hstack(
            [(np.linspace(45, 315, len(names_s[:-len(uni_sub)]))).astype(int)[::+1] % 360, node_angles[-len(uni_sub):]])
        node_colors_s = node_colors[node_names != '']
        if make_plot:
            fig, axes = plot_connectivity_circle(c_v[good_in_index], names_s, indices=(c_v_in_s, c_v_out_s),
                                                 node_width=node_width, node_linewidth=0, node_colors=node_colors_s,
                                                 facecolor='white', textcolor='k',
                                                 linewidth=linewidth, node_angles=node_angles_s, colormap=linecolormap,
                                                 vmin=0, vmax=1, show=False, fontsize_names=fontsize_names)
            fig.set_size_inches(10, 10)
            if '/' in ls_t:
                ls_t = ls_t.replace('/', '-')
            fig.savefig(f'./feature_relevance_figures/{technique}_type_{ls_t}_only_show_good.pdf')
    return all_type_features


def markers_chord_plot(adatas_all, predict_label, predict_label_anno, major_dict, potential_features=10,
                       subset_feature=True):
    pr_ty_dict = dict(zip(predict_label, predict_label_anno))
    p_l = np.vectorize(pr_ty_dict.get)(predict_label)
    p_l_less = np.vectorize(major_dict.get)(p_l)

    p_fe = []
    p_fe_idx = []
    adatas_all_new = []
    for ad_x in adatas_all:
        ad_x.obs['predict_sub'] = p_l
        ad_x.obs['predict_sub_less'] = p_l_less
        if subset_feature:
            sc.pp.highly_variable_genes(ad_x)
            markers = ad_x.var_names[ad_x.var['highly_variable']].values
            markers_idx = np.where(ad_x.var['highly_variable'])[0]
        else:
            markers = ad_x.var_names.values
            markers_idx = np.arange(len(ad_x.var_names))
        p_fe.append(markers)
        p_fe_idx.append(markers_idx)
        adatas_all_new.append(ad_x)
    return adatas_all_new, p_fe, p_fe_idx, p_l_less, pr_ty_dict


def feature_relevance_chord_plot(shap_values_0_1, unique_ct, var_names_all, all_type_features, technique, in_mod=0,
                                 direction='0to1', thres=None,fontsize_names=15,
                                 potential_coloarmaps=['spring', 'summer', 'winter', 'autumn']):
    scaler = MinMaxScaler()
    direction_dict = {0: int(direction.split('to')[0]),
                      1: int(direction.split('to')[-1])}
    colors_mods = [color_cm(potential_coloarmaps[direction_dict[ii]], len(c_p_fe)) for ii, c_p_fe in
                   enumerate(var_names_all)]
    target_mod = 0 if in_mod else 1
    cv_all = {}
    io_names_all = {}
    for ct in unique_ct:
        cur_marker, cur_idx = [], []
        for ii_m, v_n in enumerate(var_names_all):
            _m, _i, _ = np.intersect1d(v_n, all_type_features[ct][direction_dict[ii_m]], return_indices=True)
            cur_marker.append(_m)
            cur_idx.append(_i)

        shap_mean = []
        for sp in [shap_values_0_1[ss] for ss in cur_idx[target_mod]]:
            shap_mean.append(np.mean(np.abs(sp), axis=0))
        shap_mean = np.vstack(shap_mean)
        shap_mean_norm = scaler.fit_transform(shap_mean.T)

        if thres != None:
            n_in, n_out = np.where(shap_mean_norm > thres)
            shap_mean_norm = shap_mean_norm[np.unique(n_in), :]
            c_v_in, c_v_out = np.where(shap_mean_norm > thres)
            c_v = shap_mean_norm[shap_mean_norm > thres]
            fe_shape = shap_mean_norm.shape
            in_names = var_names_all[0][np.unique(n_in)]
            out_names = var_names_all[1][np.unique(n_out)]
            names_idx = [np.unique(n_in), np.unique(n_out)]
        else:
            c_v = shap_mean_norm[cur_idx[in_mod], :]
            fe_shape = c_v.shape
            c_v_in, c_v_out = np.where(c_v >= 0)
            c_v = c_v.ravel()
            in_names, _in_idx, _ = np.intersect1d(var_names_all[0], cur_marker[0], return_indices=True)
            out_names, _out_idx, _ = np.intersect1d(var_names_all[1], cur_marker[1], return_indices=True)
            names_idx = [_in_idx, _out_idx]

        io_names = [in_names, out_names]
        node_names = np.hstack([in_names, out_names])
        c_v_out += fe_shape[0]
        starts = [105, -75]
        ends = [255, 75]

        node_angles = []
        node_colors = []

        for ii, nm in enumerate(io_names):
            start = starts[ii]
            end = ends[ii]
            node_angles.append((np.linspace(start, end, len(io_names[ii][io_names[ii] != '']))).astype(int)[::+1] % 360)
            node_colors.append(np.array(colors_mods[ii])[names_idx[ii]])

        node_angles = np.hstack(node_angles)
        node_colors = np.hstack(node_colors)

        s_title = ct
        fig, axes = plot_connectivity_circle(c_v, node_names, indices=(c_v_in, c_v_out), node_width=5, node_linewidth=0,
                                             node_colors=node_colors, facecolor='white', textcolor='k',
                                             linewidth=2, node_angles=node_angles, colormap='Reds', vmin=0, vmax=1,
                                             show=False, fontsize_names=fontsize_names, title=s_title)
        fig.set_size_inches(10, 10)
        if '/' in ct:
            ct = ct.replace('/', '-')
        fig.savefig(f'./feature_relevance_figures/{technique}_feature_{ct}_only_show_good_{direction}.pdf')
        cv_all[ct] = c_v.reshape(fe_shape)
        io_names_all[ct] = io_names
    return cv_all, io_names_all

def merge_sub_feature(all_type_features, major_dict):
    all_type_features_mj = {}
    for ct in all_type_features.keys():
        _c = all_type_features[ct]
        if major_dict[ct] not in all_type_features_mj.keys():
            all_type_features_mj[major_dict[ct]] = {}
        for mod in _c.keys():
            _c_m = _c[mod]
            if mod not in all_type_features_mj[major_dict[ct]].keys():
                all_type_features_mj[major_dict[ct]][mod] = []
            all_type_features_mj[major_dict[ct]][mod].append(_c_m)
    for ct in all_type_features_mj.keys():
        for mod in all_type_features_mj[ct].keys():
            all_type_features_mj[ct][mod] = np.unique(all_type_features_mj[ct][mod])
    return all_type_features_mj

def merge_sub_feature_all(all_type_features, major_dict):
    all_type_features_mj = {}
    for ct in all_type_features.keys():
        _c = all_type_features[ct]
        if major_dict[ct] not in all_type_features_mj.keys():
            all_type_features_mj[major_dict[ct]] = {}
        for mod in _c.keys():
            _c_m = _c[mod]
            if mod not in all_type_features_mj[major_dict[ct]].keys():
                all_type_features_mj[major_dict[ct]][mod] = []
            all_type_features_mj[major_dict[ct]][mod].append(_c_m)
    return all_type_features_mj
