

import nibabel as nib
import os
import sys
import copy
from tqdm import tqdm
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import plotly.express as px
import math
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
sns.set(context='notebook', style='ticks', font_scale=1.5, font='sans-serif', rc={"lines.linewidth": 1.2})
sns.set_style("white")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sns.despine(left=True, bottom=True)
bg_color = (0.88, 0.85, 0.95)
bg_color = (1, 1, 1)
bg_color = 'white'
import pandas as pd

plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['axes.facecolor'] = bg_color
plt.rcParams["savefig.facecolor"] = bg_color
COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
plt.rcParams.update({"savefig.format": 'png'})

renaming_columns = {
        'Brain stem': 'BrainStem',
        'left pallidum': 'left pallidium',
        'right pallidum': 'right palladium',
        'left cerebellum exterior': 'left cerebellem exterior'
    }


# @st.cache_data
@st.cache_data
def generate_get_dataframes(imaging_features, my_shap_values, image_id, atype):
    with open("ad_top20_feature_list.txt", 'r') as f:
        ad_top20_features = f.read().split("\n")

    with open("pd_top20_feature_list.txt", 'r') as f:
        pd_top20_features = f.read().split("\n")

    ad_top20_features = [col.replace(' ', '_') for col in ad_top20_features]
    pd_top20_features = [col.replace(' ', '_') for col in pd_top20_features]
    ad_intersected_features = list(set(ad_top20_features).intersection(set(imaging_features.columns)))
    pd_intersected_features = list(set(pd_top20_features).intersection(set(imaging_features.columns)))
    F = pd.read_csv("min_max_values.csv")
    F = F.set_index('feature_name')
    F.index = F.index.map(lambda x: x.replace('id_invicrot1_', ''))
    ad_dataframe = (imaging_features[ad_intersected_features] - F.loc[ad_intersected_features]['min_value']) / (
            F.loc[ad_intersected_features]['max_value'] - F.loc[ad_intersected_features]['min_value'])
    pd_dataframe = (imaging_features[pd_intersected_features] - F.loc[pd_intersected_features]['min_value']) / (
            F.loc[pd_intersected_features]['max_value'] - F.loc[pd_intersected_features]['min_value'])

    all_intersected_features = list(set(F.index).intersection(set(imaging_features.columns)))
    overall_normalized = (imaging_features[all_intersected_features] - F.loc[all_intersected_features]['min_value']) / (
                F.loc[all_intersected_features]['max_value'] - F.loc[all_intersected_features]['min_value'])
    ad_dataframe.columns = ad_dataframe.columns.map(lambda x: f'id_invicrot1_{x}')
    pd_dataframe.columns = pd_dataframe.columns.map(lambda x: f'id_invicrot1_{x}')

    create_a_plot(F, overall_normalized.copy(), image_name=f"{image_id}_overall_normalized", cmap_name='Greys')
    if atype == 'ad':
        create_a_plot(F, my_shap_values.copy(), image_name=f"{image_id}_shapad_dataframe", cmap_name='Blues')
    elif atype == 'pd':
        create_a_plot(F, my_shap_values.copy(), image_name=f"{image_id}_shappd_dataframe", cmap_name='Greens')

    # create_a_plot(F, ad_dataframe.copy(), image_name=f"{image_id}_ad_dataframe", cmap_name='Blues')
    # create_a_plot(F, pd_dataframe.copy(), image_name=f"{image_id}_pd_dataframe", cmap_name='Greens')

    image1 = plt.imread(f'seeImages/{image_id}_overall_normalized.png')
    image2 = plt.imread(f'seeImages/{image_id}_shap{atype}_dataframe.png')

    fig, ax = plt.subplots()
    ax.imshow(image2)
    ax.imshow(image1, alpha=0.2)
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig(f"seeImages/{image_id}_shap{atype}_overlayed_image.png", dpi=300, bbox_inches='tight')
    plt.close()

    # image3 = plt.imread(f'seeImages/{image_id}_pd_dataframe.png')
    # fig, ax = plt.subplots()
    # ax.imshow(image3)
    # ax.imshow(image1, alpha=0.4)
    # ax.set_axis_off()
    # plt.savefig(f"seeImages/{image_id}_pd_overlayed_image.png", dpi=300)
    # plt.close()

    # return F, ad_dataframe, pd_dataframe, overall_normalized
    # S = pd.DataFrame(ad_dataframe.transpose().reset_index())
    # S.columns = ["region_name", "values"]


def create_a_plot(F, temp_forP, image_name, cmap_name='coolwarm'):

    label_map = pd.read_csv('labeledFeatureList.csv')
    mapping_to_map = {}
    for col in F.index:
        for col2 in label_map['region_name'].values:
            if col2.lower() in col.replace('_', ' ').lower():
                mapping_to_map[col] = col2

    P = temp_forP.values.transpose()
    P_df = pd.DataFrame(P, columns=[f's{i}' for i in range(P.shape[1])])
    P_df['region_name'] = temp_forP.columns.map(lambda x: x.replace('id_invicrot1_', ''))
    all_regions = list(set(F.index.map(lambda x: x.split('@')[-1].split('_')[0])))
    LABEL_DICT = dict(zip(list(label_map['region_name']), list(label_map['region_index'])))
    FeatureImportance = pd.DataFrame({'region_name': list(F.index)})
    FeatureImportance['short_region_index'] = FeatureImportance['region_name'].map(lambda x: mapping_to_map.get(x, "-1"))
    FeatureImportance['region_index'] = FeatureImportance['short_region_index'].map(lambda x: LABEL_DICT.get(x.split('@')[-1].split('_')[0], -1))

    P_df = pd.merge(FeatureImportance, P_df, left_on='region_name', right_on='region_name').rename(columns={'short_region_index': 'region_name'}).set_index('region_name')
    # P_df = pd.concat([FeatureImportance[['short_region_index', 'region_index']], P_df], axis=1).rename(columns={'short_region_index': 'region_name'}).set_index('region_name')
    mean_df = P_df.reset_index().groupby(['region_name', 'region_index']).agg('mean').transpose()
    mean_df = mean_df.iloc[:, ~ (mean_df.columns.get_level_values('region_index') == -1)]
    from collections import defaultdict
    df_dict = defaultdict(list)
    for i in mean_df.columns:
        df_dict['region_name'].append(i[0])
        df_dict['region_index'].append(i[1])

    S = pd.DataFrame(df_dict)
    absmean_df = P_df.abs().reset_index().groupby(['region_name', 'region_index']).agg('mean').transpose()
    absmean_df = absmean_df.iloc[:, ~ (absmean_df.columns.get_level_values('region_index') == 1)]
    global_metadata = pd.DataFrame({"SELECT": ["A"]*len(absmean_df), "AGE": [100] * len(absmean_df)}, index=absmean_df.index)
    absK = pd.concat([absmean_df, global_metadata], axis=1)
    meanK = pd.concat([mean_df, global_metadata], axis=1)
    selected_columns = [col for col in absK.columns if type(col) is tuple]
    group = {'name': 'SELECT', 'filter': []}
    groupy_columns = ['SELECT', 'AGE']
    mean_mean_df = mean_df.copy()
    dataset_groupby = {}
    data_d = {'mean_mean_df': meanK}

    for key, value in data_d.items():
        dataset_groupby[key] = {}
        dataset_groupby[key]['unnormalized_data'] = value.groupby(groupy_columns).agg('mean')[selected_columns] # value.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})

    fpath1 = "shapBrainInterpretation/LabeledImages/OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2/OASIS-TRT-20-1_in_MNI152.nii.gz"
    brain_image = nib.load(fpath1).get_fdata()
    fpath2 = "shapBrainInterpretation/LabeledImages/OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2/OASIS-TRT-20-1_DKT31_CMA_labels_in_MNI152.nii.gz"
    segmented_image = nib.load(fpath2).get_fdata()
    absK['counts'] = 0
    absK['PATID'] = absK.index
    absK['class_indicator'] = 0

    number_images = absK.groupby(groupy_columns).agg({'counts': 'count', 'PATID': lambda x: len(np.unique(x)), 'class_indicator': 'sum'})

    from itertools import product
    all_params = list(product(*[sorted(absK[col].dropna().unique()) for col in groupy_columns]))
    prediction_task = "ADTASK"
    sns.set(context='notebook', style='ticks', font_scale=2, font='sans-serif', rc={"lines.linewidth": 1.2})

    for key, value in dataset_groupby.items():
        for key2, value2 in value.items():
            n_columns = 3
            num_rows = 3
            fig, axs = plt.subplots(nrows=num_rows, ncols=n_columns, figsize=(182 * n_columns / 30, 9 * num_rows))
            plt.subplots_adjust(hspace=0)
            axslist = axs.reshape(-1)
            e = 0
            for slice_number, axis_number in product([60, 90, 120], [0, 1, 2]):
                ax = axslist[e]
                print(group['name'], axis_number, slice_number, key, key2)
                if axis_number == 0:
                    numpy_image = segmented_image[slice_number, :, :]
                elif axis_number == 1:
                    numpy_image = segmented_image[:, slice_number, :]
                else:
                    numpy_image = np.rot90(segmented_image[:, :, slice_number])
                G = value2.copy()
                SEG_NUMPY = pd.DataFrame(numpy_image.astype(int))
                A = np.unique(SEG_NUMPY)
                vmin_min = np.percentile(G.values, 5)  # G.min().min()
                vmax_max = np.percentile(G.values, 95)  # G.max().max()
                ax.set_axis_off()
                replacement_dict = dict(zip(G.columns.get_level_values(-1).map(lambda x: x[1]), G.loc[all_params[0]]))
                for i in A:
                    replacement_dict[i] = replacement_dict.get(i, np.nan)
                V = SEG_NUMPY.copy().replace(replacement_dict).values
                cmap = copy.copy(mpl.cm.get_cmap(cmap_name))
                cmap.set_bad("white")
                ax.set_axis_off()
                if False and e == 8: # (e + 1) % n_columns == 0:
                    # import pdb; pdb.set_trace()
                    sns.heatmap(V, cmap=cmap, mask=V == np.nan, ax=ax, vmax=vmax_max, vmin=vmin_min, center=0,
                                rasterized=True,
                                cbar_kws=dict(use_gridspec=False, location="bottom", fraction=0.05))
                else:
                    sns.heatmap(V, cmap=cmap, mask=V == np.nan, ax=ax, vmax=vmax_max, vmin=vmin_min, center=0,
                                cbar=None, cbar_ax=None, rasterized=True)

                e += 1
            fig.tight_layout()
            plt.savefig(f"seeImages/{image_name}.png", dpi=100 // num_rows)
            plt.close()
            # import sys; sys.exit()








# plt.savefig(f'image_progression/{prediction_task}&{group}&{class_label}.svg', dpi=200)
# plt.savefig(f'image_progression/{prediction_task}&{group}&{class_label}.pdf', dpi=400)


# FeatureImportance['region_index'] = FeatureImportance['region_name'].map(lambda x: LABEL_DICT.get(x.split('@')[-1].split('_')[0], -1))
# FeatureImportance['short_region_index'] = F.map(lambda x: x.split('@')[-1].split('_')[0])
"""
    FeatureImportance.loc[396] = ['bias', 1000000, 'bias']
    results_data = {}
    for class_label, class_index in class_label_dict.items():
        if not class_label == select_class_label:
            continue
        results_data[class_label] = {}
        P = shap_data[0][0]['shap_values_rep'][:, class_index, :].transpose()
        P_df = pd.DataFrame(P, columns=[f's{i}' for i in range(P.shape[1])])
        P_df = pd.concat([FeatureImportance[['short_region_index', 'region_index']], P_df], axis=1).rename(
            columns={'short_region_index': 'region_name'}).set_index('region_name')
        P_df = P_df.iloc[:-1, :]
        results_data[class_label]['SHAP_df'] = P_df.copy()

    mean_df = P_df.reset_index().groupby(['region_name', 'region_index']).agg('mean').transpose()
    mean_df = mean_df.iloc[:, ~ (mean_df.columns.get_level_values('region_index') == -1)]
    from collections import defaultdict
    df_dict = defaultdict(list)
    for i in mean_df.columns:
        df_dict['region_name'].append(i[0])
        df_dict['region_index'].append(i[1])
    S = pd.DataFrame(df_dict)
    S['symbol'] = S['region_name'].map(lambda x: ''.join([i[0].upper() for i in x.split()]))
    S.to_csv('feature_mapping.csv', index=False)

    # mean_df.columns = mean_df.columns.get_level_values('region_name')
    # selected_columns = mean_df.columns.get_level_values('region_name')

    absmean_df = P_df.abs().reset_index().groupby(['region_name', 'region_index']).agg('mean').transpose()
    absmean_df = absmean_df.iloc[:, ~ (absmean_df.columns.get_level_values('region_index') == 1)]
    # absmean_df.columns = absmean_df.columns.get_level_values('region_name')

    absK = pd.concat([absmean_df, global_metadata], axis=1)
    meanK = pd.concat([mean_df, global_metadata], axis=1)

    selected_columns = [col for col in absK.columns if type(col) is tuple]

    def normalize(data, axis=0):
        if axis == 0:
            return (data - data.mean(axis=0)) / data.std(axis=0)
        else:
            return ((data.transpose() - data.transpose().mean(axis=0)) / (data.transpose().std(axis=0))).transpose()

    def minmax_normalize(data, axis=0):
        if axis == 0:
            return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        else:
            for i in range(len(data)):
                max_value = data.iloc[i].quantile(0.95)
                data.iloc[i] = data.iloc[i].map(lambda x: min(x, max_value))

            return ((data.transpose() - data.transpose().min(axis=0)) / (
                        data.transpose().max(axis=0) - data.transpose().min(axis=0))).transpose()

    def sum_normalize(data, axis=0):
        if axis == 0:
            return (data) / (data.sum(axis=0))
        else:
            for i in range(len(data)):
                max_value = data.iloc[i].quantile(0.95)
                data.iloc[i] = data.iloc[i].map(lambda x: min(x, max_value))

            return ((data.transpose()) / (data.transpose().sum(axis=0))).transpose()

    if len(group) == 0:
        groupy_columns = ['SELECT', 'AGE']
    else:
        groupy_columns = [group['name'], 'AGE']

    meanK = meanK.dropna(subset=groupy_columns)
    g = list(meanK.groupby(groupy_columns))
    average_risk = {}
    for i in range(len(g)):
        average_risk[g[i][0]] = np.mean(g[i][1][selected_columns].values)

    relative_meanK = meanK.copy()
    z = relative_meanK.set_index(groupy_columns).index
    for col in selected_columns:
        relative_meanK[col] = relative_meanK[col] - z.map(lambda x: average_risk[x])

    # meanK = relative_meanK.copy()
    if not len(group) == 0:
        if not len(group['filter']) == 0:
            absK = absK[absK[group['name']].isin(group['filter'])]
            meanK = meanK[meanK[group['name']].isin(group['filter'])]
            relative_meanK = relative_meanK[relative_meanK[group['name']].isin(group['filter'])]

    # absmean_absmean_df = absK.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})
    mean_mean_df = meanK.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})
    # import pdb; pdb.set_trace()
    relativemean_mean_df = relative_meanK.groupby(groupy_columns).agg(
        {col: lambda x: x.mean(axis=0) for col in selected_columns})
    # abs_mean_df = meanK.groupby(groupy_columns).agg({col: lambda x: x.abs().mean(axis=0) for col in selected_columns})
    # import pdb; pdb.set_trace()
    # L = meanK.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})
    # L = (L.transpose() - L.transpose().mean(axis=0)).transpose()
    # relativeabs_mean_df = abs_mean_df.copy()
    # import pdb; pdb.set_trace()
    # relativeabs_mean_df = abs_mean_df.copy()
    # for i in range(len(relativeabs_mean_df)):
    #     relativeabs_mean_df.iloc[i] = relativeabs_mean_df.iloc[i] - relativeabs_mean_df.mean(axis=1).iloc[i]

    dataset_groupby = {}
    # data_d = {'absmean_absmean_df': absmean_absmean_df, 'absmean_mean_df': abs_mean_df, 'mean_mean_df': mean_mean_df}}
    # data_d = {'relativemean_mean_df': relativemean_mean_df, 'mean_mean_df': mean_mean_df}# , 'relativeabsmean_mean_df': relativeabs_mean_df}
    data_d = {'relativemean_mean_df': relative_meanK,
              'mean_mean_df': meanK}  # , 'relativeabsmean_mean_df': relativeabs_mean_df}

    for key, value in data_d.items():
        dataset_groupby[key] = {}
        # value_temp = value.copy()
        # value_temp.loc[:, selected_columns] = normalize( value_temp[selected_columns] , axis=0)
        # dataset_groupby[key]['norm_axis0'] = value_temp.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})

        # value_temp = value.copy()
        # value_temp.loc[:, selected_columns] = normalize( value_temp[selected_columns] , axis=1)
        # dataset_groupby[key]['norm_axis1'] = value_temp.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})

        # value_temp = value.copy()
        # value_temp.loc[:, selected_columns] = minmax_normalize( value_temp[selected_columns] , axis=0)
        # dataset_groupby[key]['minmax_axis0'] = value_temp.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})

        # value_temp = value.copy()
        # value_temp.loc[:, selected_columns] = minmax_normalize( value_temp[selected_columns] , axis=1)
        # dataset_groupby[key]['minmax_axis1'] = value_temp.groupby(groupy_columns).agg({col: lambda x: x.mean(axis=0) for col in selected_columns})

        dataset_groupby[key]['unnormalized_data'] = value.groupby(groupy_columns).agg(
            {col: lambda x: x.mean(axis=0) for col in selected_columns})
        # import pdb; pdb.set_trace()
        # dataset_groupby[key]['norm_axis1_minmax_axis0'] = minmax_normalize(dataset_groupby[key]['norm_axis1'], axis=0)
    absK['counts'] = 0
    number_images = absK.groupby(groupy_columns).agg(
        {'counts': 'count', 'PATID': lambda x: len(np.unique(x)), 'class_indicator': 'sum'})

    from itertools import product
    all_params = list(product(*[sorted(absK[col].dropna().unique()) for col in groupy_columns]))

    fpath1 = "LabeledImages/OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2/OASIS-TRT-20-1_in_MNI152.nii.gz"
    brain_image = nib.load(fpath1).get_fdata()
    fpath2 = "LabeledImages/OASIS-TRT-20_DKT31_CMA_labels_in_MNI152_v2/OASIS-TRT-20-1_DKT31_CMA_labels_in_MNI152.nii.gz"
    segmented_image = nib.load(fpath2).get_fdata()
    from itertools import product
    for key, value in dataset_groupby.items():
        for key2, value2 in value.items():
            for slice_number, axis_number in product([60, 90, 120], [0, 1, 2]):
                print(group['name'], axis_number, slice_number, key, key2)
                if len(value2) == 0 or os.path.exists(
                        f"my_image_progression/{axis_number}&{slice_number}&{prediction_task}&{group['name']}&{key}_{key2}.png"):
                    continue
                if axis_number == 0:
                    numpy_image = segmented_image[slice_number, :, :]
                elif axis_number == 1:
                    numpy_image = segmented_image[:, slice_number, :]
                else:
                    numpy_image = np.rot90(segmented_image[:, :, slice_number])
                G = value2.copy()
                SEG_NUMPY = pd.DataFrame(numpy_image.astype(int))
                # import pdb; pdb.set_trace()
                # SEG_NUMPY = np.rot90(SEG_NUMPY)
                A = np.unique(SEG_NUMPY)
                n_columns = 6
                sns.set(context='notebook', style='ticks', font_scale=2, font='sans-serif', rc={"lines.linewidth": 1.2})
                if 'norm_' in key2:
                    # import pdb; pdb.set_trace()
                    vmin_min = -1.5
                    # vmin_min = np.percentile(G.values, 5)# G.min().min()
                    vmax_max = 1.5
                    # vmax_max = np.percentile(G.values, 95) # G.max().max()
                elif 'minmax' in key2:
                    vmin_min = 0
                    vmax_max = 1
                else:
                    vmin_min = np.percentile(G.values, 5)  # G.min().min()
                    vmax_max = np.percentile(G.values, 95)  # G.max().max()
                if len(group) == 0:
                    num_rows = 1
                    fig, axs = plt.subplots(nrows=1, ncols=n_columns, figsize=(182 * n_columns / 30, 9))
                else:
                    num_rows = len(all_params) // n_columns
                    if num_rows == 0:
                        fig, axs = plt.subplots(nrows=1, ncols=n_columns, figsize=(182 * n_columns / 30, 9))
                    else:
                        fig, axs = plt.subplots(nrows=len(all_params) // n_columns, ncols=n_columns,
                                                figsize=(182 * n_columns / 30, 9 * len(all_params) / (n_columns)))

                plt.subplots_adjust(hspace=0.1)
                axslist = axs.reshape(-1)
                for e, ax in enumerate(axslist):
                    title = '_'.join([f"{name}: {all_params[e][enm]}" for enm, name in enumerate(G.index.names)])
                    if not all_params[e] in G.index:
                        title += "_Num images=0"
                        ax.set_axis_off()
                    elif number_images.loc[all_params[e]][0] == 0:
                        title += "_Num images=0"
                        ax.set_axis_off()


                    else:
                        title += '_Num images=' + str(
                            number_images.loc[all_params[e]][0]) + '\nNum images(+ class)=' + str(
                            number_images.loc[all_params[e]][2]) + '\nNum subjects=' + str(
                            number_images.loc[all_params[e]][1])
                        replacement_dict = dict(zip(G.columns.get_level_values(-1), G.loc[all_params[e]]))
                        for i in A:
                            replacement_dict[i] = replacement_dict.get(i, np.nan)
                        V = SEG_NUMPY.copy().replace(replacement_dict).values
                        cmap = copy.copy(mpl.cm.get_cmap("coolwarm"))
                        # cmap = copy.copy(mpl.cm.get_cmap("PiYG"))
                        # import pdb; pdb.set_trace()
                        cmap.set_bad("white")
                        ax.set_axis_off()
                        if (e + 1) % n_columns == 0:
                            sns.heatmap(V, cmap=cmap, mask=V == np.nan, ax=ax, vmax=vmax_max, vmin=vmin_min, center=0,
                                        rasterized=True,
                                        cbar_kws=dict(use_gridspec=False, location="bottom", fraction=0.05))
                        else:
                            sns.heatmap(V, cmap=cmap, mask=V == np.nan, ax=ax, vmax=vmax_max, vmin=vmin_min, center=0,
                                        cbar=None, cbar_ax=None, rasterized=True)
                    if e == 0:
                        if not num_rows == 1:
                            ax.set_title(title.replace('_', '\n'))
                        else:
                            ax.set_title(title.split('_')[1] + '\n' + title.split('_')[-1])
                    elif e // n_columns == 0:
                        ax.set_title(title.split('_')[1] + '\n' + title.split('_')[-1])
                    elif e % n_columns == 0:
                        ax.set_title(title.split('_')[0] + '\n' + title.split('_')[-1])
                    else:
                        ax.set_title(title.split('_')[-1])
                    fig.tight_layout()

                plt.savefig(
                    f"my_image_progression/{axis_number}&{slice_number}&{prediction_task}&{group['name']}&{key}_{key2}.png",
                    dpi=100 // num_rows)

        # plt.savefig(f'image_progression/{prediction_task}&{group}&{class_label}.svg', dpi=200)
        # plt.savefig(f'image_progression/{prediction_task}&{group}&{class_label}.pdf', dpi=400)


if sys.argv[1] == 'updrs':
    prediction_task = 'GPU_MDS-UPDRSPartIII_2GROUPS_PREDICTION'
    select_class_label = '15-95'
    ename = "oct11"
    groups = [{'name': "SELECT", 'filter': []}]
    groups = [{'name': "SELECT", 'filter': []}, {'name': "LRRK2 Mutation", 'filter': []},
              {'name': "APOE4 Mutation", 'filter': []}, {'name': "GBA Mutation", 'filter': []},
              {'name': 'STUDY', 'filter': []}, {'name': 'DIAGNOSIS', 'filter': ['MCI', 'Control', 'Dementia', 'PD']}]
    for e in range(len(groups)):
        generate_progression_plots(prediction_task, ename, select_class_label, group=groups[e])

# , "Gene Mutation",  "LRRK2 Mutation"]# , "GBA Mutation", "DIAGNOSIS"]# "APOE4 Mutation" ] # selections = [[], ["PD", "Control"], [0, 1], [0, 1]]

elif sys.argv[1] == 'moca':
    prediction_task = 'GPU_MOCA_3GROUPS_PREDICTION'
    groups = [{'name': "SELECT", 'filter': []}]
    groups = [{'name': "SELECT", 'filter': []}, {'name': "LRRK2 Mutation", 'filter': []},
              {'name': "APOE4 Mutation", 'filter': []}, {'name': "GBA Mutation", 'filter': []},
              {'name': 'STUDY', 'filter': []}, {'name': 'DIAGNOSIS', 'filter': ['MCI', 'Control', 'Dementia', 'PD']}]
    select_class_label = '-1-24'
    ename = "oct10"
    # groups = ['DIAGNOSIS']# ['MOCA_3GROUPS', ]#"DIAGNOSIS"]#, 'APOE4 Mutation']# , "Gene Mutation", "STUDY", "DIAGNOSIS", "LRRK2 Mutation", "GBA Mutation", 'APOE4 Mutation' ]
    # selections = [['Dementia', 'Control'], ["ADNI", "PPMI"], [], [], [], [], []]
    for e in range(len(groups)):
        generate_progression_plots(prediction_task, ename, select_class_label, group=groups[e])

elif sys.argv[1] == 'diagnosis':
    prediction_task = 'GPU_DIAGNOSIS_PREDICTION'
    groups = ["SELECT", "STUDY", "DIAGNOSIS", "Gene Mutation", "LRRK2 Mutation", "GBA Mutation", 'APOE4 Mutation']
    selections = [[], ["ADNI", "PPMI"], [], [], [], [], [], []]  # , ["Dementia", "MCI", "PD", "Control"] ]
    for e in range(len(groups)):
        plot_image_progression(prediction_task, ename='oct10', group=groups[e], selection=selections[e])



"""