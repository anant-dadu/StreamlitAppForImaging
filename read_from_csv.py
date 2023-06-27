import pandas as pd
with open("ad_top20_feature_list.txt", 'r') as f:
    ad_top20_features = f.read().split("\n")

with open("pd_top20_feature_list.txt", 'r') as f:
    pd_top20_features = f.read().split("\n")

imaging_features = pd.read_csv("28364-00000000-T1w-00.nii.gz_extracted_image_features.csv")
imaging_features.columns = [col.replace('/', '_').replace('-', '_').lower() for  col in imaging_features.columns]
ad_top20_features = [col.replace(' ', '_') for col in ad_top20_features]
pd_top20_features = [col.replace(' ', '_') for col in pd_top20_features]
ad_intersected_features = list(set(ad_top20_features).intersection(set(imaging_features.columns)))
pd_intersected_features = list(set(pd_top20_features).intersection(set(imaging_features.columns)))
F = pd.read_csv("min_max_values.csv")
F = F.set_index('feature_name')
F.index = F.index.map(lambda x: x.replace('id_invicrot1_', ''))
ad_dataframe = (imaging_features[ad_intersected_features] - F.loc[ad_intersected_features]['min_value']) / (F.loc[ad_intersected_features]['max_value'] - F.loc[ad_intersected_features]['min_value'])
pd_dataframe = (imaging_features[pd_intersected_features] - F.loc[pd_intersected_features]['min_value']) / (F.loc[pd_intersected_features]['max_value'] - F.loc[pd_intersected_features]['min_value'])


# set(pd_top20_features).difference(set(imaging_features.columns))
# set(ad_top20_features).difference(set(imaging_features.columns))
import anndata as ad
adata = ad.read("adata.h5ad")
raw_data = pd.DataFrame(adata.X, columns=adata.var.index)

F1 = raw_data.max(axis=0).reset_index()
F1.columns = ['feature_name', 'max_value']

F2 = raw_data.min(axis=0).reset_index()
F2.columns = ['feature_name', 'min_value']
F = pd.merge(F1, F2)
F.to_csv("/Users/dadua2/Projects/WebDevelopmentStreamlits/nddsimaging/StreamlitAppForImaging/min_max_values.csv", index=False, sep=',')

