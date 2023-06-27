import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
import plotly.express as px
from pathlib import Path
from collections import defaultdict
import plotly
import copy
import matplotlib.pyplot as plt
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
import joblib
import joblib
import lightgbm as lgb
import seaborn as sns
import plotly.express as px
import altair
import subprocess
import requests
import pipes

def process_name(x):
    return x.replace('id_invicrot1_', '').replace('_', ' ')

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def store_data(uploaded_file, save_name):
    with open(save_name, 'wb') as out:
        out.write(uploaded_file.getbuffer())

    result = subprocess.run(f"scp {save_name} dadua2@helix.nih.gov:/home/dadua2/Deployment/driveData/rawImages/", shell=True)
    if not result.stderr:
        st.info("File transferred to server")
    else:
        raise Exception('File transfer fail')

def performAPI(image_id):
    r_file_exists = requests.get(f"http://127.0.0.1:45227/checkImagingRaw/{image_id}")
    r_features_exists = requests.get(f"http://127.0.0.1:45227/checkImagingFeatures/{image_id}")
    file_exists = 0 if r_file_exists.status_code == 404 else 1
    features_exists = 0 if r_features_exists.status_code == 404 else 1
    if file_exists:
        if not features_exists:
            r_status = requests.get(f"http://127.0.0.1:45227/imageStatus/{image_id}")
            if r_status.status_code == 404:
                st.info("Preprocessing has not started yet.")
                if st.button("Click to submit"):
                    response = requests.get(f"http://127.0.0.1:45227/extractImagingFeatures/{image_id}")
                    if response.status_code == 200:
                        st.info(f"Your image processing job is submitted. Please note the link to check the results. It will take about 30-45 minutes to process. Link here {image_id}")
            else:
                temp = r_status.json()['job_status'].strip()
                if temp == "running":
                    st.info("Your image is under process...")
                elif temp == "completed with error":
                    st.info("Some error occured. Please rerun")
                    if st.button("Click to re-submit"):
                        response = requests.get(f"http://127.0.0.1:45227/extractImagingFeatures/{image_id}")
                        if response.status_code == 200:
                            st.info(f"Your image processing job is submitted. Please note the link to check the results. It will take about 30-45 minutes to process. Link here {image_id}")
                else:
                        st.info("should be successfully completed")
        else:
            st.info("Congrats! your image has processed. Check diagnostic reports.")
    else:
        st.info("No such image exists.")
    return features_exists, r_features_exists

def app():
    st.markdown("""<style>.big-font {font-size:100px !important;}</style>""", unsafe_allow_html=True) 
    st.markdown(
        """<style>
        .boxBorder {
            border: 2px solid #990066;
            padding: 10px;
            outline: #990066 solid 5px;
            outline-offset: 5px;
            font-size:25px;
        }</style>
        """, unsafe_allow_html=True) 
    st.markdown('<div class="boxBorder"><font color="RED">Disclaimer: This predictive tool is only for research purposes</font></div>', unsafe_allow_html=True)
    st.write("## Model Perturbation Analysis")

    get_params = st.experimental_get_query_params()
    # st.experimental_set_query_params({})
    image_id = None
    features_exists = False
    # st.write(get_params)
    if not len(get_params) == 0:
        image_id = get_params['image_id'][0]
        features_exists, r_features_exists = performAPI(image_id)
    else:
        uploaded_file = st.file_uploader("Upload nifti file")
        if not uploaded_file:
            pass
        else:
            image_id = uploaded_file.name
            r_file_exists = requests.get(f"http://127.0.0.1:45227/checkImagingRaw/{image_id}")
            file_exists = 0 if r_file_exists.status_code == 404 else 1
            if file_exists:
                features_exists, r_features_exists = performAPI(image_id)
            else:
                # if data is not there submit it as process cannot be running
                # should use features_exists, r_features_exists = performAPI(image_id)
                # but the problem is uploading time and performAPI shows no file exists
                store_data(uploaded_file, image_id)
                response = requests.get(f"http://127.0.0.1:45227/extractImagingFeatures/{image_id}")
                if not features_exists and response.status_code==200:
                    st.info(f"Your image processing job is submitted. Please note the link to check the results. It will take about 30-45 minutes to process. Link here {image_id}")

    if features_exists:
        with open("ad_top20_feature_list.txt", 'r') as f:
            ad_top20_features = f.read().split("\n")

        with open("pd_top20_feature_list.txt", 'r') as f:
            pd_top20_features = f.read().split("\n")
        imaging_features = pd.read_json(r_features_exists.json()['imaging_features'])
        imaging_features.columns = [col.replace('/', '_').replace('-', '_').lower() for col in imaging_features.columns]
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

        ad_dataframe.columns = ad_dataframe.columns.map(lambda x: f'id_invicrot1_{x}')
        pd_dataframe.columns = pd_dataframe.columns.map(lambda x: f'id_invicrot1_{x}')
        # st.write(ad_dataframe)
        # st.write(X)
        # st.write(ad_dataframe)
        # st.write(ad_dataframe[X.columns])
        # st.write(X)
        csv = convert_df(imaging_features)
        # col1, col2 = st.columns(2)
        st.download_button(
            label="Download extracted features as CSV",
            data=csv,
            file_name=f'{image_id}_extracted_image_features.csv',
            mime='text/csv',
        )

    else:
        image_id = None

    select_disease = st.selectbox("Select the disease", options=['ADRD', 'PD'])
    if select_disease == 'ADRD':
        # fname = "/Users/dadua2/EssentialCodeBase/project_MLPhenotypesMRIGWAS/figures4paper/0.FINALFOLDER/LIGHTGBM_MODEL/lightgbm_ad_shap.pkl"
        lightgbm_model = joblib.load('ad_reduced_lgb.pkl')
    else:
        # fname = "/Users/dadua2/EssentialCodeBase/project_MLPhenotypesMRIGWAS/figures4paper/0.FINALFOLDER/LIGHTGBM_MODEL/lightgbm_pd_shap.pkl"
        lightgbm_model = joblib.load('pd_reduced_lgb.pkl')

    X = pd.read_csv(f"sample_dataset_{select_disease}.csv")
    X.index = list(range(X.shape[0]))
    # get_params = st.experimental_get_query_params()
    # if get_params['image_id']
    # st.experimental_set_query_params(
    #     image_id="IIII",
    # )
    # response = requests.get("https://antspyt1w-htco5r3hya-ue.a.run.app/justants")
    # st.write(response.status_code)
    # st.write(response.json())


    if image_id is not None:
        X.loc[len(X)] = list(ad_dataframe[X.columns].iloc[0]) if select_disease == 'ADRD' else list(pd_dataframe[X.columns].iloc[0])
        select_patient = len(X) - 1
        st.write("### Features filled using normalized values of uploaded image.")

    else:
        st.write('### Please enter the following {} factors to perform prediction or select a random patient'.format(len(X.columns)))
        if st.button("Random Patient"):
            import random
            select_patient = random.choice(list(X.index))
        else:
            select_patient = list(X.index)[0]
            # select_patient = 904

    feature_mapping = {i: process_name(i) for i in X.columns}
    categorical_columns = []
    numerical_columns = []
    X_new = X.fillna('Not available')
    for col in X_new.columns:
        numerical_columns.append(col)

    # select_patient_index = ids.index(select_patient)
    new_feature_input = defaultdict(list) 

    st.write('--'*10)
    st.write('##### Note: X denoted NA values')
    col1, col2, col3, col4 = st.columns(4)
    
    for col in numerical_columns:
        X_new[col] = X_new[col].map(lambda x: float(x) if not x=='Not available' else np.nan)

    for i in range(0, len(numerical_columns), 4):
        with col1:
            if (i+0) >= len(numerical_columns):
                continue
            c1 = numerical_columns[i+0] 
            idx = X_new.loc[select_patient, c1]
            # f1 = st.number_input("{}".format(feature_mapping[c1]), min_value=X_new[c1].min(),  max_value=X_new[c1].max(), value=idx)
            f1 = st.number_input("{}".format(feature_mapping[c1]), min_value=0.0,  max_value=1.0, value=idx)

            new_feature_input[c1].append(f1)
        with col2:
            if (i+1) >= len(numerical_columns):
                continue
            c2 = numerical_columns[i+1] 
            idx = X_new.loc[select_patient, c2]
            f2 = st.number_input("{}".format(feature_mapping[c2]), min_value=0.0,  max_value=1.0, value=idx)
            new_feature_input[c2].append(f2)
        with col3:
            if (i+2) >= len(numerical_columns):
                continue
            c3 = numerical_columns[i+2] 
            idx = X_new.loc[select_patient, c3]
            f3 = st.number_input("{}".format(feature_mapping[c3]), min_value=0.0,  max_value=1.0, value=idx)
            new_feature_input[c3].append(f3)
        with col4:
            if (i+3) >= len(numerical_columns):
                continue
            c4 = numerical_columns[i+3] 
            idx = X_new.loc[select_patient, c4]
            f4 = st.number_input("{}".format(feature_mapping[c4]), min_value=0.0,  max_value=1.0, value=idx)
            new_feature_input[c4].append(f4)
    
    st.write('--'*10)
    st.write("### Do you want to see the effect of changing a factor on this patient?")
    color_discrete_map = {}
    color_discrete_map_list = ["red", "green", "blue", "goldenred", "magenta", "yellow", "pink", "grey"]
    class_names = ['Control', select_disease]
    for e, classname in enumerate(class_names):
        color_discrete_map[classname] = color_discrete_map_list[e] 
    
    show_whatif = st.checkbox("Enable what-if analysis")
    col01, col02 = st.columns(2)
    with col01:
        st.write('### Prediction on actual feature values')
        if not show_whatif:
            dfl = pd.DataFrame(new_feature_input)
            ndfl = dfl.copy()
            feature_print_what = ndfl.iloc[0].fillna('Not available')
            feature_print_what.index = feature_print_what.index.map(lambda x: feature_mapping[x])
            feature_print_what = feature_print_what.reset_index()
            feature_print_what.columns = ["Feature Name", "Feature Value"]
            feature_print = feature_print_what.copy()
            dfl = dfl[X.columns].replace('Not available', np.nan)
            predicted_prob = {'predicted_probability': [], 'classname': []}
            predicted_class = -1
            max_val = -1

            outi = lightgbm_model.predict(dfl.iloc[0, :].values.reshape(1, -1))
            trans = lambda x: (2 ** x) / (1 + 2 ** x)
            predicted_prob['predicted_probability'].append(trans(outi))
            predicted_prob['classname'].append(select_disease)
            predicted_prob['predicted_probability'].append(1-trans(outi))
            predicted_prob['classname'].append('Control')
            predicted_class = select_disease if trans(outi) > 0.5 else 'Control'
            max_val = max(trans(outi), 1-trans(outi))

            K = pd.DataFrame(predicted_prob)
            K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
            K['color'] = ['zed' if i == predicted_class else 'red' for i in list(predicted_prob['classname'])]
            t1 = dfl.copy()
            t2 = ndfl.copy().fillna('Not available')
        else:
            # st.write(X_new.loc[select_patient, :])
            # X_new.to_csv('/app/HELLOJI.csv', index=False)
            # print ('oHELKLO')
            # X_new.loc[select_patient, :] =  [np.nan, 'definite', 'bulbar', 'bulbar', np.nan, 2, 92, 75, 0, 72.833, 314]
            feature_print = X_new.loc[select_patient, :].fillna('Not available')
            # feature_print.iloc[:, 1] = ['never', 'definite', 'bulbar', 'bulbar', 'Not available', '2']
            feature_print.index = feature_print.index.map(lambda x: feature_mapping[x])
            feature_print = feature_print.reset_index()
            feature_print.columns = ["Feature Name", "Feature Value"]
            # feature_print.
            predicted_prob = {'predicted_probability': [], 'classname': []}
            predicted_class = -1
            max_val = -1

            outi = lightgbm_model.predict(X.loc[select_patient, :].values.reshape(1, -1))
            trans = lambda x: (2 ** x) / (1 + 2 ** x)
            predicted_prob['predicted_probability'].append(trans(outi))
            predicted_prob['classname'].append(select_disease)
            predicted_prob['predicted_probability'].append(1 - trans(outi))
            predicted_prob['classname'].append('Control')
            predicted_class = select_disease if trans(outi) > 0.5 else 'Control'
            max_val = max(trans(outi), 1 - trans(outi))
            K = pd.DataFrame(predicted_prob)
            K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
            K['color'] = ['zed' if i == predicted_class else 'red' for i in list(predicted_prob['classname'])]
            t1 = pd.DataFrame(X.loc[select_patient, :]).T
            t2 = pd.DataFrame(X_new.loc[select_patient, :].fillna('Not available')).T

        st.table(feature_print.set_index("Feature Name").astype(str))

        K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"}).copy()
        K['Predicted Probability'] = K['Predicted Probability'].map(float)
        # st.write(K)
        # st.write(list(K['Class Labels']))
        # st.write(pd.DataFrame(K).iloc[0])
        # fig, ax = plt.subplots()
        # sns.barplot(x=list(K['Class Labels']), y=list(K['Predicted Probability']))
        # st.pyplot(fig)
        # fig = px.bar(K, x='Class Labels', y='Predicted Probability')
        # st.plotly_chart(fig, use_container_width=True)


        f = altair.Chart(K).mark_bar().encode(
                     y=altair.Y('Class Labels:N',sort=altair.EncodingSortField(field="Predicted Probability", order='descending')),
                     x=altair.X('Predicted Probability:Q', scale=altair.Scale(domain=[0, 1])),
                     color=altair.Color('color', legend=None),
                 ).properties(width=500, height=300)
        st.write(f)
        # st.write('#### Trajectory for Predicted Class')
        st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))

        explainer = shap.TreeExplainer(lightgbm_model)
        my_shap_values = explainer.shap_values(t1.values)
        t1 = t2.copy()
        t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
        shap.force_plot(explainer.expected_value, my_shap_values, t1.round(2), show=False, matplotlib=True, contribution_threshold=0.1)
        st.pyplot()
        t2.columns = t2.columns.map(lambda x: feature_mapping.get(x, x))
        r = shap.decision_plot(explainer.expected_value, my_shap_values, t2.round(2), return_objects=True, new_base_value=0, highlight=0)
        st.pyplot()

    if show_whatif:
        with col02:
            dfl = pd.DataFrame(new_feature_input)
            ndfl = dfl.copy()
            st.write('### Prediction with what-if analysis')
            t2 = ndfl.copy().fillna('Not available')
            feature_print_what = ndfl.iloc[0].fillna('Not available')
            feature_print_what.index = feature_print_what.index.map(lambda x: feature_mapping[x])
            feature_print_what = feature_print_what.reset_index()
            feature_print_what.columns = ["Feature Name", "Feature Value"] 
            selected = []
            for i in range(len(feature_print_what)):
                if feature_print.iloc[i]["Feature Value"] == feature_print_what.iloc[i]["Feature Value"]:
                    pass
                else:
                    selected.append(feature_print.iloc[i]["Feature Name"])

            # st.table(feature_print)

            st.table(feature_print_what.astype(str).set_index("Feature Name").style.apply(lambda x: ['background: yellow' if (x.name in selected) else 'background: lightgreen' for i in x], axis=1))
            dfl = dfl[X.columns].replace('Not available', np.nan)
            predicted_prob = defaultdict(list)
            predicted_class = -1
            max_val = -1

            outi = lightgbm_model.predict(dfl.iloc[0, :].values.reshape(1, -1))
            trans = lambda x: (2 ** x) / (1 + 2 ** x)
            predicted_prob['predicted_probability'].append(trans(outi))
            predicted_prob['classname'].append(select_disease)
            predicted_prob['predicted_probability'].append(1 - trans(outi))
            predicted_prob['classname'].append('Control')
            predicted_class =select_disease if trans(outi) > 0.5 else 'Control'
            max_val = max(trans(outi), 1 - trans(outi))


            K = pd.DataFrame(predicted_prob)
            K['predicted_probability'] = K['predicted_probability'] / K['predicted_probability'].sum()
            K['color'] = ['zed' if i==predicted_class else 'red' for i in list(predicted_prob['classname']) ]
            K = K.rename(columns={"classname": "Class Labels", "predicted_probability": "Predicted Probability"})
            K['Predicted Probability'] = K['Predicted Probability'].map(float)
            f = altair.Chart(K).mark_bar().encode(
                y=altair.Y('Class Labels:N',sort=altair.EncodingSortField(field="Predicted Probability", order='descending')),
                    x=altair.X('Predicted Probability:Q', scale=altair.Scale(domain=[0, 1])),
                    color=altair.Color('color', legend=None),
                ).properties( width=500, height=300)
            st.write(f)

            st.write('#### Model Output Trajectory for {} Class using SHAP values'.format(predicted_class))

            t1 = dfl.copy()
            explainer = shap.TreeExplainer(lightgbm_model)
            my_shap_values = explainer.shap_values(t1.values)
            # t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
            t1 = t2.copy()  # ndfl.copy().fillna('Not available')
            t1.columns = t1.columns.map(lambda x: feature_mapping.get(x, x).split(' (')[0])
            shap.force_plot(explainer.expected_value, my_shap_values, t1.round(2), show=False, matplotlib=True, contribution_threshold=0.1) # , link='logit'
            st.pyplot()
            # plt.savefig("/app/mar4_force_plot.pdf", bbox_inches='tight')
            # plt.savefig("/app/mar4_force_plot.eps", bbox_inches='tight')
            t2.columns = t2.columns.map(lambda x: feature_mapping.get(x, x))
            _ = shap.decision_plot(explainer.expected_value, my_shap_values, t2.round(2), feature_order=r.feature_idx, return_objects=True, new_base_value=0, highlight=0) # , link='logit'
            # fig.savefig('/app/mar4_decisionplot.pdf', bbox_inches='tight')
            # fig.savefig('/app/mar4_decisionplot.eps', bbox_inches='tight')
            st.pyplot()


app()