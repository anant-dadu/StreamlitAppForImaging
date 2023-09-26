import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

import warnings
warnings.filterwarnings("ignore")
import logging


logger = logging.getLogger()
logger.disabled = True


st.title("Prediction, prognosis and monitoring of neurodegeneration at biobank-scale via machine learning and imaging.")
st.header("Summary")
st.subheader("Background")
st.write("Alzheimer’s disease and related dementias (ADRD) and Parkinson’s disease (PD) are the most common neurodegenerative disorders. These are multisystem disorders affecting different body parts and functions. Patients with ADRD or PD have long asymptomatic phases and exhibit significant etiology or clinical manifestations heterogeneity. Hence, quantitative measures that can provide early disease indicators are necessary to improve patient stratification, clinical care, and clinical trial design. This work uses machine learning techniques to derive such a quantitative marker from T1-weighted (T1w) brain Magnetic resonance imaging (MRI).")

st.subheader("Methods")
st.write("In this retrospective study, we developed a machine learning (ML) based score of T1w brain MRI image utilizing disease-specific Parkinson's Disease Progression Marker Initiative (PPMI) and Alzheimer's Disease Neuroimaging Initiative (ADNI) cohorts. Then, we evaluated the potential of ML-based scores for early diagnosis, prognosis, and monitoring of ADRD and PD in an independent large-scale population-based cohort, UK Biobank, using longitudinal data.")

st.subheader("Findings")
st.write("In this analysis, 1,826 dementia (from 731 participants), 3,161 healthy controls images (925 participants) from the ADNI cohort, 684 PD (319 participants), 232 healthy controls (145 participants) from PPMI cohort were used to train machine learning models. The classification performance is 0.94 [95% CI: 0.93-0.96] area under the ROC Curve (AUC) for ADRD detection and 0.63 [95% CI: 0.57-0.71] for PD detection using 790 extracted structural brain features. We identified the hippocampus and temporal brain regions as significantly affected by ADRD and the substantia nigra region by PD. The normalized ML model’s probabilistic output (ADRD and PD imaging scores) was evaluated on 42,835 participants with imaging data from UK Biobank. For diagnosis occurrence events within 5 years, the integrated survival model achieves a time-dependent AUC of 0.86 [95% CI: 0.80-0.92] for dementia and 0.89 [95% CI: 0.85-0.94] for PD. ADRD imaging score is strongly associated with dementia free survival (hazard ratio (HR) 1.76 [95% CI: 1.50-2.05] per S.D. of imaging score), and PD imaging score shows association with PD free survival (hazard ratio 2.33 [95% CI: 1.55-3.50]) in our integrated model. HR and prevalence increased stepwise over imaging score quartiles for PD, demonstrating heterogeneity. The scores are associated with multiple clinical assessments such as Mini-Mental State Examination (MMSE), Alzheimer’s Disease Assessment Scale-cognitive subscale (ADAS-Cog), and pathological markers, including amyloid and tau. Finally, imaging scores are associated with polygenic risk scores for multiple diseases. Our results indicate that we can use imaging scores to assess the genetic architecture of such disorders in the future.")

st.subheader("Interpretation")
st.write("Our study demonstrates the use of quantitative markers generated using machine learning techniques for ADRD and PD. We show that disease probability scores obtained from brain structural features are useful for early detection, prognosis prediction, and monitoring disease progression.")

st.subheader("Funding")
st.write("US National Institute on Aging, and US National Institutes of Health.")
# """
# from MachineLearningStreamlitBase.multiapp import MultiApp
# from MachineLearningStreamlitBase.apps import streamlit_shapley_component
# from MachineLearningStreamlitBase.apps import streamlit_shapley_component
#
# import copy
# from apps import select
# app = MultiApp()
#
# ##TODO: UPDATE TITLE
# # st.write('# Machine Learning for ALS')
# app.add_app("Home", select.app)
# app.add_app("Scientific background", streamlit_shapley_component.app)
# from MachineLearningStreamlitBase.apps import streamlit_prediction_component_multiclass
# app.add_app("Predict ADRD/PD disease", streamlit_prediction_component_multiclass.app)
# ##TODO: Add any apps you like
# # app.add_app("Explore the ALS subtype topological space", topological_space.app)
# app.run()

