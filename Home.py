import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

import warnings
warnings.filterwarnings("ignore")
import logging


logger = logging.getLogger()
logger.disabled = True


st.title("Prediction, prognosis and monitoring of neurodegeneration at biobank-scale via machine learning and imaging.")

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

