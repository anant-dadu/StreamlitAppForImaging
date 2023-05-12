import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import hashlib
# import plotly.express as px
# import plotly
import copy
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

import warnings
warnings.filterwarnings("ignore")
import logging


logger = logging.getLogger()
logger.disabled = True

from MachineLearningStreamlitBase.multiapp import MultiApp
from MachineLearningStreamlitBase.apps import streamlit_shapley_component
import copy
from apps import select
app = MultiApp()

##TODO: UPDATE TITLE
# st.write('# Machine Learning for ALS')
app.add_app("Home", select.app)
app.add_app("Scientific background", streamlit_shapley_component.app)
from MachineLearningStreamlitBase.apps import streamlit_prediction_component_multiclass
app.add_app("Predict ADRD/PD disease", streamlit_prediction_component_multiclass.app)
##TODO: Add any apps you like
# app.add_app("Explore the ALS subtype topological space", topological_space.app)
app.run()

