import streamlit as st
import joblib
import os
import numpy as np

st.session_state['x'] = None
st.session_state['model'] = joblib.load(os.path.join(os.getcwd(),"model.pkl"))

st.session_state['x'] = st.number_input("Input a value for x", value=None, placeholder="Type a number...")

if st.session_state['x'] is not None:
    st.session_state['x'] = np.array(st.session_state['x'])
    st.session_state['x'] = np.reshape(st.session_state['x'], (-1,1))
    st.write("The output is ", st.session_state['model'].predict(st.session_state['x']))
