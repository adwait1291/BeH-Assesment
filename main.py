import numpy as np
import pandas as pd
import pickle
import base64
import streamlit as st 
import string


pickle_in = open("vectorizer.pkl","rb")
vect=pickle.load(pickle_in)

pickle_in = open("NaiveBayes.pkl","rb")
model = pickle.load(pickle_in)

pickle_in = open("label.pkl","rb")
label = pickle.load(pickle_in)

html_temp = """
    <div style="background-color: pink;padding:10px; border-radius: 5px;
    height: 100%;
    width: 100%;">
    <h2 style="color:white;text-align:center;">Question Tagger</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)


#---------------Input--------------#

question = st.text_input("Enter any question"," ")
question = np.array([question])
question =vect.transform(question)     

#---------------Predicting output--------------#
if st.button("Predict"):
    pred = model.predict(question)
    pred = label.inverse_transform(pred)
    st.success('prediction: {}'.format(pred))
        





