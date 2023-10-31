import pandas as pd
import numpy as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

parkinsons_model=pickle.load(open('parkinsons_model.sav','rb'))
heart_model=pickle.load(open('heart_model.sav','rb'))
diabetes_model=pickle.load(open('diabetes_model.sav','rb'))

with st.sidebar:
    selected=option_menu('Multiple Disease Prediction',['Diabetes Disease','Heart Disease','Parkinsons Disease'],icons=['activity','heart','person'],default_index=0 )


if (selected=='Diabetes Disease'):

    st.title('Diabetes Disease Prediction Useing ML')

    col1,col2,col3=st.columns(3)
    with col1:
        Pregnancies=st.text_input('Number of Pregnancies')
    with col2:
        Glucose=st.text_input('Glucose Level')
    with col3:
        BloodPressure=st.text_input('BloodPressure Value')
    with col1:
        SkinThickness=st.text_input('SkinThickness Value')
    with col2:
        Insulin=st.text_input('Insulin Level')
    with col3:
        BMI=st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction Value')
    with col2:
        Age=st.text_input('Age Of Person')

    diabet_dignosis=''
    if st.button('Diabetes Disease Predict'):
        diabet_predict=diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

        if(diabet_predict[0]==1):
            diabet_dignosis = 'The Person is Diabetic'
        else:
            diabet_dignosis = 'The Person is Not Diabetic'

    st.success(diabet_dignosis)


if (selected == 'Heart Disease'):

    st.title('Heart Disease Prediction Useing ML')

    col1,col2,col3=st.columns(3)
    with col1:
        age=st.text_input('Age of Person')
    with col2:
        sex=st.text_input('Sex')
    with col3:
        cp=st.text_input('CP Value')
    with col1:
        trestbps=st.text_input('Trestbps Value')
    with col2:
        chol=st.text_input(' Chol Value')
    with col3:
        fbs=st.text_input('Fbs Value')
    with col1:
        restecg=st.text_input('Restecg Value')
    with col2:
        thalach=st.text_input('Thalach Value')
    with col3:
        exang=st.text_input('Exang Value')
    with col1:
        oldpeak=st.text_input('Oldpeak Value')
    with col2:
        slope=st.text_input('Slope Value')
    with col3:
        ca=st.text_input('Ca Value')
    with col1:
        thal=st.text_input('Thal Value')

    heart_dignosis = ''
    if st.button('Heart Disease Predict'):
        heart_predict = heart_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

        if (heart_predict[0] == 1):
            heart_dignosis = 'The Person is Heart disease'
        else:
            heart_dignosis = 'The Person is Not Heart disease'

    st.success( heart_dignosis)


if (selected == 'Parkinsons Disease'):

    st.title('Parkinsons Disease Prediction Useing ML')

    col1,col2,col3,col4=st.columns(4)
    with col1:
        MDVP_Fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        MDVP_Fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        MDVP_Flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        MDVP_Jitter = st.text_input('MDVP:Jitter(%)')
    with col1:
        MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col2:
        MDVP_RAP = st.text_input('MDVP:RAP')
    with col3:
        MDVP_PPQ = st.text_input('MDVP:PPQ')
    with col4:
        Jitter_DDP = st.text_input('Jitter:DDP')
    with col1:
        MDVP_Shimmer = st.text_input('MDVP:Shimmer')
    with col2:
        MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col3:
        Shimmer_APQ3 = st.text_input('Shimmer:APQ3')
    with col4:
        Shimmer_APQ5 = st.text_input('Shimmer:APQ5')
    with col1:
        MDVP_APQ = st.text_input('MDVP:APQ')
    with col2:
        Shimmer_DDA = st.text_input('Shimmer:DDA')
    with col3:
        NHR = st.text_input('NHR')
    with col4:
        HNR = st.text_input('HNR')
    with col1:
        RPDE = st.text_input('RPDE')
    with col2:
        DFA = st.text_input('DFA')
    with col3:
        Spread1 = st.text_input('spread1')
    with col4:
        Spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE=st.text_input('PPE')

    parkinson_dignosis = ''
    if st.button('Heart Disease Predict'):
        parkinsons_predict = parkinsons_model.predict([[MDVP_Fo,MDVP_Fhi,MDVP_Flo,MDVP_Jitter,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,Spread1,Spread2,D2,PPE]])

        if (parkinsons_predict[0] == 1):
            parkinson_dignosis = 'The Person is parkinson disease'
        else:
            parkinson_dignosis = 'The Person is Not parkinson disease'

    st.success(parkinson_dignosis)

