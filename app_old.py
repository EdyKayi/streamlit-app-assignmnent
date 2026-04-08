import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import os

# -------------------------------------------------------
# Helper functions  (no caching decorator – it was deprecated)
# -------------------------------------------------------
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(page_title="Loan Prediction App", layout="wide")

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if app_mode == 'Home':
    st.title('LOAN PREDICTION :')

    # Show image only if the file exists (avoids FileNotFoundError)
    if os.path.exists('loan_image.jpg'):
        st.image('loan_image.jpg')
    else:
        st.info("ℹ️ Place a 'loan_image.jpg' in the same folder to display the banner image.")

    st.write('@DSU for learning purposes only')

# -------------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------------
elif app_mode == 'Prediction':

    # Load CSV only if it exists
    if os.path.exists('test.csv'):
        csv = pd.read_csv("test.csv")
        st.write(csv)
    else:
        st.warning("⚠️ 'informations.csv' not found. Place it in the same folder.")

    st.subheader('Sir/Mme, YOU need to fill all necessary information in order to get a reply to your loan request!')
    st.sidebar.header("Information about the client:")

    # Dictionaries
    gender_dict  = {"Male": 1, "Female": 2}
    feature_dict = {"No": 1, "Yes": 2}
    edu          = {'Graduate': 1, 'Not Graduate': 2}
    prop         = {'Rural': 1, 'Urban': 2, 'Semiurban': 3}

    # Sidebar inputs
    Gender             = st.sidebar.radio('Gender',        tuple(gender_dict.keys()))
    Married            = st.sidebar.radio('Married',       tuple(feature_dict.keys()))
    Self_Employed      = st.sidebar.radio('Self Employed', tuple(feature_dict.keys()))
    Dependents         = st.sidebar.radio('Dependents',    options=['0', '1', '2', '3+'])
    Education          = st.sidebar.radio('Education',     tuple(edu.keys()))
    ApplicantIncome    = st.sidebar.slider('Applicant Income',   0,      10000, 0)
    CoapplicantIncome  = st.sidebar.slider('Coapplicant Income', 0,      10000, 0)
    LoanAmount         = st.sidebar.slider('Loan Amount (K$)',   9.0,  700.0,  200.0)
    Loan_Amount_Term   = st.sidebar.selectbox('Loan Amount Term',
                            (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History     = st.sidebar.radio('Credit History', (0.0, 1.0))
    Property_Area      = st.sidebar.radio('Property Area', tuple(prop.keys()))

    # Encode Dependents
    class_0 = class_1 = class_2 = class_3 = 0
    if   Dependents == '0':  class_0 = 1
    elif Dependents == '1':  class_1 = 1
    elif Dependents == '2':  class_2 = 1
    else:                    class_3 = 1

    # Encode Property Area
    Rural = Urban = Semiurban = 0
    if   Property_Area == 'Urban':    Urban    = 1
    elif Property_Area == 'Semiurban': Semiurban = 1
    else:                              Rural    = 1

    # Build feature vector
    feature_list = [
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        get_value(Gender,   gender_dict),
        get_fvalue(Married),
        class_0, class_1, class_2, class_3,
        get_value(Education, edu),
        get_fvalue(Self_Employed),
        Rural, Urban, Semiurban
    ]

    single_sample = np.array(feature_list).reshape(1, -1)

    # -------------------------------------------------------
    # Predict button
    # -------------------------------------------------------
    if st.button("Predict"):

        # Check model file exists
        if not os.path.exists('RF.sav'):
            st.error("❌ Model file 'RF.sav' not found. Place it in the same folder.")
        else:
            loaded_model = pickle.load(open('RF.sav', 'rb'))
            prediction   = loaded_model.predict(single_sample)

            if prediction[0] == 0:
                st.error('According to our Calculations, you will NOT get the loan from the Bank.')

                # Show GIF only if file exists
                if os.path.exists("green-cola-no.gif"):
                    with open("green-cola-no.gif", "rb") as f:
                        data_url_no = base64.b64encode(f.read()).decode("utf-8")
                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url_no}" alt="no gif">',
                        unsafe_allow_html=True,
                    )

            elif prediction[0] == 1:
                st.success('Congratulations!! You WILL get the loan from the Bank.')

                # Show GIF only if file exists
                if os.path.exists("6m-rain.gif"):
                    with open("6m-rain.gif", "rb") as f:
                        data_url = base64.b64encode(f.read()).decode("utf-8")
                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" alt="yes gif">',
                        unsafe_allow_html=True,
                    )
