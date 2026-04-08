import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import os

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def get_fvalue(val):
    """Encodes Yes/No to 2/1"""
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val)

def get_value(val, my_dict):
    """Returns encoded value from a given dictionary"""
    return my_dict.get(val)

def load_gif(filepath):
    """Safely loads and base64-encodes a GIF file"""
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
st.sidebar.title("🏦 Loan Prediction")
app_mode = st.sidebar.selectbox('📌 Select Page', ['Home', 'Prediction'])

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if app_mode == 'Home':

    st.title('🏦 LOAN PREDICTION APP')
    st.write("---")

    # Banner Image
    if os.path.exists('loan_image.jpg'):
        st.image('loan_image.jpg', use_container_width=True)
    else:
        st.info("ℹ️ Add a 'loan_image.jpg' to your project folder to display a banner.")

    st.write("---")
    st.write("*@DSU — For learning purposes only*")

    # Dataset Preview
    st.markdown("### 📂 Dataset Preview")
    if os.path.exists('test.csv'):
        data = pd.read_csv('test.csv')
        st.write(data.head())

        # Bar Chart
        st.markdown("### 📊 Applicant Income VS Loan Amount (Top 20)")
        chart_data = data[['ApplicantIncome', 'LoanAmount']].head(20).reset_index(drop=True)
        st.bar_chart(chart_data)

        # Extra: Summary statistics
        st.markdown("### 📈 Dataset Summary Statistics")
        st.write(data.describe())

    else:
        st.warning("⚠️ 'test.csv' not found. Please place it in the same folder as app.py.")

# -------------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------------
elif app_mode == 'Prediction':

    st.title("🔮 Loan Eligibility Prediction")
    st.write("---")
    st.subheader('Sir/Mme, please fill in all necessary information to get a reply to your loan request!')

    # Sidebar inputs
    st.sidebar.header("📋 Client Information")

    # Dictionaries for encoding
    gender_dict  = {"Male": 1, "Female": 2}
    feature_dict = {"No": 1, "Yes": 2}
    edu          = {'Graduate': 1, 'Not Graduate': 2}
    prop         = {'Rural': 1, 'Urban': 2, 'Semiurban': 3}

    # --- Numeric Inputs ---
    st.sidebar.markdown("#### 💰 Financial Details")
    ApplicantIncome   = st.sidebar.slider('Applicant Income ($)',    0,     10000, 0)
    CoapplicantIncome = st.sidebar.slider('Co-applicant Income ($)', 0,     10000, 0)
    LoanAmount        = st.sidebar.slider('Loan Amount (K$)',        9.0,   700.0, 200.0)
    Loan_Amount_Term  = st.sidebar.selectbox('Loan Amount Term (months)',
                            (12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0))
    Credit_History    = st.sidebar.radio('Credit History', (0.0, 1.0),
                            help="1.0 = Good credit history, 0.0 = Poor credit history")

    # --- Categorical Inputs ---
    st.sidebar.markdown("#### 👤 Personal Details")
    Gender        = st.sidebar.radio('Gender',        tuple(gender_dict.keys()))
    Married       = st.sidebar.radio('Married',       tuple(feature_dict.keys()))
    Self_Employed = st.sidebar.radio('Self Employed', tuple(feature_dict.keys()))
    Dependents    = st.sidebar.radio('Dependents',    options=['0', '1', '2', '3+'])
    Education     = st.sidebar.radio('Education',     tuple(edu.keys()))
    Property_Area = st.sidebar.radio('Property Area', tuple(prop.keys()))

    # --- One-Hot Encoding: Dependents ---
    class_0 = class_1 = class_2 = class_3 = 0
    if   Dependents == '0': class_0 = 1
    elif Dependents == '1': class_1 = 1
    elif Dependents == '2': class_2 = 1
    else:                   class_3 = 1

    # --- One-Hot Encoding: Property Area ---
    Rural = Urban = Semiurban = 0
    if   Property_Area == 'Urban':     Urban     = 1
    elif Property_Area == 'Semiurban': Semiurban = 1
    else:                              Rural     = 1

    # --- Data Dictionary ---
    data1 = {
        'Gender':            Gender,
        'Married':           Married,
        'Dependents':        [class_0, class_1, class_2, class_3],
        'Education':         Education,
        'ApplicantIncome':   ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'Self Employed':     Self_Employed,
        'LoanAmount':        LoanAmount,
        'Loan_Amount_Term':  Loan_Amount_Term,
        'Credit_History':    Credit_History,
        'Property_Area':     [Rural, Urban, Semiurban],
    }

    # --- Feature Vector (must match training column order) ---
    feature_list = [
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        get_value(Gender,        gender_dict),
        get_fvalue(Married),
        data1['Dependents'][0],
        data1['Dependents'][1],
        data1['Dependents'][2],
        data1['Dependents'][3],
        get_value(Education,     edu),
        get_fvalue(Self_Employed),
        data1['Property_Area'][0],
        data1['Property_Area'][1],
        data1['Property_Area'][2],
    ]

    single_sample = np.array(feature_list).reshape(1, -1)

    # --- Show client summary before predicting ---
    st.markdown("### 📝 Your Input Summary")
    summary_df = pd.DataFrame({
        'Feature': [
            'Gender', 'Married', 'Dependents', 'Education', 'Self Employed',
            'Applicant Income', 'Co-applicant Income', 'Loan Amount (K$)',
            'Loan Term (months)', 'Credit History', 'Property Area'
        ],
        'Value': [
            Gender, Married, Dependents, Education, Self_Employed,
            f"${ApplicantIncome}", f"${CoapplicantIncome}", f"${LoanAmount}K",
            Loan_Amount_Term, Credit_History, Property_Area
        ]
    })
    st.table(summary_df)

    # -------------------------------------------------------
    # Predict Button
    # -------------------------------------------------------
    st.write("---")
    if st.button("🔍 Predict Loan Eligibility"):

        if not os.path.exists('RF.sav'):
            st.error("❌ Model file 'RF.sav' not found. Please place it in the same folder as app.py.")
        else:
            with st.spinner("Analyzing your application..."):
                loaded_model = pickle.load(open('RF.sav', 'rb'))
                prediction   = loaded_model.predict(single_sample)

            # --- RESULT ---
            if prediction[0] == 0:
                st.error('❌ According to our calculations, you will NOT get the loan from the Bank.')
                data_url_no = load_gif("green-cola-no.gif")
                if data_url_no:
                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url_no}" alt="rejected gif" width="300">',
                        unsafe_allow_html=True,
                    )

            elif prediction[0] == 1:
                st.success('✅ Congratulations!! You WILL get the loan from the Bank!')
                st.balloons()
                data_url = load_gif("6m-rain.gif")
                if data_url:
                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" alt="approved gif" width="300">',
                        unsafe_allow_html=True,
                    )
