import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'healthcare-dataset-stroke-data.csv'
data = pd.read_csv(file_path)

# Preprocess the data
data = data.drop(columns=['id'])
imputer = SimpleImputer(strategy='mean')
data['bmi'] = imputer.fit_transform(data[['bmi']])
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop(columns=['stroke'])
y = data['stroke']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app
st.title("Stroke Prediction")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 50)
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 50.0, 300.0, 100.0)
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    hypertension = st.sidebar.selectbox('Hypertension', ('Yes', 'No'))
    heart_disease = st.sidebar.selectbox('Heart Disease', ('Yes', 'No'))
    ever_married = st.sidebar.selectbox('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    residence_type = st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))
    smoking_status = st.sidebar.selectbox('Smoking Status', ('formerly smoked', 'never smoked', 'smokes'))

    data = {'age': age,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married_Yes': 1 if ever_married == 'Yes' else 0,
            'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
            'work_type_Private': 1 if work_type == 'Private' else 0,
            'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
            'work_type_children': 1 if work_type == 'children' else 0,
            'Residence_type_Urban': 1 if residence_type == 'Urban' else 0,
            'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
            'smoking_status_never smoked': 1 if smoking_status == 'never smoked' else 0,
            'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Ensure all features are present in the input dataframe
input_df = input_df.reindex(columns=data.drop(columns=['stroke']).columns, fill_value=0)

# Preprocess user input
input_scaled = scaler.transform(input_df)

# Predict stroke when the button is pressed
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction')
    stroke_status = np.array(['No Stroke', 'Stroke'])
    prediction_result = stroke_status[prediction][0]
    st.write(prediction_result)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
    
    # Provide suggestions based on prediction
    st.subheader('Health Suggestions')
    if prediction_result == 'Stroke':
        st.write("""
            - Engage in regular physical activity (e.g., walking, swimming, cycling) to improve cardiovascular health.
            - Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.
            - Monitor and control blood pressure, cholesterol, and blood sugar levels.
            - Avoid smoking and limit alcohol consumption.
            - Manage stress through relaxation techniques such as meditation, yoga, or deep breathing exercises.
            - Follow your healthcare provider's advice and take prescribed medications as directed.
        """)
    else:
        st.write("""
            - Continue leading a healthy lifestyle to maintain your good health.
            - Engage in regular physical activity to keep your cardiovascular system strong.
            - Eat a balanced diet and monitor your blood pressure, cholesterol, and blood sugar levels.
            - Avoid smoking and limit alcohol consumption.
            - Manage stress and maintain a healthy weight.
            - Regular check-ups with your healthcare provider can help catch any potential issues early.
        """)

