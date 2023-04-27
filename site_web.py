import pandas as pd
import streamlit as st
import pickle
import keras
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

jobRoleOptions = ['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                  'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist',
                  'Sales Executive', 'Sales Representative'
                  ]

educationFieldOptions = ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'
                         ]

departmentOptions = ['Human Resources', 'Research & Development', 'Sales']

jobInvolvementOptions = ['Low', 'Medium', 'High', 'Very High']

businessTravelOptions = ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely']

maritalStatusOptions = ["Divorced", "Married", "Single"]

educationOptions = ["Below College", "College", "Bachelor", "Master", "Doctor"]

environmentSatisfactionOptions = ["Low", "Medium", "High", "Very High"]

jobSatisfactionOptions = ["Low", "Medium", "High", "Very High"]

relationshipSatisfactionOptions = ["Low", "Medium", "High", "Very High"]

workLifeBalanceOptions = ["Bad", "Good", "Better", "Best"]

performanceRatingOptions = ["Excellent", "Outstanding"]


def user_input_features():
    data = {}

    data['Age'] = [st.sidebar.slider('Age', min_value=18, max_value=100)]
    data['DistanceFromHome'] = [st.sidebar.number_input('Distance From Home', step=1, min_value=0)]
    data['Education'] = [st.sidebar.selectbox('Education', range(1, len(educationOptions) + 1),
                                              format_func=lambda x: educationOptions[x - 1])]

    data['EnvironmentSatisfaction'] = [st.sidebar. \
                                           selectbox('Environment Satisfaction',
                                                     range(1, len(environmentSatisfactionOptions) + 1),
                                                     format_func=lambda x: environmentSatisfactionOptions[x - 1])]
    data['Gender'] = [1] if st.sidebar.selectbox('Gender', ['Female', 'Male']) == 'Male' else [0]
    data['JobInvolvement'] = [st.sidebar.selectbox('Job Involvement', range(1, len(jobInvolvementOptions) + 1),
                                                   format_func=lambda x: jobInvolvementOptions[x - 1])]
    data['JobLevel'] = [st.sidebar.number_input('Job level', step=1, min_value=0, max_value=5)]
    data['JobSatisfaction'] = [st.sidebar.selectbox('Job Satisfaction', range(1, len(jobSatisfactionOptions) + 1),
                                                    format_func=lambda x: jobSatisfactionOptions[x - 1])]
    data['MonthlyIncome'] = [st.sidebar.number_input('Monthly Income ($)', min_value=0, step=1)]
    data['OverTime'] = [1] if st.sidebar.selectbox('OverTime', ['Yes', 'No']) == 'Yes' else [0]
    data['PerformanceRating'] = [st.sidebar.selectbox('Performance Rating', [3, 4],
                                                      format_func=lambda x: performanceRatingOptions[x - 3])]
    data['RelationshipSatisfaction'] = [
        st.sidebar.selectbox('Relationship Satisfaction', range(1, len(relationshipSatisfactionOptions) + 1),
                             format_func=lambda x: relationshipSatisfactionOptions[x - 1])]
    data['TotalWorkingYears'] = [st.sidebar.number_input('Total Working Years', min_value=0, step=1)]
    data['WorkLifeBalance'] = [st.sidebar.selectbox('Work Life Balance', range(1, len(workLifeBalanceOptions) + 1),
                                                    format_func=lambda x: workLifeBalanceOptions[x - 1])]
    data['YearsAtCompany'] = [st.sidebar.number_input('Years at company', min_value=0, step=1)]
    data['YearsInCurrentRole'] = [st.sidebar.number_input('Years In Current Role', min_value=0, step=1)]
    data['YearsSinceLastPromotion'] = [st.sidebar.number_input('Years Since Last Promotion', min_value=0, step=1)]
    data['YearsWithCurrManager'] = [st.sidebar.number_input('Years with current manager', min_value=0, step=1)]
    data['MaritalStatus'] = st.sidebar.selectbox('Marital Status', maritalStatusOptions)
    data['JobRole'] = st.sidebar.selectbox('Job Level', jobRoleOptions)
    data['EducationField'] = st.sidebar.selectbox('Education Field', educationFieldOptions)
    data['Department'] = st.sidebar.selectbox('Departament', departmentOptions)
    data['BusinessTravel'] = st.sidebar.selectbox('Business Travel',
                                                  businessTravelOptions)

    for option in maritalStatusOptions:
        data['MaritalStatus_' + option] = [1] if option == data['MaritalStatus'] else [0]

    data.pop('MaritalStatus')

    for option in jobRoleOptions:
        data['JobRole_' + option] = [1] if option == data['JobRole'] else [0]

    data.pop('JobRole')

    for option in educationFieldOptions:
        data['EducationField_' + option] = [1] if option == data['EducationField'] else [0]

    data.pop('EducationField')

    for option in departmentOptions:
        data['Department_' + option] = [1] if option == data['Department'] else [0]

    data.pop('Department')

    for option in businessTravelOptions:
        data['BusinessTravel_' + option] = [1] if option == data['BusinessTravel'] else [0]

    data.pop('BusinessTravel')

    return pd.DataFrame(data, index=[0])


def load_model():
    return pickle.load(open('model.pickle', 'rb'))


def load_scaler():
    return pickle.load(open('scaler.pickle', 'rb'))


def main():
    model = load_model()
    scaler = load_scaler()

    st.write("""
    # Departamento de Recursos Humanos
    # """)
    st.write('---')

    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Escolha de paramentros para Predição')

    df = user_input_features()
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    st.header('Parametros especificados')
    st.write(df)
    st.write('---')

    prediction = model.predict(df)
    
    st.header('Resultado previsto:')
    st.write('O funcionário irá ' + ('continuar na empresa' if prediction < 0.5 else "sair da empresa"))
    st.write('---')


if __name__ == '__main__':
    main()
