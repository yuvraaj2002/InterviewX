import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb
import pickle
import joblib
import os


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.0rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

with open("artifacts/industry_types.pkl", "rb") as f:
    industry_types = pickle.load(f)


@st.cache_resource
def load_model_gbc():
    classifier = joblib.load('artifacts/GB_classifier_model.pkl')
    return classifier




@st.cache_resource
def load_encodings_pipeline():

    with open("artifacts/oe_edu.pkl", "rb") as f:
        oe_edu = pickle.load(f)

    with open("artifacts/oe_emptype.pkl", "rb") as f:
        oe_emptype = pickle.load(f)

    with open("artifacts/oe_exp.pkl", "rb") as f:
        oe_exp = pickle.load(f)

    with open("artifacts/te_industry.pkl", "rb") as f:
        te_industry = pickle.load(f)

    with open("artifacts/Processing_pipeline.pkl", "rb") as f:
        scaling_pipe = pickle.load(f)

    return [oe_edu, oe_emptype, oe_exp, te_industry, scaling_pipe]


def create_non_text_features(
        telecommuting,
        has_company_logo,
        has_questions,
        employment_type,
        required_experience,
        required_education,
        industry,
        Salary_range_provided,
        department_mentioned,
):
    """
    This method will take the input variables and will return a DataFrame with input feature values.
    :return: DataFrame
    """
    yes_no_mapping = {"Yes": 1.0, "No": 0.0}

    # Create a dictionary with your input variables
    data = {
        "telecommuting" : [yes_no_mapping[telecommuting]],
        "has_company_logo": [yes_no_mapping[has_company_logo]],
        "has_questions": [yes_no_mapping[has_questions]],
        "employment_type": [employment_type],
        "required_experience": [required_experience],
        "required_education": [required_education],
        "industry": [industry],
        "Salary_range_provided": [yes_no_mapping[Salary_range_provided]],
        "department_mentioned": [yes_no_mapping[department_mentioned]],
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df



def process_predict_non_text_feature(df):

    # Loading the encodings and pipeline
    oe_edu, oe_emptype, oe_exp, te_industry, Scaling_pipeline = (
        load_encodings_pipeline()
    )

    # Encoding the categorical features
    df["required_experience"] = pd.Series(
        oe_exp.transform(df["required_experience"].values.reshape(-1, 1)).reshape(-1)
    )
    df["required_education"] = pd.Series(
        oe_edu.transform(df["required_education"].values.reshape(-1, 1)).reshape(-1)
    )
    df["employment_type"] = pd.Series(
        oe_emptype.transform(df["employment_type"].values.reshape(-1, 1)).reshape(-1)
    )
    df["industry"] = pd.Series(
        te_industry.transform(df["industry"].values.reshape(-1, 1)).reshape(-1)
    )

    # Scaling the features
    Input = Scaling_pipeline.transform(df)

    # Loading the classifier
    classifier1 = load_model_gbc()

    # Assuming 'Input' is your numpy.ndarray
    model1_output = classifier1.predict(Input)
    return model1_output


def spot_scam_page():
    st.markdown(
        "<h1 style='text-align: center; font-size: 55px;'>Spot the Scam🕵️</h1>",
        unsafe_allow_html=True,
    )
    # st.write("adsfsa jjajdsjfk sadf jlskdjfklsaj lkfsldkjglsk dg jsj dlkgjsd lgs")
    st.markdown(
        "<p style='font-size: 22px; text-align: center;padding-left: 2rem;padding-right: 2rem;'>In times of tough market situations, fake job postings and scams often spike, posing a significant threat to job seekers. To combat this, I've developed a user-friendly module designed to protect individuals from falling prey to such fraudulent activities. This module requires users to input details about the job posting they're considering. Behind the scenes, two powerful AI models thoroughly analyze the provided information. Once completed, users receive a clear indication of whether the job posting is genuine or potentially deceptive.</p>",
        unsafe_allow_html=True,
    )
    st.write("***")

    configuration_col, input_col = st.columns(spec=(0.8, 2), gap="large")
    with configuration_col:
        st.markdown(
            "<p class='center' style='font-size: 18px; background-color: #C3E8FF; padding:1rem;'>To obtain predictions regarding the current state of the plant, you need to upload the image below. This image should ideally capture the entire plant, ensuring clar.</p>",
            unsafe_allow_html=True,
        )
        model_output_wt = st.slider(
            label="",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            key="Model_wt",
            label_visibility="collapsed",
        )
        pie_data = {
            "Categories": ["Model1", "Model2"],
            "Weights": [
                model_output_wt,
                100 - model_output_wt,
            ],
        }

        # Create a DataFrame from the data
        pie_df = pd.DataFrame(pie_data)
        custom_colors = ["#C3E8FF", "#43B7FF"]

        # Create a dynamic pie chart using Plotly Express
        fig = px.pie(
            pie_df,
            names="Categories",
            values="Weights",
            color_discrete_sequence=custom_colors,
            height=380,  # Adjust the height as per your requirement
            width=380,  # Adjust the width as per your requirement
        )
        st.plotly_chart(fig, use_container_width=True)

    with input_col:

        input_col1, input_col2 = st.columns(spec=(1, 1), gap="large")
        with input_col1:
            employment_type = st.selectbox(
                "What type of employment is specified in the job posting?",
                ("Full-time", "Contract", "Part-time", "Temporary", "Other"),
            )
            required_experience = st.selectbox(
                "What level of experience is required for this position?",
                (
                    "Mid-Senior level",
                    "Entry level",
                    "Associate",
                    "Not Applicable",
                    "Director",
                    "Internship",
                    "Executive",
                ),
            )
            required_education = st.selectbox(
                "What level of education is required for this position?",
                (
                    "Bachelor's Degree",
                    "High School or equivalent",
                    "Unspecified",
                    "Master's Degree",
                    "Associate Degree",
                    "Certification",
                    "Some College Coursework Completed",
                    "Professional",
                    "Vocational",
                    "Some High School Coursework",
                    "Doctorate",
                    "Vocational - HS Diploma",
                    "Vocational - Degree",
                ),
            )
            has_company_logo = st.selectbox(
                "Is there company logo on job post?", ("Yes", "No")
            )
            department_mentioned = st.selectbox(
                "Did they specify which department is involved in the job listing?",
                ("Yes", "No"),
            )
            Salary_range_provided = st.selectbox(
                "Was the salary range mentioned in the job description?", ("Yes", "No")
            )

        with input_col2:
            industry = st.selectbox("Select the industry", (industry_types))
            telecommuting = st.selectbox(
                "Did they have telecommuting in the job listing?",
                ("Yes", "No"),
            )
            has_questions = st.selectbox(
                "Do they have questions in the job listing?",
                ("Yes", "No"),
            )
            # description = st.text_input("Enter the job description mentioned")
            # requirements = st.text_input("Enter the mentioned requirements in job post")
            # benefits = st.text_input(
            #     "Enter benefits they are claiming they will provide"
            # )

            st.write("***")
            predict_bt = st.button("Analyze job post🔎", use_container_width=True)
            if predict_bt:
                if any(
                    [
                        telecommuting == None,
                        has_company_logo == None,
                        has_questions == None,
                        employment_type == None,
                        required_experience == None,
                        required_education == None,
                        industry == None,
                        Salary_range_provided == None,
                        department_mentioned == None,
                    ]
                ):
                    st.error("Please enter all the fields")

                else:
                    # Calling function to get the non text features dataframe
                    model1_input_df = create_non_text_features(
                        telecommuting,
                        has_company_logo,
                        has_questions,
                        employment_type,
                        required_experience,
                        required_education,
                        industry,
                        Salary_range_provided,
                        department_mentioned,
                    )

                    model1_output = process_predict_non_text_feature(model1_input_df)
                    if model1_output == 1.0:
                        st.error("The job listing is likely fake. Please proceed with caution.")
                    else:
                        st.success("The job listing appears to be genuine.")
                  

spot_scam_page()
