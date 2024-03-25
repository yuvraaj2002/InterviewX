import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
import tensorflow as tf
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv
load_dotenv()

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
with open("artifacts/stop_words.pkl", "rb") as f:
    stop_words = pickle.load(f)


@st.cache_resource
def load_model_xgb():
    xgb_model = xgb.Booster(model_file="artifacts/xgboost_classifier_model.bin")
    return xgb_model


@st.cache_resource
def load_model_blstm():
    blstm_model = tf.keras.models.load_model("artifacts/model.keras")
    return blstm_model


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
    has_company_logo,
    Salary_range_provided,
    department_mentioned,
    employment_type,
    required_experience,
    required_education,
    industry,
):
    """
    This method will take the input variables and will return a DataFrame with input feature values.
    :return: DataFrame
    """
    yes_no_mapping = {"Yes": 1.0, "No": 0.0}

    # Create a dictionary with your input variables
    data = {
        "has_company_logo": [yes_no_mapping[has_company_logo]],
        "Salary_range_provided": [yes_no_mapping[Salary_range_provided]],
        "department_mentioned": [yes_no_mapping[department_mentioned]],
        "employment_type": [employment_type],
        "required_experience": [required_experience],
        "required_education": [required_education],
        "industry": [industry],
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df


# Remove stop words
def remove_stopwords(token):
    return tf.math.logical_not(tf.reduce_any(tf.math.equal(stop_words, token)))


def process_text(input_data):
    """
    Output: Cleaned text tensor

    Description: This function will take a single raw text as input, remove all stopwords and punctuation, then lowercase the words to eliminate any ambiguity.
    Ultimately clean text will be returned as a tensor.
    """

    # Lowercase the input data
    lowercase = tf.strings.lower(input_data)

    # Remove punctuation
    no_punctuation = tf.strings.regex_replace(
        lowercase, "[%s]" % re.escape(string.punctuation), ""
    )

    # Tokenizing the words in the strings
    tokens = tf.strings.split(no_punctuation)

    filtered_tokens = tf.map_fn(remove_stopwords, tokens, fn_output_signature=tf.bool)

    # Filter tokens based on the boolean mask
    filtered_tokens = tf.boolean_mask(tokens, filtered_tokens)

    processed_text = tf.strings.reduce_join(filtered_tokens, separator=" ", axis=-1)
    return processed_text


def process_predict_text_feature(text):
    """
    This function takes a question as input and returns the predicted category.
    """

    # Creating tensor from the string
    Input_text = tf.constant(text)

    # Cleaning the text
    Input_text = process_text(Input_text)

    # Fixing the input dimension
    Input_text = tf.expand_dims(Input_text, axis=0)

    # Getting predictions from the model
    blstm_model = load_model_blstm()
    predictions = blstm_model.predict(Input_text)

    # Extracting the predicted class index (as a scalar)
    # predicted_class_index = np.argmax(predictions, axis=1)[0]  # Access the first element

    # Mapping the index to the corresponding label
    # return class_labels[predicted_class_index]
    return predictions


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
    classifier1 = load_model_xgb()

    # Assuming 'Input' is your numpy.ndarray
    data_matrix = xgb.DMatrix(Input)
    model1_output = classifier1.predict(data_matrix)
    return model1_output


def text_feature():
    pass


def spot_scam_page():
    st.markdown(
        "<h1 style='text-align: center; font-size: 60px;'>Spot the Scam🕵️</h1>",
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
            "<p class='center' style='font-size: 18px; background-color: #CEFCBA; padding:1rem;'>To obtain predictions regarding the current state of the plant, you need to upload the image below. This image should ideally capture the entire plant, ensuring clar.</p>",
            unsafe_allow_html=True,
        )
        model_output_wt = st.slider(
            label="",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            key="facilities_recommendation_wt",
            label_visibility="collapsed",
        )
        data = {
            "Categories": ["Model1", "Model2"],
            "Weights": [
                model_output_wt,
                100 - model_output_wt,
            ],
        }

        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        custom_colors = ["#AEF359", "#03C04A"]

        # Create a dynamic pie chart using Plotly Express
        fig = px.pie(
            df,
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
            description = st.text_input("Enter the job description mentioned")
            requirements = st.text_input("Enter the mentioned requirements in job post")
            benefits = st.text_input(
                "Enter benefits they are claiming they will provide"
            )

            st.write("***")
            predict_bt = st.button("Analyze job post🔎", use_container_width=True)
            if predict_bt:

                if any(
                    [
                        employment_type == None,
                        required_experience == None,
                        required_education == None,
                        has_company_logo == None,
                        department_mentioned == None,
                        Salary_range_provided == None,
                        industry == None,
                        description == "",
                        requirements == "",
                        benefits == "",
                    ]
                ):
                    st.error("Please enter all the fields")

                else:

                    # Calling function to get the non text features dataframe
                    model1_input_df = create_non_text_features(
                        has_company_logo,
                        Salary_range_provided,
                        department_mentioned,
                        employment_type,
                        required_experience,
                        required_education,
                        industry,
                    )
                    model1_output = process_predict_non_text_feature(model1_input_df)
                    st.write(model1_output)

                    raw_text = description + " " + requirements + " " + benefits
                    model2_output = process_predict_text_feature(raw_text)
                    st.write(model2_output)
                    # model2_input_df = create_text_feature(description,requirements,benefits)


spot_scam_page()
