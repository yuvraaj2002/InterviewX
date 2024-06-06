import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_dataset():
    df = pd.read_csv("Dataset/Job_features_num_cat.csv")
    df.drop(['Unnamed: 0'],axis=1,inplace=True)
    return df


def univariate_analysis(df):
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Introductory Analysis</h1>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(spec=(2, 1), gap="large")
    with col1:
        st.dataframe(df.head(8))
    with col2:
        st.markdown(
            "<p style='font-size: 16px; text-align: left;background-color:#C3E8FF;padding:1rem;'>Explore clustering algorithms—K-Means, DBSCAN, and AGNES—by selecting one from the dropdown menu and clicking Visualize to see how each partitions the data. This interactive exploration reveals unique cluster structures formed by each algorithm, providing valuable insights into their behaviors.</p>",
            unsafe_allow_html=True,
        )
        with st.expander(label = "What is the overall dimensionality of the dataset ?"):
            st.write(df.shape,"Which means there are around 17k rows and 10 features")
        with st.expander(label = "What's the count of categorical/numerical features in our data ?"):
            st.write("All the 10 features are categorical in nature, but out of all 3 features are ordinal features,1 is nominal feature and remaining 6 are simple binary categorical features")



def multivariate_analysis(df):
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Multivariate Analysis</h1>",
        unsafe_allow_html=True,
    )

    col1,col2 = st.columns(spec=(1, 1), gap="small")
    with col1:

        # Stacked Bar Chart for Employment Type
        with st.container():
            st.header('Employment Type')
            fig1 = px.histogram(df, x='employment_type', color='fraudulent', barmode='group',
                                title='Fraudulent vs Non-Fraudulent Postings by Employment Type',
                                labels={'fraudulent': 'Fraudulent'})
            st.plotly_chart(fig1, use_container_width=True)

        # Stacked Bar Chart for Company Logo Presence
        with st.container():
            st.header('Company Logo Presence')
            fig2 = px.histogram(df, x='has_company_logo', color='fraudulent', barmode='group',
                                title='Fraudulent vs Non-Fraudulent Postings by Company Logo Presence',
                                labels={'has_company_logo': 'Has Company Logo', 'fraudulent': 'Fraudulent'})
            st.plotly_chart(fig2, use_container_width=True)

        # Stacked Bar Chart for Required Education
        with st.container():
            st.header('Required Education')
            fig3 = px.histogram(df, x='required_education', color='fraudulent', barmode='group',
                                title='Fraudulent vs Non-Fraudulent Postings by Required Education',
                                labels={'required_education': 'Required Education', 'fraudulent': 'Fraudulent'})
            st.plotly_chart(fig3, use_container_width=True)

        # Stacked Bar Chart for Required Experience
        with st.container():
            st.header('Required Experience')
            fig4 = px.histogram(df, x='required_experience', color='fraudulent', barmode='group',
                                title='Fraudulent vs Non-Fraudulent Postings by Required Experience',
                                labels={'required_experience': 'Required Experience', 'fraudulent': 'Fraudulent'})
            st.plotly_chart(fig4, use_container_width=True)




def feature_selection(df):
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Feature Selection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 17px; text-align: left;'>A heatmap is a graphical representation of data where individual values are depicted as colors, allowing for quick visualization of patterns and relationships within a dataset. It is particularly useful for displaying the correlation matrix of features in a dataset, where the color intensity represents the strength of the correlation between pairs of features. For feature selection, heatmaps help identify highly correlated features, which can be redundant, enabling data scientists to simplify models by removing one of the correlated features. This process enhances model performance and interpretability by reducing multicollinearity and retaining only the most relevant features.</p>",
        unsafe_allow_html=True,
    )

    # Heatmap for Feature Correlations
    with st.container():
        df_encoded = df.copy()
        df_encoded['employment_type'] = df_encoded['employment_type'].astype('category').cat.codes
        df_encoded['required_experience'] = df_encoded['required_experience'].astype('category').cat.codes
        df_encoded['required_education'] = df_encoded['required_education'].astype('category').cat.codes
        df_encoded['industry'] = df_encoded['industry'].astype('category').cat.codes

        corr_matrix = df_encoded.corr()
        fig5 = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(2).values,
            colorscale='Viridis'
        )
        fig5.update_layout(title='', xaxis_title='Features', yaxis_title='Features', height=600)
        st.plotly_chart(fig5, use_container_width=True)



def visualization():

    # Calling the function to load the dataframe
    df = load_dataset()

    # Calling function for doing univariate analysis
    univariate_analysis(df)

    # Calling function for diong multivarite analysis
    multivariate_analysis(df)

    # Feature selection charts
    feature_selection(df)


visualization()