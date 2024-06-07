import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc


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
    df = pd.read_csv("Dataset/Job_features_NText_.csv")
    return df


def back_encoding(df):

    cols = ['has_company_logo', 'Salary_range_provided', 'department_mentioned']
    for col in cols:
        df[col] = df[col].replace({1: 'Yes', 0: 'No'})

    # Only considering the floor value
    cols = ['employment_type', 'required_experience', 'required_education']
    for col in cols:
        df[col] = np.floor(df[col])

    df['required_experience'] = df['required_experience'].replace({
        0: "Not Applicable",
        1: "Internship",
        2: "Entry level",
        3: "Mid-Senior level",
        4: "Associate",
        5: "Director",
        6: "Executive"})

    df['employment_type'] = df['employment_type'].replace(
        {0: "Other", 1: "Temporary", 2: "Part-time", 3: "Contract", 4: "Full-time"})

    df['required_education'] = df['required_education'].replace({
        0: "Unspecified",
        1: "Some High School Coursework",
        2: "High School or equivalent",
        3: "Some College Coursework Completed",
        4: "Vocational",
        5: "Vocational - HS Diploma",
        6: "Vocational - Degree",
        7: "Associate Degree",
        8: "Certification",
        9: "Professional",
        10: "Bachelor's Degree",
        11: "Master's Degree",
        12: "Doctorate"})
    return df


def univariate_analysis(df):
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Introductory Analysis</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(spec=(2, 1), gap="small")
    with col1:
        st.dataframe(df.head(8))
    with col2:
        st.markdown(
            "<p style='font-size: 17px; text-align: left;background-color:#C3E8FF;padding:1rem;'>Welcome to the Univariate Analysis Module! When it comes to understanding data, focusing on one thing at a time is key. It's all about looking closely at one variable at a time, helping us uncover important patterns and insights. Think of it as the first step in exploring data, like peeling back layers to reveal what's underneath and discover the secrets hidden within your data!.</p>",
            unsafe_allow_html=True,
        )
        with st.expander(label = "What is the overall dimensionality of the dataset ?"):
            st.write(df.shape,"Which means there are around 17k rows and 10 features")
        with st.expander(label = "What's the count of categorical/numerical features in our data ?"):
            st.write("All the 8 features are categorical in nature, but out of all 3 features are ordinal features,1 is nominal feature and remaining 4 are simple binary categorical features")

    pie_col1, pie_col2,pie_col3 = st.columns(spec=(1,1,1), gap="large")

    with pie_col1:
        # Plot pie chart for employment_type
        fig1 = px.pie(df, names='employment_type', title='Distribution of Employment Type')
        st.plotly_chart(fig1, use_container_width=True)

    with pie_col2:
        # Plot pie chart for required_experience
        fig2 = px.pie(df, names='required_experience', title='Distribution of Required Experience')
        st.plotly_chart(fig2, use_container_width=True)

    with pie_col3:
        # Plot pie chart for required_education
        fig3 = px.pie(df, names='required_education', title='Distribution of Required Education')
        st.plotly_chart(fig3, use_container_width=True)


    # Create column layout
    bar_col1, bar_col2, bar_col3 = st.columns(spec=(1, 1, 1), gap="large")

    # Plot bar plot for has_company_logo
    with bar_col1:
        fig5 = px.bar(df['has_company_logo'].value_counts().reset_index(), x='has_company_logo', y='count',
                      labels={'index': 'Has Company Logo', 'has_company_logo': 'Count'},
                      title='Distribution of Has Company Logo', width=200)
        st.plotly_chart(fig5, use_container_width=True)

    # Plot bar plot for Salary_range_provided
    with bar_col2:
        fig7 = px.bar(df['Salary_range_provided'].value_counts().reset_index(), x='Salary_range_provided', y='count',
                      labels={'index': 'Salary Range Provided', 'Salary_range_provided': 'Count'},
                      title='Distribution of Salary Range Provided', width=200)
        st.plotly_chart(fig7, use_container_width=True)

    # Plot bar plot for department_mentioned
    with bar_col3:
        fig8 = px.bar(df['department_mentioned'].value_counts().reset_index(), x='department_mentioned', y='count',
                      labels={'index': 'Department Mentioned', 'department_mentioned': 'Count'},
                      title='Distribution of Department Mentioned', width=200)
        st.plotly_chart(fig8, use_container_width=True)


    # Calculate value counts for the industry feature
    ind_counts_filtered = pd.read_csv("Dataset/Industry_counts.csv")

    # Create bubble plot
    fig = px.scatter(ind_counts_filtered, x='industry', y=ind_counts_filtered.index, size='count',
                     labels={'industry': 'Industry', 'count': 'Count'},
                     title='Understanding the frequency of industry domains',
                     size_max=50)

    # Update layout
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    # Show plot
    st.plotly_chart(fig, use_container_width=True)





def multivariate_analysis(processed_df):
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Multivariate Analysis</h1>",
        unsafe_allow_html=True,
    )

    col1,col2 = st.columns(spec=(1, 1), gap="small")
    with col1:

        # Stacked Bar Chart for Employment Type
        with st.container():
            pivot = processed_df.pivot_table(index='employment_type', columns='required_experience', values='fraudulent',
                                     aggfunc='mean')
            # Plot heatmap
            fig = px.imshow(pivot, labels={'color': 'Fraud Rate'},
                            title='Heatmap of Fraud by Employment Type and Experience Level')
            st.plotly_chart(fig)




        # Stacked Bar Chart for Required Education
        with st.container():
            # Group by telecommuting and company logo
            grouped = processed_df.groupby(['telecommuting', 'has_company_logo']).mean().reset_index()

            # Plot grouped bar chart
            fig = px.bar(grouped, x='has_company_logo', y='fraudulent', color='telecommuting', barmode='group',
                         title='Grouped Bar Chart of Fraud by Telecommuting and Company Logo')
            st.plotly_chart(fig)

        # Stacked Bar Chart for Required Experience
        with st.container():
            # Plot scatter plot matrix
            fig = px.scatter_matrix(processed_df, dimensions=['department_mentioned', 'Salary_range_provided'],
                                    color='fraudulent',
                                    title='Scatter Plot Matrix of Fraud by Department Mentioned and Salary Range Provided')
            st.plotly_chart(fig)

    with col2:
        pass
        # with st.container():
        #     # Plot facet grid (using scatter plot with facet row and facet col)
        #     fig = px.scatter(processed_df, x='telecommuting', y='fraudulent', facet_row='required_experience',
        #                      facet_col='required_education', color='fraudulent',
        #                      title='Facet Grid of Fraud by Experience, Education, and Telecommuting')
        #
        #     # Streamlit
        #     st.plotly_chart(fig)


def feature_selection(df):
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Feature Selection</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 18px; text-align: left;'>A heatmap is a graphical representation of data where individual values are depicted as colors, allowing for quick visualization of patterns and relationships within a dataset. It is particularly useful for displaying the correlation matrix of features in a dataset, where the color intensity represents the strength of the correlation between pairs of features. For feature selection, heatmaps help identify highly correlated features, which can be redundant, enabling data scientists to simplify models by removing one of the correlated features. This process enhances model performance and interpretability by reducing multicollinearity and retaining only the most relevant features.</p>",
        unsafe_allow_html=True,
    )

    # Heatmap for Feature Correlations
    # Encoding categorical features
    df_encoded = df.copy()
    df_encoded['employment_type'] = df_encoded['employment_type'].astype('category').cat.codes
    df_encoded['required_experience'] = df_encoded['required_experience'].astype('category').cat.codes
    df_encoded['required_education'] = df_encoded['required_education'].astype('category').cat.codes
    df_encoded['industry'] = df_encoded['industry'].astype('category').cat.codes

    # Compute correlation matrix
    corr_matrix = df_encoded.corr()

    # Create annotated heatmap
    fig5 = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.round(2).values,
        colorscale='Viridis',
        showscale=True  # Add color scale (color bar)
    )

    # Update layout for better readability
    fig5.update_layout(
        title='Heatmap for Feature Correlations',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        "<p style='font-size: 18px; text-align: left;'>The feature selection techniques used in this analysis include RF_Importance, which measures feature importance based on the decrease in node impurity in a Random Forest, and GB_Importance, which assesses feature importance based on each feature's contribution to predictions in Gradient Boosting. Perm_importance evaluates feature importance by measuring the decrease in model performance when the feature values are randomly shuffled, while RFE_importance (Recursive Feature Elimination) ranks features by recursively fitting the model and removing the least important feature(s). These techniques provide different perspectives on feature significance, enhancing the robustness of the feature selection process..</p>",
        unsafe_allow_html=True,
    )

    # Now we will define the barplots
    data = {
        'Features': ['telecommuting', 'has_company_logo', 'has_questions', 'Salary_range_provided',
                     'department_mentioned', 'employment_type', 'required_experience', 'required_education',
                     'industry'],
        'RF_Importance': [0.011477, 0.067639, 0.020699, 0.019789, 0.023458, 0.121970, 0.247988,
                          0.182749, 0.304231],
        'GB_Importance': [0.001457, 0.159125, 0.007848, 0.003218, 0.008740, 0.251612, 0.422582,
                          0.056006, 0.089412],
        'Perm_importance': [-0.007493, 0.018411, 0.011561, 0.008778, 0.030614, 0.167202, 0.398844,
                            0.107686, 0.264397],
        'RFE_importance': [0.011477, 0.067639, 0.020699, 0.019789, 0.023458, 0.121970, 0.247988,
                           0.182749, 0.304231]
    }
    df = pd.DataFrame(data)

    # Define the techniques
    techniques = ['RF_Importance', 'GB_Importance', 'Perm_importance', 'RFE_importance']

    # Create subplots: 1 row, 5 columns
    fig = make_subplots(rows=1, cols=5, subplot_titles=techniques)

    # Add bars to each subplot
    for i, technique in enumerate(techniques, start=1):
        fig.add_trace(
            go.Bar(x=df['Features'], y=df[technique], name=technique),
            row=1, col=i
        )

    # Update layout for better readability
    fig.update_layout(
        height=500,
        width=2200,
        title_text='Feature Importance by Different Techniques',
        showlegend=False
    )

    # Rotate x-axis labels for all subplots
    for i in range(1, 5):
        fig.update_xaxes(tickangle=-45, row=1, col=i)

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.write(" ")



def model_performance():
    st.markdown(
        "<h2 style='text-align: left; font-size: 40px; '>Model Performance</h1>",
        unsafe_allow_html=True,
    )
    #

    col1,col2 = st.columns(spec=(1, 1), gap="large")
    with col1:
        y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])

        # Compute confusion matrix
        cm = np.zeros((2, 2))
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1

        # Create Plotly heatmap
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues'
        )

        # Update layout
        fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted label', yaxis_title='True label')

        # Display plot
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.3, 0.7, 0.6, 0.2, 0.4, 0.8])

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # Create Plotly figure
        fig = go.Figure()

        # Add ROC curve
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='blue', width=2),
                                 name=f'ROC Curve (AUC={roc_auc:.2f})'))
        # Add random guess line
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1, dash='dash'),
                      name='Random Guess')

        # Update layout
        fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                          xaxis_title='False Positive Rate (FPR)',
                          yaxis_title='True Positive Rate (TPR)',
                          xaxis=dict(range=[0, 1], constrain='domain'),
                          yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
                          width=800, height=600)

        # Display plot
        st.plotly_chart(fig, use_container_width=True)



def visualization():

    # Calling the function to load the dataframe
    df = load_dataset()
    temp_df = back_encoding(df.copy())

    # Calling function for doing univariate analysis
    univariate_analysis(temp_df)

    # Calling function for doing multivariate analysis
    #multivariate_analysis(df)

    # Feature selection charts
    feature_selection(df)

    # Calling function for showing the performance
    model_performance()


visualization()