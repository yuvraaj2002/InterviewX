import streamlit as st
import tensorflow as tf
from tensorflow import keras
import time

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

def segmentation_UI():
    st.markdown(
        "<h1 style='text-align: center; font-size: 50px;'>QueryLabeler️</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 21px; text-align: center;padding-left: 2rem;padding-right: 2rem;margin-bottom: 2rem;'>Welcome to QuestClassify, the ultimate tool for efficiently organizing and categorizing your questions. Whether you're managing survey responses, curating educational content, or handling a large database of FAQs, QuestClassify streamlines your workflow and enhances productivity. Simply input your questions, and our advanced neural network will accurately classify each one into predefined categories: AI, Behavioural, CS Fundamentals, DSA (Data Structures and Algorithms), and System Design. QuestClassify automatically segments and organizes your questions based on their labels, ensuring easy retrieval and utilization. After the segmentation process, conveniently download a ZIP folder containing separate files for each category, simplifying data management and enhancing accessibility.</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        que_col, add_col, segment_col = st.columns(spec=(5,1,1), gap="medium")
        with que_col:
            text = st.text_input("Enter the text")
        with add_col:
            st.write("")
            st.write("")
            add_que_bt = st.button("Add Question ✅",use_container_width=True)
        with segment_col:
            st.write("")
            st.write("")
            segment_bt = st.button("Perform Segmentation 🏷️",use_container_width=True)

    if segment_bt:
        progress_text = "Segmenting questions. This may take a few moments. Please wait..."
        bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.001)
            bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        bar.empty()
        st.success("Segmentation Done succesfully")
        st.write("")

        que_categories = ['Artificial Intelligence', 'Behavioural', 'CS fundamentals', 'Data structures and Algorithm', 'System Design']
        row = st.columns(5)
        index = 0
        for col in row:
            tile = col.container(height=200)  # Adjust the height as needed
            tile.markdown(
                "<p style='text-align: center; font-size: 18px; background-color: #C3E8FF;padding:0.5rem;'>"
                + str(que_categories[index])
                + "</p>",
                unsafe_allow_html=True,
            )

            # if index == 4:
            #     tile.metric(
            #         label="Similarity Score",
            #         value=0,
            #         delta="Base line score",
            #     )
            # else:
            #     tile.metric(
            #         label="Similarity Score",
            #         value=round(facilities_results["SimilarityScore"][index], 3),
            #         delta=round(
            #             facilities_results["SimilarityScore"][index]
            #             - baseline_similarity_score,
            #             5,
            #         ),
            #     )
            index = index + 1

    
    # loaded_model = keras.models.load_model("artifacts/Que_Classifier.keras")
    # st.write(loaded_model.summary())


segmentation_UI()