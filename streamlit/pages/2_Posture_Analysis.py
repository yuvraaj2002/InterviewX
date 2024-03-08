import cv2
import mediapipe as mp
import streamlit as st
import tempfile
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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def process_video(file_path, skip_count=2):
    # Decode video bytes into frames
    cap = cv2.VideoCapture(file_path)

    with mp_pose.Pose(
        min_detection_confidence=0.8, min_tracking_confidence=0.8
    ) as pose:
        frame_count = 0
        skip_count = 2  # Number of frames to skip between processing

        # Create a placeholder for the image
        image_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Process only every skip_count frame
            if frame_count % skip_count == 0:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Make detections
                results = pose.process(image)

                # Extract landmarks and render them (optional)
                try:
                    landmarks = results.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(245, 117, 66), thickness=2, circle_radius=2
                        ),
                        mp_drawing.DrawingSpec(
                            color=(245, 66, 230), thickness=2, circle_radius=2
                        ),
                    )
                except:
                    pass

                # Display the processed frame
                image_placeholder.image(image, channels="RGB")

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def posture_analysis_page():
    st.markdown(
        "<h1 style='text-align: left; font-size: 60px;'>Posture analysis🕵️</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 22px; text-align: left;padding-right: 2rem;padding-bottom: 1rem;'>In times of tough market situations, fake job postings and scams often spike, posing a significant threat to job seekers. To combat this, I've developed a user-friendly module designed to protect individuals from falling prey to such fraudulent activities. This module requires users to input details about the job posting they're considering. Behind the scenes, two powerful AI models thoroughly analyze the provided information. Once completed, users receive a clear indication of whether the job posting is is genuine or potentially decepti.</p>",
        unsafe_allow_html=True,
    )

    input_col, configuration_col = st.columns(spec=(2, 1.7), gap="large")
    with input_col:
        pass

    temp_file_path = None  # Initialize temp_file_path outside the if block

    with configuration_col:
        video = st.file_uploader("Upload the video")
        analyze_video = st.button("Analyze", use_container_width=True)
        if video is not None and analyze_video:
            video_bytes = video.read()

            # Save uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(video_bytes)
                temp_file_path = temp_file.name

            with input_col:
                process_video(temp_file_path)

            # Remove the temporary file after processing if temp_file_path is defined
            if temp_file_path:
                os.unlink(temp_file_path)

        row = st.columns(4)
        index = 0
        for col in row:
            tile = col.container(height=180)  # Adjust the height as needed
            tile.markdown(
                "<p style='text-align: left; font-size: 18px; '>This</p>",
                unsafe_allow_html=True,
            )
            index = index + 1

        video_download_col, statistics_download_col = st.columns(
            spec=(1, 1), gap="large"
        )
        with video_download_col:
            orignal_video = st.button("Play original video", use_container_width=True)
            if orignal_video:
                with input_col:
                    st.video(video)

            st.button("Download analysis chart", use_container_width=True)

        with statistics_download_col:
            processed_video = st.button(
                "Play processed video", use_container_width=True
            )
            if processed_video:
                with input_col:
                    st.video("output/processed_video.mp4")
                st.info("Click 'Analyze' to display processed frames.")
            st.button("Download processed video", use_container_width=True)


posture_analysis_page()
