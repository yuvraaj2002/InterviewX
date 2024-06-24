import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import math
import base64

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


def angle_between_points(p1, p2):
    """
    Calculate the angle between two points in 3D space.

    Args:
        p1 (Point): First point, with attributes x, y, and z representing its coordinates.
        p2 (Point): Second point, with attributes x, y, and z representing its coordinates.

    Returns:
        float: Angle between the two points in degrees.

    Notes:
        - Requires the 'math' module.
        - The function uses the dot product of the vectors formed by the two points
          to calculate the angle between them.
    """
    x1, y1, z1 = p1.x, p1.y, p1.z
    x2, y2, z2 = p2.x, p2.y, p2.z

    # Calculate the dot product of the two vectors
    dot_product = x1 * x2 + y1 * y2 + z1 * z2

    # Calculate the magnitude of each vector
    magnitude1 = math.sqrt(x1**2 + y1**2 + z1**2)
    magnitude2 = math.sqrt(x2**2 + y2**2 + z2**2)

    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Use arccos to get the angle in radians
    angle_rad = math.acos(cosine_angle)

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def process_video(
    file_path, angles_shoulders, angles_lse, angles_rse, angles_lew, angles_rew
):

    # Decode video bytes into frames
    cap = cv2.VideoCapture(file_path)

    with mp_pose.Pose(
        min_detection_confidence=0.8, min_tracking_confidence=0.8
    ) as pose:
        frame_count = 0
        skip_count = 5  # Number of frames to skip between processing

        # Create a placeholder for the image
        image_placeholder = st.empty()

        # Clear the lists before starting the loop
        angles_shoulders.clear()
        angles_lse.clear()
        angles_rse.clear()
        angles_lew.clear()
        angles_rew.clear()

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
                            color=(245, 117, 66), thickness=4, circle_radius=3
                        ),
                        mp_drawing.DrawingSpec(
                            color=(245, 66, 230), thickness=4, circle_radius=3
                        ),
                    )
                    # Retrieve left and right shoulder coordinates
                    ls_cord = landmarks[
                        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
                    ]
                    rs_cord = landmarks[
                        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
                    ]

                    le_cord = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    re_cord = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

                    lw_cord = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    rw_cord = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                    angles_shoulders.append(angle_between_points(ls_cord, rs_cord))
                    angles_lse.append(angle_between_points(ls_cord, le_cord))
                    angles_rse.append(angle_between_points(rs_cord, re_cord))
                    angles_lew.append(angle_between_points(lw_cord, re_cord))
                    angles_rew.append(angle_between_points(rw_cord, re_cord))
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

    return [angles_shoulders, angles_lse, angles_rse, angles_lew, angles_rew]


def posture_analysis_page():
    st.markdown(
        "<h1 style='text-align: left; font-size: 52px;'>Posture analysis🕵️</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size: 22px; text-align: left;padding-right: 2rem;padding-bottom: 1rem;'>In today's competitive job market, where first impressions matter immensely, it's crucial for candidates to convey confidence through their body language and posture during interviews. Recognizing the importance of this aspect, our module is designed to assist candidates in analyzing their posture while addressing interview questions.By simply uploading a brief 20-30 second video and initiating the analysis through the click of a button, our algorithm delves into a detailed examination of the upper body. Specifically, it meticulously scrutinizes the angles formed between the shoulders, elbows, and wrists.</p>",
        unsafe_allow_html=True,
    )

    # Initialize lists to store left-shoulder, right-shoulder, shoulder, right-elbow-hand, left-elbow-hand
    angles_lse = []
    angles_rse = []
    angles_shoulders = []
    angles_rew = []
    angles_lew = []

    input_col, configuration_col = st.columns(spec=(2, 1.9), gap="large")
    with input_col:
        pass

    temp_file_path = None  # Initialize temp_file_path outside the if block

    with configuration_col:
        video = st.file_uploader("Upload the video")
        analyze_video = st.button("Analyze", use_container_width=True)
        video_processed = False
        if video is not None:
            if analyze_video:
                video_bytes = video.read()

                # Save uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(video_bytes)
                    temp_file_path = temp_file.name

                with input_col:
                    with st.spinner("Processing video 🔎"):
                        angles_shoulders, angles_lse, angles_rse, angles_lew, angles_rew = (
                            process_video(
                                temp_file_path,
                                angles_shoulders,
                                angles_lse,
                                angles_rse,
                                angles_lew,
                                angles_rew,
                            )
                        )
                    video_processed = True
                    st.success("Video processed successfully 👏")

                # Remove the temporary file after processing if temp_file_path is defined
                if temp_file_path:
                    os.unlink(temp_file_path)
        else:
            st.error("Upload the video and then press analyze button❗")

        # Convert lists to NumPy arrays
        if video_processed:
            angles_shoulders_array = np.array(angles_shoulders)
            angles_lse_array = np.array(angles_lse)
            angles_rse_array = np.array(angles_rse)
            angles_lew_array = np.array(angles_lew)
            angles_rew_array = np.array(angles_rew)

            # Calculate the mean of each array
            avg_angles = {
                "Shoulders angle": angles_shoulders_array.mean(),
                "Left shoulder-elbow angle": angles_lse_array.mean(),
                "Right shoulder-elbow angle": angles_rse_array.mean(),
                "Left elbow-wrist angle": angles_lew_array.mean(),
                "Right elbow-wrist angle": angles_rew_array.mean(),
            }

            st.write("***")
            row = st.columns(5)
            index = 0
            for name, avg_angle, col in zip(
                avg_angles.keys(), avg_angles.values(), row
            ):
                tile = col.container(height=123)  # Adjust the height as needed
                tile.markdown(
                    f"<p style='text-align: left; font-size: 18px; '>Average {name}: <b>{avg_angle:.2f}</b></p>",
                    unsafe_allow_html=True,
                )
                index += 1

            feedback_dict = {
                "Shoulder alignment": "",
                "Hand gestures": "",
                "Left Shoulder-Elbow": "",
                "Right Shoulder-Elbow": "",
            }

            if avg_angles["Shoulders angle"] > 0 and avg_angles["Shoulders angle"] < 15:
                feedback_dict["Shoulder alignment"] = "✅"
            else:
                feedback_dict["Shoulder alignment"] = "❌"

            if (
                avg_angles["Left elbow-wrist angle"] != 0
                and avg_angles["Right elbow-wrist angle"] != 0
            ):
                feedback_dict["Hand gestures"] = "✅"
            else:
                feedback_dict["Hand gestures"] = "❌"

            if (
                avg_angles["Left shoulder-elbow angle"] > 0
                and avg_angles["Left shoulder-elbow angle"] < 20
            ):
                feedback_dict["Left Shoulder-Elbow"] = "✅"
            else:
                feedback_dict["Left Shoulder-Elbow"] = "❌"

            if (
                avg_angles["Right shoulder-elbow angle"] > 0
                and avg_angles["Right shoulder-elbow angle"] < 20
            ):
                feedback_dict["Right Shoulder-Elbow"] = "✅"
            else:
                feedback_dict["Right Shoulder-Elbow"] = "❌"

            feedback_df = pd.DataFrame(feedback_dict, index=["Feedback"])
            st.dataframe(feedback_df)


posture_analysis_page()
