import streamlit as st
import subprocess
import tempfile
import os
import uuid
import shutil


def run_yolo_command(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def convert_avi_to_mp4(input_file, output_file):
    command = f"ffmpeg -i {input_file} {output_file}"
    subprocess.run(command, shell=True)


def face_gesture_recognition():
    video_col, config_col = st.columns([3, 1], gap="large")

    with video_col:
        st.title("Face")
        video_displayed = False  # Variable to control video display

    with config_col:
        st.title("Configuration")
        st.markdown(
            "<p style='background-color: #d9b3ff; padding: 20px; border-radius: 10px; font-size: 20px;'>URLs submitted undergo HTTPS verification; if successful and devoid of paywalls, the article's content is extracted. A model generates variable chunks of summarized text for efficient data loading. These summarized chunks are stored in a text file for user access..</p>",
            unsafe_allow_html=True,
        )
        video_upload = st.file_uploader("Upload your video (MP4 only)", type=["mp4"])
        analyze_gestures_bt = st.button("Analyze the video", use_container_width=True)

        if analyze_gestures_bt:
            if video_upload is not None:
                video_bytes = video_upload.read()
                video_displayed = True  # Set to True when video is uploaded

                temp_dir = "/home/yuvraj/Documents/AI/AI_Projects/AI_Tutor/Temp"
                temp_raw_file = os.path.join(temp_dir, "raw.mp4")

                # Save the uploaded video to a temporary file
                with open(temp_raw_file, "wb") as f:
                    f.write(video_bytes)

                # Execute YOLO command
                # Execute YOLO command
                yolo_command = f"yolo task=detect mode=predict model=/home/yuvraj/Documents/AI/AI_Projects/AI_Tutor/Artifacts/best.pt conf=0.30 source={temp_raw_file} save=True"
                output, error = run_yolo_command(yolo_command)
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("YOLO command executed successfully!")

                runs_dir = "/home/yuvraj/Documents/AI/AI_Projects/AI_Tutor/runs"
                prediction_folder = os.path.join(runs_dir, "detect", "predict")
                temp_pred_file_avi = os.path.join(
                    prediction_folder, os.listdir(prediction_folder)[0]
                )
                temp_pred_file_mp4 = os.path.join(temp_dir, "processed_video.mp4")
                convert_avi_to_mp4(temp_pred_file_avi, temp_pred_file_mp4)

                # Delete the temporary folder, and runs folder
                os.remove(temp_raw_file)
                shutil.rmtree(
                    "/home/yuvraj/Documents/AI/AI_Projects/AI_Tutor/runs/detect"
                )

    # Display the video in the video_col if it's uploaded and button clicked
    if video_displayed:
        with video_col:
            st.video(temp_pred_file_mp4)
            os.remove(temp_pred_file_mp4)
