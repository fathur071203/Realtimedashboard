import numpy as np
import cv2
import streamlit as st
import pandas as pd
from tensorflow import keras
from keras.models import model_from_json
import plotly.express as px
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import time
import threading

# load model
emotion_dict = ["marah", "risih", "takut", "senyum", "netral", "sedih", "terkejut"]
emotion_prediction = [[0, 0, 0, 0, 0, 0, 0]]
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("emotion_model.h5")

# load face
try:
    face_cascade = cv2.CascadeClassifier('hc.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activities = ["Home", "Webcam Face Detection"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")

        # Use st.columns to create a layout with two columns
        b1, b2 = st.columns((3, 7))

        # Put the webrtc_streamer in the first column
        with b1:
            st.write("Tekan Start Untuk melakukan Deteksi Muka")
            webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                            video_processor_factory=Faceemotion)

        # Put the bar chart in the second column
        with b2:
            st.header("REMOSTO Emotion Analysis Chart")
            # Additional code for displaying bar chart
            expression_df = pd.DataFrame(columns=emotion_dict)

            mean_values = [0] * len(emotion_dict)  # Initialize with zeros

            colors = ['red', 'green', 'grey', 'orange', 'lightgrey', 'blue', 'purple']

            # Bar chart
            fig = px.bar(x=emotion_dict, y=mean_values,
                         color=emotion_dict, color_discrete_sequence=colors,
                         text_auto='.2s')

            fig.update_xaxes(title_text=None)
            fig.update_yaxes(title_text=None)
            fig.update_layout(showlegend=False)

            fig.update_layout(
                width=600,
                height=350,
                title_text="Visitor Emotions",
                title_x=0,
                title_y=0.865,
                title_font=dict(size=27),
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                margin=dict(l=10, r=10, t=110, b=0)
            )

            emoji_mapping = {
                'marah': '😡',
                'risih': '😒',
                'takut': '😱',
                'senyum': '😊',
                'netral': '😐',
                'sedih': '😢',
                'terkejut': '😲'
            }

            for i, expression in enumerate(emotion_dict):
                emoji = emoji_mapping.get(expression, '')  # Get the emoji corresponding to the expression
                fig.add_annotation(
                    x=i,
                    y=mean_values[i] + 0.5,
                    text=emoji,
                    showarrow=False,
                    font=dict(family="Arial", size=14),
                    xanchor="center"
                )

            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

            chart = st.plotly_chart(fig, use_container_width=True)

            # Start a separate thread to update the emotion chart
            if not hasattr(st.session_state, "update_thread"):
                st.session_state.update_thread = True
                st.session_state.expression_df = expression_df

                def update_chart():
                    while True:
                        # Get the current emotion DataFrame
                        emotion_df = st.session_state.expression_df

                        # Perform the emotion detection and update the DataFrame
                        # (You may need to replace this part with your live emotion detection logic)
                        current_emotion = [0.2, 0.3, 0.1, 0.15, 0.05, 0.1, 0.2]
                        emotion_df = emotion_df.append(pd.Series(current_emotion, index=emotion_dict), ignore_index=True)

                        # Calculate mean values
                        mean_values = emotion_df.mean()

                        # Update the chart data
                        fig.data[0].y = mean_values

                        # Wait for 3 seconds before the next update
                        time.sleep(3)

                st.session_state.chart_update_thread = threading.Thread(target=update_chart)
                st.session_state.chart_update_thread.start()
    else:
        pass

if __name__ == "__main__":
    main()
