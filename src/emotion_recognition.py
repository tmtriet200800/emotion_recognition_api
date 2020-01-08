import cv2
import os
from keras.models import load_model
import numpy as np
import sys

from src.utils.datasets import get_labels
from src.utils.inference import detect_faces
from src.utils.inference import draw_text
from src.utils.inference import draw_bounding_box
from src.utils.inference import apply_offsets
from src.utils.inference import load_detection_model
from src.utils.inference import load_image
from src.utils.preprocessor import preprocess_input

from keras import backend as K


def process_image(image):
    K.clear_session()

    # parameters for loading data and images
    if sys.path[1] == '/app':
        #load model for heroku
        detection_model_path = sys.path[1] + '/trained_models/detection_models/haarcascade_frontalface_default.xml'
        emotion_model_path = sys.path[1] + '/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    else:
        detection_model_path = sys.path[-1] + '/trained_models/detection_models/haarcascade_frontalface_default.xml'
        emotion_model_path = sys.path[-1] + '/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

    emotion_labels = get_labels('fer2013')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # loading images
    image_array = np.fromstring(image, np.uint8)
    unchanged_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(unchanged_image, cv2.COLOR_BGR2GRAY)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    faces = detect_faces(face_detection, gray_image)
    emotion_text_arr = []
    
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_text_arr.append(emotion_text)

        color=(255,0,0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    K.clear_session()

    return (bgr_image, emotion_text_arr)
