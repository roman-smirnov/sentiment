from statistics import mode
import cv2
from utils import *

FRAME_WINDOW = 10
EMOTION_OFFSETS = (20, 40)
NUM_HANDS = 2
MIN_HAND_SCORE = 0.27

expression_score = 50
hands_score = 50
speech_score = 50

face_detector, hand_detector, hand_sess, expression_classifier = load_models()

# label dictionary
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# starting lists for calculating modes
expression_window = []

# output window frame
cv2.namedWindow('window_frame')

# starting video streaming
video_capture = cv2.VideoCapture(0)

while True:
    bgr_image = video_capture.read()[1]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    hand_detections = detect_objects(rgb_image, hand_detector, hand_sess)

    for box, score in zip(hand_detections[0], hand_detections[1]):
        if score < MIN_HAND_SCORE:
            continue
        y1, x1, y2, x2 = box
        x1, x2 = int(x1 * 1280), int(x2 * 1280)
        y1, y2 = int(y1 * 720), int(y2 * 720)
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=7, minSize=(23, 23))

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, EMOTION_OFFSETS)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (64, 64))
        except:
            continue

        gray_face = gray_face.astype('float32') / 127 - 1
        expression_prediction = expression_classifier.predict(gray_face[None, ..., None])

        emotion_probability = np.max(expression_prediction)
        expression_label_arg = np.argmax(expression_prediction)
        emotion_text = labels[expression_label_arg]
        expression_window.append(emotion_text)

        if len(expression_window) > FRAME_WINDOW:
            expression_window.pop(0)
        try:
            emotion_mode = mode(expression_window)
        except:
            continue

        cv2.rectangle(rgb_image, (x1, y1 + 10), (x2, y2), (0, 255, 0), 1)
        cv2.putText(rgb_image, emotion_mode, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
