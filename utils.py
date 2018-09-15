import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf

detection_graph = tf.Graph()


# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory
    print("> ====== loading HAND frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('./models/hand_model.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
    return detection_graph, sess


def detect_objects(image_np, detection_graph, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 1)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0, font_scale=1, thickness=1):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
                cv2.LINE_AA)


def updated_mean(mean1, var1, mean2, var2):
    new_mean = (mean1 * var2 + mean2 * var1) / (var1 + var2)
    return new_mean


def updated_var(var1, var2):
    new_var = 1 / ((1 / var1) + (1 / var2))
    return new_var


def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]


def load_models():
    face_detector = cv2.CascadeClassifier('./models/face_haar.xml')
    hand_detector, hand_sess = load_inference_graph()
    expression_classifier = load_model('./models/emotion_model.hdf5', compile=False)
    return face_detector, hand_detector, hand_sess, expression_classifier
