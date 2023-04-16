import cv2
from config.core import config, get_class_labels
from predict import predict_objects, load_yolo_model


# load class names
classes = get_class_labels()

# get YOLO model
net = load_yolo_model()

# Open video stream
cap = cv2.VideoCapture(config.cam_config.cam_schody)  # Replace with your video URL or video file path

detections = predict_objects(cap=cap,net=net, classes=classes)
print(detections)

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
