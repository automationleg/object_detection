import cv2
import numpy as np
from config.core import config
import time
import logging

# Configure logging
logging.basicConfig(filename='detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    n = net.getUnconnectedOutLayers()
    # Get the indices of the output layers, i.e., the layers that produce the final detections
    return [layersNames[i-1] for i in net.getUnconnectedOutLayers()]


# Load YOLO model
def load_yolo_model():
    net = cv2.dnn.readNetFromDarknet(config.model_config.yolo_config_file, config.model_config.yolo_weights_file)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net


def predict_objects(cap: cv2.VideoCapture, net, classes) -> dict:
    ret, frame = cap.read()

    if ret:
        # Get current time
        current_time = time.time()

        # Convert frame to blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Set the input blob to the network
        net.setInput(blob)

        # Forward pass and get the output from the output layers
        outs = net.forward(getOutputsNames(net))

        # Process each output layer
        objects_detected = {}
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                # Filter out weak detections
                if confidence > 0.5:
                    # Get bounding box coordinates
                    x, y, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x, y, w, h = int(x), int(y), int(w), int(h)

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, classes[classId], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    logging.info('object="{}" detected at x={}, y={}, width={}, height={}, confidence={}, current_time={:.2f}s'.format(classes[classId],x, y, w, h, confidence, current_time))
                    objects_detected["object"] = classes[classId]
                    objects_detected["confidence"] = confidence
        
        return objects_detected