from flask import Flask, jsonify
import cv2
from config.core import config, get_class_labels
from predict import predict_objects, load_yolo_model


app = Flask(__name__)

# Function to be executed before the first request
# @app.before_first_request
# def setup():
#     # Perform any setup logic here
#     pass

@app.route('/detect/<camera_name>')
def detect(camera_name):
    # Access the initialized objects in your script logic
    # load class names
    # return "200"
    classes = get_class_labels()

    # get YOLO model
    net = load_yolo_model()

    cam_urls = [cam.url
               for cam in config.cam_config.cameras 
               if cam.name == camera_name
               ]
    if cam_urls:
        cam_url = cam_urls[0]

    # Open video stream
    cap = cv2.VideoCapture(cam_url)  # Replace with your video URL or video file path

    detections = predict_objects(cap=cap,net=net, classes=classes)
    print(detections)
    
    # Release video capture and close all OpenCV windows
    cap.release()
    # cv2.destroyAllWindows()

    # Return the result in JSON format
    return jsonify(detections['object'])

if __name__ == '__main__':
    app.run(debug=True)