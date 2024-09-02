from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
import threading
import cv2
import numpy as np
import os
import time
import paho.mqtt.client as mqtt # type: ignore
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MQTT configuration
broker = "192.168.1.120"  # Replace with your MQTT broker address
port = 1883  # Replace with your MQTT broker port
topic = "motion/detection"

# Global dictionary to keep track of threads and stop flags
tasks_threads = {}

# Define the MQTT client callbacks
def on_connect(client, userdata, flags, rc):
    client_id = client._client_id.decode()
    print(f"Connected with result code {rc} and client id {client_id}")
    client.subscribe(topic)  # Subscribe to the topic when connected

def on_disconnect(client, userdata, rc):
    print("Disconnected with result code " + str(rc))
    if rc != 0:
        print("Unexpected disconnection. Attempting to reconnect...")
        try:
            client.reconnect()
        except Exception as e:
            print(f"Reconnection failed: {e}")
    else:
        print("Disconnected successfully.")

# Initialize MQTT client and set up callbacks
mqtt_client = mqtt.Client(client_id="Gaurav")
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect

mqtt_client.connect(broker, port, keepalive=300)  # Set to 300 seconds or as needed

# Start the MQTT loop and keep it running in a background thread
mqtt_client.loop_start()

# Ensure directories exist
image_dir = "images"
video_dir = "videos"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

def capture_image(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(image_dir, f"motion_{timestamp}.jpg")
    cv2.imwrite(image_filename, frame)
    absolute_image_path = os.path.abspath(image_filename)
    print(f"Captured image: {absolute_image_path}")
    return absolute_image_path

def capture_video(rtsp_url):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(video_dir, f"motion_{timestamp}.mp4")
    
    # Use the MP4V codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    cap_video = cv2.VideoCapture(rtsp_url)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a VideoWriter object with MP4 format
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

    start_time = time.time()
    while int(time.time() - start_time) < 5:  # Capture for 5 seconds
        ret, frame = cap_video.read()
        if not ret:
            break
        out.write(frame)

    cap_video.release()
    out.release()
    absolute_video_path = os.path.abspath(video_filename)
    print(f"Captured video: {absolute_video_path}")
    return absolute_video_path

def publish_message(motion_type, rtsp_url, camera_id, image_filename, video_filename):
    message = {
        "rtsp_link": rtsp_url,
        "cameraId": camera_id,
        "type": motion_type,
        "image": image_filename,
        "video": video_filename
    }
    mqtt_client.publish(topic, json.dumps(message))
    print(f"Published message: {json.dumps(message)}")

def set_roi_based_on_points(points, coordinates):
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]
    
    scaled_points = []
    for point in points:
        scaled_x = int(point[0] + x_offset)
        scaled_y = int(point[1] + y_offset)
        scaled_points.append((scaled_x, scaled_y))

    return scaled_points

def detect_motion(rtsp_url, camera_id, coordinates, motion_type, stop_event):
    global roi_points, min_area

    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increase buffer size

    # Parameters for motion detection
    threshold_value = 16
    min_area_full_frame = 1200

    # Initialize previous frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the stream.")
        return

    # Get display width and height from API coordinates
    display_width = coordinates["display_width"]
    display_height = coordinates["display_height"]

    # Resize the frame to match the display size
    frame = cv2.resize(frame, (display_width, display_height))

    # Set ROI based on the points provided by the API
    roi_points_from_api = coordinates["points"]
    roi_points = set_roi_based_on_points(roi_points_from_api, coordinates)

    # Convert roi_points to a numpy array of type int32
    roi_points = np.array(roi_points, dtype=np.int32)

    # Create an ROI mask from the points
    roi_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_points], (255, 255, 255))

    # Calculate the area of the ROI
    roi_area = cv2.countNonZero(roi_mask)
    full_frame_area = display_width * display_height
    min_area = (min_area_full_frame / full_frame_area) * roi_area

    # Initialize previous frame for comparison
    prev_frame_gray = None

    last_detection_time = 0

    window_name = f"Motion Detection - Camera {camera_id}"

    # Main motion detection loop
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (display_width, display_height))
        display_frame = frame.copy()

        masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

        gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # If prev_frame_gray is None, set it and skip this iteration
        if prev_frame_gray is None:
            prev_frame_gray = gray_frame
            continue

        # Compare the current frame with the previous frame
        frame_diff = cv2.absdiff(prev_frame_gray, gray_frame)
        _, thresh_frame = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_detected = True

        current_time = time.time()
        if motion_detected and (current_time - last_detection_time > 60):
            cv2.putText(display_frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Capture image and video in the background
            frame_copy = frame.copy()
            image_filename = capture_image(frame_copy)
            video_filename = capture_video(rtsp_url)

            # Publish MQTT message asynchronously with image and video filenames
            publish_message(motion_type, rtsp_url, camera_id, image_filename, video_filename)

            last_detection_time = current_time

        # Draw all ROIs on the display frame
        cv2.polylines(display_frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)

        # Display frame
        cv2.imshow(window_name, display_frame)

        # Update the previous frame to the current one
        prev_frame_gray = gray_frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyWindow(window_name)  # Close only the specific window for this camera

# Function to run the detection in a new thread
def start_detection(task):
    camera_id = task["cameraId"]
    stop_event = threading.Event()
    tasks_threads[camera_id] = stop_event
    rtsp_url = task["rtsp_link"]
    motion_type = task["type"]
    coordinates = task["co_ordinates"]
    detect_motion(rtsp_url, camera_id, coordinates, motion_type, stop_event)

# Endpoint to receive an array of motion detection tasks
@app.route('/start', methods=['POST'])
def detect_motion_endpoint():
    tasks = request.json

    # Start each detection task in a separate thread
    for task in tasks:
        camera_id = task["cameraId"]
        if camera_id not in tasks_threads:
            thread = threading.Thread(target=start_detection, args=(task,))
            thread.start()

    # Immediately return a response to the client
    return jsonify({"status": "Motion detection tasks started"}), 200

@app.route('/stop', methods=['POST'])
def stop_motion_detection():
    camera_ids = request.json.get('cameraIds', [])  # Get the array of camera IDs from the request
    
    # Check if the cameraIds is a list
    if not isinstance(camera_ids, list):
        return jsonify({"error": "cameraIds should be an array"}), 400

    stopped_tasks = []
    not_found_tasks = []

    for camera_id in camera_ids:
        if camera_id in tasks_threads:
            tasks_threads[camera_id].set()  # Set the stop event for the corresponding task
            del tasks_threads[camera_id]  # Remove the task from the dictionary
            stopped_tasks.append(camera_id)
            cv2.destroyWindow(f"Motion Detection - Camera {camera_id}")  # Close the window for this camera
        else:
            not_found_tasks.append(camera_id)

    # Determine success based on whether any tasks were stopped
    success = len(stopped_tasks) > 0

    # Prepare the response
    response = {
        "success": success,
        "stopped": stopped_tasks,
        "not_found": not_found_tasks
    }

    return jsonify(response), 200

if __name__ == '__main__':
    from waitress import serve  # type: ignore
    serve(app, host='0.0.0.0', port=5000)
