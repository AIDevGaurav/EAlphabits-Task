import cv2
import numpy as np
import os
import time
import threading
import paho.mqtt.client as mqtt  # type: ignore
from concurrent.futures import ThreadPoolExecutor

# MQTT configuration
broker = "192.168.1.75"  # Replace with your MQTT broker address
port = 1883  # Replace with your MQTT broker port
topic = "motion/detection"

# Define the MQTT client callbacks
def on_connect(client, userdata, flags, rc):
    client_id = client._client_id.decode()
    print(f"Connected with result code {rc} and client id {client_id}")
    client.subscribe(topic)  # Subscribe to the topic when connected

def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} -> {msg.payload.decode()}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")
    else:
        print("Disconnected successfully.")

# Initialize MQTT client and set up callbacks
mqtt_client = mqtt.Client(client_id="Gaurav")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

# Connect to the broker
mqtt_client.connect(broker, port, 60)

# Function to publish a message
def publish_message(message):
    mqtt_client.publish(topic, message)
    print(f"Published message: {message}")

# Start the MQTT client loop in a separate thread
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
    print(f"Captured image: {image_filename}")
    return image_filename

def capture_video(rtsp_url):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(video_dir, f"motion_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap_video = cv2.VideoCapture(rtsp_url)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))

    start_time = time.time()
    while int(time.time() - start_time) < 5:
        ret, frame = cap_video.read()
        if not ret:
            break
        out.write(frame)

    cap_video.release()
    out.release()
    print(f"Captured video: {video_filename}")
    return video_filename

# Function to set ROI based on provided points from API
def set_roi_based_on_points(points, coordinates):
    x_offset = coordinates["x"]
    y_offset = coordinates["y"]
    
    scaled_points = []
    for point in points:
        scaled_x = int(point[0] + x_offset)
        scaled_y = int(point[1] + y_offset)
        scaled_points.append((scaled_x, scaled_y))

    return scaled_points

# Function to detect motion
def detect_motion(rtsp_url, camera_id, coordinates):
    global roi_points, min_area

    cap = cv2.VideoCapture(rtsp_url)

    # Parameters for motion detection
    threshold_value = 16
    min_area_full_frame = 1200

    # Initialize previous frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the stream.")
        return

    original_height, original_width = frame.shape[:2]

    # Get display width and height from API coordinates
    display_width = coordinates["display_width"]
    display_height = coordinates["display_height"]

    # Calculate the resizing factors dynamically
    fx = display_width / original_width
    fy = display_height / original_height

    # Resize the frame to match the display size
    frame = cv2.resize(frame, (display_width, display_height))

    # Set ROI based on the points provided by the API
    roi_points_from_api = coordinates["points"]
    roi_points = set_roi_based_on_points(roi_points_from_api, coordinates)

    # Create an ROI mask from the points
    roi_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_points)], (255, 255, 255))

    # Calculate the area of the ROI
    roi_area = cv2.countNonZero(roi_mask)
    full_frame_area = display_width * display_height
    min_area = (min_area_full_frame / full_frame_area) * roi_area

    prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

    last_detection_time = 0

    # Create a thread pool executor for background tasks
    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (display_width, display_height))
            display_frame = frame.copy()  # Ensure display_frame is updated every iteration

            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

            gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

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
                frame_copy = frame.copy()  # Create a copy to pass to the background thread
                executor.submit(capture_image, frame_copy)
                executor.submit(capture_video, rtsp_url)

                # Publish MQTT message asynchronously without waiting for image/video capture completion
                executor.submit(publish_message, f"Motion detected! Camera ID: {camera_id}")

                last_detection_time = current_time

            # Draw all ROIs on the display frame
            cv2.polylines(display_frame, [np.array(roi_points)], isClosed=True, color=(255, 0, 0), thickness=2)

            # Display frame
            cv2.imshow("Motion Detection", display_frame)
            
            # Update the previous frame to the current one
            prev_frame_gray = gray_frame.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Motion Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
if __name__ == '__main__':
    # Example RTSP URL and Camera ID
    rtsp_url = "rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live"
    camera_id = 20

    # Example coordinates as you would receive from an API
    coordinates = {
        "x": 529.8125,
        "y": 343,
        "width": 333,
        "height": 281,
        "points": [
            [0, 0],
            [101, -174],
            [345, -218],
            [376, -41],
            [36, 104],
            [0, 0]
        ],
        "display_width": 1382,  # Taken from API
        "display_height": 777   # Taken from API
    }

    # Start motion detection
    detect_motion(rtsp_url, camera_id, coordinates)

# Stop the MQTT client loop and disconnect
mqtt_client.loop_stop()
mqtt_client.disconnect()
