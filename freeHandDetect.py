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

# Function to handle mouse events for drawing the ROI
def draw_roi(event, x, y, flags, param):
    global drawing, current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_points.append((x, y))
            cv2.line(display_frame, current_points[-2], current_points[-1], (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_points.append((x, y))
        cv2.line(display_frame, current_points[-2], current_points[-1], (0, 255, 0), 2)
        roi_points.append(current_points)  # Save the current shape
        cv2.fillPoly(roi_mask, [np.array(current_points)], (255, 255, 255))

# Function to detect motion
def detect_motion():
    global roi_mask, display_frame, roi_points, drawing, min_area

    cap = cv2.VideoCapture(rtsp_url)

    # Parameters for motion detection
    threshold_value = 16
    min_area_full_frame = 1200

    # Initialize previous frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the stream.")
        return

    frame = cv2.resize(frame, None, fx=2.3, fy=1.3)
    display_frame = frame.copy()
    roi_mask = np.zeros_like(frame, dtype=np.uint8)

    # Set up the mouse callback to draw the ROI
    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", draw_roi)

    while True:
        cv2.imshow("Draw ROI", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to finish drawing
            break

    cv2.destroyWindow("Draw ROI")

    # Resize the ROI mask to match any resized frames
    roi_mask = cv2.resize(roi_mask, (frame.shape[1], frame.shape[0]))

    # Calculate the area of the ROI
    roi_area = sum(cv2.contourArea(np.array(shape)) for shape in roi_points)
    full_frame_width = 1920
    full_frame_height = 1080
    full_frame_area = full_frame_width * full_frame_height
    min_area = (min_area_full_frame / full_frame_area) * roi_area

    masked_frame = cv2.bitwise_and(frame, roi_mask)
    prev_frame_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

    last_detection_time = 0

    # Create a thread pool executor for background tasks
    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, None, fx=2.3, fy=1.3)
            roi_mask_resized = cv2.resize(roi_mask, (frame.shape[1], frame.shape[0]))

            display_frame = frame.copy()  # Copy for displaying with rectangles
            masked_frame = cv2.bitwise_and(frame, roi_mask_resized)
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
            if motion_detected and (current_time - last_detection_time > 10):  # replace 60 with your desired second delay....
                cv2.putText(display_frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Capture image and video in the background
                future_image = executor.submit(capture_image, frame.copy())
                future_video = executor.submit(capture_video, rtsp_url)

                # Prepare and publish MQTT message after capturing image and video
                def publish_mqtt_message(future_image, future_video):
                    image_filename = future_image.result()
                    video_filename = future_video.result()
                    message = {
                        "motion": "Motion detected!",
                        "rtsp_link": rtsp_url,
                        "cameraId": 20,
                        "image": image_filename,
                        "video": video_filename
                    }
                    publish_message(str(message))

                executor.submit(publish_mqtt_message, future_image, future_video)

                last_detection_time = current_time

            # Draw all ROIs on the display frame
            for shape in roi_points:
                cv2.polylines(display_frame, [np.array(shape)], isClosed=True, color=(255, 0, 0), thickness=2)

            # Display frame
            cv2.imshow("Motion Detection", display_frame)

            prev_frame_gray = gray_frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Motion Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

# Global variables for drawing the ROIs
roi_points = []  # Store all shapes here
current_points = []  # Points for the current shape being drawn
drawing = False

# RTSP URL for motion detection
rtsp_url = "rtsp://user:Admin$123@103.162.197.92:600/media/video2"  # Replace with your RTSP URL

# Start motion detection
detect_motion()

# Stop the MQTT client loop and disconnect
mqtt_client.loop_stop()
mqtt_client.disconnect()
