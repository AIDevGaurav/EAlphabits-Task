from flask import Flask, request, jsonify
import threading
from main import detect_motion

app = Flask(__name__)

# Global list to store threads for each RTSP stream
threads = []

# Function to start the detection process for a given RTSP link
def start_detection(rtsp_url, camera_id, detection_type, coordinates):
    # Depending on the 'detection_type', you can start the appropriate detection logic.
    # For this example, we assume `detection_type` can be "PERSON_DETECTION" or others.
    
    if detection_type == "PERSON_DETECTION":
        # Example of calling your existing detect_motion function
        detect_motion(rtsp_url, camera_id, coordinates)
    else:
        print(f"Detection type {detection_type} is not supported.")

# Flask route to start detection on a new RTSP link
@app.route('/start', methods=['POST'])
def start():
    data = request.json
    rtsp_url = data.get('rtsp_link')
    camera_id = data.get('cameraId')
    detection_type = data.get('type')
    coordinates = data.get('co-ordinates')

    # Validate the input data
    if not rtsp_url or not camera_id or not detection_type or not coordinates:
        return jsonify({"error": "Please provide rtsp_link, cameraId, type, and coordinates"}), 400

    # Start a new thread for this RTSP link and detection type
    thread = threading.Thread(target=start_detection, args=(rtsp_url, camera_id, detection_type, coordinates))
    thread.start()
    threads.append(thread)
    return jsonify({"message": f"Started {detection_type} for Camera ID {camera_id} at {rtsp_url}"}), 200

# Flask route to stop all running threads (optional, if needed)
@app.route('/stop', methods=['POST'])
def stop():
    # Logic to stop threads (not implemented here, as stopping threads cleanly can be complex)
    return jsonify({"message": "Stopping all streams is not implemented"}), 501

# Main function to run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
