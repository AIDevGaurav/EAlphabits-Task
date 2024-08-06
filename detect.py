import cv2
import threading

rtsp_url = "rtsp://admin:admin@789@192.168.1.199:554/unicast/c1/s0/live"  # Replace with your RTSP URL

def detectMotion():
    # Replace 'your_rtsp_url' with the actual RTSP stream URL
    cap = cv2.VideoCapture(rtsp_url)

    # Parameters for motion detection
    threshold_value = 5  # Adjust Based on FPS
    min_area_full_frame = 50  # Set a lower min_area_full_frame if you want to detect smaller movements or objects.

    # Define the ROI (top-left and bottom-right corners)
    roi_top_left = (200, 150)
    roi_bottom_right = (600, 450)

    # Calculate ROI dimensions
    roi_width = roi_bottom_right[0] - roi_top_left[0]
    roi_height = roi_bottom_right[1] - roi_top_left[1]
    roi_area = roi_width * roi_height

    # Adjust min_area for the ROI
    full_frame_width = 1920  # Replace with your full frame width
    full_frame_height = 1080  # Replace with your full frame height
    full_frame_area = full_frame_width * full_frame_height

    min_area = (min_area_full_frame / full_frame_area) * roi_area

    # Initialize previous frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the stream.")
        return

    # Resize the current frame to the desired window size
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    # Extract the ROI from the frame
    roi_frame = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    # Convert to grayscale and blur the previous frame
    prev_frame_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

    while True:
        # Read current frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the current frame to the desired window size
        frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

        # Draw the ROI rectangle on the frame
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)

        # Extract the ROI from the frame
        roi_frame = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

        # Convert to grayscale
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Calculate frame difference
        frame_diff = cv2.absdiff(prev_frame_gray, gray_frame)
        _, thresh_frame = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                (x, y, w, h) = cv2.boundingRect(contour)

                # Adjust coordinates to be relative to the full frame
                full_frame_x = x + roi_top_left[0]
                full_frame_y = y + roi_top_left[1]

                # Draw the rectangle on the full frame
                cv2.rectangle(frame, (full_frame_x, full_frame_y), (full_frame_x + w, full_frame_y + h), (0, 255, 0), 2)
                motion_detected = True
                print((full_frame_x, full_frame_y, w, h))

                # x = threading.Thread(target=takePicture, args=())
                # x.start()

        if motion_detected:
            # Do action whatever you want
            # (e.g take recording video, take capture images, etc...)

            cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame
        cv2.imshow("Motion Detection", frame)

        # Update previous frame
        prev_frame_gray = gray_frame

        # Exit on key press (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Motion Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

detectMotion()
