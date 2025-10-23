import sys
from pathlib import Path
import cv2

def main():
    # Replace this URL with your camera's RTSP/HTTP address
    cam_url = "rtsp://192.168.136.100:554/live/0"
    cap = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        print(f"Error: Unable to connect to camera at {cam_url}")
        sys.exit(1)

    print(f"Connected to {cam_url}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame")
            break

        cv2.imshow("IP Camera Feed", frame)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

