import cv2

# --- 1) Try opening your camera (change 0 → 1,2… if needed) ---
cam_index = 1 
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera at index {cam_index}")
    exit(1)

# --- 2) Grab the first frame safely ---
ret, prev = cap.read()
if not ret or prev is None:
    print("ERROR: Failed to read first frame")
    cap.release()
    exit(1)

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# --- 3) Processing loop (frame-difference example) ---
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("ERROR: Lost camera connection")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Frame-difference
    diff = cv2.absdiff(gray, prev_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_pixels = cv2.countNonZero(thresh)
    running = motion_pixels > 5000

    # Simple direction: compare left/right halves
    h, w = thresh.shape
    left  = cv2.countNonZero(thresh[:, :w//2])
    right = cv2.countNonZero(thresh[:, w//2:])
    direction = "CW" if right > left else "CCW"

    print(f"{'Running' if running else 'Stopped'}, {direction}")

    prev_gray = gray

# --- 4) Clean up ---
cap.release()

