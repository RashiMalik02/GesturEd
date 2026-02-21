# reactions/opencv_handler.py
# This file is inside your `reactions` app folder. REPLACE the existing reactions/opencv_handler.py with this.

import cv2
from .cv_modules import HandTracker, VirtualLab


def generate_frames(reaction_type):
    """
    Generator that captures webcam frames, runs them through Person 1's
    CV modules, and yields processed JPEG frames as a multipart stream.
    Camera is always released, even if the client disconnects mid-stream.
    """
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return

    # Initialize Person 1's modules ONCE per stream session
    tracker = HandTracker()
    lab = VirtualLab()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            # 1. Get hand tracking data
            hand_pos = tracker.get_hand_position(frame)

            # 2. Draw test tube, detect collision, and change litmus color
            # The lab module handles the collision and color change internally
            frame = lab.draw_elements(frame, hand_pos, reaction_type)

            # 3. Encode and yield the processed frame
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        camera.release()