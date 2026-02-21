# backend/reactions/cv_modules.py
# Place in: backend/reactions/cv_modules.py — REPLACE existing file entirely.
import cv2
import math
import numpy as np
import mediapipe as mp


class HandTracker:
    """Detects hand, draws landmarks, returns tilt angle + wrist pixel coords."""

    def __init__(self, mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_hand_position(self, frame):
        """
        Returns {"angle": float, "x": int, "y": int} or None if no hand found.
        - angle: tilt of hand (0–90°), used to determine pouring
        - x, y:  wrist pixel coords, used to anchor the test tube
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results or not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        wrist    = hand.landmark[0]
        fingertip = hand.landmark[12]

        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)

        dy = (wrist.y * h) - (fingertip.y * h)
        dx = (fingertip.x * w) - (wrist.x * w)
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle

        return {"angle": angle, "x": wrist_x, "y": wrist_y}


class TestTube:
    """Test tube that follows the hand and reports where the liquid stream lands."""

    def __init__(self, width=50, height=180):
        self.width = width
        self.height = height
        self.liquid_level = 0.7
        self.current_angle = 0
        self.is_pouring = False

    def set_angle(self, angle):
        self.current_angle = angle if angle is not None else 0
        self.is_pouring = angle is not None and angle > 34

    def draw(self, frame, hand_x, hand_y, liquid_color):
        """
        Draws tube anchored at (hand_x, hand_y).
        Returns (is_pouring, stream_end_x, stream_end_y).
        stream_end coords are only meaningful when is_pouring is True.
        """
        x, y = hand_x - self.width // 2, hand_y - self.height // 2

        liquid_height = int(self.height * self.liquid_level)
        liquid_y = y + self.height - liquid_height

        # Liquid fill
        cv2.rectangle(frame,
                      (x + 3, liquid_y),
                      (x + self.width - 3, y + self.height - 5),
                      liquid_color, -1)

        # Glass outline
        cv2.rectangle(frame,
                      (x, y),
                      (x + self.width, y + self.height),
                      (100, 100, 100), 3)

        # Rounded bottom
        cv2.ellipse(frame,
                    (x + self.width // 2, y + self.height),
                    (self.width // 2, 12),
                    0, 0, 180,
                    (100, 100, 100), 3)

        stream_end_x, stream_end_y = x + self.width // 2, y + self.height

        if self.is_pouring:
            stream_end_x, stream_end_y = self._draw_pouring_effect(frame, x, y, liquid_color)

        return self.is_pouring, stream_end_x, stream_end_y

    def _draw_pouring_effect(self, frame, x, y, liquid_color):
        """Draws a downward stream from the tube mouth; returns landing coords."""
        stream_start_x = x + self.width // 2
        stream_start_y = y + self.height

        # Stream falls straight down
        stream_end_x = stream_start_x
        stream_end_y = stream_start_y + 120

        cv2.line(frame,
                 (stream_start_x, stream_start_y),
                 (stream_end_x, stream_end_y),
                 liquid_color, 5)

        cv2.circle(frame, (stream_end_x, stream_end_y), 8, liquid_color, -1)

        return stream_end_x, stream_end_y


class VirtualLab:
    """Manages litmus paper state and collision detection."""

    # Litmus paper fixed at bottom-left of frame
    PAPER_X, PAPER_Y, PAPER_W, PAPER_H = 20, 350, 80, 100

    def __init__(self):
        self.test_tube = TestTube()
        self.reaction_triggered = False

    def draw_elements(self, frame, hand_pos, reaction_type):
        angle = hand_pos["angle"] if hand_pos else None
        hand_x = hand_pos["x"] if hand_pos else frame.shape[1] // 2
        hand_y = hand_pos["y"] if hand_pos else frame.shape[0] // 2

        self.test_tube.set_angle(angle)

        # BGR color pairs: (liquid_color, paper_initial, paper_final)
        if reaction_type == "red_litmus":
            # Base: red paper turns blue
            liquid_color     = (220, 80,  40)   # blue liquid (BGR)
            initial_color    = (40,  40,  220)   # red paper
            triggered_color  = (220, 80,  40)    # blue paper
        else:
            # Acid: blue paper turns red
            liquid_color     = (40,  40,  220)   # red liquid (BGR)
            initial_color    = (220, 80,  40)    # blue paper
            triggered_color  = (40,  40,  220)   # red paper

        # Draw test tube
        is_pouring, stream_end_x, stream_end_y = self.test_tube.draw(
            frame, hand_x, hand_y, liquid_color
        )

        # Collision detection
        px, py, pw, ph = self.PAPER_X, self.PAPER_Y, self.PAPER_W, self.PAPER_H
        if is_pouring and not self.reaction_triggered:
            if px <= stream_end_x <= px + pw and py <= stream_end_y <= py + ph:
                self.reaction_triggered = True

        # Draw litmus paper
        paper_color = triggered_color if self.reaction_triggered else initial_color
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), paper_color, -1)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (200, 200, 200), 2)

        label = "Red Litmus" if reaction_type == "red_litmus" else "Blue Litmus"
        cv2.putText(frame, label, (px, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if self.reaction_triggered:
            cv2.putText(frame, "Reaction Complete!", (px - 10, py + ph + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return frame