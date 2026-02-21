import cv2
import numpy as np
import math

class TestTube:
    def __init__(self, x=300, y=300, width=60, height=200):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.liquid_level = 0.7
        self.liquid_color = (200, 200, 255)
        self.display_angle = 0
        self.current_angle = 0
        self.is_pouring = False
        self.MAX_ANGLE = 90

    def set_angle(self, angle):
        if angle is None:
            # When no hand detected, slowly return to 0
            self.current_angle = 0
            self.is_pouring = False
            return
        self.current_angle = min(angle, self.MAX_ANGLE)
        self.is_pouring = self.current_angle > 40

    def draw(self, frame):
        self.display_angle += (self.current_angle - self.display_angle) * 0.1

        # Lower threshold — starts pouring at 25 degrees instead of 40
        if self.display_angle > 25 and self.liquid_level > 0:
            self.is_pouring = True
            self.liquid_level -= 0.0008
            self.liquid_level = max(0, self.liquid_level)
        else:
            self.is_pouring = False

        frame = self._draw_rotated(frame)

        if self.display_angle > 25 and self.liquid_level > 0:
            self._draw_pouring_effect(frame)

        return frame

    def _draw_rotated(self, frame):
        h, w = frame.shape[:2]
        temp = np.zeros((h, w, 3), dtype=np.uint8)

        self._draw_tube_components(temp)

        pivot = (self.x + self.width // 2, self.y)
        rotation_matrix = cv2.getRotationMatrix2D(pivot, self.display_angle, 1.0)

        rotated = cv2.warpAffine(
            temp, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        mask = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(frame_bg, rotated)
        return frame

    def _draw_tube_components(self, frame):
        self._draw_liquid_with_gravity(frame)
        self._fill_rounded_bottom(frame)
        self._draw_glass_outline(frame)

        cv2.ellipse(frame,
                    (self.x + self.width // 2, self.y + self.height),
                    (self.width // 2, 12),
                    0, 0, 180,
                    (100, 100, 100), 3)

    def _draw_liquid_with_gravity(self, frame):
        # Negate angle so surface slopes OPPOSITE to tube tilt (stays horizontal in world)
        angle_rad = math.radians(-self.display_angle)

        tube_left  = self.x + 3
        tube_right = self.x + self.width - 3
        tube_bottom = self.y + self.height
        tube_top    = self.y

        liquid_height = int(self.height * self.liquid_level)
        surface_center_y = tube_bottom - liquid_height
        cx = (tube_left + tube_right) / 2

        # Surface slope — negated so it opposes tube rotation
        surface_slope = math.tan(angle_rad)

        for y in range(tube_bottom, tube_top, -1):
            y_surface_left  = surface_center_y - (tube_left  - cx) * surface_slope
            y_surface_right = surface_center_y - (tube_right - cx) * surface_slope

            left_filled  = y >= y_surface_left
            right_filled = y >= y_surface_right

            if not left_filled and not right_filled:
                continue

            if left_filled and right_filled:
                draw_left  = tube_left
                draw_right = tube_right
            elif left_filled:
                if abs(surface_slope) > 0.001:
                    x_cross    = int(cx - (y - surface_center_y) / surface_slope)
                    draw_left  = tube_left
                    draw_right = min(tube_right, x_cross)
                else:
                    draw_left, draw_right = tube_left, tube_right
            else:
                if abs(surface_slope) > 0.001:
                    x_cross    = int(cx - (y - surface_center_y) / surface_slope)
                    draw_left  = max(tube_left, x_cross)
                    draw_right = tube_right
                else:
                    draw_left, draw_right = tube_left, tube_right

            if draw_right <= draw_left:
                continue

            progress   = (tube_bottom - y) / max(liquid_height, 1)
            brightness = 1 - (abs(0.5 - progress) * 0.4)
            color      = tuple(int(c * brightness) for c in self.liquid_color)
            cv2.line(frame, (draw_left, y), (draw_right, y), color, 1)

        # Draw liquid surface highlight line
        for x in range(tube_left, tube_right):
            y_surface = int(surface_center_y - (x - cx) * surface_slope)
            if tube_top < y_surface < tube_bottom:
                cv2.circle(frame, (x, y_surface), 1, (220, 220, 255), -1)

    def _fill_rounded_bottom(self, frame):
        if self.liquid_level > 0:
            cv2.ellipse(frame,
                        (self.x + self.width // 2, self.y + self.height),
                        (self.width // 2 - 3, 10),
                        0, 0, 180,
                        self.liquid_color, -1)

    def _draw_glass_outline(self, frame):
        cv2.rectangle(frame,
                      (self.x, self.y),
                      (self.x + self.width, self.y + self.height),
                      (80, 80, 80), 3)
        shine_x = self.x + 5
        cv2.line(frame,
                 (shine_x, self.y + 10),
                 (shine_x, self.y + self.height - 20),
                 (180, 180, 180), 2)

    def _draw_pouring_effect(self, frame):
        import time
        angle_rad = math.radians(self.display_angle)
        pivot_x = self.x + self.width // 2
        pivot_y = self.y

        # Mouth = left edge of tube opening after rotation
        mouth_offset_x = -(self.width // 2)
        mouth_offset_y = 0

        stream_x = int(pivot_x
                    + mouth_offset_x * math.cos(angle_rad)
                    - mouth_offset_y * math.sin(angle_rad))
        stream_y = int(pivot_y
                    + mouth_offset_x * math.sin(angle_rad)
                    + mouth_offset_y * math.cos(angle_rad))

        # ---- 3D STREAM using bezier curve with gravity ----
        # Control point curves the stream downward naturally
        ctrl_x = stream_x - 20
        ctrl_y = stream_y + 50
        end_x  = stream_x - 45
        end_y  = stream_y + 130

        # Draw stream as thick bezier with 3D shading (light center, dark edges)
        steps = 30
        prev_x, prev_y = stream_x, stream_y

        for i in range(1, steps + 1):
            t = i / steps
            # Quadratic bezier
            bx = int((1-t)**2 * stream_x + 2*(1-t)*t * ctrl_x + t**2 * end_x)
            by = int((1-t)**2 * stream_y + 2*(1-t)*t * ctrl_y + t**2 * end_y)

            # Stream gets narrower as it falls (tapers)
            width = max(1, int(6 * (1 - t * 0.6)))

            # 3D shading: draw dark outer, then bright core
            dark_color  = tuple(int(c * 0.6) for c in self.liquid_color)
            bright_color = tuple(min(255, int(c * 1.2)) for c in self.liquid_color)

            cv2.line(frame, (prev_x, prev_y), (bx, by), dark_color, width + 2)
            cv2.line(frame, (prev_x, prev_y), (bx, by), self.liquid_color, width)
            cv2.line(frame, (prev_x, prev_y), (bx, by), bright_color, max(1, width - 2))

            prev_x, prev_y = bx, by

        # ---- ANIMATED DROPS with 3D sphere shading ----
        t_now = time.time()
        for i in range(5):
            # Each drop has a phase offset so they fall at different times
            phase = (t_now * 2 + i * 0.4) % 1.0

            drop_x = end_x + int(math.sin(phase * math.pi) * 5) - i * 3
            drop_y = end_y + int(phase * 80)
            radius = max(2, 7 - i)

            # 3D sphere effect: dark base, liquid color mid, bright highlight
            dark_color   = tuple(int(c * 0.5) for c in self.liquid_color)
            bright_color = tuple(min(255, int(c * 1.3)) for c in self.liquid_color)

            cv2.circle(frame, (drop_x, drop_y), radius, dark_color, -1)
            cv2.circle(frame, (drop_x, drop_y), max(1, radius - 1), self.liquid_color, -1)
            # Highlight dot (top-left of sphere = light source)
            cv2.circle(frame, (drop_x - radius//3, drop_y - radius//3),
                    max(1, radius // 3), bright_color, -1)

        # ---- SPLASH at the bottom ----
        splash_y = end_y + 85
        for i in range(6):
            splash_angle = math.radians(180 + i * 30)
            sx = int(end_x + math.cos(splash_angle) * (8 + i * 2))
            sy = int(splash_y + math.sin(splash_angle) * 4)
            cv2.circle(frame, (sx, sy), 2, self.liquid_color, -1)