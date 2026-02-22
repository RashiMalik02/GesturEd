import cv2
from hand_tracker import HandTracker
from test_tube import TestTube
from litmus_paper import LitmusPaper


def main():
    cap = cv2.VideoCapture(2)
    tracker = HandTracker()
    tube = TestTube(x=350, y=200)
    paper   = LitmusPaper(x=310, y=420, width=90, height=130)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Hand tracking
        frame = tracker.find_hands(frame)
        angle = tracker.get_hand_angle(frame)
        
        # Update tube angle
        tube.set_angle(angle)

        frame = paper.draw(frame)
        
        # Draw tube
        frame = tube.draw(frame)

        
        # Display info
        if tube.is_pouring and tube.liquid_level > 0:
            import math
            angle_rad  = math.radians(tube.display_angle)
            pivot_x    = tube.x + tube.width // 2
            pivot_y    = tube.y
            mouth_off_x = -(tube.width // 2)
            stream_x   = int(pivot_x + mouth_off_x * math.cos(angle_rad))
            stream_y   = int(pivot_y + mouth_off_x * math.sin(angle_rad))
            # Bezier end point (same as in _draw_pouring_effect)
            end_x = stream_x - 45
            end_y = stream_y + 130

            paper.receive_liquid(end_x, end_y + 85, tube.liquid_color)

        if angle is not None:
            cv2.putText(frame, f"Angle: {angle:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if tube.is_pouring:
                cv2.putText(frame, "POURING!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Virtual Chemistry Lab", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()