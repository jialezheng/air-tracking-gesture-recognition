import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        # Fingertip landmark indices
        self.tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    def find_hands(self, frame):
        """Detect hands and return landmarks. Also draws skeleton on frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        
        return frame

    def get_positions(self, frame, hand_index=0):
        """Return list of (id, x_pixel, y_pixel) for each landmark."""
        positions = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = frame.shape
            for id, lm in enumerate(hand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                positions.append((id, px, py))
        return positions

    def fingers_up(self, positions):
        """Return list of 5 booleans: which fingers are up [thumb, index, middle, ring, pinky]."""
        if len(positions) == 0:
            return [False] * 5

        fingers = []

        # Thumb: compare tip x vs IP joint x (horizontal check)
        # This works for right hand — tip is to the RIGHT of IP when open
        if positions[4][1] > positions[3][1]:
            fingers.append(True)
        else:
            fingers.append(False)

        # Other 4 fingers: tip is ABOVE (lower y) the PIP joint when open
        for tip_id in [8, 12, 16, 20]:
            if positions[tip_id][2] < positions[tip_id - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)

        return fingers