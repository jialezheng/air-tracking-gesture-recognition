import cv2
import pyautogui
import time
from hand_tracker import HandTracker

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Screen and camera
screen_w, screen_h = pyautogui.size()
cam_w, cam_h = 640, 480

# Smoothing
smoothing = 0.3
prev_x, prev_y = 0, 0

# Active zone
frame_margin = 100

# Click cooldown — prevents spamming clicks every frame
click_cooldown = 0.5  # seconds between clicks
last_click_time = 0

# Fist quit — must hold fist for 2 seconds to quit
fist_start_time = None
FIST_HOLD_DURATION = 2.0

# Scroll tracking
scroll_start_y = None

tracker = HandTracker()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)


def get_distance(pos1, pos2):
    """Distance between two landmarks. pos format: (id, x, y)"""
    return ((pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2) ** 0.5


def map_to_screen(index_x, index_y):
    """Map finger position from camera active zone to screen coordinates."""
    index_x = max(frame_margin, min(index_x, cam_w - frame_margin))
    index_y = max(frame_margin, min(index_y, cam_h - frame_margin))

    screen_x = int((index_x - frame_margin) / (cam_w - 2 * frame_margin) * screen_w)
    screen_y = int((index_y - frame_margin) / (cam_h - 2 * frame_margin) * screen_h)
    return screen_x, screen_y


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = tracker.find_hands(frame)
    positions = tracker.get_positions(frame)

    mode = "NO HAND"
    mode_color = (150, 150, 150)

    if positions:
        fingers = tracker.fingers_up(positions)
        now = time.time()

        # --- QUIT: Fist (all fingers down) held for 2 seconds ---
        if not any(fingers):
            if fist_start_time is None:
                fist_start_time = now
            held = now - fist_start_time
            remaining = FIST_HOLD_DURATION - held
            if remaining <= 0:
                mode = "QUITTING..."
                mode_color = (0, 0, 255)
                cv2.putText(frame, mode, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
                cv2.imshow("Air Mouse v2", frame)
                cv2.waitKey(500)
                break
            else:
                mode = f"FIST - quit in {remaining:.1f}s"
                mode_color = (0, 100, 255)
        else:
            fist_start_time = None  # reset if any finger goes up

            # --- MOVE: Only index finger up ---
            if fingers[1] and not fingers[2] and not fingers[0]:
                target_x, target_y = map_to_screen(positions[8][1], positions[8][2])

                curr_x = prev_x + (target_x - prev_x) * smoothing
                curr_y = prev_y + (target_y - prev_y) * smoothing

                pyautogui.moveTo(int(curr_x), int(curr_y))
                prev_x, prev_y = curr_x, curr_y

                mode = "MOVE"
                mode_color = (0, 255, 0)

            # --- LEFT CLICK: Pinch thumb + index ---
            elif fingers[0] and fingers[1]:
                thumb_tip = positions[4]
                index_tip = positions[8]
                distance = get_distance(thumb_tip, index_tip)

                if distance < 40 and (now - last_click_time) > click_cooldown:
                    pyautogui.click()
                    last_click_time = now
                    mode = "LEFT CLICK!"
                    mode_color = (0, 0, 255)
                elif distance < 40:
                    mode = "CLICK (cooldown)"
                    mode_color = (100, 100, 255)
                else:
                    mode = "PINCH TO CLICK"
                    mode_color = (0, 255, 255)

            # --- RIGHT CLICK: Index + middle up, pinch them together ---
            elif fingers[1] and fingers[2] and not fingers[3]:
                index_tip = positions[8]
                middle_tip = positions[12]
                distance = get_distance(index_tip, middle_tip)

                if distance < 40 and (now - last_click_time) > click_cooldown:
                    pyautogui.rightClick()
                    last_click_time = now
                    mode = "RIGHT CLICK!"
                    mode_color = (255, 0, 0)
                elif distance < 40:
                    mode = "R-CLICK (cooldown)"
                    mode_color = (255, 100, 100)
                else:
                    mode = "PINCH FOR R-CLICK"
                    mode_color = (255, 150, 0)

            # --- SCROLL: Open palm (all fingers up) ---
            elif all(fingers):
                wrist_y = positions[0][2]
                if scroll_start_y is None:
                    scroll_start_y = wrist_y

                diff = scroll_start_y - wrist_y
                if abs(diff) > 20:
                    pyautogui.scroll(int(diff / 10))
                    scroll_start_y = wrist_y

                mode = "SCROLL"
                mode_color = (255, 0, 255)

            else:
                scroll_start_y = None
    else:
        fist_start_time = None
        scroll_start_y = None

    # --- Draw UI ---
    # Active zone
    cv2.rectangle(frame, (frame_margin, frame_margin),
                  (cam_w - frame_margin, cam_h - frame_margin), (255, 0, 255), 2)

    # Mode display
    cv2.putText(frame, mode, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

    # Gesture guide at bottom
    cv2.putText(frame, "Point=Move | Pinch=Click | 2-Pinch=RClick | Palm=Scroll | Fist=Quit",
                (10, cam_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Air Mouse v2", frame)

    # ESC key to quit (key code 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()