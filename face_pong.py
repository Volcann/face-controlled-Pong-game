import cv2
import numpy as np
import os

# === High‐score persistence ===
HS_FILE = "highscore.txt"

def load_high_score():
    if not os.path.exists(HS_FILE):
        return 0
    try:
        with open(HS_FILE, "r") as f:
            return int(f.read().strip())
    except:
        return 0

def save_high_score(hs):
    with open(HS_FILE, "w") as f:
        f.write(str(hs))

high_score = load_high_score()

# === Initialize webcam & window ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
win_name = "Face Pong"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 800, 600)

# create a sensitivity trackbar (scaled by 10)
cv2.createTrackbar("Sensitivity×0.1", win_name, 18, 100, lambda x: None)

# === Load face detector ===
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    print("ERROR loading cascade")
    exit(1)

# === Other settings ===
scale_factor  = 1.1
min_neighbors = 3
downscale     = 0.5

# === Game state ===
score = 0

def reset_ball(frame_w, frame_h):
    pos = np.array([frame_w/2, frame_h/2], dtype=float)
    vel = np.array([4, 4], dtype=float)
    return pos, vel

# prime first frame
ret, frame = cap.read()
h, w = frame.shape[:2]
ball_pos, ball_vel = reset_ball(w, h)

paddle_w = int(w * 0.03)
paddle_h = int(h * 0.15)
paddle_x = int(w * 0.05)
paddle_y = h // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # read sensitivity from trackbar
    raw = cv2.getTrackbarPos("Sensitivity×0.1", win_name)
    sensitivity = max(raw, 1) / 10.0  # avoid zero

    # draw center line
    cv2.line(frame, (w//2, 0), (w//2, h), (60, 60, 60), 2, cv2.LINE_AA)

    # face detect on real frame
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0,0), fx=downscale, fy=downscale)
    faces = face_cascade.detectMultiScale(
        small, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )

    if len(faces) > 0:
        x, y, fw, fh = faces[0]
        center_y = int((y + fh/2) / downscale)
        delta    = center_y - paddle_y
        # amplify movement
        paddle_y += int(sensitivity * delta)

    # clamp paddle
    paddle_y = max(paddle_h//2, min(paddle_y, h - paddle_h//2))

    # draw paddle
    top    = (paddle_x, paddle_y - paddle_h//2)
    bottom = (paddle_x, paddle_y + paddle_h//2)
    cv2.ellipse(frame, top,    (paddle_w//2, paddle_w//2), 0, 180, 360, (200,50,50), -1, cv2.LINE_AA)
    cv2.ellipse(frame, bottom, (paddle_w//2, paddle_w//2), 0,   0, 180, (200,50,50), -1, cv2.LINE_AA)
    cv2.rectangle(frame,
                  (paddle_x - paddle_w//2, paddle_y - paddle_h//2),
                  (paddle_x + paddle_w//2, paddle_y + paddle_h//2),
                  (200,50,50), -1, cv2.LINE_AA)

    # move ball
    ball_pos += ball_vel
    if ball_pos[1] <= 0 or ball_pos[1] >= h:
        ball_vel[1] *= -1
    if ball_pos[0] >= w:
        ball_vel[0] *= -1

    # collision
    if (paddle_x - paddle_w//2 <= ball_pos[0] <= paddle_x + paddle_w//2 and
        paddle_y - paddle_h//2 <= ball_pos[1] <= paddle_y + paddle_h//2):
        ball_vel[0] *= -1
        score += 1
        if score > high_score:
            high_score = score
            save_high_score(high_score)

    # reset if missed
    if ball_pos[0] < 0:
        ball_pos, ball_vel = reset_ball(w, h)
        score = 0

    # draw ball
    cv2.circle(frame, tuple(ball_pos.astype(int)), int(w*0.015), (50,200,50), -1, cv2.LINE_AA)

    # overlay scores
    cv2.putText(frame, f"Score: {score}",      (w-200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240,240,240), 2, cv2.LINE_AA)
    cv2.putText(frame, f"High Score: {high_score}", (w-300, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Sens: {sensitivity:.1f}", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
