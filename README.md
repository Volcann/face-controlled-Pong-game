````markdown
# Face‑Controlled Pong

A modern, responsive Pong‑style game you control with your face via your webcam. Built with OpenCV and NumPy, it tracks your current score and persists your all‑time high score between sessions.

---

## Features

- **Face‑controlled paddle**: Move the paddle up/down by moving your face in front of the camera.
- **Responsive design**: Scales to any camera resolution.
- **Modern look**: Dark background, center line, anti‑aliased shapes, rounded paddle ends.
- **Score tracking**: Shows current score and all‑time high score.
- **Persistent high score**: Saved to `highscore.txt` automatically.

---

## Prerequisites

- **Python 3.7+**
- A working **webcam**
- **git**, **pip** (or other Python package manager)

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face-pong.git
   cd face-pong
   ```
````

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   # venv\Scripts\activate     # Windows (PowerShell)
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify high‑score file**

   - On first run, `highscore.txt` will be created automatically with `0`.
   - To reset, simply delete or edit `highscore.txt`.

---

## Running the Game

```bash
python face_pong.py
```

- **Quit**: Press `q` in the game window
- **Resize**: Drag window corners (OpenCV’s `WINDOW_NORMAL`)

---

## Project Structure

```
face-pong/
├─ face_pong.py       # Main game script
├─ highscore.txt      # Auto-generated high score storage
├─ requirements.txt   # Pip freeze of your venv
└─ README.md          # This file
```

---

## Configuration & Customization

- **Detection sensitivity**
  – `sensitivity`, `scale_factor`, `min_neighbors`, `downscale` in `face_pong.py`
- **Visual styles**
  – Colors (RGB tuples) and sizing ratios (relative to frame) are easy to tweak
- **Window size**
  – Adjust the initial `cv2.resizeWindow()` dimensions or remove to use native camera size

---

## Troubleshooting

- **No camera detected**
  – Ensure webcam isn’t used by another app and drivers are installed.
- **Slow performance**
  – Increase `downscale` (e.g. to `0.4`) for faster face detection.
- **Cascade load error**
  – Confirm `opencv-python` is installed; cascade is loaded via `cv2.data.haarcascades`.

---

Enjoy controlling Pong with your face—and beat your personal high score! 😄
