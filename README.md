# Fitness Posture Tracker (MediaPipe + OpenCV)

A real-time fitness posture tracking app in Python using MediaPipe Pose and OpenCV. It detects 33 landmarks, overlays a skeleton, validates posture for 10 exercises, counts reps, and provides real-time feedback. Progress can be logged to SQLite.

## Features

- Real-time webcam-based pose tracking (33 landmarks)
- Skeleton overlay and problematic joints highlighted in red
- 10 exercises with posture rules and automatic rep counting:
  1. Squats
  2. Push-ups
  3. Plank (hold-based; reps visualize time in good form)
  4. Lunges
  5. Bicep curls
  6. Shoulder press
  7. Side lateral raises
  8. Deadlifts
  9. Jumping jacks
  10. Mountain climbers
- Real-time feedback messages (e.g., "Straighten your back", "Lower deeper", "Good rep!")
- Heads-up display with exercise name, rep count, accuracy %, and FPS
- Optional SQLite logging of workout summary (date, exercise, total reps, average accuracy)

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --list
python main.py --exercise squats --save-db --mirror
```

Keys while running:
- q or Esc: Quit
- r: Reset rep count
- e: Switch to next exercise

## Example: Joint Angle Calculation

We use the angle ABC with B as the vertex, computed from 2D pixel coordinates:

```python
from pose_utils import calculate_angle

# A, B, C are (x, y) pixels
angle = calculate_angle(A, B, C)
print(f"Angle at B is {angle:.1f} degrees")
```

This angle feeds the exercise rules. For squats, we roughly track:
- Knee angle approaching ~90° at the bottom
- Hips at/below knee height at the bottom (heuristic)
- Back straightness via shoulder-hip-knee angle

Pseudo-logic excerpt (see `exercise_rules.py`):

```python
# knee_angle < 95 -> "down" stage
# knee_angle > 160 after being down -> count 1 rep

if knee_angle < 95:
    stage = "down"
if stage == "down" and knee_angle > 160:
    stage = "up"
    reps += 1  # Good rep!
```

## Notes and Tips

- Camera placement: side view works better for push-ups, curls, deadlifts; front view helps for jumping jacks.
- The app uses simple heuristics and angle thresholds; performance may vary by camera angle, distance, and lighting.
- If landmarks are missing or visibility is low, form checks may be skipped that frame.
- Accuracy score is a simple 0–100 heuristic aggregating relevant angles.

## Extending

- Add new exercises by subclassing `ExerciseBase` in `exercise_rules.py` and implementing `update(...)`.
- Add audio feedback by integrating a TTS library (e.g., `pyttsx3`) or beeps.
- Persist per-set or per-rep analytics by expanding `ProgressDB`.

## Troubleshooting

- If you see "Move into frame", ensure the entire body is within view.
- On high CPU usage, lower camera resolution (e.g., `--width 960 --height 540`) or set `model_complexity=0` in `main.py`.