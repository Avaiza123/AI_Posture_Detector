import shutil
import sqlite3
from exercise_rules import list_exercises
from main_fitness_app import build_exercise_by_key, RobustExerciseClassifier
from pose_utils import get_pose_landmarks, mp_pose
import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import tempfile
import time
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

# --- Global Initialization ---
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

app = FastAPI(title="Enhanced Fitness Tracker API")
classifier = RobustExerciseClassifier()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session Management for Rep Counting ---
class ExerciseSession:
    def __init__(self, exercise_key: str):
        self.exercise_key = exercise_key
        self.exercise = build_exercise_by_key(exercise_key)
        self.start_time = time.time()
        self.frame_count = 0
        self.total_reps = 0
        self.last_rep_time = 0
        self.rep_accuracies = [] # List of (accuracy, timestamp) for each completed rep
        self.session_accuracies = [] # List of frame-level accuracies
        self.phase_history = []
        self.last_phase = None
        self.consecutive_frames_same_phase = 0
        self.last_frame_timestamp = 0
        self.feedback_history = []
        # New for tracking individual rep details more precisely
        self.rep_details: List[Dict[str, Any]] = []


    def _get_current_phase(self, landmarks_px, visibility_scores) -> str:
        """Determines the current phase of the exercise."""
        try:
            if hasattr(self.exercise, 'posture_state'):
                return self.exercise.posture_state.current_state
            elif hasattr(self.exercise, 'current_state'):
                return self.exercise.current_state
            elif hasattr(self.exercise, 'phase'):
                return self.exercise.phase
            else:
                # Fallback: Try to determine phase from angles (e.g., for squats)
                if hasattr(self.exercise, '_calculate_current_angles'):
                    angles = self.exercise._calculate_current_angles(landmarks_px, visibility_scores)
                    if angles and 'knee_angle' in angles and angles['knee_angle']:
                        knee_angle = angles['knee_angle']
                        if knee_angle < 90:
                            return "down"
                        elif knee_angle > 160:
                            return "up"
                        else:
                            return "active"
        except:
            pass # Ignore calculation errors for phase
        return "ready"

    def update(self, landmarks_px, visibility_scores, frame_shape, timestamp: float):
        """Update exercise state and return current analysis"""
        self.frame_count += 1
        self.last_frame_timestamp = timestamp

        # Update the exercise with current frame
        rep_inc, feedback_text, highlight_ids, form_accuracy = self.exercise.update(
            landmarks_px,
            visibility_scores,
            frame_shape,
            timestamp
        )

        rep_just_completed = False

        # --- 1. Rep Counting and Accuracy Tracking ---
        if rep_inc > 0:
            self.total_reps += rep_inc
            self.last_rep_time = timestamp
            rep_just_completed = True
            print(f"🎉 REP COUNTED! Total: {self.total_reps}")

            # Store accuracy for this completed rep
            rep_accuracy = 0.0
            if hasattr(self.exercise, 'rep_accuracies') and self.exercise.rep_accuracies:
                rep_accuracy = self.exercise.rep_accuracies[-1]
                # self.rep_accuracies stores accuracy, not (accuracy, timestamp) anymore for simplicity
                self.rep_accuracies.append(rep_accuracy)

                # Store detailed rep information
            self.rep_details.append({
                "number": self.total_reps,
                "accuracy": rep_accuracy,
                "timestamp": timestamp
            })

        # --- 2. Phase Tracking ---
        current_phase = self._get_current_phase(landmarks_px, visibility_scores)

        self.phase_history.append(current_phase)
        if len(self.phase_history) > 20: # Keep last 20 phases
            self.phase_history.pop(0)

        # --- 3. Session Accuracy Calculation ---
        if form_accuracy and form_accuracy > 0:
            self.session_accuracies.append(form_accuracy)
            if len(self.session_accuracies) > 100:
                self.session_accuracies.pop(0)

        # --- 4. Generate Feedback ---
        if isinstance(feedback_text, list):
            feedback_messages = feedback_text
        elif feedback_text:
            feedback_messages = [feedback_text]
        else:
            feedback_messages = ["Keep going!"]

        # Add rep completion message if needed
        if rep_just_completed:
            rep_accuracy = self.rep_details[-1]['accuracy'] if self.rep_details else form_accuracy
            feedback_messages.insert(0, f"🎉 REP {self.total_reps} completed! ({rep_accuracy:.1f}% accuracy)")

        return {
            "reps": self.total_reps,
            "phase": current_phase,
            "form_accuracy": form_accuracy,
            "feedback": feedback_messages,
            "rep_just_completed": rep_just_completed,
            "highlight_ids": highlight_ids,
            "exercise_instance": self.exercise
        }

# Global session storage
sessions: Dict[str, ExerciseSession] = {}

def get_session(exercise_key: str, session_id: Optional[str] = None) -> ExerciseSession:
    """Get or create a session for exercise tracking"""
    session_key = session_id or exercise_key

    if session_key not in sessions:
        print(f"🆕 Creating new session for: {exercise_key}")
        sessions[session_key] = ExerciseSession(exercise_key)

    # Clean old sessions (older than 30 minutes)
    current_time = time.time()
    expired_keys = [
        key for key, session in sessions.items()
        if current_time - session.start_time > 1800 # 30 minutes
    ]
    for key in expired_keys:
        del sessions[key]
        print(f"🗑️ Removed expired session: {key}")

    return sessions[session_key]

# --- Database Functions ---
def get_db_connection():
    try:
        conn = sqlite3.connect('workouts.sqlite')
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"❌ Database connection failed: {e}", file=sys.stderr)
        # Initialize DB structure if it fails because table doesn't exist
        try:
            conn = sqlite3.connect('workouts.sqlite')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workout_sessions (
                    id INTEGER PRIMARY KEY,
                    exercise_id TEXT NOT NULL,
                    exercise_name TEXT NOT NULL,
                    total_reps INTEGER NOT NULL,
                    avg_accuracy REAL NOT NULL,
                    duration REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workout_reps (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER NOT NULL,
                    rep_number INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES workout_sessions(id)
                )
            ''')
            conn.commit()
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as init_e:
            print(f"❌ Database initialization failed: {init_e}", file=sys.stderr)
            raise Exception(f"Database connection and initialization failed: {e} / {init_e}")


def save_workout_session(session_data: Dict[str, Any]):
    """Save workout session to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO workout_sessions
            (exercise_id, exercise_name, total_reps, avg_accuracy, duration, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_data.get('exercise_id'),
            session_data.get('exercise_name'),
            session_data.get('total_reps', 0),
            session_data.get('avg_accuracy', 0.0),
            session_data.get('duration', 0),
            datetime.now().isoformat()
        ))

        session_id = cursor.lastrowid

        # Save individual reps
        reps = session_data.get('reps', [])
        for rep in reps:
            cursor.execute('''
                INSERT INTO workout_reps
                (session_id, rep_number, accuracy, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                session_id,
                rep.get('number'),
                rep.get('accuracy', 0.0),
                rep.get('timestamp')
            ))

        conn.commit()
        conn.close()
        print(f"💾 Saved workout session {session_id} to database")
        return session_id
    except Exception as e:
        print(f"❌ Error saving workout: {e}")
        traceback.print_exc()
        return None

# --- Utility Functions ---
# ... (extract_xy, convert_landmarks_to_tuples remain unchanged) ...
def extract_xy(point):
    """Extract x, y coordinates from any format"""
    if isinstance(point, (tuple, list)):
        return float(point[0]), float(point[1])
    elif isinstance(point, dict):
        return float(point.get('x', 0.0)), float(point.get('y', 0.0))
    elif hasattr(point, 'x') and hasattr(point, 'y'):
        return float(point.x), float(point.y)
    raise ValueError(f"Cannot extract x, y from {type(point)}")

def convert_landmarks_to_tuples(results, frame_shape):
    """Convert MediaPipe landmarks to pixel coordinate tuples"""
    h, w, _ = frame_shape
    landmarks_px = []
    visibility_scores = []

    if not results.pose_landmarks:
        return [], []

    for landmark in results.pose_landmarks.landmark:
        x_pixel = int(landmark.x * w)
        y_pixel = int(landmark.y * h)
        landmarks_px.append((x_pixel, y_pixel))
        visibility_scores.append(landmark.visibility)

    return landmarks_px, visibility_scores

# --- API Endpoints ---
# ... (root, get_exercises remain unchanged) ...
@app.get("/")
def root():
    return {
        "status": "Backend Running",
        "timestamp": time.time(),
        "active_sessions": len(sessions),
        "available_exercises": list(list_exercises().keys())
    }

@app.get("/exercises")
async def get_exercises():
    try:
        exercises = list_exercises()
        exercise_list = []
        for key, data in exercises.items():
            exercise_list.append({
                "id": key,
                "name": data["name"],
                "description": f"AI-guided {data['name'].lower()}"
            })
        return {
            "success": True,
            "exercises": exercise_list
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/analyze")
async def analyze_frame(data: dict):
    # This endpoint handles the LIVE rep counting and accuracy updates
    try:
        print("\n" + "="*60)
        print("🎯 Analyze endpoint called")
        print("="*60)

        exercise_id = data.get("exercise_id", "squats")
        exercise_name = data.get("exercise_name") or exercise_id.title()
        frame_data = data.get("frame")
        session_id = data.get("session_id") # Optional session ID for tracking

        if not frame_data:
            raise HTTPException(
                status_code=400,
                detail="Missing frame data."
            )

        # --- 1. Decode Frame Data ---
        try:
            if "base64," in frame_data:
                frame_data = frame_data.split("base64,")[1]

            frame_bytes = base64.b64decode(frame_data)
            print(f"📦 Decoded {len(frame_bytes)} bytes")

            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image data after imdecode.")

            print(f"🖼️ Image decoded: {frame.shape}")

        except Exception as e:
            print(f"❌ Frame decoding error: {e}")
            return {"success": False, "error": f"Frame decoding failed: {str(e)}"}

        # --- 2. Pose Detection ---
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print("✅ Image converted to RGB")

            print("🔍 Detecting pose...")
            results = pose_detector.process(frame_rgb)

            if results.pose_landmarks is None or len(results.pose_landmarks.landmark) < 1:
                print("⚠️ No pose detected")
                return {
                    "success": True,
                    "pose_detected": False,
                    "exercise_id": exercise_id,
                    "exercise_name": exercise_name,
                    "message": "No person detected. Please ensure you're visible in frame.",
                    "feedback": ["Stand in clear view", "Ensure good lighting"],
                    "rep_count": 0,
                    "reps": 0
                }

            # --- 3. Landmark Conversion ---
            landmarks_px, visibility_scores = convert_landmarks_to_tuples(
                results,
                frame_rgb.shape
            )
            print(f"✅ Converted {len(landmarks_px)} landmarks")

        except Exception as e:
            print(f"❌ Pose detection/conversion error: {e}")
            traceback.print_exc()
            return {"success": False, "error": f"Pose processing failed: {str(e)}"}

        # --- 4. Get Session and Update Exercise (LIVE COUNTING) ---
        try:
            # Get or create session for rep counting
            session = get_session(exercise_id, session_id)
            print(f"📊 Using session: {exercise_id} (frame: {session.frame_count + 1})")

            # Update session with current frame
            timestamp = time.time()
            analysis = session.update(
                landmarks_px,
                visibility_scores,
                frame_rgb.shape,
                timestamp
            )

            exercise_instance = analysis["exercise_instance"]
            form_accuracy = analysis["form_accuracy"]
            feedback_messages = analysis["feedback"]
            rep_just_completed = analysis["rep_just_completed"]

            # Calculate overall session accuracy
            session_accuracy = 0.0
            if session.session_accuracies:
                session_accuracy = sum(session.session_accuracies) / len(session.session_accuracies)

            # Get current angles for detailed feedback
            current_angles = {}
            try:
                if hasattr(exercise_instance, '_calculate_current_angles'):
                    current_angles = exercise_instance._calculate_current_angles(landmarks_px, visibility_scores)
            except:
                pass

            # Enhanced feedback based on angles
            enhanced_feedback = feedback_messages
            if current_angles:
                # Keep original enhanced feedback logic for client display
                if 'knee_angle' in current_angles and current_angles['knee_angle']:
                    knee_angle = current_angles['knee_angle']
                    if knee_angle < 90:
                        enhanced_feedback.append(f"Knee angle: {knee_angle:.1f}° (down position)")
                    elif knee_angle > 160:
                        enhanced_feedback.append(f"Knee angle: {knee_angle:.1f}° (standing)")
                    else:
                        enhanced_feedback.append(f"Knee angle: {knee_angle:.1f}°")

            # Prepare normalized landmarks for Flutter
            normalized_landmarks = []
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    normalized_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': visibility_scores[i]
                    })

            print(f"✅ Exercise analysis successful!")
            print(f" ✓ Reps: {session.total_reps}")
            print(f" ✓ Phase: {analysis['phase']}")
            print(f" ✓ Form Accuracy: {form_accuracy}%")
            print(f" ✓ Session Accuracy: {session_accuracy:.1f}%")
            if rep_just_completed:
                print(f" 🎉 REP {session.total_reps} JUST COMPLETED!")

            print(f"✅ ANALYSIS COMPLETE")
            print("="*60 + "\n")

            # --- 5. Return Results ---
            return {
                "success": True,
                "pose_detected": True,
                "exercise_id": exercise_id,
                "exercise_name": exercise_name,
                "phase": analysis['phase'],
                "form_accuracy": form_accuracy,
                "overall_accuracy": session_accuracy,
                "feedback": enhanced_feedback[:5], # Limit to 5 messages
                "rep_count": session.total_reps,
                "reps": session.total_reps,
                "landmarks": normalized_landmarks,
                "rep_just_completed": rep_just_completed,
                "session_id": session_id or exercise_id,
                "session_stats": {
                    "total_frames": session.frame_count,
                    "session_duration": timestamp - session.start_time,
                    "average_accuracy": session_accuracy,
                    "rep_accuracies": session.rep_accuracies[-5:] if session.rep_accuracies else [] # Last 5 reps
                },
                "message": "Analysis successful"
            }

        except Exception as e:
            print(f"⚠️ Exercise analysis failed: {e}")
            traceback.print_exc()

            # Still return pose data even if analysis fails
            normalized_landmarks = []
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    normalized_landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': visibility_scores[i]
                    })

            return {
                "success": True,
                "pose_detected": True,
                "exercise_id": exercise_id,
                "exercise_name": exercise_name,
                "phase": "active",
                "form_accuracy": 70.0,
                "overall_accuracy": 70.0,
                "feedback": ["Pose detected!", "Analysis in progress"],
                "rep_count": 0,
                "reps": 0,
                "landmarks": normalized_landmarks,
                "rep_just_completed": False,
                "message": f"Pose detected but analysis error: {str(e)}"
            }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        return {"success": False, "error": f"Server error: {str(e)}"}

# ... (start_session remains unchanged) ...
@app.post("/start_session")
async def start_session(data: dict):
    """Start a new exercise session with proper rep counting"""
    try:
        exercise_id = data.get("exercise_id", "squats")
        session_id = f"{exercise_id}_{int(time.time())}" # Create unique session ID

        # Create new session
        session = get_session(exercise_id, session_id)

        print(f"🚀 Started new session: {session_id}")
        print(f" Exercise: {exercise_id}")
        print(f" Start time: {datetime.fromtimestamp(session.start_time)}")

        return {
            "success": True,
            "session_id": session_id,
            "exercise_id": exercise_id,
            "start_time": session.start_time,
            "message": f"Started {exercise_id} session"
        }

    except Exception as e:
        print(f"❌ Error starting session: {e}")
        return {"success": False, "error": str(e)}

@app.post("/end_session")
async def end_session(data: dict):
    # This endpoint handles the ACCURATE upload/saving of data
    """End an exercise session and save results"""
    try:
        session_id = data.get("session_id")
        exercise_id = data.get("exercise_id")

        if not session_id and not exercise_id:
            return {"success": False, "error": "session_id or exercise_id required"}

        session_key = session_id or exercise_id

        if session_key in sessions:
            session = sessions[session_key]
            session_duration = time.time() - session.start_time

            # Calculate average accuracy
            avg_accuracy = sum(session.session_accuracies) / len(session.session_accuracies) if session.session_accuracies else 0

            # Prepare session data for saving
            session_data = {
                "exercise_id": session.exercise_key,
                "exercise_name": list_exercises().get(session.exercise_key, {}).get("name", session.exercise_key),
                "total_reps": session.total_reps,
                "avg_accuracy": avg_accuracy,
                "duration": session_duration,
                # Use the precise rep_details captured during the session
                "reps": session.rep_details
            }

            # Save to database (Upload)
            save_workout_session(session_data)

            # Remove session
            del sessions[session_key]

            print(f"✅ Session ended: {session_key}")
            print(f" Total reps: {session.total_reps}")
            print(f" Duration: {session_duration:.1f}s")
            print(f" Avg accuracy: {session_data['avg_accuracy']:.1f}%")

            return {
                "success": True,
                "session_id": session_key,
                "total_reps": session.total_reps,
                "avg_accuracy": session_data['avg_accuracy'],
                "duration": session_duration,
                "message": "Session saved successfully"
            }
        else:
            return {"success": False, "error": "Session not found"}

    except Exception as e:
        print(f"❌ Error ending session: {e}")
        return {"success": False, "error": str(e)}

# ... (get_session_stats, reset_rep_counter remain unchanged) ...
@app.get("/session_stats/{session_id}")
async def get_session_stats(session_id: str):
    """Get current session statistics"""
    try:
        if session_id in sessions:
            session = sessions[session_id]
            session_duration = time.time() - session.start_time

            avg_accuracy = 0.0
            if session.session_accuracies:
                avg_accuracy = sum(session.session_accuracies) / len(session.session_accuracies)

            return {
                "success": True,
                "session_id": session_id,
                "exercise_id": session.exercise_key,
                "total_reps": session.total_reps,
                "frame_count": session.frame_count,
                "avg_accuracy": avg_accuracy,
                "duration": session_duration,
                "rep_accuracies": session.rep_accuracies,
                "start_time": session.start_time
            }
        else:
            return {"success": False, "error": "Session not found"}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/reset_rep_counter")
async def reset_rep_counter(data: dict):
    """Reset rep counter for a session"""
    try:
        session_id = data.get("session_id")
        exercise_id = data.get("exercise_id")

        if not session_id and not exercise_id:
            return {"success": False, "error": "session_id or exercise_id required"}

        session_key = session_id or exercise_id

        if session_key in sessions:
            sessions[session_key].total_reps = 0
            sessions[session_key].rep_accuracies = []
            sessions[session_key].rep_details = [] # Reset new details list
            print(f"🔄 Reset rep counter for session: {session_key}")
            return {"success": True, "message": "Rep counter reset"}
        else:
            # Create new session if doesn't exist
            session = get_session(exercise_id or "squats", session_key)
            return {"success": True, "message": "Created new session with zero reps"}

    except Exception as e:
        print(f"❌ Error resetting rep counter: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Ensure database tables exist on startup
    try:
        conn = get_db_connection()
        conn.close()
        print("✅ Database connection successful and tables checked/initialized.")
    except Exception as e:
        print(f"❌ Database initialization failed on startup: {e}")
        sys.exit(1)

    import uvicorn
    print("🚀 Starting Enhanced Fitness Tracker API...")
    print(f"📊 Available exercises: {list(list_exercises().keys())}")
    print("🌐 Server running on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)