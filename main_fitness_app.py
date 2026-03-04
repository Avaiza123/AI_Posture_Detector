"""
Main Fitness Application (Updated)
- Startup menu: Live Camera Feed or Uploaded Video File
- Uses exercise rules from `exercise_rules.py` for strict form validation
- Preview overlay option (no zoom/cropping): shows full frame scaled down only when necessary
- Modular: run_session handles both live and video modes
- Optional voice feedback (pyttsx3) if installed

Usage:
- Run the script and use the GUI to choose mode and upload files.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import threading
import time
import os
import cv2
import numpy as np

from pose_utils import draw_pose, draw_highlights, get_pose_landmarks, mp_pose
from exercise_rules import list_exercises
from exercise_rules import list_exercises, build_exercise_by_key
from robust_exercise_classifier import RobustExerciseClassifier

def speak(text: str):
    """Text-to-speech disabled - voice feedback removed"""
    pass


class MainFitnessAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Fitness Posture Tracker")
        self.root.geometry("700x540")
        self.root.configure(bg="#f6f7f9")

        self.classifier = RobustExerciseClassifier()
        self.video_path = None
        self.preview_overlays = tk.BooleanVar(value=True)
        self.selected_exercise = None
        # Voice feedback removed

        self._running = False
        self._capture = None
        self._display_window = "Fitness Preview"
        self._fullscreen = True  # Start in fullscreen by default

        self.setup_exercise_selection()

    def setup_exercise_selection(self):
        """Step 1: Exercise selection screen"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f6f7f9", padx=40, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="Smart Fitness Tracker", 
                        font=("Helvetica", 24, "bold"), bg="#f6f7f9", fg="#2c3e50")
        title.pack(pady=(0, 10))
        
        subtitle = tk.Label(main_frame, text="Step 1: Select Your Exercise", 
                           font=("Helvetica", 16), bg="#f6f7f9", fg="#7f8c8d")
        subtitle.pack(pady=(0, 30))
        
        # Exercise buttons frame
        exercises_frame = tk.Frame(main_frame, bg="#f6f7f9")
        exercises_frame.pack(pady=20)
        
        # Get available exercises
        exercises = list_exercises()
        
        # Create exercise buttons in a grid
        row = 0
        col = 0
        for key, exercise_data in exercises.items():
            btn = tk.Button(exercises_frame, 
                           text=exercise_data["name"],
                           font=("Helvetica", 12, "bold"),
                           bg="#3498db", fg="white",
                           width=20, height=2,
                           command=lambda k=key: self.select_exercise(k),
                           cursor="hand2")
            btn.grid(row=row, column=col, padx=10, pady=10)
            
            col += 1
            if col >= 2:  # 2 columns
                col = 0
                row += 1
        
    def select_exercise(self, exercise_key):
        """Handle exercise selection and move to input method selection"""
        self.selected_exercise = exercise_key
        exercise_name = list_exercises()[exercise_key]["name"]
        print(f"Selected exercise: {exercise_name}")
        
        # Move to step 2: input method selection
        self.setup_input_method_selection()
    
    def setup_input_method_selection(self):
        """Step 2: Input method selection (video upload or live camera)"""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f6f7f9", padx=40, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Show selected exercise
        exercise_name = list_exercises()[self.selected_exercise]["name"]
        selected_label = tk.Label(main_frame, text=f"Selected: {exercise_name}", 
                                 font=("Helvetica", 16, "bold"), bg="#f6f7f9", fg="#27ae60")
        selected_label.pack(pady=(0, 20))
        
        # Title for step 2
        title = tk.Label(main_frame, text="Step 2: Choose Input Method", 
                        font=("Helvetica", 20, "bold"), bg="#f6f7f9", fg="#2c3e50")
        title.pack(pady=(0, 30))
        
        # Input method buttons
        buttons_frame = tk.Frame(main_frame, bg="#f6f7f9")
        buttons_frame.pack(pady=40)
        
        # Upload video button
        upload_btn = tk.Button(buttons_frame, 
                              text="📁 Upload Video File",
                              font=("Helvetica", 14, "bold"),
                              bg="#e74c3c", fg="white",
                              width=25, height=3,
                              command=self.choose_video_upload,
                              cursor="hand2")
        upload_btn.pack(pady=15)
        
        # Live camera button  
        camera_btn = tk.Button(buttons_frame,
                              text="📹 Live Camera Feed", 
                              font=("Helvetica", 14, "bold"),
                              bg="#2ecc71", fg="white",
                              width=25, height=3,
                              command=self.choose_live_camera,
                              cursor="hand2")
        camera_btn.pack(pady=15)
        
        # Back button
        back_btn = tk.Button(buttons_frame,
                            text="← Back to Exercise Selection",
                            font=("Helvetica", 10),
                            bg="#95a5a6", fg="white", 
                            command=self.setup_exercise_selection,
                            cursor="hand2")
        back_btn.pack(pady=(30, 0))

    def choose_video_upload(self):
        """Handle video upload selection"""
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv")]
        )
        if self.video_path:
            self.setup_ui()
            
    def choose_live_camera(self):
        """Handle live camera selection"""
        self.video_path = None  # None indicates live camera
        self.setup_ui()

    def setup_ui(self):
        """Main UI after exercise and input method are selected"""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        frame = tk.Frame(self.root, bg="#f6f7f9")
        frame.pack(fill='both', expand=True, padx=12, pady=12)

        # Show selected exercise
        exercise_name = list_exercises()[self.selected_exercise]["name"]
        input_method = "Live Camera" if self.video_path is None else "Video Upload"
        
        title = tk.Label(frame, text=f"Analyzing: {exercise_name}", font=("Arial", 18, "bold"), bg="#f6f7f9", fg="#2c3e50")
        title.pack(pady=8)

        description = tk.Label(frame, text=f"Input: {input_method} | AI-powered form analysis with real-time feedback", 
                              font=("Arial", 11), bg="#f6f7f9", fg="#7f8c8d")
        description.pack(pady=4)

        # Start session automatically based on selection
        start_frame = tk.Frame(frame, bg="#f6f7f9")
        start_frame.pack(pady=12)
        
        start_btn = tk.Button(start_frame, text="🚀 Start Analysis", width=30, height=2,
                             font=("Arial", 12, "bold"), bg="#27ae60", fg="white",
                             command=self.start_selected_session, cursor="hand2")
        start_btn.pack()

        options_frame = tk.Frame(frame, bg="#f6f7f9")
        options_frame.pack(pady=6)

        tk.Checkbutton(options_frame, text="Preview Overlays (no zoom)", variable=self.preview_overlays, bg="#f6f7f9").pack(side="left", padx=8)

        self.status_text = tk.Text(frame, height=14, width=82, bg="#ffffff", fg="#333333")
        self.status_text.pack(pady=(10,0))
        self.status_text.insert(tk.END, f"Status: Ready to analyze {exercise_name}\n")
        self.status_text.insert(tk.END, f"Input method: {input_method}\n")
        if self.video_path:
            self.status_text.insert(tk.END, f"Video file: {os.path.basename(self.video_path)}\n")
        self.status_text.config(state=tk.DISABLED)

        # Control buttons
        bottom_frame = tk.Frame(frame, bg="#f6f7f9")
        bottom_frame.pack(fill='x', pady=(8,0))
        
        back_btn = tk.Button(bottom_frame, text="← Change Selection", command=self.setup_exercise_selection, bg="#95a5a6", fg='white')
        back_btn.pack(side='left', padx=8)
        
        stop_btn = tk.Button(bottom_frame, text="⏹ Stop Session", command=self.stop_session, bg="#e74c3c", fg='white')
        stop_btn.pack(side='right', padx=8)
        
    def start_selected_session(self):
        """Start the analysis session with pre-selected exercise and input"""
        if self.video_path is None:
            # Live camera mode
            threading.Thread(target=self.run_session, kwargs={"camera_index": 0, "exercise_key": self.selected_exercise}, daemon=True).start()
        else:
            # Video upload mode  
            threading.Thread(target=self.run_session, kwargs={"video_path": self.video_path, "exercise_key": self.selected_exercise}, daemon=True).start()

    def log(self, msg: str):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def on_live(self):
        # Ask user to choose exercise or auto-detect
        exercise_key = self.ask_exercise_choice()
        if exercise_key is None:
            return
        # Start camera session in background thread
        threading.Thread(target=self.run_session, kwargs={"camera_index": 0, "exercise_key": exercise_key}, daemon=True).start()

    def on_upload(self):
        # Prompt for file
        file_types = [('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv'), ('All files', '*.*')]
        path = filedialog.askopenfilename(title="Select Exercise Video", filetypes=file_types)
        if not path:
            return
        self.video_path = path
        exercise_key = self.ask_exercise_choice()
        if exercise_key is None:
            return
        threading.Thread(target=self.run_session, kwargs={"video_path": path, "exercise_key": exercise_key}, daemon=True).start()

    def ask_exercise_choice(self):
        """Prompt user for an exercise key or 'auto'"""
        exercises = list_exercises()
        keys = list(exercises.keys())
        # Build selection string
        choices = "\n".join([f"{i+1}. {exercises[k]['name']} (key='{k}')" for i, k in enumerate(keys)])
        prompt = f"Choose exercise by number or enter 'auto' to auto-detect:\n\n{choices}\n\nEnter choice (default=auto):"
        ans = simpledialog.askstring("Exercise Selection", prompt, parent=self.root)
        if ans is None:
            return None
        ans = ans.strip().lower()
        if ans == "" or ans == "auto":
            return "auto"
        if ans.isdigit():
            idx = int(ans) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
            else:
                messagebox.showinfo("Info", "Invalid number - falling back to auto-detect")
                return "auto"
        # treat string as key
        if ans in exercises:
            return ans
        messagebox.showinfo("Info", "Unknown option - falling back to auto-detect")
        return "auto"

    def stop_session(self):
        """Enhanced session stopping with complete cleanup"""
        self.log("Stopping session...")
        self._running = False
        
        # Release capture if exists
        try:
            if self._capture:
                self._capture.release()
                self._capture = None
        except Exception as e:
            self.log(f"Error releasing capture: {e}")
        
        # Close preview window and all OpenCV windows
        try:
            cv2.destroyWindow(self._display_window)
            cv2.destroyAllWindows()  # Ensure all windows are closed
            cv2.waitKey(1)  # Process the destroy events
        except Exception as e:
            self.log(f"Error closing windows: {e}")
        
        # Force GUI to close if needed
        try:
            if hasattr(self, 'root') and self.root:
                self.root.quit()  # Exit the mainloop
                self.root.destroy()  # Destroy the window
        except Exception as e:
            self.log(f"Error closing GUI: {e}")
        
        self.log("Session stopped successfully")

    def run_session(self, video_path: str = None, camera_index: int = None, exercise_key: str = "auto"):
        """Unified session runner for camera or video file.
        - If exercise_key == 'auto' the system will auto-detect the exercise then run.
        - If preview_overlays is True, a preview window is shown (scaled down only) with overlays, no cropping.
        """
        # Prevent simultaneous sessions
        if self._running:
            self.log("A session is already running. Stop it first.")
            return
        self._running = True

        # Initialize MediaPipe Pose here (use classifier's pose for consistency)
        pose = self.classifier.pose

        # Determine source
        if camera_index is not None:
            cap = cv2.VideoCapture(camera_index)
        elif video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            self.log("No input source specified")
            self._running = False
            return

        self._capture = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # If exercise_key is auto-detect, do a short classification pass
        chosen_key = exercise_key
        if exercise_key == "auto":
            self.log("Auto-detecting exercise (short sample)...")
            # Use classifier to detect (only for file - for camera use short live sample)
            if video_path:
                ex_name, conf, _ = self.classifier.get_stable_classification(video_path, max_frames=int(fps * 4))
            else:
                # For live camera, sample a few frames
                sample_frames = []
                sample_count = int(fps * 3)
                for i in range(sample_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    lm, vis = get_pose_landmarks(results, frame.shape)
                    sample_frames.append((lm, vis))
                # fallback: call classifier.get_stable_classification on an on-disk buffer not available here
                ex_name, conf = None, 0.0
            if ex_name and conf >= 0.5:
                chosen_key = ex_name
                self.log(f"Auto-detected: {chosen_key} ({conf:.1%})")
            else:
                chosen_key = simpledialog.askstring("Choose exercise", "Auto-detect failed; enter exercise key (e.g. squats, pushups):", parent=self.root) or "squats"
                chosen_key = chosen_key.strip().lower()
                self.log(f"Using exercise: {chosen_key}")

        # Build exercise instance
        try:
            exercise = build_exercise_by_key(chosen_key)
        except Exception as e:
            self.log(f"Could not initialize exercise '{chosen_key}': {e}")
            self._running = False
            cap.release()
            return

        # Preview window sizing limits (we will only scale down, never crop)
        max_display_w, max_display_h = 1600, 900  # Increased display size for better visibility

        # Session stats
        frame_count = 0
        pose_detected = 0
        start_time = time.time()

        # Interactive feedback timer to avoid spamming voice
        last_voice_time = 0
        voice_interval = 2.0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                landmarks_px, visibility = get_pose_landmarks(results, frame.shape)

                feedback_text = ""
                highlight_ids = []
                accuracy = None

                if landmarks_px is not None:
                    pose_detected += 1
                    timestamp = frame_count / (fps or 30.0)

                    rep_inc, feedback_text, highlight_ids, accuracy = exercise.update(landmarks_px, visibility, frame.shape, timestamp)

                    # Handle new list-based feedback from enhanced exercises
                    if isinstance(feedback_text, list):
                        # Convert list to string for compatibility with existing voice/logging systems
                        feedback_text_str = " | ".join(feedback_text[:3])  # Limit for readability
                        legacy_feedback_list = feedback_text  # Keep list for UI
                    else:
                        # Legacy string feedback 
                        feedback_text_str = feedback_text
                        legacy_feedback_list = [feedback_text] if feedback_text else []

                    # Voice feedback (optional)
                    if False:  # Voice feedback disabled
                        threading.Thread(target=speak, args=(feedback_text_str,), daemon=True).start()
                        last_voice_time = time.time()

                else:
                    feedback_text_str = "Move into frame - pose not detected"
                    legacy_feedback_list = ["Move into frame - pose not detected"]

                # Enhanced logging with frame-level details
                if frame_count % int(fps * 2) == 0:  # Log every 2 seconds
                    cur_reps = exercise.reps
                    avg_acc = np.mean(exercise.session_accuracies) if exercise.session_accuracies else 0.0
                    pose_detection_rate = (pose_detected / frame_count) * 100
                    
                    # Detailed status logging
                    status_msg = (f"Frame {frame_count}: reps={cur_reps} | "
                                f"avg_accuracy={avg_acc:.1f}% | "
                                f"pose_detection={pose_detection_rate:.1f}% | "
                                f"current_accuracy={accuracy:.1f}%" if accuracy else "N/A")
                    
                    self.log(status_msg)
                    
                    # Additional debug info if pose detected
                    if landmarks_px is not None and hasattr(exercise, 'posture_state'):
                        state_info = f"State: {exercise.posture_state.current_state}"
                        if hasattr(exercise, '_in_rep'):
                            state_info += f" | in_rep: {exercise._in_rep}"
                        if hasattr(exercise, '_valid_start_frames'):
                            state_info += f" | valid_frames: {exercise._valid_start_frames}"
                        self.log(state_info)

                # Real-time rep logging with detailed information
                if accuracy is not None:
                    timestamp = frame_count / (fps or 30.0)
                    
                    # Log rep completions immediately with context
                    if hasattr(exercise, 'rep_timestamps') and exercise.rep_timestamps:
                        last_rep_time = exercise.rep_timestamps[-1]
                        if abs(last_rep_time - timestamp) < 0.1:  # Recently completed rep
                            rep_details = (f"🎉 REP #{exercise.reps} completed at {timestamp:.2f}s | "
                                         f"Accuracy: {exercise.rep_accuracies[-1]:.1f}% | "
                                         f"Total session accuracy: {np.mean(exercise.session_accuracies):.1f}%")
                            self.log(rep_details)

                # If preview overlays requested, render without cropping (scale down only to maintain aspect ratio)
                if self.preview_overlays.get():
                    # Draw pose and highlights on a copy to not modify original frame used for analysis
                    disp = frame.copy()
                    if landmarks_px is not None:
                        draw_pose(disp, results)
                        if highlight_ids:
                            draw_highlights(disp, landmarks_px, highlight_ids, color=(0,0,255))

                    # Calculate enhanced feedback and accuracies
                    if landmarks_px is not None:
                        # Calculate current angles for the exercise
                        current_angles = {}
                        try:
                            if hasattr(exercise, '_calculate_current_angles'):
                                current_angles = exercise._calculate_current_angles(landmarks_px, visibility)
                            else:
                                current_angles = exercise._calculate_basic_angles(landmarks_px, visibility)
                        except:
                            # Fallback to basic angle calculation
                            try:
                                current_angles = exercise._calculate_basic_angles(landmarks_px, visibility)
                            except:
                                # Final fallback with empty angles
                                current_angles = {'knee_angle': None, 'hip_angle': None, 'elbow_angle': None}
                        
                        # Determine current phase with better fallback
                        phase = "active"  # Default phase
                        try:
                            if hasattr(exercise, '_determine_phase') and current_angles:
                                if 'hip_angle' in current_angles and 'knee_angle' in current_angles:
                                    if current_angles['hip_angle'] is not None and current_angles['knee_angle'] is not None:
                                        phase = exercise._determine_phase(current_angles['hip_angle'], current_angles['knee_angle'])
                                elif 'elbow_angle' in current_angles and current_angles['elbow_angle'] is not None:
                                    # For push-ups/bicep curls
                                    elbow = current_angles['elbow_angle']
                                    phase = "middle" if elbow < 120 else "start"
                            
                            # Exercise-specific phase detection
                            if hasattr(exercise, 'is_down') and hasattr(exercise, 'is_up'):
                                if exercise.is_down:
                                    phase = "middle"
                                elif exercise.is_up:
                                    phase = "start"
                        except Exception as e:
                            pass  # Keep default phase
                        
                        # Calculate enhanced accuracies with error handling
                        try:
                            form_accuracy, joint_deviations = exercise.compute_form_accuracy(current_angles, phase)
                        except:
                            form_accuracy, joint_deviations = None, {}
                        
                        # Prepare movement data for simple accuracy
                        movement_data = {'primary_angle': None}
                        if hasattr(exercise, 'down_threshold'):
                            if 'knee_angle' in current_angles and current_angles['knee_angle'] is not None:
                                movement_data['primary_angle'] = current_angles['knee_angle']
                            elif 'elbow_angle' in current_angles and current_angles['elbow_angle'] is not None:
                                movement_data['primary_angle'] = current_angles['elbow_angle']
                        
                        # Get phase transition history
                        if not hasattr(exercise, 'phase_history'):
                            exercise.phase_history = []
                        exercise.phase_history.append(phase)
                        if len(exercise.phase_history) > 10:  # Keep last 10 phases
                            exercise.phase_history.pop(0)
                        
                        try:
                            simple_accuracy = exercise.compute_simple_accuracy(movement_data, exercise.phase_history)
                        except:
                            simple_accuracy = None
                        
                        # Generate NEW COMPREHENSIVE FEEDBACK MESSAGES
                        try:
                            if form_accuracy is not None and simple_accuracy is not None:
                                # Use new comprehensive feedback system if available
                                if hasattr(exercise, 'generate_comprehensive_feedback'):
                                    feedback_messages = exercise.generate_comprehensive_feedback(
                                        landmarks_px, visibility, current_angles, joint_deviations, phase)
                                elif legacy_feedback_list:
                                    # Use enhanced feedback from exercise.update() if available
                                    feedback_messages = legacy_feedback_list
                                else:
                                    # Fallback to legacy system for exercises not yet enhanced
                                    feedback_messages = exercise.generate_feedback_message(current_angles, joint_deviations, phase)
                            else:
                                # Check if user has any reps - if so, encourage to continue
                                if hasattr(exercise, 'reps') and exercise.reps > 0:
                                    feedback_messages = ["💪 Keep going!", "✨ Continue your exercise"]
                                else:
                                    feedback_messages = ["🏃‍♂️ Get into position", "💪 Start your exercise"]
                        except Exception as e:
                            # Enhanced error handling
                            print(f"Feedback generation error: {e}")
                            if form_accuracy is not None and simple_accuracy is not None:
                                feedback_messages = ["✅ Exercising detected", "💪 Keep moving"]
                            else:
                                if hasattr(exercise, 'reps') and exercise.reps > 0:
                                    feedback_messages = ["💪 Keep going!", "✨ Continue your exercise"]
                                else:
                                    feedback_messages = ["🏃‍♂️ Get into position", "💪 Start your exercise"]
                        
                        # Store accuracies for session tracking (only when exercising)
                        if form_accuracy is not None and simple_accuracy is not None:
                            exercise.session_accuracies.append((form_accuracy + simple_accuracy) / 2)
                            if len(exercise.session_accuracies) > 100:  # Keep last 100 measurements
                                exercise.session_accuracies.pop(0)
                        
                        # Render modern responsive UI system
                        try:
                            # Use default values when not exercising
                            display_form_accuracy = form_accuracy if form_accuracy is not None else 0
                            display_simple_accuracy = simple_accuracy if simple_accuracy is not None else 0
                            
                            # Use new responsive UI system that adapts to video orientation
                            disp = exercise.draw_ui(
                                disp, exercise.name, exercise.reps, 
                                display_form_accuracy, display_simple_accuracy, 
                                feedback_messages, phase
                            )
                        except Exception as e:
                            # Fallback to basic responsive UI
                            # FORCE Pic 1 style comprehensive UI for ALL videos (especially vertical)
                            try:
                                disp = exercise.draw_ui(
                                    disp, exercise.name, exercise.reps, 
                                    display_form_accuracy, display_simple_accuracy, 
                                    feedback_messages, "active"
                                )
                            except Exception as e:
                                print(f"UI Error (using fallback): {e}")
                                # Force comprehensive UI even on errors - don't use simple fallback
                                try:
                                    disp = exercise.render_vertical_ui_panel(
                                        disp, display_form_accuracy, display_simple_accuracy,
                                        feedback_messages, exercise.reps, "active"
                                    )
                                except:
                                    # Last resort - still use comprehensive UI but with safe defaults
                                    self._draw_comprehensive_ui_safe(disp, exercise.name, exercise.reps, 
                                                                display_form_accuracy, display_simple_accuracy, feedback_messages)
                    else:
                        # Fallback to simple feedback display for no pose detection
                        self._draw_simple_feedback_on_frame(disp, feedback_text, accuracy, exercise)

                    # Enhanced display scaling: maintain aspect ratio, ensure full body visible
                    h, w = disp.shape[:2]
                    
                    # Calculate scale factor to fit within display limits while preserving aspect ratio
                    scale_w = max_display_w / w
                    scale_h = max_display_h / h
                    scale = min(scale_w, scale_h, 1.0)  # Never scale up, only down
                    
                    # Only resize if frame is larger than display limits
                    if scale < 1.0:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        disp = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                        # Optional: Add letterboxing if needed to center in display area
                        if new_w < max_display_w or new_h < max_display_h:
                            # Create letterbox background
                            letterbox = np.zeros((max_display_h, max_display_w, 3), dtype=np.uint8)
                            
                            # Center the scaled frame
                            y_offset = (max_display_h - new_h) // 2
                            x_offset = (max_display_w - new_w) // 2
                            letterbox[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = disp
                            disp = letterbox

                    # Create fullscreen window for better visibility
                    cv2.namedWindow(self._display_window, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(self._display_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    
                    # Display with consistent window name
                    cv2.imshow(self._display_window, disp)
                    
                    # Enhanced keyboard controls + window close detection
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Check if window was closed by user (X button)
                    try:
                        if cv2.getWindowProperty(self._display_window, cv2.WND_PROP_VISIBLE) < 1:
                            self.log("Window closed by user")
                            self.stop_session()
                            break
                    except cv2.error:
                        # Window was destroyed externally
                        self.log("Window destroyed externally")
                        self.stop_session() 
                        break
                    
                    if key == ord('q'):
                        self.stop_session()
                        break
                    elif key == ord('f'):
                        # Toggle fullscreen
                        current_prop = cv2.getWindowProperty(self._display_window, cv2.WND_PROP_FULLSCREEN)
                        if current_prop == cv2.WINDOW_FULLSCREEN:
                            cv2.setWindowProperty(self._display_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        else:
                            cv2.setWindowProperty(self._display_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    elif key == ord('r'):
                        # Reset rep counter
                        if exercise:
                            exercise.reset()
                            self.log("Rep counter reset")
                    elif key == 27:  # ESC key
                        self.stop_session()
                        break

                # If not previewing overlays we still must sleep a bit to avoid hogging CPU
                else:
                    # small sleep to yield
                    time.sleep(0.001)

            # End loop
        except Exception as e:
            self.log(f"Error during session: {e}")
        finally:
            cap.release()
            try:
                cv2.destroyWindow(self._display_window)
            except Exception:
                pass

            duration = time.time() - start_time if start_time else 0.0
            final_stats = exercise.get_session_stats()
            self.log("Session complete")
            self.log(f"Exercise: {exercise.name}")
            self.log(f"Total reps: {final_stats['total_reps']}")
            self.log(f"Average rep accuracy: {final_stats['average_rep_accuracy']:.1f}%")
            self.log(f"Average session accuracy: {final_stats['average_session_accuracy']:.1f}%")
            self._running = False

    def _draw_feedback_on_frame(self, frame: np.ndarray, feedback_text: str, accuracy: float, exercise=None):
        """Overlay feedback text on frame with adaptive layout for all video orientations"""
        # Get original frame dimensions
        h, w = frame.shape[:2]
        
        # Detect video orientation
        is_vertical = h > w * 1.2
        is_square = abs(h - w) < min(h, w) * 0.2
        
        # Adaptive overlay dimensions
        if is_vertical:
            overlay_height = min(h // 3, 250)  # Use bottom third for vertical
            overlay_y = h - overlay_height - 10
            overlay_x = 5
            overlay_width = w - 10
        else:
            overlay_height = min(200, h // 4)  # Larger overlay height for landscape
            overlay_y = 10
            overlay_x = 10
            overlay_width = w - 20
            
        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (overlay_x, overlay_y), (overlay_x + overlay_width, overlay_y + overlay_height), (20, 20, 20), -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Adaptive font scaling based on orientation
        if is_vertical:
            font_scale = max(0.7, min(1.2, w / 400))  # Smaller scale for vertical
            x_start = overlay_x + 10
        else:
            font_scale = max(1.0, min(2.0, w / 600))  # Original for landscape
            x_start = overlay_x + 10
            
        thickness = max(2, int(font_scale * 2.5))
        y_start = overlay_y + 30
        line_height = int((30 if is_vertical else 35) * font_scale)
        
        # Display rep counter with adaptive sizing
        if exercise is not None:
            rep_text = f"REPS: {exercise.reps}"
            rep_font_scale = font_scale * (1.2 if is_vertical else 1.6)  # Smaller for vertical
            rep_thickness = thickness + (1 if is_vertical else 2)
            cv2.putText(frame, rep_text, (x_start, y_start), cv2.FONT_HERSHEY_DUPLEX, 
                       rep_font_scale, (0, 0, 0), rep_thickness + 2, cv2.LINE_AA)  # Black outline
            cv2.putText(frame, rep_text, (x_start, y_start), cv2.FONT_HERSHEY_DUPLEX, 
                       rep_font_scale, (0, 255, 0), rep_thickness, cv2.LINE_AA)  # Green text
            y_start += int(line_height * (1.3 if is_vertical else 1.6))
        
        # Display exercise name with adaptive sizing
        if exercise is not None and hasattr(exercise, 'name'):
            ex_text = exercise.name if is_vertical else f"Exercise: {exercise.name}"  # Shorter for vertical
            ex_font_scale = font_scale * (0.9 if is_vertical else 1.1)
            cv2.putText(frame, ex_text, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 
                       ex_font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)  # Black outline
            cv2.putText(frame, ex_text, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 
                       ex_font_scale, (255, 255, 0), thickness, cv2.LINE_AA)  # Cyan text
            y_start += int(line_height * (1.0 if is_vertical else 1.1))

        # Helper function for text wrapping in vertical videos
        def wrap_line(text, max_width_chars):
            if len(text) <= max_width_chars:
                return [text]
            words = text.split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if len(test_line) <= max_width_chars:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            return lines
        
        # Split feedback lines and apply wrapping for vertical videos
        original_lines = feedback_text.split('\n') if feedback_text else ["Keep going"]
        
        # Determine max characters per line based on orientation
        max_chars = int(overlay_width / (font_scale * 8)) if is_vertical else 50
        
        # Process and wrap lines
        wrapped_lines = []
        for line in original_lines:
            if line.strip():
                wrapped_lines.extend(wrap_line(line, max_chars))
        
        # Calculate available space and max lines
        available_height = overlay_height - (y_start - overlay_y) - 10
        max_lines = max(2, available_height // line_height)
        
        # Display wrapped feedback lines
        displayed_lines = 0
        for line in wrapped_lines:
            if displayed_lines >= max_lines or y_start + line_height > overlay_y + overlay_height - 10:
                break
                
            if not line.strip():
                continue
                
            # Color coding for different message types
            color = (220, 220, 220)  # Default bright gray
            if line.startswith('🎉'):
                color = (0, 255, 0)  # Bright green for excellent
            elif line.startswith('✅'):
                color = (50, 255, 50)  # Lighter green for good
            elif line.startswith('⚠'):
                color = (0, 200, 255)  # Bright orange for warnings
            elif line.startswith('❌'):
                color = (0, 50, 255)  # Bright red for errors
            elif line.startswith('📍') or line.startswith('⏳'):
                color = (255, 255, 0)  # Yellow for info
            
            # Adaptive font scale for feedback
            feedback_font_scale = font_scale * (0.7 if is_vertical else 0.8)
            feedback_thickness = max(2, int(feedback_font_scale * 2))
            
            # Draw text with outline for visibility
            cv2.putText(frame, line, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 
                       feedback_font_scale, (0, 0, 0), feedback_thickness + 1, cv2.LINE_AA)  # Black outline
            cv2.putText(frame, line, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 
                       feedback_font_scale, color, feedback_thickness, cv2.LINE_AA)  # Colored text
            
            y_start += line_height
            displayed_lines += 1

    def _draw_comprehensive_ui_safe(self, frame, exercise_name, reps, form_accuracy, accuracy, feedback_messages):
        """
        Safe comprehensive UI that matches Pic 1 style - ALWAYS shows full panel for vertical videos
        This is the last resort to ensure vertical videos never show minimal UI
        """
        import cv2
        
        h, w = frame.shape[:2]
        is_vertical = h > w  # Force detection
        
        if is_vertical:
            # FORCE PIC 1 STYLE PANEL FOR VERTICAL VIDEOS
            panel_height = int(min(500, max(320, h // 2.2)))
            panel_x = 10
            panel_y = 80  # Shift down to avoid any text overlays
            panel_width = w - 20
            
            # Ensure all coordinates are integers
            panel_x = int(panel_x)
            panel_y = int(panel_y)
            panel_width = int(panel_width)
            panel_height = int(panel_height)
            
            # Semi-transparent background (Pic 1 style)
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (15, 15, 15), -1)
            cv2.addWeighted(frame, 0.65, overlay, 0.35, 0, frame)
            
            # Professional border
            cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                         (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                         (80, 140, 200), 2)
            
            # Content with Pic 1 style layout
            content_x = panel_x + 15
            y_pos = panel_y + 30
            
            # Exercise name (Big title)
            cv2.putText(frame, exercise_name.upper(), (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 3)
            y_pos += 60
            
            # Reps (Large number)
            cv2.putText(frame, f"REPS: {reps}", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 3)
            y_pos += 50
            
            # FORM ACCURACY with bar
            cv2.putText(frame, f"FORM ACCURACY: {form_accuracy:.0f}%", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 120), 2)
            y_pos += 30
            
            # Form accuracy bar
            bar_width = int(min(panel_width - 40, 300))
            bar_height = 15
            cv2.rectangle(frame, (int(content_x), int(y_pos)), (int(content_x + bar_width), int(y_pos + bar_height)), (80, 80, 80), -1)
            form_fill = int((form_accuracy / 100.0) * bar_width)
            if form_fill > 0:
                cv2.rectangle(frame, (int(content_x), int(y_pos)), (int(content_x + form_fill), int(y_pos + bar_height)), (120, 255, 120), -1)
            y_pos += 50
            
            # ACCURACY with bar
            cv2.putText(frame, f"ACCURACY: {accuracy:.0f}%", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 120), 2)
            y_pos += 30
            
            # Overall accuracy bar
            cv2.rectangle(frame, (int(content_x), int(y_pos)), (int(content_x + bar_width), int(y_pos + bar_height)), (80, 80, 80), -1)
            overall_fill = int((accuracy / 100.0) * bar_width)
            if overall_fill > 0:
                cv2.rectangle(frame, (int(content_x), int(y_pos)), (int(content_x + overall_fill), int(y_pos + bar_height)), (120, 255, 120), -1)
            y_pos += 50
            
            # FEEDBACK section
            cv2.putText(frame, "FEEDBACK:", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 35
            
            # Feedback messages
            for i, msg in enumerate(feedback_messages[:4]):  # Limit to 4 messages
                if y_pos + 25 < panel_y + panel_height - 10:
                    clean_msg = str(msg)[:60] + "..." if len(str(msg)) > 60 else str(msg)
                    cv2.putText(frame, f"• {clean_msg}", (content_x, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 35
        else:
            # Horizontal layout (unchanged)
            self._draw_clean_fallback_ui_horizontal(frame, exercise_name, reps, form_accuracy, accuracy, feedback_messages)

    def _draw_clean_fallback_ui_horizontal(self, frame, exercise_name, reps, form_accuracy, accuracy, feedback_messages):
        """
        Clean fallback UI with modern design for horizontal videos
        """
        import cv2
        
        h, w = frame.shape[:2]
        
        # Right panel layout for horizontal videos
        panel_width = min(300, w // 3)
        panel_x = w - panel_width - 20
        panel_y = 20
        panel_height = min(400, h - 40)
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.45, overlay, 0.55, 0, frame)
        
        # Content
        content_x = panel_x + 15
        y_pos = panel_y + 40
        
        # Exercise name
        cv2.putText(frame, exercise_name.upper(), (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 60
        
        # Reps
        cv2.putText(frame, f"REPS: {reps}", (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        y_pos += 60
        
        # Accuracies
        if form_accuracy > 0:
            cv2.putText(frame, f"FORM: {form_accuracy:.0f}%", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            y_pos += 40
            cv2.putText(frame, f"OVERALL: {accuracy:.0f}%", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        else:
            cv2.putText(frame, "START EXERCISING", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    def _draw_simple_feedback_on_frame(self, frame: np.ndarray, feedback_text: str, accuracy: float, exercise=None):
        """
        Simple fallback feedback display for when pose detection fails
        """
        h, w = frame.shape[:2]
        
        # Simple overlay
        overlay_height = 80
        cv2.rectangle(frame, (10, 10), (w-10, overlay_height), (40, 40, 40), -1)
        
        # Clean "no pose detected" message with professional styling
        panel_width = min(400, w // 3)
        panel_x = w - panel_width - 20
        panel_y = 20
        
        # Semi-transparent background for no-pose message
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + 100), (30, 30, 30), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        # Large, clear message
        no_pose_font = 0.9
        no_pose_thickness = 2
        cv2.putText(frame, "MOVE INTO FRAME", (panel_x + 20, panel_y + 40), 
                   cv2.FONT_HERSHEY_DUPLEX, no_pose_font, (255, 255, 255), no_pose_thickness)
        cv2.putText(frame, "Pose not detected", (panel_x + 20, panel_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, no_pose_font * 0.7, (200, 200, 200), no_pose_thickness)
        
        if exercise and hasattr(exercise, 'reps'):
            reps_font = no_pose_font * 0.8
            reps_thickness = max(2, int(reps_font * 2))
            cv2.putText(frame, f"Current Reps: {exercise.reps}", (20, 65 + int(no_pose_font * 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, reps_font, (0, 0, 0), reps_thickness + 1)  # Black outline
            cv2.putText(frame, f"Current Reps: {exercise.reps}", (20, 65 + int(no_pose_font * 25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, reps_font, (255, 255, 255), reps_thickness)
        
        # Add instructions at the bottom with proper font scaling
        instructions = "Controls: 'q' quit | 'ESC' quit | 'f' toggle fullscreen | 'r' reset reps | Click X to close"
        inst_font_scale = max(0.4, min(0.6, w / 1200))  # Smaller font for instructions
        inst_thickness = max(1, int(inst_font_scale * 2))
        cv2.putText(frame, instructions, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   inst_font_scale, (0, 0, 0), inst_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, instructions, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   inst_font_scale, (180, 180, 180), inst_thickness, cv2.LINE_AA)


def main():
    root = tk.Tk()
    app = MainFitnessAppGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_session)
    root.mainloop()


if __name__ == "__main__":
    main()