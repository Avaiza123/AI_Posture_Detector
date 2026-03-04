"""
Robust Multi-Exercise Classifier
Simplified but robust exercise detection system
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
from collections import deque, Counter
import time

from pose_utils import get_pose_landmarks, L, calculate_angle
from exercise_rules import list_exercises, build_exercise_by_key, safe_angle, avg_ignore_none


class RobustExerciseClassifier:
    """Robust multi-exercise classifier with temporal smoothing"""
    
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.classification_history = deque(maxlen=30)
        
    def extract_movement_features(self, landmarks, visibility) -> Dict[str, float]:
        """Extract key movement features from pose landmarks"""
        features = {}
        
        try:
            # Knee angles (primary for squats, lunges, deadlifts)
            left_knee = safe_angle(landmarks, L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value, visibility)
            right_knee = safe_angle(landmarks, L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value, visibility)
            features['knee_angle'] = avg_ignore_none([left_knee, right_knee])
            
            # Elbow angles (primary for pushups, bicep curls, shoulder press)
            left_elbow = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value, visibility)
            right_elbow = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value, visibility)
            features['elbow_angle'] = avg_ignore_none([left_elbow, right_elbow])
            features['elbow_min'] = min([x for x in [left_elbow, right_elbow] if x is not None], default=180)
            
            # Hip angles (for hip hinge movements)
            features['hip_angle'] = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value, visibility)
            
            # Body orientation (vertical vs horizontal exercises)
            body_line = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_ANKLE.value, visibility)
            features['body_line'] = body_line
            
            # NEW: Simple torso orientation check
            # For pushups: torso is horizontal (shoulders and hips at similar Y positions)
            # For squats: torso is vertical (shoulders significantly above hips)
            left_shoulder = landmarks[L.LEFT_SHOULDER.value]
            left_hip = landmarks[L.LEFT_HIP.value]
            
            # Y coordinate difference (positive when shoulder is above hip)
            y_diff = left_hip[1] - left_shoulder[1]  # Hip Y - Shoulder Y
            features['torso_vertical_diff'] = y_diff  # Positive = vertical, near 0 = horizontal
            
            # Arm positions
            left_wrist = landmarks[L.LEFT_WRIST.value]
            right_wrist = landmarks[L.RIGHT_WRIST.value]
            left_shoulder = landmarks[L.LEFT_SHOULDER.value]
            right_shoulder = landmarks[L.RIGHT_SHOULDER.value]
            
            # Arms overhead detection
            arms_up = (left_wrist[1] < left_shoulder[1]) and (right_wrist[1] < right_shoulder[1])
            features['arms_overhead'] = 1.0 if arms_up else 0.0
            
            # Hip height for movement range
            left_hip = landmarks[L.LEFT_HIP.value]
            right_hip = landmarks[L.RIGHT_HIP.value]
            features['hip_height'] = (left_hip[1] + right_hip[1]) / 2
            
            # Body stability
            shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
            features['shoulder_stability'] = max(0, 100 - shoulder_level * 2)
            
        except (IndexError, TypeError) as e:
            print(f"Warning: Error extracting features: {e}")
            
        return features
    
    def classify_exercise_frame(self, landmarks, visibility) -> Dict[str, float]:
        """Classify exercise for a single frame"""
        features = self.extract_movement_features(landmarks, visibility)
        
        scores = {}
        
        # Squats classification (IMPROVED ELBOW PENALTY)
        squat_score = 0
        
        # PRIMARY: Knee bending (KEY for squats)
        if features.get('knee_angle'):
            knee = features['knee_angle']
            if 90 <= knee <= 140:  # Deep squat position
                squat_score += 60  # Strong indicator
            elif 140 <= knee <= 170:  # Standing/shallow squat
                squat_score += 40
            elif knee > 170:  # Very straight legs
                squat_score += 15  # Could be standing phase
                
        # SECONDARY: Arms are not primary movers (KEY DIFFERENTIATOR)
        if features.get('elbow_angle'):
            elbow = features['elbow_angle']
            if elbow > 140:  # Arms extended/stable
                squat_score += 30
            elif 100 <= elbow <= 140:  # Arms partially bent (normal)
                squat_score += 15
            elif elbow < 100:  # Very bent arms (PUSHUP INDICATOR)
                squat_score -= 40  # Strong penalty for bent elbows
                
        # TERTIARY: Body alignment
        if features.get('body_line') and features['body_line'] > 160:
            squat_score += 15
            
        # PENALTY: Arms overhead (unless overhead squats)
        if features.get('arms_overhead', False):
            squat_score -= 15
            
        scores['squats'] = max(0, squat_score)
        
        # Pushups classification (CAMERA ANGLE FIXED)
        pushup_score = 0
        
        # PRIMARY: Elbow movement pattern (MOST RELIABLE for pushups)
        if features.get('elbow_angle'):
            elbow = features['elbow_angle']
            if 60 <= elbow <= 120:  # Active push range (bent elbows)
                pushup_score += 70  # Very strong indicator
            elif 120 <= elbow <= 160:  # Mid-range
                pushup_score += 50
            elif 160 <= elbow <= 180:  # Extended position
                pushup_score += 25
                
        # SECONDARY: Body alignment (straight line)
        if features.get('body_line'):
            body_angle = features['body_line']
            if body_angle > 165:  # Very straight body line
                pushup_score += 40
            elif body_angle > 150:  # Reasonably straight
                pushup_score += 20
                
        # TERTIARY: Leg stability (straight legs in pushups)
        if features.get('knee_angle'):
            knee = features['knee_angle']
            if knee > 160:  # Straight legs (good for pushups)
                pushup_score += 30
            elif knee < 120:  # Very bent knees (bad for pushups)
                pushup_score -= 25
                
        # PENALTY: Arms overhead
        if features.get('arms_overhead', False):
            pushup_score -= 40
        else:
            pushup_score += 15  # Bonus for arms not overhead
            
        scores['pushups'] = max(0, pushup_score)
        
        # Plank classification
        plank_score = 0
        if features.get('body_line') and 165 <= features['body_line'] <= 185:
            plank_score += 50
        if features.get('elbow_angle') and 80 <= features['elbow_angle'] <= 100:
            plank_score += 30
        if features.get('shoulder_stability') and features['shoulder_stability'] > 80:
            plank_score += 20
        scores['plank'] = max(0, plank_score)
        
        # Bicep curls classification
        bicep_score = 0
        if features.get('elbow_min'):
            elbow = features['elbow_min']
            if 30 <= elbow <= 170:  # Curling range
                bicep_score += 40
            if features.get('body_line') and features['body_line'] > 160:  # Upright
                bicep_score += 30
            if features.get('shoulder_stability') and features['shoulder_stability'] > 70:
                bicep_score += 20
        scores['bicep_curls'] = max(0, bicep_score)
        
        # Jumping jacks classification
        jacks_score = 0
        if features.get('arms_overhead'):
            jacks_score += 40
        if features.get('body_line') and features['body_line'] > 150:  # Upright
            jacks_score += 30
        # Add more dynamic movement detection for jumping jacks
        scores['jumping_jacks'] = max(0, jacks_score)
        
        # Shoulder press classification
        shoulder_score = 0
        if features.get('elbow_angle') and features['elbow_angle'] > 90:
            shoulder_score += 30
        if features.get('arms_overhead'):
            shoulder_score += 40
        if features.get('body_line') and features['body_line'] > 160:
            shoulder_score += 20
        scores['shoulder_press'] = max(0, shoulder_score)
        
        # Add other exercises with basic scoring
        scores['lunges'] = max(0, scores['squats'] * 0.7)  # Similar to squats but different
        scores['deadlifts'] = max(0, scores['squats'] * 0.6)  # Hip hinge movement
        scores['side_lateral_raises'] = max(0, scores['shoulder_press'] * 0.8)
        scores['mountain_climbers'] = max(0, scores['plank'] * 0.7)
        
        return scores
    
    def get_stable_classification(self, video_path: str, max_frames: int = 120) -> Tuple[str, float, Dict]:
        """Analyze video and return stable exercise classification"""
        
        print("🔍 ROBUST EXERCISE CLASSIFICATION")
        print("=" * 60)
        print(f"📹 Video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_to_analyze = min(max_frames, total_frames)
        print(f"📊 Analyzing {frames_to_analyze} frames at {fps:.1f} FPS")
        
        # Collect all frame classifications
        all_scores = {ex: [] for ex in list_exercises().keys()}
        frame_count = 0
        valid_frames = 0
        
        while frame_count < frames_to_analyze:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            landmarks_px, visibility = get_pose_landmarks(results, frame.shape)
            
            if landmarks_px is not None:
                # Classify frame
                frame_scores = self.classify_exercise_frame(landmarks_px, visibility)
                
                # Store scores
                for exercise, score in frame_scores.items():
                    all_scores[exercise].append(score)
                
                valid_frames += 1
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / frames_to_analyze) * 100
                print(f"⏳ Classification progress: {progress:.0f}%")
        
        cap.release()
        
        # Calculate final statistics
        exercise_stats = {}
        for exercise, scores in all_scores.items():
            if scores:
                mean_score = np.mean(scores)
                max_score = np.max(scores)
                std_score = np.std(scores)
                consistency = max(0, 100 - std_score)
                frames_above_threshold = len([s for s in scores if s > 50])
                
                exercise_stats[exercise] = {
                    'mean_score': mean_score,
                    'max_score': max_score,
                    'consistency': consistency,
                    'frames_detected': frames_above_threshold,
                    'detection_rate': frames_above_threshold / len(scores) if scores else 0
                }
        
        # Determine best exercise
        if exercise_stats:
            # Sort by mean score
            sorted_exercises = sorted(exercise_stats.items(), 
                                    key=lambda x: x[1]['mean_score'], reverse=True)
            
            best_exercise = sorted_exercises[0]
            exercise_name = best_exercise[0]
            stats = best_exercise[1]
            
            # Calculate confidence based on multiple factors
            confidence = (stats['mean_score'] / 100.0) * 0.6 + \
                        (stats['consistency'] / 100.0) * 0.2 + \
                        (stats['detection_rate']) * 0.2
            
            confidence = min(confidence, 1.0)
            
            print(f"✅ CLASSIFICATION COMPLETE")
            print(f"   Best Match: {exercise_name.upper()}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Mean Score: {stats['mean_score']:.1f}%")
            
            return exercise_name, confidence, exercise_stats
        else:
            print(f"❌ No valid classifications found")
            return None, 0.0, {}
    
    def analyze_video_with_classification(self, video_path: str, max_seconds: int = None) -> Dict:
        """Complete video analysis with classification and rep counting"""
        
        print("🤖 ROBUST FITNESS TRACKER")
        print("=" * 60)
        
        # Step 1: Classify exercise
        exercise_name, confidence, exercise_stats = self.get_stable_classification(video_path)
        
        if not exercise_name or confidence < 0.6:
            return {
                'error': 'Could not reliably classify exercise',
                'detected_exercise': exercise_name,
                'classification_confidence': confidence,
                'exercise_stats': exercise_stats
            }
        
        print(f"\n🏋️ PROCEEDING WITH: {exercise_name.upper()}")
        print("=" * 60)
        
        # Step 2: Initialize exercise counter
        try:
            exercise_instance = build_exercise_by_key(exercise_name)
            exercise_instance.debug_enabled = True
        except Exception as e:
            return {
                'error': f'Could not initialize {exercise_name}: {str(e)}',
                'detected_exercise': exercise_name,
                'classification_confidence': confidence
            }
        
        # Step 3: Count reps
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        if max_seconds and max_seconds > 0:
            frames_to_process = min(int(fps * max_seconds), total_frames)
            duration_to_process = min(max_seconds, total_duration)
        else:
            frames_to_process = total_frames
            duration_to_process = total_duration
        
        print(f"📊 Rep counting: {frames_to_process} frames ({duration_to_process:.1f}s)")
        
        frame_count = 0
        initial_reps = exercise_instance.reps
        pose_detected_count = 0
        
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            landmarks_px, visibility = get_pose_landmarks(results, frame.shape)
            
            if landmarks_px is not None:
                pose_detected_count += 1
                timestamp = frame_count / fps
                
                # Update exercise with pose data
                rep_inc, feedback, highlights, accuracy = exercise_instance.update(
                    landmarks_px, visibility, frame.shape, timestamp
                )
                
                if rep_inc > 0:
                    print(f"🔥 REP #{exercise_instance.reps} at {timestamp:.1f}s (accuracy: {accuracy:.1f}%)")
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % int(fps * 5) == 0:
                progress = (frame_count / frames_to_process) * 100
                current_reps = exercise_instance.reps - initial_reps
                print(f"⏳ Progress: {progress:.0f}% | Reps: {current_reps}")
        
        cap.release()
        
        # Final results
        final_reps = exercise_instance.reps - initial_reps
        pose_detection_rate = (pose_detected_count / frame_count) * 100
        session_stats = exercise_instance.get_session_stats()
        
        print(f"\n🎉 ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"🎯 Exercise: {exercise_name.upper()}")
        print(f"🔥 Total Reps: {final_reps}")
        print(f"📊 Classification Confidence: {confidence:.1%}")
        print(f"👀 Pose Detection: {pose_detection_rate:.1f}%")
        print(f"📈 Average Accuracy: {session_stats['average_session_accuracy']:.1f}%")
        
        return {
            'detected_exercise': exercise_name,
            'confidence': confidence,
            'total_reps': final_reps,
            'accuracy_score': session_stats['average_session_accuracy'],
            'rep_timestamps': session_stats.get('rep_timestamps', []),
            'pose_detection_rate': pose_detection_rate,
            'session_stats': session_stats,
            'exercise_stats': exercise_stats,
            'frames_analyzed': frame_count,
            'duration_analyzed': duration_to_process
        }


def main():
    """Test the robust classifier"""
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("🧪 Robust Exercise Classifier Test")
        print("Usage: python robust_exercise_classifier.py <video_path> [max_seconds]")
        
        # Show available videos
        videos = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if videos:
            print(f"\n📹 Available videos:")
            for i, video in enumerate(videos, 1):
                print(f"  {i}. {video}")
        return
    
    video_path = sys.argv[1]
    max_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    classifier = RobustExerciseClassifier()
    results = classifier.analyze_video_with_classification(video_path, max_seconds)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Exercise: {results['detected_exercise']}")
        print(f"   Reps: {results['total_reps']}")
        print(f"   Accuracy: {results['average_accuracy']:.1f}%")
        print(f"   Confidence: {results['classification_confidence']:.1%}")


if __name__ == "__main__":
    main()