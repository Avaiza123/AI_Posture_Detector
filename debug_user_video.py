#!/usr/bin/env python3
"""
DEBUG USER VIDEO - Analyze Rep Counting Issues
==============================================

This script will analyze the specific user video and identify rep counting problems.
"""

import cv2
import mediapipe as mp
import numpy as np
from exercise_rules import *
import os
import time

def analyze_user_video():
    """Analyze the user's video to debug rep counting issues"""
    
    video_path = r'c:\Users\E.xone\AppData\Local\Packages\Microsoft.ScreenSketch_8wekyb3d8bbwe\TempState\Recordings\20251105-1801-34.2197432.mp4'
    
    print("🔍 DEBUGGING USER VIDEO REP COUNTING")
    print("=" * 50)
    print(f"📁 Video: {os.path.basename(video_path)}")
    
    if not os.path.exists(video_path):
        print("❌ Video file not found!")
        return
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open video file!")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"📊 Video Properties:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {duration:.2f} seconds")
    print()
    
    # Test different exercises to see which one might work
    exercises_to_test = [
        ("squats", SquatsExercise()),
        ("pushups", PushUpsExercise()),
        ("lunges", LungesExercise()),
        ("bicep_curls", BicepCurlsExercise()),
        ("shoulder_press", ShoulderPressExercise()),
        ("jumping_jacks", JumpingJacksExercise())
    ]
    
    print("🧪 TESTING DIFFERENT EXERCISES:")
    print("-" * 40)
    
    for exercise_name, exercise in exercises_to_test:
        print(f"\n📋 Testing {exercise_name}...")
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Reset exercise state
        exercise.reps = 0
        exercise.rep_accuracies = []
        exercise.rep_timestamps = []
        exercise.session_accuracies = []
        
        frame_num = 0
        rep_events = []
        accuracy_samples = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_num += 1
            timestamp = frame_num / fps
            
            # Skip some frames for faster processing
            if frame_num % 3 != 0:
                continue
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Convert landmarks to the format expected by exercises
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                visibility = [landmark.visibility for landmark in results.pose_landmarks.landmark]
                
                # Test the exercise
                try:
                    rep_increment, feedback, highlights, accuracy = exercise._update_exercise_logic(
                        landmarks, visibility, frame.shape, timestamp
                    )
                    
                    if rep_increment > 0:
                        rep_events.append({
                            'time': timestamp,
                            'frame': frame_num,
                            'rep_num': exercise.reps,
                            'accuracy': accuracy
                        })
                    
                    if accuracy is not None:
                        accuracy_samples.append(accuracy)
                        
                except Exception as e:
                    # Exercise doesn't support new interface, try old one
                    try:
                        rep_increment, feedback, highlights, accuracy = exercise.update(
                            landmarks, visibility, frame.shape
                        )
                        
                        if rep_increment > 0:
                            rep_events.append({
                                'time': timestamp,
                                'frame': frame_num,
                                'rep_num': exercise.reps,
                                'accuracy': accuracy
                            })
                        
                        if accuracy is not None:
                            accuracy_samples.append(accuracy)
                            
                    except Exception as e2:
                        print(f"   ❌ Error testing {exercise_name}: {e2}")
                        break
            
            # Limit processing to first 30 seconds for quick analysis
            if timestamp > 30:
                break
        
        # Report results
        avg_accuracy = np.mean(accuracy_samples) if accuracy_samples else 0
        
        print(f"   📊 Results for {exercise_name}:")
        print(f"      Reps detected: {exercise.reps}")
        print(f"      Rep events: {len(rep_events)}")
        print(f"      Average accuracy: {avg_accuracy:.1f}%")
        
        if rep_events:
            print(f"      Rep timings:")
            for event in rep_events[:5]:  # Show first 5 reps
                print(f"         Rep {event['rep_num']}: {event['time']:.2f}s (acc: {event['accuracy']:.1f}%)")
        
        if exercise.reps > 0:
            print(f"   ✅ {exercise_name} detected {exercise.reps} reps - POTENTIAL MATCH!")
        else:
            print(f"   ⚠ {exercise_name} detected no reps")
    
    cap.release()
    pose.close()
    
    print("\n" + "=" * 50)
    print("🔍 ANALYSIS COMPLETE")
    print("\nRecommendations:")
    print("1. Check which exercise type had the most reps detected")
    print("2. If no exercise detected reps, the movements might be too subtle")
    print("3. Consider adjusting thresholds or improving camera angle")
    print("4. Ensure the person performing exercise is clearly visible")

if __name__ == "__main__":
    analyze_user_video()