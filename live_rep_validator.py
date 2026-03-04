#!/usr/bin/env python3
"""
LIVE CAMERA REP VALIDATION SYSTEM
================================
Enhanced rep detection with form validation for live camera mode only.
Video mode logic remains unchanged.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import deque

class LiveRepValidator:
    """
    Enhanced rep validation system specifically for live camera mode.
    Adds form validation, angle smoothing, and stricter rep detection.
    """
    
    def __init__(self, exercise_name: str, smoothing_window: int = 5):
        self.exercise_name = exercise_name
        self.smoothing_window = smoothing_window
        
        # Angle smoothing buffers
        self.angle_buffers = {}
        
        # Form validation state
        self.is_in_valid_start = False
        self.is_in_valid_down = False
        self.rep_locked = False
        self.last_validation_time = 0.0
        
        # Exercise-specific thresholds with tolerance
        self.form_thresholds = self._get_exercise_thresholds()
        
    def _get_exercise_thresholds(self) -> Dict:
        """Get exercise-specific form validation thresholds"""
        thresholds = {
            "Squats": {
                "start_angle_min": 155.0,  # Standing knee angle minimum
                "start_angle_max": 180.0,  # Standing knee angle maximum
                "down_angle_min": 90.0,    # Squat knee angle minimum
                "down_angle_max": 155.0,   # Squat knee angle maximum
                "tolerance": 10.0,         # ±10° tolerance
                "body_alignment_min": 160.0  # Body line angle minimum
            },
            "Push-ups": {
                "start_angle_min": 140.0,  # Extended elbow angle minimum
                "start_angle_max": 180.0,  # Extended elbow angle maximum  
                "down_angle_min": 45.0,    # Bent elbow angle minimum
                "down_angle_max": 100.0,   # Bent elbow angle maximum
                "tolerance": 10.0,         # ±10° tolerance
                "body_alignment_min": 160.0  # Body line angle minimum
            },
            "Lunges": {
                "start_angle_min": 150.0,  # Standing knee angle minimum
                "start_angle_max": 180.0,  # Standing knee angle maximum
                "down_angle_min": 80.0,    # Lunge knee angle minimum
                "down_angle_max": 120.0,   # Lunge knee angle maximum
                "tolerance": 10.0,         # ±10° tolerance
                "body_alignment_min": 160.0  # Body line angle minimum
            },
            "Jumping jacks": {
                "start_angle_min": 160.0,  # Arms down angle minimum
                "start_angle_max": 180.0,  # Arms down angle maximum
                "down_angle_min": 30.0,    # Arms up angle minimum  
                "down_angle_max": 60.0,    # Arms up angle maximum
                "tolerance": 15.0,         # ±15° tolerance (more dynamic)
                "body_alignment_min": 160.0  # Body line angle minimum
            },
            "Bicep curls": {
                "start_angle_min": 140.0,  # Extended elbow angle minimum
                "start_angle_max": 180.0,  # Extended elbow angle maximum
                "down_angle_min": 30.0,    # Curled elbow angle minimum
                "down_angle_max": 80.0,    # Curled elbow angle maximum
                "tolerance": 10.0,         # ±10° tolerance
                "body_alignment_min": 160.0  # Body line angle minimum
            }
        }
        
        # Default thresholds for other exercises
        default = {
            "start_angle_min": 140.0,
            "start_angle_max": 180.0,
            "down_angle_min": 60.0,
            "down_angle_max": 120.0,
            "tolerance": 10.0,
            "body_alignment_min": 160.0
        }
        
        return thresholds.get(self.exercise_name, default)
    
    def smooth_angle(self, angle_name: str, current_angle: float) -> float:
        """Apply moving average smoothing to reduce jitter"""
        if angle_name not in self.angle_buffers:
            self.angle_buffers[angle_name] = deque(maxlen=self.smoothing_window)
        
        buffer = self.angle_buffers[angle_name]
        buffer.append(current_angle)
        
        # Return smoothed average
        return sum(buffer) / len(buffer)
    
    def is_valid_start_position(self, primary_angle: float, body_alignment: Optional[float] = None) -> bool:
        """Check if user is in valid start position for the exercise"""
        thresholds = self.form_thresholds
        tolerance = thresholds["tolerance"]
        
        # Check primary angle (knee for squats/lunges, elbow for pushups/curls, etc.)
        angle_valid = (thresholds["start_angle_min"] - tolerance <= primary_angle <= 
                      thresholds["start_angle_max"] + tolerance)
        
        # Check body alignment if provided
        alignment_valid = True
        if body_alignment is not None:
            alignment_valid = body_alignment >= (thresholds["body_alignment_min"] - tolerance)
        
        return angle_valid and alignment_valid
    
    def is_valid_down_position(self, primary_angle: float, body_alignment: Optional[float] = None) -> bool:
        """Check if user is in valid down position for the exercise"""
        thresholds = self.form_thresholds
        tolerance = thresholds["tolerance"]
        
        # Check primary angle
        angle_valid = (thresholds["down_angle_min"] - tolerance <= primary_angle <= 
                      thresholds["down_angle_max"] + tolerance)
        
        # Check body alignment if provided
        alignment_valid = True
        if body_alignment is not None:
            alignment_valid = body_alignment >= (thresholds["body_alignment_min"] - tolerance)
        
        return angle_valid and alignment_valid
    
    def within_tolerance(self, value: float, target: float, tolerance: float) -> bool:
        """Check if value is within tolerance of target"""
        return abs(value - target) <= tolerance
    
    def validate_rep_transition(self, primary_angle: float, body_alignment: Optional[float], 
                               frame_timestamp: float) -> Tuple[bool, str, bool]:
        """
        Validate complete rep transition with form checking.
        Returns: (rep_completed, status_message, form_valid)
        """
        
        # Smooth the primary angle to reduce jitter
        smoothed_angle = self.smooth_angle("primary", primary_angle)
        
        # Check current form validity
        start_form_valid = self.is_valid_start_position(smoothed_angle, body_alignment)
        down_form_valid = self.is_valid_down_position(smoothed_angle, body_alignment)
        
        rep_completed = False
        status_message = ""
        overall_form_valid = start_form_valid or down_form_valid
        
        # State machine for rep validation
        if not self.is_in_valid_start and not self.is_in_valid_down:
            # Waiting for valid start position
            if start_form_valid:
                self.is_in_valid_start = True
                self.rep_locked = False
                status_message = f"Ready position ({smoothed_angle:.1f}°) - good form!"
            else:
                status_message = f"Get into start position ({smoothed_angle:.1f}°)"
                
        elif self.is_in_valid_start and not self.is_in_valid_down:
            # In start position, waiting for valid down movement
            if down_form_valid:
                self.is_in_valid_down = True
                self.is_in_valid_start = False
                status_message = f"Good down position ({smoothed_angle:.1f}°)"
            elif not start_form_valid:
                # Lost start position without reaching down - reset
                self.is_in_valid_start = False
                status_message = f"Return to start position ({smoothed_angle:.1f}°)"
            else:
                status_message = f"In start position ({smoothed_angle:.1f}°) - go down"
                
        elif not self.is_in_valid_start and self.is_in_valid_down:
            # In down position, waiting for return to start
            if start_form_valid and not self.rep_locked:
                # Complete rep cycle detected!
                rep_completed = True
                self.rep_locked = True
                self.is_in_valid_start = True
                self.is_in_valid_down = False
                self.last_validation_time = frame_timestamp
                status_message = f"🎉 REP COMPLETED! ({smoothed_angle:.1f}°)"
            elif not down_form_valid and not start_form_valid:
                # Lost form completely - reset
                self.is_in_valid_down = False
                status_message = f"Form lost - return to start ({smoothed_angle:.1f}°)"
            else:
                status_message = f"In down position ({smoothed_angle:.1f}°) - return up"
        
        # Unlock rep after cooldown period
        if self.rep_locked and (frame_timestamp - self.last_validation_time) > 1.0:
            if start_form_valid:
                self.rep_locked = False
        
        return rep_completed, status_message, overall_form_valid
    
    def get_corrective_feedback(self, primary_angle: float, body_alignment: Optional[float]) -> List[str]:
        """Generate corrective feedback based on current form"""
        feedback = []
        thresholds = self.form_thresholds
        tolerance = thresholds["tolerance"]
        
        # Check if angle is too high/low for any valid position
        if primary_angle > thresholds["start_angle_max"] + tolerance:
            if self.exercise_name in ["Squats", "Lunges"]:
                feedback.append("Bend knees slightly - you're too upright")
            elif self.exercise_name in ["Push-ups", "Bicep curls"]:
                feedback.append("Bend arms slightly - too extended")
                
        elif primary_angle < thresholds["down_angle_min"] - tolerance:
            if self.exercise_name in ["Squats", "Lunges"]:
                feedback.append("Stand up more - too low")
            elif self.exercise_name in ["Push-ups", "Bicep curls"]:
                feedback.append("Extend arms more - too bent")
        
        # Check body alignment
        if body_alignment is not None and body_alignment < thresholds["body_alignment_min"] - tolerance:
            feedback.append("Keep body straight - better posture")
        
        return feedback[:2]  # Limit to 2 feedback items
    
    def reset_state(self):
        """Reset validation state (useful when switching exercises)"""
        self.is_in_valid_start = False
        self.is_in_valid_down = False
        self.rep_locked = False
        self.angle_buffers.clear()


def is_live_camera_mode(frame_timestamp: float) -> bool:
    """
    Detect if we're in live camera mode vs video mode.
    Live camera typically has more frequent, real-time timestamps.
    Video mode has consistent frame-based timestamps.
    """
    # Simple heuristic: live camera mode typically has timestamps with millisecond precision
    # and shows more variation in timing
    return True  # For now, assume we can detect this or it will be passed as a parameter


# Helper functions for consistent behavior across exercises
def within_tolerance_range(value: float, min_val: float, max_val: float, tolerance: float = 10.0) -> bool:
    """Check if value is within tolerance range"""
    return (min_val - tolerance) <= value <= (max_val + tolerance)


def calculate_body_alignment(landmarks, visibility) -> Optional[float]:
    """Calculate body alignment angle for form validation"""
    try:
        from exercise_rules import safe_angle, L
        # Use shoulder-hip-ankle line as body alignment indicator
        body_line = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_ANKLE.value, visibility)
        if body_line is None:
            body_line = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_ANKLE.value, visibility)
        return body_line
    except:
        return None


if __name__ == "__main__":
    # Test the validator
    validator = LiveRepValidator("Squats")
    
    print("🧪 Testing LiveRepValidator")
    print("=" * 50)
    
    # Test valid start position
    print("Test 1: Valid start position")
    is_valid = validator.is_valid_start_position(170.0, 175.0)
    print(f"   Standing (170°, body 175°): {is_valid}")
    
    # Test valid down position  
    print("Test 2: Valid down position")
    is_valid = validator.is_valid_down_position(120.0, 165.0)
    print(f"   Squatting (120°, body 165°): {is_valid}")
    
    # Test invalid positions
    print("Test 3: Invalid positions")
    is_valid = validator.is_valid_start_position(100.0, 140.0)
    print(f"   Bad start (100°, body 140°): {is_valid}")
    
    print("\n✅ LiveRepValidator ready for integration!")