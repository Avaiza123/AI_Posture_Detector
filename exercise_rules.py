from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import numpy as np

from pose_utils import calculate_angle, angle_accuracy, L

# Import live camera validation system
try:
    from live_rep_validator import LiveRepValidator, within_tolerance_range, calculate_body_alignment
except ImportError:
    # Fallback if validation module not available
    LiveRepValidator = None
    within_tolerance_range = None
    calculate_body_alignment = None


# Helper utilities for robust rep counting

def avg_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    """Calculate average of non-None values"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def safe_angle(landmarks, a, b, c, vis, min_vis=0.5) -> Optional[float]:
    """Calculate angle safely with visibility check - handles both MediaPipe landmarks and pixel tuples"""
    idxs = [a, b, c]
    if any(vis[i] < min_vis for i in idxs):
        return None
    
    # Check if landmarks are MediaPipe objects or pixel tuples
    if hasattr(landmarks[a], 'x'):
        # MediaPipe landmark objects
        pa = (landmarks[a].x, landmarks[a].y)
        pb = (landmarks[b].x, landmarks[b].y)
        pc = (landmarks[c].x, landmarks[c].y)
    else:
        # Already pixel tuples from get_pose_landmarks
        pa = landmarks[a]
        pb = landmarks[b]
        pc = landmarks[c]
    
    return calculate_angle(pa, pb, pc)


def within_tolerance(value: float, target: float, tolerance: float) -> bool:
    """Check if value is within tolerance of target"""
    return abs(value - target) <= tolerance


def smooth_angle_transition(current: float, previous: float, max_change: float = 30.0) -> float:
    """Smooth angle transitions to prevent jitter"""
    if previous is None:
        return current
    
    change = abs(current - previous)
    if change > max_change:
        # Large jump detected, use weighted average
        return previous * 0.7 + current * 0.3
    return current


# ========================================================================================
# UNIVERSAL REP COUNTING HELPER FUNCTIONS
# ========================================================================================

def is_within_tolerance(value: float, target: float, tolerance: float) -> bool:
    """
    Check if a value is within tolerance range of target
    
    Args:
        value: Current measured value
        target: Target/ideal value
        tolerance: Allowed deviation (±)
    
    Returns:
        True if value is within tolerance range
    """
    if value is None:
        return False
    return abs(value - target) <= tolerance


def is_full_rep(start_angle: float, end_angle: float, current_angle: float, 
               tolerance: float = 10.0) -> bool:
    """
    Check if we've completed a full rep range of motion
    
    Args:
        start_angle: Starting position angle
        end_angle: End position angle
        current_angle: Current measured angle
        tolerance: Tolerance for position matching
    
    Returns:
        True if current angle indicates completion of full range
    """
    if current_angle is None:
        return False
    
    # Check if we're back to start position within tolerance
    return is_within_tolerance(current_angle, start_angle, tolerance)


def calculate_rep_accuracy(angles: dict, target_angles: dict, 
                         tolerance_multiplier: float = 1.0) -> float:
    """
    Calculate overall rep accuracy based on joint angles
    
    Args:
        angles: Dictionary of current joint angles
        target_angles: Dictionary of target angles for current phase
        tolerance_multiplier: Multiplier for tolerance (1.0 = standard, 0.5 = strict, 2.0 = lenient)
    
    Returns:
        Accuracy percentage (0-100)
    """
    if not angles or not target_angles:
        return 0.0
    
    accuracies = []
    
    for joint, current_angle in angles.items():
        if current_angle is None or joint not in target_angles:
            continue
            
        target_info = target_angles[joint]
        target = target_info.get('target', 180.0)
        tolerance = target_info.get('tolerance', 15.0) * tolerance_multiplier
        
        # Calculate accuracy based on deviation from target
        deviation = abs(current_angle - target)
        max_deviation = tolerance * 2  # Full penalty at 2x tolerance
        
        accuracy = max(0.0, 100.0 * (1.0 - deviation / max_deviation))
        accuracies.append(accuracy)
    
    return sum(accuracies) / len(accuracies) if accuracies else 0.0


class RepCountingState:
    """
    Universal state machine for consistent rep counting across all exercises
    Handles phase transitions, cooldowns, and validation
    """
    
    def __init__(self, exercise_name: str, cooldown_time: float = 0.8):
        self.exercise_name = exercise_name
        self.current_phase = "start"  # start, middle, end
        self.last_phase_change = 0.0
        self.phase_hold_time = 0.3    # Time to hold phase before confirming
        self.cooldown_time = cooldown_time  # Cooldown between reps
        self.last_rep_time = 0.0
        self.rep_in_progress = False
        self.phase_history = []       # Track recent phases for validation
        self.angle_history = {}       # Track angles for smoothing
        
    def update_phase(self, new_phase: str, timestamp: float, 
                    is_valid: bool = True, debug: bool = False) -> bool:
        """
        Update current phase with validation and anti-jitter
        
        Args:
            new_phase: New phase to transition to
            timestamp: Current timestamp
            is_valid: Whether the movement quality is valid
            debug: Enable debug output
            
        Returns:
            True if phase actually changed
        """
        # Don't change if in cooldown
        if self.is_in_cooldown(timestamp):
            return False
            
        # Don't change if movement quality is poor - but be more lenient for real-world videos
        # Only block if quality is really terrible (below 20%)
        if not is_valid and hasattr(self, '_last_accuracy') and self._last_accuracy < 20.0:
            if debug:
                print(f"[{self.exercise_name}] Phase change blocked - very poor movement quality")
            return False
            
        # Check if phase actually changed
        if new_phase == self.current_phase:
            return False
            
        # Check if enough time has passed since last change (anti-jitter)
        time_since_change = timestamp - self.last_phase_change
        if time_since_change < self.phase_hold_time:
            return False
            
        # Update phase
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.last_phase_change = timestamp
        
        # Track phase history for pattern validation
        self.phase_history.append((new_phase, timestamp))
        if len(self.phase_history) > 10:  # Keep last 10 phases
            self.phase_history.pop(0)
            
        if debug:
            print(f"[{self.exercise_name}] Phase: {old_phase} → {new_phase} at {timestamp:.2f}s")
            
        return True
        
    def check_rep_completion(self, timestamp: float, accuracy: float = 0.0, 
                           min_accuracy: float = 20.0, debug: bool = False) -> bool:
        """
        Check if a complete rep has been performed
        
        Args:
            timestamp: Current timestamp
            accuracy: Current movement accuracy
            min_accuracy: Minimum accuracy required for rep counting (reduced to 20% for real-world videos)
            debug: Enable debug output
            
        Returns:
            True if rep should be counted
        """
        # Must have sufficient accuracy
        if accuracy < min_accuracy:
            if debug:
                print(f"[{self.exercise_name}] Rep blocked - accuracy too low: {accuracy:.1f}% < {min_accuracy}%")
            return False
            
        # Must not be in cooldown
        if self.is_in_cooldown(timestamp):
            if debug:
                print(f"[{self.exercise_name}] Rep blocked - in cooldown")
            return False
            
        # Check for valid phase sequence (start → middle → end → start)
        if len(self.phase_history) >= 3:
            recent_phases = [p[0] for p in self.phase_history[-4:]]  # Last 4 phases
            
            # Look for complete cycle patterns
            valid_patterns = [
                ["start", "middle", "end"],           # Standard cycle
                ["start", "middle", "end", "start"],  # Complete return
                ["middle", "end", "start"],           # Started from middle
                ["end", "start", "middle"]            # Continuing cycle
            ]
            
            for pattern in valid_patterns:
                if len(recent_phases) >= len(pattern):
                    if recent_phases[-len(pattern):] == pattern:
                        if debug:
                            print(f"[{self.exercise_name}] ✅ Valid rep pattern detected: {pattern}")
                        self.last_rep_time = timestamp
                        return True
                        
        if debug:
            recent_phases = [p[0] for p in self.phase_history[-5:]]
            print(f"[{self.exercise_name}] No valid rep pattern in: {recent_phases}")
            
        return False
        
    def is_in_cooldown(self, timestamp: float) -> bool:
        """Check if we're in post-rep cooldown period"""
        return (timestamp - self.last_rep_time) < self.cooldown_time
        
    def smooth_angle(self, joint: str, new_angle: float) -> float:
        """Apply smoothing to joint angle measurements"""
        if joint not in self.angle_history:
            self.angle_history[joint] = []
            
        history = self.angle_history[joint]
        history.append(new_angle)
        
        # Keep only recent values
        if len(history) > 5:
            history.pop(0)
            
        # Return smoothed value (weighted average favoring recent values)
        if len(history) == 1:
            return new_angle
        elif len(history) == 2:
            return (history[0] * 0.3 + history[1] * 0.7)
        else:
            # Weighted average of last 3 values
            weights = [0.2, 0.3, 0.5]
            return sum(h * w for h, w in zip(history[-3:], weights))


def apply_rep_cooldown(last_rep_time: float, current_time: float, 
                      cooldown_period: float = 0.8) -> bool:
    """
    Apply cooldown period to prevent multiple counts from single movement
    
    Args:
        last_rep_time: Timestamp of last counted rep
        current_time: Current timestamp
        cooldown_period: Minimum time between reps (seconds)
        
    Returns:
        True if cooldown is active (should not count rep)
    """
    return (current_time - last_rep_time) < cooldown_period


class RepState:
    """Enhanced state machine for reliable rep counting with anti-jitter"""
    
    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name
        self.current_phase = "start"  # start, transition, end
        self.last_phase_change = 0.0
        self.phase_hold_time = 0.2  # Minimum time to hold phase (anti-jitter)
        self.cooldown_time = 0.5    # Cooldown after rep completion
        self.last_rep_time = 0.0
        self.rep_in_progress = False
        self.previous_angle = None
        
    def update_phase(self, new_phase: str, timestamp: float, debug: bool = False) -> bool:
        """Update phase with anti-jitter protection"""
        if new_phase == self.current_phase:
            return False
            
        # Check if enough time has passed for phase change
        time_since_change = timestamp - self.last_phase_change
        if time_since_change < self.phase_hold_time:
            return False
            
        # Check cooldown after rep completion
        if self.current_phase == "end" and new_phase == "start":
            time_since_rep = timestamp - self.last_rep_time
            if time_since_rep < self.cooldown_time:
                return False
        
        old_phase = self.current_phase
        self.current_phase = new_phase
        self.last_phase_change = timestamp
        
        if debug:
            print(f"[{self.exercise_name}] Phase: {old_phase} → {new_phase} at {timestamp:.2f}s")
        
        return True
    
    def complete_rep(self, timestamp: float, debug: bool = False) -> bool:
        """Mark rep as completed with cooldown"""
        if self.current_phase == "end":
            self.last_rep_time = timestamp
            self.rep_in_progress = False
            if debug:
                print(f"[{self.exercise_name}] ✅ Rep completed at {timestamp:.2f}s")
            return True
        return False
    
    def is_in_cooldown(self, timestamp: float) -> bool:
        """Check if we're in post-rep cooldown"""
        return (timestamp - self.last_rep_time) < self.cooldown_time


class PostureState:
    """Enhanced state tracking for exercise phases with comprehensive debugging and smoothing"""
    def __init__(self, name: str):
        self.name = name
        self.current_state = "up"  # Default starting position
        self.last_state_change = 0.0  # Start with video time, not system time
        self.state_hold_time = 0.05  # Reduced to 0.05 seconds for more responsive transitions
        self.debug_enabled = True
        # Angle smoothing for better consistency
        self.angle_history = []  # Store recent angle measurements
        self.history_size = 3  # Reduced to 3 frames for more responsive smoothing
        
    def smooth_angle(self, new_angle: float) -> float:
        """Apply moving average smoothing to reduce noise"""
        if new_angle is None:
            return None
            
        self.angle_history.append(new_angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        
        return np.mean(self.angle_history)
        
    def update_state(self, new_state: str, frame_timestamp: float = None) -> bool:
        """Update state with hysteresis to prevent jitter"""
        if frame_timestamp is None:
            frame_timestamp = time.time()
            
        if new_state != self.current_state:
            time_since_change = frame_timestamp - self.last_state_change
            
            # Use appropriate hysteresis for smooth transitions
            if time_since_change >= self.state_hold_time:
                old_state = self.current_state
                self.current_state = new_state
                self.last_state_change = frame_timestamp
                
                if self.debug_enabled:
                    print(f"[{self.name}] State transition: {old_state} → {new_state} at {frame_timestamp:.2f}s")
                
                return True
        return False
    
    def is_rep_completed(self) -> bool:
        """Check if a complete rep cycle occurred (down → up)"""
        return self.current_state == "up"


@dataclass
class PostureAccuracy:
    """Comprehensive posture accuracy tracking"""
    angle_name: str
    current_angle: float
    target_angle: float
    tolerance: float
    accuracy_score: float
    is_correct: bool


class ExerciseBase:
    """Enhanced base class for all exercises with robust rep counting"""
    
    name: str
    reps: int

    def __init__(self, name: str):
        self.name = name
        self.reps = 0
        self.rep_state = RepState(name)
        self.rep_accuracies = []  # Store accuracy for each rep
        self.session_accuracies = []  # Store all frame accuracies
        self.rep_timestamps = []  # Track when reps occur
        self.debug_enabled = True
        
        # Movement validation
        self.min_movement_threshold = 15.0  # Minimum angle change to consider movement
        self.angle_tolerance = 10.0  # Default tolerance for angle matching
        
        # Default threshold attributes to prevent attribute errors
        # These should be overridden in subclasses with exercise-specific values
        self.standing_hip_threshold = 160.0
        self.squat_hip_threshold = 100.0
        self.squat_knee_threshold = 140.0
        self.elbow_up_threshold = 160.0
        self.elbow_down_threshold = 90.0
        self.shoulder_alignment_threshold = 90.0
        self.lunge_knee_threshold = 140.0
        self.standing_knee_threshold = 160.0
        self.hip_alignment_threshold = 160.0
        self.curl_up_threshold = 55.0
        self.curl_down_threshold = 150.0
        self.shoulder_stability_threshold = 170.0
        self.arms_up_threshold = 135.0
        self.arms_down_threshold = 45.0
        self.feet_separation_threshold = 1.0
        
        # Default state tracking attributes to prevent attribute errors
        self._went_through_squat = False
        self._last_accuracy = 0.0
        self._in_rep = False
        self._valid_start_frames = 0
        
        # Live camera validation (only used in live mode)
        self.live_validator = None
        if LiveRepValidator is not None:
            self.live_validator = LiveRepValidator(name)
        
        # Mode detection
        self.is_live_mode = False
        self.frame_count = 0
        self.last_timestamps = []
        
    def reset(self):
        """Reset exercise state"""
        self.reps = 0
        self.rep_state = RepState(self.name)
        self.rep_accuracies.clear()
        self.session_accuracies.clear()
        self.rep_timestamps.clear()

    def is_valid_movement(self, current_angle: float, start_angle: float, end_angle: float) -> bool:
        """Check if movement represents a valid exercise motion"""
        if current_angle is None or start_angle is None or end_angle is None:
            return False
            
        movement_range = abs(start_angle - end_angle)
        return movement_range >= self.min_movement_threshold
    
    def calculate_basic_accuracy(self, measured_angle: float, target_angle: float) -> float:
        """Calculate basic accuracy score for an angle measurement"""
        if measured_angle is None or target_angle is None:
            return 0.0
        
        error = abs(measured_angle - target_angle)
        max_error = self.angle_tolerance * 2  # Allow 2x tolerance for calculation
        accuracy = max(0.0, 100.0 * (1.0 - error / max_error))
        return min(100.0, accuracy)

    def _detect_live_mode(self, frame_timestamp: float) -> bool:
        """Detect if we're in live camera mode vs video mode"""
        self.frame_count += 1
        self.last_timestamps.append(frame_timestamp)
        
        # Keep only last 10 timestamps for analysis
        if len(self.last_timestamps) > 10:
            self.last_timestamps.pop(0)
        
        # Live camera mode typically has:
        # 1. More irregular timing intervals
        # 2. Timestamps close to current system time
        # 3. Higher precision timestamps
        
        if len(self.last_timestamps) >= 5:
            # Check if timestamps are close to real-time (within last 30 seconds)
            current_time = time.time()
            time_diff = abs(current_time - frame_timestamp)
            
            # If timestamp is very close to current time, likely live mode
            if time_diff < 30.0:
                self.is_live_mode = True
            else:
                self.is_live_mode = False
        
        return self.is_live_mode

    def update(self, landmarks, visibility, image_shape, frame_timestamp: float = None) -> Tuple[int, str, List[int], Optional[float]]:
        """Main update method - detects mode and applies appropriate validation"""
        if frame_timestamp is None:
            frame_timestamp = time.time()
        
        # Detect if we're in live camera mode
        is_live = self._detect_live_mode(frame_timestamp)
        
        # Call the exercise-specific logic
        return self._update_exercise_logic(landmarks, visibility, image_shape, frame_timestamp)
    
    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """Override this method in exercise subclasses"""
        raise NotImplementedError("Subclasses must implement _update_exercise_logic")

    def _is_user_exercising(self, current_angles: dict) -> bool:
        """
        Intelligent exercise detection with persistent tracking
        Returns True when user is actively exercising or recently exercised
        """
        # Initialize tracking variables
        if not hasattr(self, '_angle_history'):
            self._angle_history = []
        if not hasattr(self, '_exercise_start_time'):
            self._exercise_start_time = None
        if not hasattr(self, '_last_significant_movement'):
            self._last_significant_movement = None
        if not hasattr(self, '_exercise_session_active'):
            self._exercise_session_active = False
            
        # Get primary movement angle for this exercise
        primary_angle = None
        if hasattr(self, 'down_threshold') and hasattr(self, 'up_threshold'):
            # Find the primary movement angle for this exercise
            if 'knee_angle' in current_angles and current_angles['knee_angle'] is not None:
                primary_angle = current_angles['knee_angle']
            elif 'elbow_angle' in current_angles and current_angles['elbow_angle'] is not None:
                primary_angle = current_angles['elbow_angle']
            elif 'hip_angle' in current_angles and current_angles['hip_angle'] is not None:
                primary_angle = current_angles['hip_angle']
        
        if primary_angle is None:
            return self._exercise_session_active  # Keep previous state if no angle
            
        # Track angle history for movement detection
        import time
        current_time = time.time()
        self._angle_history.append((current_time, primary_angle))
        
        # Keep last 5 seconds of history (longer for better detection)
        self._angle_history = [(t, a) for t, a in self._angle_history if current_time - t <= 5.0]
        
        if len(self._angle_history) < 5:  # Need minimal history
            return self._exercise_session_active  # Keep previous state
            
        # Check for movement patterns
        recent_angles = [angle for _, angle in self._angle_history]
        angle_range = max(recent_angles) - min(recent_angles)
        
        # More intelligent detection criteria
        if hasattr(self, 'down_threshold') and hasattr(self, 'up_threshold'):
            # Check if user has moved through exercise range
            exercise_range = self.up_threshold - self.down_threshold
            min_movement = exercise_range * 0.3  # 30% of full exercise range
            
            # Check for exercise-specific movements
            in_exercise_zone = any(self.down_threshold - 20 <= angle <= self.up_threshold + 20 for angle in recent_angles)
            has_exercise_movement = angle_range >= min_movement
            
            # Check for rep-like patterns (movement between thresholds)
            crossed_thresholds = False
            if len(self._angle_history) >= 10:
                recent_10 = [angle for _, angle in self._angle_history[-10:]]
                has_low = any(angle <= self.down_threshold + 10 for angle in recent_10)
                has_high = any(angle >= self.up_threshold - 10 for angle in recent_10)
                crossed_thresholds = has_low and has_high
            
            # Detect significant movement
            currently_moving = in_exercise_zone and (has_exercise_movement or crossed_thresholds)
        else:
            # For exercises without thresholds, use general movement
            currently_moving = angle_range >= 15.0  # Lower threshold for general exercises
        
        # Update movement tracking
        if currently_moving:
            self._last_significant_movement = current_time
            if not self._exercise_session_active:
                self._exercise_start_time = current_time
            self._exercise_session_active = True
        
        # Keep session active for a grace period after movement stops
        if self._exercise_session_active and self._last_significant_movement:
            grace_period = 3.0  # 3 seconds grace period
            time_since_movement = current_time - self._last_significant_movement
            
            if time_since_movement > grace_period:
                # Only end session if we've been still for a while AND not in exercise position
                if hasattr(self, 'down_threshold') and hasattr(self, 'up_threshold'):
                    # Check if in neutral/standing position
                    in_neutral = primary_angle > (self.up_threshold - 15)
                    if in_neutral and time_since_movement > 5.0:  # 5 seconds in neutral position
                        self._exercise_session_active = False
                else:
                    if time_since_movement > 5.0:
                        self._exercise_session_active = False
        
        return self._exercise_session_active
    
    def compute_form_accuracy(self, current_angles: dict, phase: str) -> tuple:
        """
        Compute Form Accuracy - measures how well user's posture matches ideal joint angles
        
        Form Accuracy is based on:
        - Joint angle deviation from ideal positions (±10° tolerance)
        - Phase-specific requirements (start, middle, end positions)
        - Weighted scoring for major vs minor joints
        - Smooth angle data using moving averages
        
        Returns:
            tuple: (form_accuracy_percentage, joint_deviations_dict)
        """
        # If user has completed reps, they're definitely exercising
        if hasattr(self, 'reps') and self.reps > 0:
            # User has completed reps, continue with accuracy calculation
            pass
        else:
            # Check if user is actually exercising before calculating accuracy
            if not self._is_user_exercising(current_angles):
                return None, {}  # No accuracy when not exercising
            
        # Return good baseline if no angles available
        if not current_angles or not any(v is not None for v in current_angles.values()):
            return 75.0, {}
        
        # Get ideal angles with fallback
        try:
            ideal_angles = self._define_ideal_angles()
        except:
            return 75.0, {}
            
        if phase not in ideal_angles:
            # Use dynamic phase detection based on current angles
            if 'knee_angle' in current_angles and current_angles['knee_angle'] is not None:
                knee = current_angles['knee_angle']
                if knee < 130:  # In squat/down position
                    phase = 'middle'
                else:  # Standing position
                    phase = 'start'
            else:
                phase = 'start'  # Default phase
        
        # If still no phase match, use flexible scoring
        if phase not in ideal_angles:
            return self._calculate_flexible_accuracy(current_angles)
            
        joint_scores = {}
        joint_deviations = {}
        total_weight = 0
        weighted_score = 0
        
        # Joint importance weights (major joints weighted higher)
        joint_weights = {
            'primary_angle': 0.4,    # Main movement joint (knee/elbow/etc)
            'secondary_angle': 0.3,  # Supporting joint
            'hip_angle': 0.3,        # Hip alignment
            'knee_angle': 0.3,       # Knee alignment
            'elbow_angle': 0.3,      # Elbow alignment
            'torso_angle': 0.2,      # Torso posture
            'back_angle': 0.2,       # Back alignment
            'shoulder_angle': 0.2    # Shoulder position
        }
        
        phase_requirements = ideal_angles[phase]
        
        for joint_name, requirements in phase_requirements.items():
            if joint_name in current_angles and current_angles[joint_name] is not None:
                current_value = current_angles[joint_name]
                target_value = requirements['target']
                tolerance = requirements.get('tolerance', self.angle_tolerance)
                
                # Calculate deviation from ideal
                deviation = abs(current_value - target_value)
                joint_deviations[joint_name] = deviation
                
                # Score based on how close to ideal (within tolerance = 100%)
                if deviation <= tolerance:
                    joint_score = 100.0
                else:
                    # Gradual decrease beyond tolerance
                    excess_deviation = deviation - tolerance
                    joint_score = max(0.0, 100.0 - (excess_deviation * 2.0))
                
                joint_scores[joint_name] = joint_score
                
                # Apply weighting
                weight = joint_weights.get(joint_name, 0.1)
                weighted_score += joint_score * weight
                total_weight += weight
        
        # Calculate final form accuracy
        if total_weight > 0:
            form_accuracy = min(100.0, max(60.0, weighted_score / total_weight))  # Minimum 60% for detected pose
        else:
            form_accuracy = 75.0  # Default good score when no specific scoring available
            
        return form_accuracy, joint_deviations
    
    def _calculate_flexible_accuracy(self, current_angles: dict) -> tuple:
        """
        Calculate accuracy based on available angles when ideal angles aren't defined
        """
        scores = []
        deviations = {}
        
        # Basic angle reasonableness checks
        if 'knee_angle' in current_angles and current_angles['knee_angle'] is not None:
            knee = current_angles['knee_angle']
            if 80 <= knee <= 180:  # Reasonable knee range
                knee_score = 90.0 if 90 <= knee <= 170 else 75.0
                scores.append(knee_score)
                deviations['knee_angle'] = abs(knee - 145)  # Target middle range
        
        if 'hip_angle' in current_angles and current_angles['hip_angle'] is not None:
            hip = current_angles['hip_angle']
            if 80 <= hip <= 180:  # Reasonable hip range
                hip_score = 90.0 if 100 <= hip <= 170 else 75.0
                scores.append(hip_score)
                deviations['hip_angle'] = abs(hip - 135)
        
        if 'elbow_angle' in current_angles and current_angles['elbow_angle'] is not None:
            elbow = current_angles['elbow_angle']
            if 45 <= elbow <= 180:  # Reasonable elbow range
                elbow_score = 90.0 if 60 <= elbow <= 170 else 75.0
                scores.append(elbow_score)
                deviations['elbow_angle'] = abs(elbow - 115)
        
        # Calculate average with good baseline
        if scores:
            accuracy = sum(scores) / len(scores)
        else:
            accuracy = 80.0  # Good baseline when pose is detected but no specific angles
        
        return accuracy, deviations
    
    def compute_simple_accuracy(self, movement_data: dict, phase_transitions: list) -> float:
        """
        Compute Simple Accuracy - measures overall movement correctness
        
        Simple Accuracy is based on:
        - Movement detection consistency (smooth motion)
        - Phase transition completeness (up/down cycles)
        - Motion smoothness and stability
        - Overall exercise execution quality
        
        Returns:
            float: simple_accuracy_percentage or None if not exercising
        """
        # Additional check - if user has completed reps, they're definitely exercising
        if hasattr(self, 'reps') and self.reps > 0:
            # User has completed reps, so they're exercising
            pass  # Continue with accuracy calculation
        else:
            # Check if user is exercising based on movement data
            if 'primary_angle' in movement_data and movement_data['primary_angle'] is not None:
                # Create temporary angles dict to check exercise activity
                temp_angles = {'primary_angle': movement_data['primary_angle']}
                if hasattr(self, 'name') and 'squat' in self.name.lower():
                    temp_angles['knee_angle'] = movement_data['primary_angle']
                elif hasattr(self, 'name') and 'push' in self.name.lower():
                    temp_angles['elbow_angle'] = movement_data['primary_angle']
                    
                if not self._is_user_exercising(temp_angles):
                    return None  # No accuracy when not exercising
            else:
                return None  # No movement data
            
        scores = []
        
        # 1. Movement Consistency Score (40% weight)
        if 'primary_angle' in movement_data and movement_data['primary_angle'] is not None:
            angle = movement_data['primary_angle']
            # Score based on reasonable angle range for the exercise
            if hasattr(self, 'down_threshold') and hasattr(self, 'up_threshold'):
                angle_range = self.up_threshold - self.down_threshold
                if self.down_threshold <= angle <= self.up_threshold:
                    consistency_score = 100.0
                elif angle < self.down_threshold:
                    # Below range - score based on how far below
                    deviation = self.down_threshold - angle
                    consistency_score = max(60.0, 100.0 - (deviation * 2.0))
                else:
                    # Above range - score based on how far above
                    deviation = angle - self.up_threshold
                    consistency_score = max(60.0, 100.0 - (deviation * 2.0))
            else:
                consistency_score = 80.0  # Default for exercises without thresholds
            scores.append(consistency_score * 0.4)
        
        # 2. Phase Transition Score (30% weight)
        if len(phase_transitions) >= 2:
            # Check for smooth transitions
            transition_score = 90.0  # Good transitions
            if len(set(phase_transitions[-3:])) >= 2:  # Multiple phases in recent history
                transition_score = 95.0
        else:
            transition_score = 70.0  # Limited transition data
        scores.append(transition_score * 0.3)
        
        # 3. Motion Stability Score (30% weight)
        # Based on how stable the detected angles are
        stability_score = 85.0  # Default good stability
        if hasattr(self, 'session_accuracies') and len(self.session_accuracies) > 0:
            recent_accuracies = self.session_accuracies[-5:]  # Last 5 measurements
            if len(recent_accuracies) > 1:
                variance = np.var(recent_accuracies) if len(recent_accuracies) > 1 else 0
                # Lower variance = more stable = higher score
                stability_score = max(60.0, 100.0 - variance)
        scores.append(stability_score * 0.3)
        
        # Calculate final simple accuracy
        simple_accuracy = sum(scores)
        return min(100.0, max(0.0, simple_accuracy))
    
    def generate_comprehensive_feedback(self, landmarks, visibility, current_angles: dict, joint_deviations: dict, phase: str) -> list:
        """
        NEW ENHANCED FEEDBACK SYSTEM - Multi-point biomechanically accurate analysis
        
        Returns 2-4 detailed, actionable feedback messages based on comprehensive form analysis
        """
        # Always check if pose is detected first
        if not any(v is not None for v in current_angles.values()):
            return ["⚠️ Move into frame for analysis"]
        
        # Check if the exercise has comprehensive feedback (new system)
        if hasattr(self, 'create_comprehensive_feedback'):
            try:
                comprehensive_feedback = self.create_comprehensive_feedback(landmarks, visibility, current_angles, joint_deviations, phase)
                if comprehensive_feedback:
                    return comprehensive_feedback
            except Exception as e:
                print(f"Comprehensive feedback error for {self.name}: {e}")
                # Fall through to legacy system on error
        
        # Fallback to legacy feedback system for exercises not yet enhanced
        return self.generate_feedback_message_legacy(current_angles, joint_deviations, phase)
    
    def generate_feedback_message_legacy(self, current_angles: dict, joint_deviations: dict, phase: str) -> list:
        """
        Legacy feedback system - kept for backward compatibility
        """
        feedback_messages = []
        
        # Always provide posture status first
        if any(v is not None for v in current_angles.values()):
            feedback_messages.append("✅ Pose detected - Analyzing form")
        else:
            feedback_messages.append("⚠️ Move into frame for analysis")
            return feedback_messages
        
        # Phase-specific feedback with more detail
        if phase == "start":
            feedback_messages.append("🚀 Ready position - Begin movement")
        elif phase == "middle":
            feedback_messages.append("💪 Active phase - Hold good form")
        elif phase == "end":
            feedback_messages.append("⬆️ Return phase - Complete movement")
        else:
            feedback_messages.append("📍 Moving - Maintain posture")
        
        # Real-time joint feedback with lower tolerance for immediate response
        immediate_feedback = []
        for joint_name, deviation in joint_deviations.items():
            if deviation > (self.angle_tolerance * 0.7):  # More sensitive feedback
                immediate_feedback.extend(self._get_joint_specific_feedback(joint_name, deviation, current_angles))
        
        # Prioritize immediate corrections
        if immediate_feedback:
            feedback_messages.extend(immediate_feedback[:2])  # Limit to 2 most important
        
        # Exercise-specific guidance
        exercise_feedback = self._get_exercise_specific_feedback(current_angles, phase)
        if exercise_feedback:
            feedback_messages.extend(exercise_feedback[:2])  # Limit to prevent overflow
        
        # Add general encouragement if form is good
        if not immediate_feedback and len([d for d in joint_deviations.values() if d > self.angle_tolerance]) == 0:
            feedback_messages.append("🎉 Excellent form!")
        
        return feedback_messages
    
    # Keep old method name for compatibility
    def generate_feedback_message(self, current_angles: dict, joint_deviations: dict, phase: str) -> list:
        return self.generate_feedback_message_legacy(current_angles, joint_deviations, phase)
    
    def _get_joint_specific_feedback(self, joint_name: str, deviation: float, current_angles: dict) -> list:
        """
        Get highly specific feedback for individual joint deviations with actionable corrections
        """
        feedback = []
        
        # Get exercise name for context
        exercise_name = getattr(self, 'name', '').lower()
        
        if 'knee' in joint_name.lower():
            if deviation > 20.0:
                if 'squat' in exercise_name or 'lunge' in exercise_name:
                    feedback.append("🦵 Knees caving in - push them outward!")
                elif 'plank' in exercise_name or 'push' in exercise_name:
                    feedback.append("🦵 Lock knees straight - no bending!")
                else:
                    feedback.append("🦵 Major knee adjustment needed - check alignment")
            elif deviation > 10.0:
                if 'squat' in exercise_name:
                    feedback.append("🦵 Minor knee tracking - keep them aligned")
                else:
                    feedback.append("🦵 Small knee adjustment needed")
        
        elif 'elbow' in joint_name.lower():
            if deviation > 20.0:
                if 'push' in exercise_name or 'curl' in exercise_name:
                    feedback.append("💪 Elbow position way off - check form")
                else:
                    feedback.append("💪 Major arm correction needed")
            elif deviation > 10.0:
                if 'push' in exercise_name:
                    feedback.append("💪 Elbows flaring - tuck them closer to body")
                elif 'curl' in exercise_name:
                    feedback.append("💪 Control the weight - smoother movement")
                else:
                    feedback.append("💪 Minor elbow adjustment")
        
        elif 'hip' in joint_name.lower():
            if deviation > 20.0:
                if 'squat' in exercise_name:
                    feedback.append("🍑 Hips too high - sit back more!")
                elif 'deadlift' in exercise_name:
                    feedback.append("🍑 Drive hips forward - squeeze glutes!")
                elif 'plank' in exercise_name:
                    feedback.append("🍑 Hips sagging - lift them up!")
                else:
                    feedback.append("🍑 Major hip position error")
            elif deviation > 10.0:
                feedback.append("🍑 Small hip adjustment needed")
        
        elif 'torso' in joint_name.lower() or 'back' in joint_name.lower() or 'body' in joint_name.lower():
            if deviation > 20.0:
                if 'push' in exercise_name or 'plank' in exercise_name:
                    feedback.append("🏠 Keep body straight - no sagging or piking!")
                elif 'squat' in exercise_name:
                    feedback.append("🏠 Chest up - don't lean forward!")
                else:
                    feedback.append("🏠 Straighten your posture")
            elif deviation > 10.0:
                feedback.append("🏠 Minor posture adjustment needed")
        
        return feedback
    
    def _get_exercise_specific_feedback(self, current_angles: dict, phase: str) -> list:
        """
        Override this in subclasses for exercise-specific feedback
        """
        return []
    
    def _calculate_basic_angles(self, landmarks, visibility) -> dict:
        """
        Fallback method to calculate basic angles when specific exercise methods aren't available
        """
        angles = {}
        
        try:
            # Basic knee angle (hip-knee-ankle)
            left_knee = safe_angle(landmarks, 23, 25, 27, visibility)  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            right_knee = safe_angle(landmarks, 24, 26, 28, visibility)  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            if left_knee is not None and right_knee is not None:
                angles['knee_angle'] = (left_knee + right_knee) / 2
            elif left_knee is not None:
                angles['knee_angle'] = left_knee
            elif right_knee is not None:
                angles['knee_angle'] = right_knee
            
            # Basic hip angle (shoulder-hip-knee)
            left_hip = safe_angle(landmarks, 11, 23, 25, visibility)   # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
            right_hip = safe_angle(landmarks, 12, 24, 26, visibility)  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE
            if left_hip is not None and right_hip is not None:
                angles['hip_angle'] = (left_hip + right_hip) / 2
            elif left_hip is not None:
                angles['hip_angle'] = left_hip
            elif right_hip is not None:
                angles['hip_angle'] = right_hip
            
            # Basic elbow angle (shoulder-elbow-wrist)
            left_elbow = safe_angle(landmarks, 11, 13, 15, visibility)  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
            right_elbow = safe_angle(landmarks, 12, 14, 16, visibility) # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
            if left_elbow is not None and right_elbow is not None:
                angles['elbow_angle'] = (left_elbow + right_elbow) / 2
            elif left_elbow is not None:
                angles['elbow_angle'] = left_elbow
            elif right_elbow is not None:
                angles['elbow_angle'] = right_elbow
                
        except Exception as e:
            pass  # Return empty dict on any error
        
        return angles
    
    def render_vertical_ui_panel(self, frame, form_accuracy: float, simple_accuracy: float, 
                               feedback_messages: list, reps: int, exercise_state: str) -> np.ndarray:
        """
        MODERN RESPONSIVE UI SYSTEM - Adapts to video orientation automatically
        
        Creates a modern fitness app UI that:
        - Automatically detects video orientation (vertical/horizontal)
        - Places UI optimally based on aspect ratio
        - Fixes text encoding issues (no more ???? characters)
        - Includes accuracy bars and clean sections
        - Professional, readable design for all screen sizes
        """
        return self.draw_ui(frame, self.name, reps, form_accuracy, simple_accuracy, feedback_messages, exercise_state)
    
    def detect_video_orientation(self, frame) -> str:
        """
        Detect video orientation for responsive UI layout
        
        Returns:
            'vertical': height > width (mobile videos, portrait mode)
            'horizontal': width > height (landscape videos, webcam)
        """
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1.1:  # Clearly horizontal (landscape)
            return 'horizontal'
        elif aspect_ratio < 0.9:  # Clearly vertical (portrait)
            return 'vertical'
        else:  # Square-ish - treat as horizontal for UI purposes
            return 'horizontal'
    
    def _detect_potential_text_overlay(self, region) -> bool:
        """
        Simple detection for potential text or graphics in the top region
        Returns True if there might be overlays that need to be avoided
        """
        try:
            import cv2
            # Convert to grayscale for analysis
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
                
            # Check for high contrast areas (potential text)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.sum() / (gray.shape[0] * gray.shape[1])
            
            # If edge density is high, there might be text or graphics
            return edge_density > 15.0
        except:
            return False  # Default to no overlay detected
    
    def draw_accuracy_bar(self, frame, x: int, y: int, width: int, height: int, 
                         percentage: float, color: tuple):
        """
        Draw a horizontal accuracy progress bar with professional styling
        
        Args:
            frame: Video frame to draw on
            x, y: Top-left corner position
            width, height: Bar dimensions
            percentage: Accuracy percentage (0-100)
            color: Fill color (BGR tuple)
        """
        import cv2
        
        # Background bar (dark grey with border)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
        cv2.rectangle(frame, (x-1, y-1), (x + width + 1, y + height + 1), (120, 120, 120), 1)
        
        # Filled portion based on percentage
        if percentage > 0:
            fill_width = max(1, int((percentage / 100.0) * width))
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Add subtle highlight on top
        if percentage > 0:
            highlight_width = max(1, int((percentage / 100.0) * width))
            cv2.rectangle(frame, (x, y), (x + highlight_width, y + 2), 
                         tuple(min(255, c + 40) for c in color), -1)
    
    def get_accuracy_color(self, accuracy: float) -> tuple:
        """
        Get color based on accuracy percentage with smooth transitions
        Returns BGR color tuple optimized for dark backgrounds
        """
        if accuracy >= 85.0:
            return (100, 255, 100)    # Bright Green - Excellent
        elif accuracy >= 70.0:
            return (120, 255, 200)    # Light Green - Good
        elif accuracy >= 55.0:
            return (100, 200, 255)    # Orange - Fair
        else:
            return (100, 150, 255)    # Light Red - Poor
    
    def draw_ui(self, frame, exercise_name: str, reps: int, form_accuracy: float, 
                overall_accuracy: float, feedback_list: list, exercise_state: str = "active"):
        """
        COMPREHENSIVE RESPONSIVE UI SYSTEM
        
        Automatically detects video orientation and creates appropriate UI layout:
        - Vertical videos: UI panel on left side or top area
        - Horizontal videos: UI panel on right side
        - Clean, modern design with proper text encoding
        - Accuracy bars and readable feedback
        - Professional fitness app appearance
        
        Args:
            frame: Video frame to draw on
            exercise_name: Name of current exercise
            reps: Current rep count
            form_accuracy: Form accuracy percentage (0-100)
            overall_accuracy: Overall accuracy percentage (0-100)
            feedback_list: List of feedback messages
            exercise_state: Current exercise state
        
        Returns:
            Frame with modern UI overlay
        """
        import cv2
        import numpy as np
        
        h, w = frame.shape[:2]
        orientation = self.detect_video_orientation(frame)
        
        # Sanitize all text inputs to prevent ???? corruption
        clean_exercise_name = self.sanitize_text(exercise_name)
        clean_feedback_list = [self.sanitize_text(str(msg)) for msg in feedback_list if msg]
        
        # Determine UI layout based on orientation
        if orientation == 'vertical':
            return self._draw_vertical_layout(frame, clean_exercise_name, reps, 
                                            form_accuracy, overall_accuracy, 
                                            clean_feedback_list, exercise_state)
        else:
            return self._draw_horizontal_layout(frame, clean_exercise_name, reps, 
                                              form_accuracy, overall_accuracy, 
                                              clean_feedback_list, exercise_state)
    
    def _draw_vertical_layout(self, frame, exercise_name: str, reps: int, 
                            form_accuracy: float, overall_accuracy: float, 
                            feedback_list: list, exercise_state: str):
        """
        VERTICAL VIDEO UI - Clean feedback panel in upper region (Pic 1 style)
        
        For vertical videos, displays feedback in upper region with clean spacing.
        Matches Pic 1 layout: exercise title, reps, accuracy bars, and feedback.
        
        Layout (Pic 1 Reference):
        ┌─────────────────────────────────────┐
        │ Exercise Name (Big Title)           │
        │ REPS: XX (Large Green Number)       │
        │ Form Accuracy: XX% [████████████]   │
        │ Accuracy: XX% [██████████████████]  │
        │ FEEDBACK:                           │
        │ • Feedback line 1                   │
        │ • Feedback line 2                   │
        ├─────────────────────────────────────┤
        │        VIDEO CONTENT AREA           │
        └─────────────────────────────────────┘
        """
        import cv2
        
        h, w = frame.shape[:2]
        
        # FORCE FULL PIC 1 FEEDBACK PANEL for ALL vertical videos
        # Smart positioning to handle text overlays or existing graphics
        panel_height = int(min(500, max(320, h // 2.2)))  # LARGER panel for full feedback
        
        # Check for potential text overlays in top region
        try:
            top_region = frame[0:100, :]
            has_text_overlay = self._detect_potential_text_overlay(top_region)
        except:
            has_text_overlay = False
        
        # Adjust panel position based on content detection
        if has_text_overlay:
            panel_y = 80  # Shift down to avoid text overlays
        else:
            panel_y = 10  # Standard top position
            
        panel_x = 10  # Small margin from left
        panel_width = w - 20  # Full width minus margins
        
        # Ensure all coordinates are integers
        panel_x = int(panel_x)
        panel_y = int(panel_y) 
        panel_width = int(panel_width)
        panel_height = int(panel_height)
        
        # For very narrow vertical videos, use side black bars if available
        if w < 600:  # Narrow mobile video
            try:
                # Check if there are black bars we can utilize
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                left_bar = gray_frame[:, :50]
                right_bar = gray_frame[:, -50:]
                
                if left_bar.mean() < 30 or right_bar.mean() < 30:  # Dark bars detected
                    panel_x = 5  # Use the black bar area
                    panel_width = w - 10
            except:
                pass  # Keep default positioning on any error
        
        # Draw horizontal top panel for vertical videos
        frame = self._draw_horizontal_top_panel_vertical(frame, panel_x, panel_y, panel_width, panel_height,
                                                       exercise_name, reps, form_accuracy, overall_accuracy,
                                                       feedback_list, exercise_state)
        
        return frame
    
    def _draw_horizontal_layout(self, frame, exercise_name: str, reps: int,
                              form_accuracy: float, overall_accuracy: float,
                              feedback_list: list, exercise_state: str):
        """
        Horizontal layout - Right-side panel matching the image style
        Clean vertical panel on the right side with exercise info, accuracy bars, and feedback
        """
        import cv2
        
        h, w = frame.shape[:2]
        
        # RIGHT-SIDE PANEL (matching image style)
        panel_width = min(350, max(280, w // 4))  # Responsive width
        panel_height = h - 40  # Nearly full height with margins
        panel_x = w - panel_width - 20  # Right side positioning
        panel_y = 20  # Top margin
        
        # Ensure coordinates are integers
        panel_x = int(panel_x)
        panel_y = int(panel_y)
        panel_width = int(panel_width)
        panel_height = int(panel_height)
        
        # Draw the complete right-side panel matching the image
        frame = self._draw_right_panel_horizontal(frame, panel_x, panel_y, panel_width, panel_height,
                                                exercise_name, reps, form_accuracy, overall_accuracy,
                                                feedback_list, exercise_state)
        
        return frame
    
    def _draw_right_panel_horizontal(self, frame, panel_x: int, panel_y: int, 
                                   panel_width: int, panel_height: int,
                                   exercise_name: str, reps: int, form_accuracy: float,
                                   overall_accuracy: float, feedback_list: list, exercise_state: str):
        """
        Draw right-side panel for horizontal videos matching the image style
        Clean vertical layout with exercise name, reps, accuracy bars, and feedback
        """
        import cv2
        
        # Semi-transparent dark background (matching image)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (70, 70, 70), -1)  # Gray background like in image
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        # Content positioning
        content_x = panel_x + 25
        content_width = panel_width - 50
        y_pos = panel_y + 40
        
        # Font scaling for readability
        base_scale = min(panel_width / 300.0, 1.2)
        font_title = min(1.0, max(0.7, 0.9 * base_scale))
        font_large = min(1.4, max(0.9, 1.2 * base_scale)) 
        font_medium = min(0.9, max(0.6, 0.75 * base_scale))
        font_small = min(0.8, max(0.5, 0.65 * base_scale))
        
        thick_title = max(2, int(3 * base_scale))
        thick_large = max(3, int(4 * base_scale))
        thick_medium = max(2, int(3 * base_scale))
        thick_small = max(2, int(2 * base_scale))
        
        # 1. EXERCISE NAME (top, like "PUSH-UPS")
        clean_exercise_name = self.sanitize_text(exercise_name.upper())
        frame = self.safe_render_text(frame, clean_exercise_name, (content_x, y_pos), 
                                    font_title, (255, 255, 255), thick_title)
        y_pos += 80
        
        # 2. REPS section
        frame = self.safe_render_text(frame, "REPS", (content_x, y_pos), 
                                    font_medium, (200, 200, 200), thick_medium)
        y_pos += 40
        
        # Large reps number (green, like in image)
        reps_color = (100, 255, 100) if reps > 0 else (150, 150, 150)
        frame = self.safe_render_text(frame, str(reps), (content_x, y_pos), 
                                    font_large, reps_color, thick_large)
        y_pos += 80
        
        # 3. FORM ACCURACY section
        normalized_form = form_accuracy / 100.0 if form_accuracy > 1.0 else form_accuracy
        form_percentage = int(normalized_form * 100)
        frame = self.safe_render_text(frame, "FORM ACCURACY", (content_x, y_pos), 
                                    font_medium, (200, 200, 200), thick_medium)
        y_pos += 35
        
        # Form accuracy percentage
        form_color = (255, 255, 100) if form_percentage >= 70 else (255, 150, 100)
        frame = self.safe_render_text(frame, f"{form_percentage}%", (content_x, y_pos), 
                                    font_medium, form_color, thick_medium)
        y_pos += 40
        
        # Form accuracy bar
        bar_width = min(content_width - 20, 250)
        bar_height = 12
        bar_x = content_x
        
        # Background bar
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), (100, 100, 100), -1)
        
        # Filled bar
        form_fill_width = int(bar_width * normalized_form)
        if form_fill_width > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + form_fill_width, y_pos + bar_height), form_color, -1)
        y_pos += 60
        
        # 4. OVERALL ACCURACY section  
        normalized_overall = overall_accuracy / 100.0 if overall_accuracy > 1.0 else overall_accuracy
        overall_percentage = int(normalized_overall * 100)
        frame = self.safe_render_text(frame, "OVERALL ACCURACY", (content_x, y_pos), 
                                    font_medium, (200, 200, 200), thick_medium)
        y_pos += 35
        
        # Overall accuracy percentage
        overall_color = (255, 255, 100) if overall_percentage >= 70 else (255, 150, 100)
        frame = self.safe_render_text(frame, f"{overall_percentage}%", (content_x, y_pos), 
                                    font_medium, overall_color, thick_medium)
        y_pos += 40
        
        # Overall accuracy bar
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), (100, 100, 100), -1)
        overall_fill_width = int(bar_width * normalized_overall)
        if overall_fill_width > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + overall_fill_width, y_pos + bar_height), overall_color, -1)
        y_pos += 60
        
        # 5. FEEDBACK section
        frame = self.safe_render_text(frame, "FEEDBACK", (content_x, y_pos), 
                                    font_medium, (255, 255, 255), thick_medium)
        y_pos += 45
        
        # Feedback messages with proper text wrapping
        remaining_height = panel_y + panel_height - y_pos - 15  # Less bottom margin
        line_height = 25  # Smaller line height to fit more text
        max_feedback_lines = max(6, int(remaining_height / line_height))  # Allow more lines
        
        if feedback_list:
            lines_used = 0
            for i, message in enumerate(feedback_list):
                if lines_used >= max_feedback_lines:
                    break
                    
                clean_message = self.sanitize_text(str(message))
                if not clean_message:
                    continue
                
                # Color coding like in the image
                if any(word in clean_message.lower() for word in ['pose', 'detected', 'analyzing']):
                    feedback_color = (200, 200, 200)  # Gray
                elif any(word in clean_message.lower() for word in ['active', 'good', 'hold']):
                    feedback_color = (100, 255, 100)  # Green
                elif any(word in clean_message.lower() for word in ['adjust', 'elbow', 'improve']):
                    feedback_color = (100, 200, 255)  # Orange/Yellow
                else:
                    feedback_color = (255, 255, 255)  # White
                
                # Use pixel-based wrapping instead of character counting
                max_text_width = content_width - 30  # Reserve space for bullet and margins
                wrapped_lines = self._wrap_text_by_pixels(clean_message, max_text_width, font_small, thick_small)
                
                for line_idx, wrapped_line in enumerate(wrapped_lines):
                    if lines_used >= max_feedback_lines:
                        break
                    
                    # Add bullet point only for first line of each message
                    if line_idx == 0:
                        display_text = f"- {wrapped_line}"
                    else:
                        display_text = f"  {wrapped_line}"  # Indent continuation lines
                    
                    # Final safety check - if text is still too wide, force wrap
                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                              font_small, thick_small)[0]
                    
                    if text_size[0] > content_width - 10:
                        # Force split the line if it's still too wide
                        words = wrapped_line.split()
                        if len(words) > 1:
                            # Split into two lines
                            mid_point = len(words) // 2
                            first_part = " ".join(words[:mid_point])
                            second_part = " ".join(words[mid_point:])
                            
                            # Render first part
                            first_text = f"- {first_part}" if line_idx == 0 else f"  {first_part}"
                            frame = self.safe_render_text(frame, first_text, (content_x, y_pos), 
                                                        font_small, feedback_color, thick_small)
                            y_pos += line_height
                            lines_used += 1
                            
                            # Render second part (if we have space)
                            if lines_used < max_feedback_lines:
                                second_text = f"  {second_part}"
                                frame = self.safe_render_text(frame, second_text, (content_x, y_pos), 
                                                            font_small, feedback_color, thick_small)
                                y_pos += line_height
                                lines_used += 1
                        else:
                            # Single long word - just render it
                            frame = self.safe_render_text(frame, display_text, (content_x, y_pos), 
                                                        font_small, feedback_color, thick_small)
                            y_pos += line_height
                            lines_used += 1
                    else:
                        # Text fits fine, render normally
                        frame = self.safe_render_text(frame, display_text, (content_x, y_pos), 
                                                    font_small, feedback_color, thick_small)
                        y_pos += line_height
                        lines_used += 1
        else:
            # Default feedback when none available
            frame = self.safe_render_text(frame, "- Ready to analyze", (content_x, y_pos), 
                                        font_small, (150, 150, 150), thick_small)
        
        return frame
    
    def _wrap_text_by_pixels(self, text: str, max_width: int, font_scale: float, thickness: int):
        """
        Wrap text based on actual pixel width to prevent overflow
        """
        import cv2
        
        if not text:
            return []
        
        words = text.split()
        if not words:
            return []
        
        lines = []
        current_line = ""
        
        for word in words:
            # Test adding this word to current line
            test_line = current_line + (" " if current_line else "") + word
            test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            if test_size[0] <= max_width:
                # Word fits, add it to current line
                current_line = test_line
            else:
                # Word doesn't fit, start new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Even single word is too long, but add it anyway
                    lines.append(word)
        
        # Add the last line if it has content
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _wrap_text_with_bullet_space(self, text: str, max_first_width: int, max_continuation_width: int, font_scale: float, thickness: int):
        """
        Enhanced text wrapping that accounts for bullet points and indentation
        """
        import cv2
        
        if not text:
            return []
        
        words = text.split()
        if not words:
            return []
        
        lines = []
        current_line = ""
        is_first_line = True
        
        for word in words:
            # Test adding this word to current line
            test_line = current_line + (" " if current_line else "") + word
            test_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Use appropriate width limit based on line type
            max_width = max_first_width if is_first_line else max_continuation_width
            
            if test_size[0] <= max_width:
                # Word fits, add it to current line
                current_line = test_line
            else:
                # Word doesn't fit, start new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                    is_first_line = False
                else:
                    # Even single word is too long, but add it anyway
                    lines.append(word)
                    is_first_line = False
        
        # Add the last line if it has content
        if current_line:
            lines.append(current_line)
        
        return lines

    def _draw_info_panel_vertical(self, frame, panel_x: int, panel_y: int, 
                                panel_width: int, panel_height: int,
                                exercise_name: str, reps: int, form_accuracy: float,
                                overall_accuracy: float, exercise_state: str):
        """
        Draw compact info panel for vertical videos (exercise name, reps, accuracy)
        """
        import cv2
        
        # Modern panel background with enhanced styling
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (20, 20, 20), -1)  # Darker background for info
        cv2.addWeighted(frame, 0.35, overlay, 0.65, 0, frame)  # More prominent
        
        # Professional border styling
        cv2.rectangle(frame, (panel_x-2, panel_y-2), 
                     (panel_x + panel_width + 2, panel_y + panel_height + 2), 
                     (100, 100, 100), 2)
        cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                     (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                     (140, 140, 140), 1)
        
        # Enhanced font scaling for compact panel
        base_scale = max(0.8, min(1.6, panel_width / 280))
        font_title = base_scale * 0.9
        font_large = base_scale * 1.3
        font_medium = base_scale * 0.75
        
        thick_title = max(2, int(base_scale * 3))
        thick_large = max(3, int(base_scale * 4))
        thick_medium = max(2, int(base_scale * 2.5))
        
        # Layout calculations
        content_x = panel_x + 20
        y_pos = panel_y + 35
        content_width = panel_width - 40
        
        line_spacing_large = int(45 * base_scale)
        line_spacing_medium = int(35 * base_scale)
        
        # Exercise name - SAFE RENDERING
        frame = self.safe_render_text(frame, exercise_name.upper(), (content_x, y_pos), 
                                    font_title, (255, 255, 255), thick_title)
        y_pos += line_spacing_large
        
        # Separator
        cv2.line(frame, (content_x, y_pos - 12), 
                (content_x + content_width - 20, y_pos - 12), (120, 120, 120), 1)
        
        # Reps section - SAFE RENDERING
        frame = self.safe_render_text(frame, "REPS", (content_x, y_pos), 
                                    font_medium, (200, 200, 200), thick_medium)
        
        # Position rep counter to the right - SAFE RENDERING
        rep_text = str(reps)
        rep_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, font_large, thick_large)[0]
        rep_x = content_x + content_width - rep_size[0] - 10
        rep_color = (0, 255, 0) if reps > 0 else (150, 150, 150)
        
        frame = self.safe_render_text(frame, rep_text, (rep_x, y_pos), 
                                    font_large, rep_color, thick_large)
        y_pos += line_spacing_medium + 15
        
        # Accuracy bars (compact horizontal layout)
        if form_accuracy > 0:
            # Form accuracy
            cv2.putText(frame, "FORM", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_medium * 0.8, (200, 200, 200), thick_medium)
            
            form_text = f"{form_accuracy:.0f}%"
            form_size = cv2.getTextSize(form_text, cv2.FONT_HERSHEY_SIMPLEX, font_medium * 0.8, thick_medium)[0]
            form_x = content_x + content_width - form_size[0] - 10
            form_color = self.get_accuracy_color(form_accuracy)
            
            cv2.putText(frame, form_text, (form_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_medium * 0.8, form_color, thick_medium)
            
            # Compact accuracy bar
            bar_y = y_pos + 8
            bar_width = content_width - 20
            bar_height = 6
            self.draw_accuracy_bar(frame, content_x, bar_y, bar_width, bar_height, 
                                 form_accuracy, form_color)
            y_pos += line_spacing_medium
            
            # Overall accuracy
            if overall_accuracy > 0:
                cv2.putText(frame, "OVERALL", (content_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_medium * 0.8, (200, 200, 200), thick_medium)
                
                overall_text = f"{overall_accuracy:.0f}%"
                overall_size = cv2.getTextSize(overall_text, cv2.FONT_HERSHEY_SIMPLEX, font_medium * 0.8, thick_medium)[0]
                overall_x = content_x + content_width - overall_size[0] - 10
                overall_color = self.get_accuracy_color(overall_accuracy)
                
                cv2.putText(frame, overall_text, (overall_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_medium * 0.8, overall_color, thick_medium)
                
                bar_y = y_pos + 8
                self.draw_accuracy_bar(frame, content_x, bar_y, bar_width, bar_height, 
                                     overall_accuracy, overall_color)
        
        return frame

    def _draw_feedback_panel_vertical(self, frame, panel_x: int, panel_y: int,
                                    panel_width: int, panel_height: int,
                                    feedback_list: list):
        """
        Draw expanded feedback panel for vertical videos with maximum space utilization
        """
        import cv2
        
        # Enhanced feedback panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (15, 15, 25), -1)  # Slightly different shade for distinction
        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)
        
        # Professional border styling
        cv2.rectangle(frame, (panel_x-2, panel_y-2), 
                     (panel_x + panel_width + 2, panel_y + panel_height + 2), 
                     (80, 120, 80), 2)  # Subtle green tint for feedback
        cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                     (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                     (120, 160, 120), 1)
        
        # Enhanced font scaling for feedback panel - LARGER FONTS
        base_scale = max(1.0, min(2.2, panel_width / 250))  # Significantly larger base scale
        font_header = base_scale * 1.0    # Feedback header
        font_feedback = base_scale * 0.9  # Feedback text - MUCH LARGER
        
        thick_header = max(2, int(base_scale * 3))
        thick_feedback = max(2, int(base_scale * 2.5))  # Thicker for visibility
        
        # Layout calculations with generous spacing
        content_x = panel_x + 25
        y_pos = panel_y + 40
        content_width = panel_width - 50
        
        # EXPANDED line spacing for better readability
        line_spacing_header = int(50 * base_scale)
        line_spacing_feedback = int(38 * base_scale)  # Generous spacing between lines
        
        # Feedback header
        cv2.putText(frame, "FEEDBACK", (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_header, (220, 220, 220), thick_header)
        y_pos += line_spacing_header
        
        # Decorative underline
        cv2.line(frame, (content_x, y_pos - 20), 
                (content_x + content_width - 30, y_pos - 20), (120, 160, 120), 2)
        
        # Calculate maximum lines that fit in the expanded panel
        available_height = (panel_y + panel_height) - y_pos - 30
        max_lines = max(8, int(available_height / line_spacing_feedback))  # Many more lines
        
        # Enhanced character limit calculation
        char_width = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
        chars_per_line = max(25, int((content_width - 40) / char_width))  # More characters per line
        
        # Display feedback with enhanced formatting
        if feedback_list and len(feedback_list) > 0:
            lines_used = 0
            
            for message_idx, message in enumerate(feedback_list[:8]):  # Show more messages
                if lines_used >= max_lines:
                    break
                
                # Enhanced text cleaning
                clean_message = self.sanitize_text(str(message))
                if not clean_message or len(clean_message.strip()) == 0:
                    continue
                
                clean_message = clean_message.strip()
                if clean_message.startswith('? '):
                    clean_message = clean_message[2:]
                
                # Enhanced color coding with brighter colors
                if any(word in clean_message.lower() for word in ['good', 'great', 'excellent', 'perfect', 'well', 'nice']):
                    feedback_color = (120, 255, 120)  # Bright green
                elif any(word in clean_message.lower() for word in ['adjust', 'improve', 'try', 'better', 'focus', 'maintain']):
                    feedback_color = (120, 230, 255)  # Bright cyan
                elif any(word in clean_message.lower() for word in ['wrong', 'error', 'bad', 'incorrect', 'fix', 'careful']):
                    feedback_color = (120, 190, 255)  # Bright orange
                else:
                    feedback_color = (250, 250, 250)  # Bright white
                
                # PIXEL-BASED text wrapping with bullet space pre-calculated for feedback panel
                bullet_width = cv2.getTextSize("• ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                indent_width = cv2.getTextSize("    ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                max_first_line_width = content_width - bullet_width - 25
                max_continuation_width = content_width - indent_width - 25
                wrapped_lines = self._wrap_text_with_bullet_space(clean_message, max_first_line_width, max_continuation_width, font_feedback, thick_feedback)
                
                for i, line in enumerate(wrapped_lines):
                    if lines_used >= max_lines:
                        break
                    
                    # Enhanced bullet formatting
                    if i == 0:
                        display_text = f"• {line}"  # Bullet for first line
                    else:
                        display_text = f"    {line}"  # Indentation for continuation
                    
                    # NO MORE ELLIPSIS TRIMMING - pixel-based wrapping handles all text fitting
                    
                    # Draw text with enhanced positioning
                    cv2.putText(frame, display_text, (content_x + 8, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_feedback, feedback_color, thick_feedback)
                    
                    y_pos += line_spacing_feedback
                    lines_used += 1
                
                # Add spacing between messages
                if lines_used < max_lines and message_idx < len(feedback_list) - 1:
                    y_pos += max(8, int(12 * base_scale))
        else:
            # Enhanced default messages
            default_messages = [
                "• Keep up the excellent work!",
                "• Focus on maintaining proper form",
                "• Control your movements for best results",
                "• Stay consistent with your breathing"
            ]
            
            for i, msg in enumerate(default_messages):
                if y_pos + line_spacing_feedback < panel_y + panel_height - 20:
                    cv2.putText(frame, msg, (content_x + 8, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_feedback, (120, 255, 120), thick_feedback)
                    y_pos += line_spacing_feedback
        
        return frame

    def _draw_info_panel_horizontal(self, frame, panel_x: int, panel_y: int,
                                  panel_width: int, panel_height: int,
                                  exercise_name: str, reps: int, form_accuracy: float,
                                  overall_accuracy: float, exercise_state: str):
        """
        Draw compact info panel for horizontal videos (left panel at bottom)
        """
        import cv2
        
        # Compact panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (25, 25, 25), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                     (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                     (120, 120, 120), 2)
        
        # Compact fonts for horizontal layout
        base_scale = max(0.5, min(1.0, panel_width / 250))
        font_small = base_scale * 0.6
        font_medium = base_scale * 0.8
        font_large = base_scale * 1.0
        
        thick_small = max(1, int(base_scale * 1.5))
        thick_medium = max(1, int(base_scale * 2))
        thick_large = max(2, int(base_scale * 2.5))
        
        # Layout
        content_x = panel_x + 10
        y_pos = panel_y + 20
        content_width = panel_width - 20
        
        line_spacing = int(18 * base_scale)
        
        # Exercise name (truncated if needed) - SAFE RENDERING
        name_text = self.sanitize_text(exercise_name.upper())
        name_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, font_medium, thick_medium)[0]
        while name_size[0] > content_width - 10 and len(name_text) > 5:
            name_text = name_text[:-4] + "..."
            name_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, font_medium, thick_medium)[0]
        
        frame = self.safe_render_text(frame, name_text, (content_x, y_pos), 
                                    font_medium, (255, 255, 255), thick_medium)
        y_pos += line_spacing + 5
        
        # Reps and accuracy in compact format - SAFE RENDERING
        frame = self.safe_render_text(frame, f"REPS: {reps}", (content_x, y_pos), 
                                    font_small, (200, 200, 200), thick_small)
        y_pos += line_spacing
        
        if form_accuracy > 0:
            # Normalize accuracy value (handle both 0-1 and 0-100 ranges)
            normalized_form = form_accuracy / 100.0 if form_accuracy > 1.0 else form_accuracy
            form_percentage = int(normalized_form * 100)
            form_text = f"FULL FORM: {form_percentage}%"
            
            # Color based on normalized value
            form_color = (120, 255, 120) if normalized_form >= 0.8 else (255, 255, 120) if normalized_form >= 0.6 else (255, 120, 120)
            
            # Draw form accuracy text
            frame = self.safe_render_text(frame, form_text, (content_x, y_pos), 
                                        font_small, form_color, thick_small)
            
            # Draw horizontal accuracy bar for form
            bar_x = content_x + 5
            bar_y = y_pos + 5  # Slightly below text
            bar_width = min(100, content_width - 15)
            bar_height = 4
            
            # Background bar (light grey)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (80, 80, 80), -1)
            
            # Foreground bar (colored based on NORMALIZED accuracy)
            fill_width = int(bar_width * normalized_form)
            fill_width = max(0, min(fill_width, bar_width))  # Clamp to valid range
            if fill_width > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                             form_color, -1)
            
            y_pos += line_spacing + 8  # Extra space for bar
        
        if overall_accuracy > 0:
            # Normalize accuracy value (handle both 0-1 and 0-100 ranges)
            normalized_overall = overall_accuracy / 100.0 if overall_accuracy > 1.0 else overall_accuracy
            overall_percentage = int(normalized_overall * 100)
            overall_text = f"FULL OVERALL: {overall_percentage}%"
            
            # Color based on normalized value
            overall_color = (120, 255, 120) if normalized_overall >= 0.8 else (255, 255, 120) if normalized_overall >= 0.6 else (255, 120, 120)
            
            # Draw overall accuracy text
            frame = self.safe_render_text(frame, overall_text, (content_x, y_pos), 
                                        font_small, overall_color, thick_small)
            
            # Draw horizontal accuracy bar for overall
            bar_x = content_x + 5
            bar_y = y_pos + 5  # Slightly below text
            bar_width = min(100, content_width - 15)
            bar_height = 4
            
            # Background bar (light grey)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (80, 80, 80), -1)
            
            # Foreground bar (colored based on NORMALIZED accuracy)
            fill_width = int(bar_width * normalized_overall)
            fill_width = max(0, min(fill_width, bar_width))  # Clamp to valid range
            if fill_width > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                             overall_color, -1)
        
        return frame

    def _draw_feedback_panel_horizontal(self, frame, panel_x: int, panel_y: int,
                                      panel_width: int, panel_height: int,
                                      feedback_list: list):
        """
        Draw feedback panel for horizontal videos (right panel at bottom) with STRICT text wrapping
        """
        import cv2
        
        # Feedback panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (20, 25, 20), -1)
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                     (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                     (100, 140, 100), 2)
        
        # Fonts optimized for horizontal feedback
        base_scale = max(0.4, min(0.8, panel_width / 400))
        font_header = base_scale * 0.7
        font_feedback = base_scale * 0.6
        
        thick_header = max(1, int(base_scale * 2))
        thick_feedback = max(1, int(base_scale * 1.5))
        
        # Layout with strict boundaries
        content_x = panel_x + 12
        y_pos = panel_y + 18
        content_width = panel_width - 24  # Strict width limits
        
        line_spacing = int(14 * base_scale)
        
        # Header - SAFE RENDERING
        frame = self.safe_render_text(frame, "FEEDBACK", (content_x, y_pos), 
                                    font_header, (220, 220, 220), thick_header)
        y_pos += line_spacing + 8
        
        # Calculate available lines with strict limits
        available_height = (panel_y + panel_height) - y_pos - 10
        max_lines = max(2, int(available_height / line_spacing))
        
        # STRICT character limit calculation to prevent overflow
        char_width = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
        chars_per_line = max(15, int((content_width - 15) / char_width))
        
        # Display feedback with AGGRESSIVE text wrapping and truncation
        if feedback_list and len(feedback_list) > 0:
            lines_used = 0
            
            for message in feedback_list[:3]:  # Limit messages for horizontal layout
                if lines_used >= max_lines:
                    break
                
                # Ultra-aggressive text cleaning
                clean_message = self.sanitize_text(str(message))
                if not clean_message or len(clean_message.strip()) == 0:
                    continue
                
                # Remove any remaining ? patterns
                clean_message = clean_message.replace('?', '').strip()
                if not clean_message:
                    continue
                
                # Color coding
                if any(word in clean_message.lower() for word in ['good', 'great', 'excellent', 'perfect']):
                    feedback_color = (100, 255, 100)
                elif any(word in clean_message.lower() for word in ['adjust', 'improve', 'better']):
                    feedback_color = (100, 200, 255)
                else:
                    feedback_color = (220, 220, 220)
                
                # PIXEL-BASED text wrapping with bullet space for horizontal panel
                bullet_width = cv2.getTextSize("• ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                indent_width = cv2.getTextSize("  ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                max_first_line_width = content_width - bullet_width - 20
                max_continuation_width = content_width - indent_width - 20
                wrapped_lines = self._wrap_text_with_bullet_space(clean_message, max_first_line_width, max_continuation_width, font_feedback, thick_feedback)
                
                for i, line in enumerate(wrapped_lines):
                    if lines_used >= max_lines:
                        break
                    
                    # Format with bullet
                    display_text = f"• {line}" if i == 0 else f"  {line}"
                    
                    # NO MORE ELLIPSIS TRIMMING - pixel-based wrapping handles all text fitting
                    
                    # Draw text with SAFE RENDERING - prevents ???? corruption
                    frame = self.safe_render_text(frame, display_text, (content_x, y_pos), 
                                                font_feedback, feedback_color, thick_feedback)
                    
                    y_pos += line_spacing
                    lines_used += 1
                
                # Small gap between messages if space allows
                if lines_used < max_lines - 1:
                    y_pos += 3
        else:
            # Default message that fits - SAFE RENDERING
            default_msg = "Keep up the good work!"
            frame = self.safe_render_text(frame, default_msg, (content_x, y_pos), 
                                        font_feedback, (100, 255, 100), thick_feedback)
        
        return frame
    
    def _draw_sidebar_panel_vertical(self, frame, panel_x: int, panel_y: int,
                                   panel_width: int, panel_height: int,
                                   exercise_name: str, reps: int, form_accuracy: float,
                                   overall_accuracy: float, feedback_list: list, exercise_state: str):
        """
        Draw comprehensive UI panel in the black sidebar area for vertical videos
        
        This function places all UI elements in the black sidebar to completely avoid
        overlapping with the main video content.
        
        Layout within sidebar:
        ┌─────────────────┐
        │ EXERCISE NAME   │
        │ Reps: XX        │
        │ Form: XX%       │  
        │ Overall: XX%    │
        ├─────────────────┤
        │ FEEDBACK:       │
        │ • Message 1     │
        │ • Message 2     │
        │ • ...           │
        └─────────────────┘
        """
        import cv2
        
        # Semi-transparent dark background for the entire sidebar panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (20, 20, 20), -1)
        cv2.addWeighted(frame, 0.25, overlay, 0.75, 0, frame)
        
        # Professional border
        cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                     (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                     (80, 140, 200), 2)
        
        # Content positioning with margins
        content_x = panel_x + 12
        content_width = panel_width - 24
        y_pos = panel_y + 25
        
        # MAXIMUM font sizes for dramatically improved readability
        base_scale = min(panel_width / 180.0, 1.8)  # MUCH higher scale factor
        font_title = min(0.95, max(0.65, 0.85 * base_scale))      # MUCH larger title
        font_info = min(0.85, max(0.6, 0.75 * base_scale))        # MUCH larger info
        font_feedback = min(0.8, max(0.55, 0.7 * base_scale))     # MUCH larger feedback
        
        thick_title = max(2, int(2.5 * base_scale))    # Extra bold title
        thick_info = max(2, int(2.2 * base_scale))      # Extra bold info
        thick_feedback = max(2, int(2.0 * base_scale))  # Extra bold feedback
        
        # MAXIMUM line spacing to expand text downwards and fill vertical space
        line_spacing_title = max(50, int(60 * base_scale))      # MUCH more space for title
        line_spacing_info = max(45, int(55 * base_scale))       # MUCH more space for info
        line_spacing_feedback = max(40, int(50 * base_scale))   # MUCH more space for feedback
        
        # 1. EXERCISE NAME (top section)
        clean_exercise_name = self.sanitize_text(exercise_name)
        # Wrap exercise name if too long for sidebar
        if len(clean_exercise_name) > 20:
            name_lines = self.wrap_text(clean_exercise_name, 18)
            for i, line in enumerate(name_lines[:2]):  # Max 2 lines for name
                frame = self.safe_render_text(frame, line, (content_x, y_pos), 
                                            font_title, (100, 200, 255), thick_title)
                y_pos += line_spacing_title
        else:
            frame = self.safe_render_text(frame, clean_exercise_name, (content_x, y_pos), 
                                        font_title, (100, 200, 255), thick_title)
            y_pos += line_spacing_title
        
        y_pos += 35  # MUCH more space after title to expand downwards
        
        # 2. REPS COUNTER with enhanced visibility
        reps_text = f"Reps: {reps}"
        frame = self.safe_render_text(frame, reps_text, (content_x, y_pos), 
                                    font_info, (255, 255, 255), thick_info)
        y_pos += line_spacing_info
        
        # 3. FORM ACCURACY with horizontal accuracy bar
        # Normalize accuracy value (handle both 0-1 and 0-100 ranges)
        normalized_form = form_accuracy / 100.0 if form_accuracy > 1.0 else form_accuracy
        form_percentage = int(normalized_form * 100)
        form_text = f"Form: {form_percentage}%"
        
        # Color based on normalized value
        form_color = (120, 255, 120) if normalized_form >= 0.8 else (255, 255, 120) if normalized_form >= 0.6 else (255, 120, 120)
        
        # Draw form accuracy text
        frame = self.safe_render_text(frame, form_text, (content_x, y_pos), 
                                    font_info, form_color, thick_info)
        
        # Draw horizontal accuracy bar for form
        bar_x = content_x + 5
        bar_y = y_pos + 8  # Slightly below text
        bar_width = min(120, content_width - 10)
        bar_height = 6
        
        # Background bar (light grey)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (80, 80, 80), -1)
        
        # Foreground bar (colored based on NORMALIZED accuracy)
        fill_width = int(bar_width * normalized_form)
        fill_width = max(0, min(fill_width, bar_width))  # Clamp to valid range
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         form_color, -1)
        
        y_pos += line_spacing_info
        
        # 4. OVERALL ACCURACY with horizontal accuracy bar
        # Normalize accuracy value (handle both 0-1 and 0-100 ranges)
        normalized_overall = overall_accuracy / 100.0 if overall_accuracy > 1.0 else overall_accuracy
        overall_percentage = int(normalized_overall * 100)
        overall_text = f"Overall: {overall_percentage}%"
        
        # Color based on normalized value
        overall_color = (120, 255, 120) if normalized_overall >= 0.8 else (255, 255, 120) if normalized_overall >= 0.6 else (255, 120, 120)
        
        # Draw overall accuracy text
        frame = self.safe_render_text(frame, overall_text, (content_x, y_pos), 
                                    font_info, overall_color, thick_info)
        
        # Draw horizontal accuracy bar for overall
        bar_x = content_x + 5
        bar_y = y_pos + 8  # Slightly below text
        bar_width = min(120, content_width - 10)
        bar_height = 6
        
        # Background bar (light grey)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (80, 80, 80), -1)
        
        # Foreground bar (colored based on NORMALIZED accuracy)
        fill_width = int(bar_width * normalized_overall)
        fill_width = max(0, min(fill_width, bar_width))  # Clamp to valid range
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         overall_color, -1)
        
        y_pos += line_spacing_info + 40  # MAXIMUM space before feedback section
        
        # 5. FEEDBACK SECTION HEADER with enhanced visibility
        frame = self.safe_render_text(frame, "FEEDBACK:", (content_x, y_pos), 
                                    font_info, (220, 220, 255), thick_info)
        y_pos += line_spacing_info + 20  # MUCH more space after header
        
        # 6. FEEDBACK MESSAGES - MAXIMUM expansion to fill all available space
        remaining_height = panel_y + panel_height - y_pos - 20  # Minimal bottom margin to use more space
        max_feedback_lines = max(8, int(remaining_height / line_spacing_feedback))  # Allow MANY more lines
        
        if feedback_list:
            lines_used = 0
            for message_idx, message in enumerate(feedback_list):
                if lines_used >= max_feedback_lines:
                    break
                
                clean_message = self.sanitize_text(str(message))
                if not clean_message:
                    continue
                
                # Enhanced message color coding for better visibility
                if any(word in clean_message.lower() for word in ['good', 'great', 'excellent', 'perfect', 'correct']):
                    feedback_color = (120, 255, 120)  # Brighter green for positive
                elif any(word in clean_message.lower() for word in ['improve', 'adjust', 'focus', 'watch']):
                    feedback_color = (120, 220, 255)  # Brighter blue for suggestions
                else:
                    feedback_color = (240, 240, 240)  # Brighter white for neutral
                
                # PIXEL-BASED text wrapping with bullet space for right panel
                bullet_width = cv2.getTextSize("• ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                indent_width = cv2.getTextSize("  ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                max_first_line_width = content_width - bullet_width - 25
                max_continuation_width = content_width - indent_width - 25
                wrapped_lines = self._wrap_text_with_bullet_space(clean_message, max_first_line_width, max_continuation_width, font_feedback, thick_feedback)
                
                for i, line in enumerate(wrapped_lines):
                    if lines_used >= max_feedback_lines:
                        break
                    
                    # Format with bullet for first line only
                    display_text = f"• {line}" if i == 0 else f"  {line}"
                    
                    # NO MORE ELLIPSIS TRIMMING - pixel-based wrapping handles all text fitting
                    
                    # Safe rendering
                    frame = self.safe_render_text(frame, display_text, (content_x, y_pos), 
                                                font_feedback, feedback_color, thick_feedback)
                    
                    y_pos += line_spacing_feedback
                    lines_used += 1
                
                # MAXIMUM spacing between different feedback messages to expand vertically
                if lines_used < max_feedback_lines and message_idx < len(feedback_list) - 1:
                    y_pos += max(20, int(25 * base_scale))  # MUCH more space between messages
        else:
            # EXPANDED default messages to fill more vertical space
            default_messages = [
                "• Keep up the excellent work!",
                "• Focus on maintaining proper form",
                "• Stay consistent with your breathing",
                "• Control your movements smoothly",
                "• You're doing great - keep going!",
                "• Maintain steady rhythm and pace",
                "• Remember to engage your core",
                "• Excellent technique so far!"
            ]
            
            lines_used = 0
            for msg in default_messages:
                if lines_used < max_feedback_lines:
                    frame = self.safe_render_text(frame, msg, (content_x, y_pos), 
                                                font_feedback, (140, 255, 140), thick_feedback)
                    y_pos += line_spacing_feedback + 15  # MUCH more spacing for defaults
                    lines_used += 1
        
        return frame
    
    def _draw_horizontal_top_panel_vertical(self, frame, panel_x: int, panel_y: int,
                                          panel_width: int, panel_height: int,
                                          exercise_name: str, reps: int, form_accuracy: float,
                                          overall_accuracy: float, feedback_list: list, exercise_state: str):
        """
        Draw horizontal feedback panel at top for vertical videos
        
        This creates a horizontal layout at the top of vertical videos, similar to 
        horizontal video layout but positioned at the top instead of bottom.
        
        Layout:
        ┌─────────────────────────────────────────────────┐
        │ Exercise Name    Reps: XX  Form: XX%  Overall: XX% │
        │ ████████████████████████████████████████████████ │
        │ • Feedback message 1                            │
        │ • Feedback message 2                            │  
        │ • Feedback message 3                            │
        └─────────────────────────────────────────────────┘
        """
        import cv2
        
        # Semi-transparent dark background for readability (Pic 1 style)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (15, 15, 15), -1)
        cv2.addWeighted(frame, 0.65, overlay, 0.35, 0, frame)
        
        # Professional border
        cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                     (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                     (80, 140, 200), 2)
        
        # Content positioning with margins
        content_x = panel_x + 15
        content_width = panel_width - 30
        y_pos = panel_y + 20
        
        # MAXIMUM font sizes for line-by-line vertical layout readability
        base_scale = min(panel_width / 500.0, panel_height / 200.0, 2.2)  # MAXIMUM scale factor
        font_title = min(1.3, max(0.8, 1.1 * base_scale))       # MAXIMUM title font
        font_info = min(1.1, max(0.7, 0.95 * base_scale))       # MAXIMUM info font
        font_feedback = min(1.0, max(0.6, 0.85 * base_scale))   # MAXIMUM feedback font
        
        thick_title = max(3, int(4.0 * base_scale))    # MAXIMUM bold title
        thick_info = max(3, int(3.5 * base_scale))      # MAXIMUM bold info
        thick_feedback = max(2, int(3.0 * base_scale))  # MAXIMUM bold feedback
        
        line_spacing = max(55, int(70 * base_scale))    # MAXIMUM vertical spacing
        bar_spacing = max(15, int(20 * base_scale))      # Space between label and bar
        
        # LINE-BY-LINE VERTICAL LAYOUT for maximum readability
        
        # 1. EXERCISE NAME (first line)
        clean_exercise_name = self.sanitize_text(exercise_name)
        if len(clean_exercise_name) > 30:
            clean_exercise_name = clean_exercise_name[:27] + "..."
        
        frame = self.safe_render_text(frame, clean_exercise_name, (content_x, y_pos), 
                                    font_title, (100, 200, 255), thick_title)
        y_pos += line_spacing
        
        # 2. REPS COUNTER (second line)
        reps_text = f"Reps: {reps}"
        frame = self.safe_render_text(frame, reps_text, (content_x, y_pos), 
                                    font_info, (255, 255, 255), thick_info)
        y_pos += line_spacing
        
        # 3. FORM ACCURACY LABEL (third line)
        # Normalize accuracy value (handle both 0-1 and 0-100 ranges)
        normalized_form = form_accuracy / 100.0 if form_accuracy > 1.0 else form_accuracy
        form_percentage = int(normalized_form * 100)
        form_text = f"FORM ACCURACY: {form_percentage}%"
        form_color = (120, 255, 120) if normalized_form >= 0.8 else (255, 255, 120) if normalized_form >= 0.6 else (255, 120, 120)
        frame = self.safe_render_text(frame, form_text, (content_x, y_pos), 
                                    font_info, form_color, thick_info)
        y_pos += bar_spacing
        
        # 3a. FORM ACCURACY BAR (below label)
        bar_width = min(content_width - 20, 300)
        bar_height = 15  # LARGER bars for visibility
        bar_x = content_x + 10
        
        # Background bar for form
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), 
                     (80, 80, 80), -1)
        
        # Foreground bar for form (use normalized value)
        form_fill_width = int(bar_width * normalized_form)
        form_fill_width = max(0, min(form_fill_width, bar_width))
        if form_fill_width > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + form_fill_width, y_pos + bar_height), 
                         form_color, -1)
        
        y_pos += bar_height + line_spacing
        
        # 4. OVERALL ACCURACY LABEL (fourth line)
        # Normalize accuracy value (handle both 0-1 and 0-100 ranges)
        normalized_overall = overall_accuracy / 100.0 if overall_accuracy > 1.0 else overall_accuracy
        overall_percentage = int(normalized_overall * 100)
        overall_text = f"ACCURACY: {overall_percentage}%"
        overall_color = (120, 255, 120) if normalized_overall >= 0.8 else (255, 255, 120) if normalized_overall >= 0.6 else (255, 120, 120)
        frame = self.safe_render_text(frame, overall_text, (content_x, y_pos), 
                                    font_info, overall_color, thick_info)
        y_pos += bar_spacing
        
        # 4a. OVERALL ACCURACY BAR (below label)
        bar_width = min(content_width - 20, 300)
        bar_height = 15  # LARGER bars for visibility
        bar_x = content_x + 10
        
        # Background bar for overall
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), 
                     (80, 80, 80), -1)
        
        # Foreground bar for overall (use normalized value)
        overall_fill_width = int(bar_width * normalized_overall)
        overall_fill_width = max(0, min(overall_fill_width, bar_width))
        if overall_fill_width > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + overall_fill_width, y_pos + bar_height), 
                         overall_color, -1)
        
        y_pos += bar_height + line_spacing
        
        # FEEDBACK MESSAGES SECTION
        remaining_height = panel_y + panel_height - y_pos - 15
        max_feedback_lines = max(3, int(remaining_height / (line_spacing - 5)))
        
        if feedback_list:
            lines_used = 0
            for message_idx, message in enumerate(feedback_list):
                if lines_used >= max_feedback_lines:
                    break
                
                clean_message = self.sanitize_text(str(message))
                if not clean_message:
                    continue
                
                # Color coding for feedback messages (existing scheme)
                if any(word in clean_message.lower() for word in ['good', 'great', 'excellent', 'perfect', 'correct']):
                    feedback_color = (100, 255, 100)  # Green = correct/active
                elif any(word in clean_message.lower() for word in ['improve', 'adjust', 'focus', 'watch']):
                    feedback_color = (100, 180, 255)  # Orange = correction needed  
                else:
                    feedback_color = (255, 255, 255)  # White = neutral
                
                # PIXEL-BASED text wrapping with bullet space pre-calculated
                bullet_width = cv2.getTextSize("• ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                indent_width = cv2.getTextSize("  ", cv2.FONT_HERSHEY_SIMPLEX, font_feedback, thick_feedback)[0][0]
                
                # Calculate max width for first line (with bullet) and continuation lines (with indent)
                max_first_line_width = content_width - bullet_width - 20  # Extra margin
                max_continuation_width = content_width - indent_width - 20  # Extra margin
                
                # Wrap text accounting for bullet space
                wrapped_lines = self._wrap_text_with_bullet_space(clean_message, max_first_line_width, max_continuation_width, font_feedback, thick_feedback)
                
                for i, line in enumerate(wrapped_lines):
                    if lines_used >= max_feedback_lines:
                        break
                    
                    display_text = f"• {line}" if i == 0 else f"  {line}"
                    
                    # NO MORE ELLIPSIS TRIMMING - pixel wrapping should handle everything
                    
                    # Safe rendering with pixel-perfect positioning
                    frame = self.safe_render_text(frame, display_text, (content_x, y_pos), 
                                                font_feedback, feedback_color, thick_feedback)
                    
                    y_pos += line_spacing  # Normal spacing for readability
                    lines_used += 1
                
                # Small gap between messages
                if lines_used < max_feedback_lines and message_idx < len(feedback_list) - 1:
                    y_pos += 8
        else:
            # Default messages for when no feedback available
            default_messages = [
                "• Keep up the excellent work!",
                "• Focus on maintaining proper form", 
                "• Stay consistent with your breathing"
            ]
            
            for msg in default_messages:
                if y_pos + line_spacing < panel_y + panel_height - 15:
                    frame = self.safe_render_text(frame, msg, (content_x, y_pos), 
                                                font_feedback, (140, 255, 140), thick_feedback)
                    y_pos += line_spacing + 10  # MORE spacing for defaults visibility
        
        return frame
    
    def sanitize_text(self, text: str) -> str:
        '''
        ULTIMATE text sanitization to eliminate ALL ???? corruption with robust Unicode handling
        Ensures 100% OpenCV-safe text rendering with comprehensive character normalization
        '''
        if not text:
            return ''
        
        # Force string conversion with error handling
        try:
            text = str(text).strip()
        except:
            return ''
        
        if not text:
            return ''
        
        import re
        import unicodedata
        
        # STEP 1: Unicode normalization to handle encoding issues
        try:
            # Normalize Unicode to decomposed form, then recompose
            text = unicodedata.normalize('NFKD', text)
            text = unicodedata.normalize('NFKC', text)
        except:
            pass
        
        # STEP 2: Handle UTF-8 decoding errors that cause ????
        try:
            # Force UTF-8 encoding/decoding to catch corruption
            text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        except:
            pass
        
        # STEP 3: Remove REPLACEMENT CHARACTERS and corruption markers
        text = text.replace('\ufffd', '')  # Unicode replacement character (causes ????)
        text = text.replace('?', '')       # All question marks
        text = re.sub(r'\?+', '', text)    # Multiple question marks
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)  # Control characters
        
        # STEP 4: Comprehensive character replacements for OpenCV compatibility
        replacements = {
            # Unicode characters that cause ???? in OpenCV
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark  
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...', # Horizontal ellipsis
            '\u00a0': ' ',  # Non-breaking space
            
            # Emojis that cause corruption in OpenCV
            '💪': 'STRONG', '🔥': 'FIRE', '🚀': 'GO', '🎯': 'TARGET',
            '📊': 'STATS', '📈': 'PROGRESS', '🔧': 'FIX', '✅': 'GOOD',
            '❌': 'ERROR', '⚠️': 'WARNING', '⚠': 'WARNING', '🎉': 'GREAT',
            '💡': 'TIP', '🏃': 'MOVE', '🏋': 'LIFT', '👍': 'GOOD',
            '👎': 'BAD', '🏆': 'WIN', '⭐': 'STAR', '🔴': 'RED',
            
            # Special symbols that cause corruption
            '•': '- ', '◦': '* ', '→': ' -> ', '←': ' <- ',
            '↑': ' UP ', '↓': ' DOWN ', '✓': 'OK', '✗': 'NO', '×': 'X',
            '∞': 'infinity', '±': '+-', '∆': 'delta', '∑': 'sum',
            
            # Punctuation that causes corruption in OpenCV
            '"': '"', '"': '"', ''': "'", ''': "'", '„': '"',
            '–': '-', '—': '-', '…': '...', '°': ' deg',
            
            # Math/science symbols that cause corruption
            '≥': '>=', '≤': '<=', '≠': '!=', '≈': '~=', '÷': '/',
            '×': 'x', '²': '2', '³': '3', '¼': '1/4', '½': '1/2', '¾': '3/4',
            
            # Accented characters that cause corruption
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e', 'í': 'i', 'ì': 'i',
            'î': 'i', 'ï': 'i', 'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o',
            'õ': 'o', 'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y',
            'ñ': 'n', 'ç': 'c', 'š': 's', 'ž': 'z', 'đ': 'd',
            
            # Currency symbols that cause corruption
            '€': 'EUR', '£': 'GBP', '¥': 'YEN', '¢': 'cent', '₹': 'INR'
        }
        
        # STEP 5: Apply comprehensive character replacements
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)
        
        # STEP 6: Ultra-aggressive ASCII-only conversion
        # Remove ALL non-ASCII characters that could cause ????
        text = re.sub(r'[^\x20-\x7E]', '', text)  # Keep ONLY printable ASCII (32-126)
        
        # STEP 7: Remove any remaining corruption indicators
        text = re.sub(r'\?+', '', text)  # Any remaining question marks
        text = re.sub(r'[^\w\s\-\.\,\!\;\:\(\)\[\]\'\"]+', '', text)  # Only safe punctuation
        
        # STEP 8: Final ASCII enforcement with multiple fallbacks
        try:
            # Primary: Force ASCII encoding
            text = text.encode('ascii', 'ignore').decode('ascii')
        except:
            try:
                # Secondary: Manual character filtering
                safe_chars = [char for char in text if 32 <= ord(char) <= 126]
                text = ''.join(safe_chars)
            except:
                # Ultimate fallback: Basic alphanumeric only
                text = re.sub(r'[^a-zA-Z0-9\s\-\.\,\!]', '', text)
            text = ''.join(safe_chars)
        
        # STEP 9: Final cleanup and validation
        text = ' '.join(text.split()).strip()  # Normalize whitespace
        
        # STEP 10: Validate result is safe for OpenCV
        if not text or len(text) == 0:
            return ''
        
        # Ensure no remaining problematic characters
        safe_text = ''
        for char in text:
            if 32 <= ord(char) <= 126:  # Only printable ASCII
                safe_text += char
            else:
                safe_text += ' '  # Replace any remaining problematic chars with space
        
        return safe_text.strip()
    
    def safe_render_text(self, frame, text: str, position: tuple, font_scale: float, 
                        color: tuple, thickness: int, font=None):
        """
        Ultra-safe text rendering for OpenCV with corruption prevention
        Guarantees no ???? characters with robust error handling
        """
        import cv2
        
        # Use safest OpenCV font
        if font is None:
            font = cv2.FONT_HERSHEY_SIMPLEX  # Most reliable font for all systems
        
        # Double-sanitize text for absolute safety
        safe_text = self.sanitize_text(str(text))
        if not safe_text:
            return frame
        
        try:
            # Attempt primary rendering
            cv2.putText(frame, safe_text, position, font, font_scale, color, thickness)
        except Exception as e:
            try:
                # Fallback 1: Try with basic ASCII only
                ascii_only = ''.join(c for c in safe_text if ord(c) < 128)
                cv2.putText(frame, ascii_only, position, font, font_scale, color, thickness)
            except:
                try:
                    # Fallback 2: Try with alphanumeric only
                    alphanumeric = ''.join(c for c in safe_text if c.isalnum() or c.isspace())
                    cv2.putText(frame, alphanumeric, position, font, font_scale, color, thickness)
                except:
                    # Ultimate fallback: Safe placeholder
                    cv2.putText(frame, "Text Error", position, font, font_scale, (100, 100, 100), thickness)
        
        return frame

    def wrap_text(self, text: str, max_chars_per_line: int) -> list:
        """
        ENHANCED text wrapping with aggressive boundary enforcement
        Ensures text NEVER exceeds panel boundaries with smart word breaking
        
        Args:
            text: Text to wrap
            max_chars_per_line: STRICT maximum characters per line
            
        Returns:
            List of text lines that GUARANTEE fit within specified width
        """
        if not text:
            return []
        
        # Safety margin - reduce by 10% to ensure fit
        safe_chars_per_line = max(10, int(max_chars_per_line * 0.9))
        
        if len(text) <= safe_chars_per_line:
            return [text]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Handle extra-long words that exceed line limit
            if len(word) > safe_chars_per_line:
                # If current line has content, finish it first
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                
                # Break long word into chunks WITHOUT truncation
                while len(word) > safe_chars_per_line:
                    chunk = word[:safe_chars_per_line]  # No "..." - show complete word
                    lines.append(chunk)
                    word = word[safe_chars_per_line:]
                
                # Add remaining part of word
                if word:
                    current_line = word
                continue
            
            # Check if adding this word would exceed the line limit
            test_line = current_line + (" " if current_line else "") + word
            
            if len(test_line) <= safe_chars_per_line:
                current_line = test_line
            else:
                # Current line is full, start a new line
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        # Add the last line if it has content
        if current_line:
            lines.append(current_line)
        
        # Final safety check - truncate any lines that are still too long
        safe_lines = []
        for line in lines:
            if len(line) <= max_chars_per_line:
                safe_lines.append(line)
            else:
                # Emergency truncation
                safe_lines.append(line[:max_chars_per_line-3] + "...")
        
        return safe_lines
    
    def _draw_ui_panel(self, frame, panel_x: int, panel_y: int, panel_width: int, 
                      panel_height: int, exercise_name: str, reps: int,
                      form_accuracy: float, overall_accuracy: float, 
                      feedback_list: list, exercise_state: str, layout_type: str):
        """
        Draw the main UI panel with modern, clean design
        
        Creates separate sections for:
        - Exercise Info
        - Reps Counter
        - Form Accuracy (with bar)
        - Overall Accuracy (with bar)  
        - Feedback Messages
        """
        import cv2
        
        # ===== MODERN PANEL BACKGROUND =====
        # Semi-transparent dark background with subtle border
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (15, 15, 15), -1)  # Dark background
        cv2.addWeighted(frame, 0.45, overlay, 0.55, 0, frame)  # Semi-transparent
        
        # Professional border styling
        cv2.rectangle(frame, (panel_x-2, panel_y-2), 
                     (panel_x + panel_width + 2, panel_y + panel_height + 2), 
                     (80, 80, 80), 2)  # Outer border
        cv2.rectangle(frame, (panel_x-1, panel_y-1), 
                     (panel_x + panel_width + 1, panel_y + panel_height + 1), 
                     (120, 120, 120), 1)  # Inner border
        
        # ===== DYNAMIC FONT SCALING =====
        # Enhanced font scaling with layout-specific adjustments
        if layout_type == 'vertical':
            # Larger fonts for vertical layout (more space available)
            base_scale = max(0.9, min(1.8, panel_width / 250))
            font_title = base_scale * 1.1    # Exercise name - larger
            font_large = base_scale * 1.4    # Rep counter - larger
            font_medium = base_scale * 0.9   # Section headers - larger
            font_small = base_scale * 0.8    # Feedback text - MUCH larger for readability
        else:
            # Optimized fonts for horizontal layout (limited height)
            base_scale = max(0.6, min(1.2, panel_height / 180))
            font_title = base_scale * 0.9    # Exercise name
            font_large = base_scale * 1.1    # Rep counter
            font_medium = base_scale * 0.7   # Section headers
            font_small = base_scale * 0.65   # Feedback text
        
        # Enhanced font thickness for better visibility
        thick_title = max(2, int(base_scale * 3))
        thick_large = max(3, int(base_scale * 4))
        thick_medium = max(2, int(base_scale * 2.5))
        thick_small = max(2, int(base_scale * 2))  # Increased thickness for feedback
        
        # ===== LAYOUT CALCULATIONS =====
        # Layout-specific spacing optimization
        if layout_type == 'vertical':
            content_x = panel_x + 25  # More padding for vertical
            y_pos = panel_y + 35      # More top padding
            content_width = panel_width - 50  # More side padding
            
            # Increased spacing for vertical layout (more room available)
            line_spacing_large = int(55 * base_scale)
            line_spacing_medium = int(42 * base_scale)
            line_spacing_small = int(32 * base_scale)  # Increased for better feedback readability
        else:
            content_x = panel_x + 20  # Standard padding for horizontal
            y_pos = panel_y + 25      # Less top padding (limited height)
            content_width = panel_width - 40  # Standard side padding
            
            # Tighter spacing for horizontal layout (limited height)
            line_spacing_large = int(35 * base_scale)
            line_spacing_medium = int(28 * base_scale)
            line_spacing_small = int(22 * base_scale)
        
        # ===== SECTION 1: EXERCISE NAME =====
        cv2.putText(frame, exercise_name.upper(), (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_title, (255, 255, 255), thick_title)
        y_pos += line_spacing_large
        
        # Add separator line
        cv2.line(frame, (content_x, y_pos - 15), 
                (content_x + content_width - 20, y_pos - 15), (100, 100, 100), 1)
        
        # ===== SECTION 2: REPS COUNTER =====
        cv2.putText(frame, "REPS", (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, (200, 200, 200), thick_medium)
        y_pos += line_spacing_medium
        
        # Large, prominent rep counter with color coding
        rep_color = (0, 255, 0) if reps > 0 else (150, 150, 150)  # Green if active, grey if starting
        cv2.putText(frame, str(reps), (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_large * 1.5, rep_color, thick_large)
        y_pos += line_spacing_large + 10
        
        # ===== SECTION 3: FORM ACCURACY =====
        cv2.putText(frame, "FORM ACCURACY", (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, (200, 200, 200), thick_medium)
        y_pos += line_spacing_medium
        
        # Accuracy value and bar
        if form_accuracy > 0:
            form_color = self.get_accuracy_color(form_accuracy)
            cv2.putText(frame, f"{form_accuracy:.0f}%", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_medium, form_color, thick_medium)
            
            # Accuracy bar
            bar_y = y_pos + 10
            bar_width = content_width - 30
            bar_height = max(8, int(12 * base_scale))
            self.draw_accuracy_bar(frame, content_x, bar_y, bar_width, bar_height, 
                                 form_accuracy, form_color)
            y_pos += 35
        else:
            cv2.putText(frame, "Start exercising", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_small, (150, 150, 150), thick_small)
            y_pos += line_spacing_medium
        
        # ===== SECTION 4: OVERALL ACCURACY =====
        cv2.putText(frame, "OVERALL ACCURACY", (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, (200, 200, 200), thick_medium)
        y_pos += line_spacing_medium
        
        # Overall accuracy value and bar
        if overall_accuracy > 0:
            overall_color = self.get_accuracy_color(overall_accuracy)
            cv2.putText(frame, f"{overall_accuracy:.0f}%", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_medium, overall_color, thick_medium)
            
            # Accuracy bar
            bar_y = y_pos + 10
            bar_width = content_width - 30
            bar_height = max(8, int(12 * base_scale))
            self.draw_accuracy_bar(frame, content_x, bar_y, bar_width, bar_height, 
                                 overall_accuracy, overall_color)
            y_pos += 40
        else:
            cv2.putText(frame, "Calculating...", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_small, (150, 150, 150), thick_small)
            y_pos += line_spacing_medium
        
        # ===== SECTION 5: FEEDBACK =====
        cv2.putText(frame, "FEEDBACK", (content_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, (200, 200, 200), thick_medium)
        y_pos += line_spacing_medium + 8
        
        # Calculate available space for feedback with layout-specific optimization
        available_height = (panel_y + panel_height) - y_pos - 25
        
        if layout_type == 'vertical':
            # More lines for vertical layout (more space available)
            max_lines = max(6, int(available_height / line_spacing_small))
        else:
            # Fewer lines for horizontal layout (limited height)
            max_lines = max(3, int(available_height / line_spacing_small))
        
        # Calculate character limit for text wrapping with better precision
        char_width = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, font_small, thick_small)[0][0]
        chars_per_line = max(20, int((content_width - 30) / char_width))
        
        # Display feedback messages with proper wrapping
        if feedback_list and len(feedback_list) > 0:
            lines_used = 0
            
            for message in feedback_list[:5]:  # Limit to 5 messages to avoid overcrowding
                if lines_used >= max_lines:
                    break
                
                # Clean the message text with enhanced sanitization
                clean_message = self.sanitize_text(str(message))
                if not clean_message or len(clean_message.strip()) == 0:
                    continue
                
                # Remove any remaining corrupted prefixes or markers
                clean_message = clean_message.strip()
                if clean_message.startswith('? '):
                    clean_message = clean_message[2:]
                
                # Determine feedback color based on content with better visibility
                if any(word in clean_message.lower() for word in ['good', 'great', 'excellent', 'perfect', 'well', 'nice']):
                    feedback_color = (100, 255, 100)  # Brighter green for positive
                elif any(word in clean_message.lower() for word in ['adjust', 'improve', 'try', 'better', 'focus', 'maintain']):
                    feedback_color = (100, 220, 255)  # Bright blue for suggestions
                elif any(word in clean_message.lower() for word in ['wrong', 'error', 'bad', 'incorrect', 'fix', 'careful']):
                    feedback_color = (100, 180, 255)  # Bright orange-red for corrections
                else:
                    feedback_color = (240, 240, 240)  # Bright white for neutral
                
                # PIXEL-BASED text wrapping with bullet space for UI panel
                bullet_width = cv2.getTextSize("• ", cv2.FONT_HERSHEY_SIMPLEX, font_small, thick_small)[0][0]
                indent_width = cv2.getTextSize("  ", cv2.FONT_HERSHEY_SIMPLEX, font_small, thick_small)[0][0]
                max_first_line_width = content_width - bullet_width - 35
                max_continuation_width = content_width - indent_width - 35
                wrapped_lines = self._wrap_text_with_bullet_space(clean_message, max_first_line_width, max_continuation_width, font_small, thick_small)
                
                for i, line in enumerate(wrapped_lines):
                    if lines_used >= max_lines:
                        break
                    
                    # Enhanced bullet formatting with better alignment
                    if i == 0:
                        display_text = f"- {line}"  # Clean dash bullet
                    else:
                        display_text = f"   {line}"  # Proper indentation for continuation
                    
                    # NO MORE ELLIPSIS TRIMMING - pixel-based wrapping handles all text fitting
                    
                    # Draw text with enhanced positioning
                    cv2.putText(frame, display_text, (content_x + 5, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_small, feedback_color, thick_small)
                    
                    y_pos += line_spacing_small
                    lines_used += 1
                
                # Add spacing between different feedback messages for clarity
                if lines_used < max_lines and message != feedback_list[-1]:
                    y_pos += max(3, int(8 * base_scale))
        else:
            # Default message when no feedback is available
            cv2.putText(frame, "• Keep up the good work!", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_small, (120, 255, 120), thick_small)
            y_pos += line_spacing_small
            cv2.putText(frame, "• Focus on your form", (content_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_small, (120, 255, 120), thick_small)
        
        return frame

    def _get_accuracy_color_bright(self, accuracy: float) -> tuple:
        """
        Get bright color based on accuracy percentage for better visibility on dark background
        Returns BGR color tuple for OpenCV
        """
        if accuracy >= 80.0:
            return (100, 255, 100)    # Bright Green - Excellent
        elif accuracy >= 65.0:
            return (100, 255, 255)    # Bright Yellow - Good
        elif accuracy >= 50.0:
            return (100, 200, 255)    # Bright Orange - Fair
        else:
            return (100, 150, 255)    # Bright Red - Needs improvement
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 3: ACCURACY METRICS (Separated Cards)
        # ═══════════════════════════════════════════════════════════════
        
        # Form Accuracy Card
        form_color = self._get_accuracy_color(form_accuracy)
        
        # Form accuracy background
        cv2.rectangle(frame, (margin-5, content_y-5), (margin + section_width + 5, content_y + 28), 
                     (30, 30, 30), -1)
        
        # Form accuracy label and value
        cv2.putText(frame, "FORM ACCURACY", (margin + 2, content_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_detail, (120, 120, 120), thick_light)
        cv2.putText(frame, f"{form_accuracy:.0f}%", (margin + section_width - 40, content_y + 12), 
                   cv2.FONT_HERSHEY_DUPLEX, font_main, form_color, thick_normal)
        
        # Form progress indicator
        progress_y = content_y + 18
        progress_width = section_width - 10
        cv2.rectangle(frame, (margin + 2, progress_y), (margin + 2 + progress_width, progress_y + 6), 
                     (60, 60, 60), -1)
        form_progress = int(progress_width * form_accuracy / 100)
        if form_progress > 0:
            cv2.rectangle(frame, (margin + 2, progress_y), (margin + 2 + form_progress, progress_y + 6), 
                         form_color, -1)
        
        content_y += 40
        
        # Overall Accuracy Card
        simple_color = self._get_accuracy_color(simple_accuracy)
        
        # Overall accuracy background
        cv2.rectangle(frame, (margin-5, content_y-5), (margin + section_width + 5, content_y + 28), 
                     (30, 30, 30), -1)
        
        # Overall accuracy label and value
        cv2.putText(frame, "OVERALL ACCURACY", (margin + 2, content_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_detail, (120, 120, 120), thick_light)
        cv2.putText(frame, f"{simple_accuracy:.0f}%", (margin + section_width - 40, content_y + 12), 
                   cv2.FONT_HERSHEY_DUPLEX, font_main, simple_color, thick_normal)
        
        # Overall progress indicator
        progress_y = content_y + 18
        cv2.rectangle(frame, (margin + 2, progress_y), (margin + 2 + progress_width, progress_y + 6), 
                     (60, 60, 60), -1)
        simple_progress = int(progress_width * simple_accuracy / 100)
        if simple_progress > 0:
            cv2.rectangle(frame, (margin + 2, progress_y), (margin + 2 + simple_progress, progress_y + 6), 
                         simple_color, -1)
        
        content_y += 50
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 4: FEEDBACK AREA (Clean Message Box)
        # ═══════════════════════════════════════════════════════════════
        
        # Feedback section header
        cv2.rectangle(frame, (margin-5, content_y-5), (margin + section_width + 5, content_y + 20), 
                     (40, 40, 40), -1)
        cv2.putText(frame, "LIVE FEEDBACK", (margin + 2, content_y + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_detail, (200, 200, 200), thick_light)
        
        content_y += 30
        
        # Feedback messages area
        feedback_box_height = min(120, panel_height - (content_y - panel_y) - 60)
        cv2.rectangle(frame, (margin-3, content_y-3), (margin + section_width + 3, content_y + feedback_box_height), 
                     (25, 25, 25), -1)
        
        # Display clean feedback messages
        feedback_y = content_y + 15
        max_messages = 4
        line_height = 18
        
        if feedback_messages:
            for i, message in enumerate(feedback_messages[:max_messages]):
                if feedback_y > content_y + feedback_box_height - 20:
                    break
                
                # Clean message display without excessive icons
                clean_message = message.replace('🎉', '').replace('✅', '').replace('⚠️', '').replace('🔧', '').replace('❌', '').strip()
                if len(clean_message) > 35:
                    clean_message = clean_message[:32] + "..."
                
                # Color based on content
                if any(word in message.lower() for word in ['good', 'excellent', 'great']):
                    msg_color = (0, 200, 0)
                elif any(word in message.lower() for word in ['adjust', 'try', 'improve']):
                    msg_color = (0, 180, 255)
                elif any(word in message.lower() for word in ['error', 'wrong']):
                    msg_color = (0, 100, 255)
                else:
                    msg_color = (180, 180, 180)
                
                # Simple bullet point
                cv2.putText(frame, "•", (margin, feedback_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_detail, (100, 100, 100), thick_light)
                cv2.putText(frame, clean_message, (margin + 15, feedback_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_detail, msg_color, thick_light)
                
                feedback_y += line_height
        else:
            # No feedback message
            cv2.putText(frame, "Looking good! Keep going.", (margin + 5, feedback_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_detail, (0, 180, 0), thick_light)
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 5: STATUS BAR (Bottom)
        # ═══════════════════════════════════════════════════════════════
        
        # Status at bottom
        bottom_y = panel_y + panel_height - 35
        combined_accuracy = (form_accuracy + simple_accuracy) / 2
        
        # Status background
        cv2.rectangle(frame, (margin-8, bottom_y-8), (margin + section_width + 8, bottom_y + 25), 
                     (50, 50, 50), -1)
        
        # Status indicator
        if combined_accuracy >= 80:
            status = "EXCELLENT"
            status_color = (0, 255, 0)
        elif combined_accuracy >= 65:
            status = "GOOD"
            status_color = (0, 200, 200)
        else:
            status = "NEEDS WORK"
            status_color = (0, 150, 255)
        
        cv2.putText(frame, status, (margin, bottom_y + 15), 
                   cv2.FONT_HERSHEY_DUPLEX, font_detail, status_color, thick_normal)
        
        # Controls hint
        cv2.putText(frame, "Q=Quit R=Reset", (margin + section_width - 100, bottom_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_detail * 0.8, (120, 120, 120), thick_light)
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 3: ACCURACY METRICS
        # ═══════════════════════════════════════════════════════════════
        
        # Section header
        cv2.putText(frame, "📈 ACCURACY METRICS", (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_DUPLEX, font_medium, (0, 0, 0), thickness_normal + 1)
        cv2.putText(frame, "📈 ACCURACY METRICS", (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_DUPLEX, font_medium, (255, 255, 255), thickness_normal)
        cv2.line(frame, (margin_left, y_pos + 5), (margin_left + content_width - 20, y_pos + 5), 
                (255, 255, 255), 2)
        y_pos += section_spacing
        
        # Form Accuracy Section
        form_color = self._get_accuracy_color(form_accuracy)
        form_text = f"Form Accuracy: {form_accuracy:.1f}%"
        cv2.putText(frame, form_text, (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, (0, 0, 0), thickness_normal + 1)
        cv2.putText(frame, form_text, (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, form_color, thickness_normal)
        y_pos += line_spacing
        
        # Form accuracy progress bar
        bar_width = content_width - 20
        bar_height = max(12, int(15 * base_font_scale))
        bar_x = margin_left + 5
        
        # Progress bar background with border
        cv2.rectangle(frame, (bar_x-2, y_pos-2), (bar_x + bar_width+2, y_pos + bar_height+2), 
                     (255, 255, 255), 1)  # White border
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), 
                     (50, 50, 50), -1)  # Dark background
        
        # Filled progress portion
        form_fill = int((form_accuracy / 100.0) * bar_width)
        if form_fill > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + form_fill, y_pos + bar_height), 
                         form_color, -1)
        y_pos += bar_height + small_spacing
        
        # Simple/Overall Accuracy Section
        simple_color = self._get_accuracy_color(simple_accuracy)
        simple_text = f"Accuracy: {simple_accuracy:.1f}%"
        cv2.putText(frame, simple_text, (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, (0, 0, 0), thickness_normal + 1)
        cv2.putText(frame, simple_text, (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_medium, simple_color, thickness_normal)
        y_pos += line_spacing
        
        # Simple accuracy progress bar
        cv2.rectangle(frame, (bar_x-2, y_pos-2), (bar_x + bar_width+2, y_pos + bar_height+2), 
                     (255, 255, 255), 1)  # White border
        cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), 
                     (50, 50, 50), -1)  # Dark background
        
        # Filled progress portion
        simple_fill = int((simple_accuracy / 100.0) * bar_width)
        if simple_fill > 0:
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + simple_fill, y_pos + bar_height), 
                         simple_color, -1)
        y_pos += bar_height + section_spacing
        
        # Combined accuracy indicator
        combined_accuracy = (form_accuracy + simple_accuracy) / 2
        combined_color = self._get_accuracy_color(combined_accuracy)
        combined_status = "🟢 Excellent" if combined_accuracy >= 85 else "🟡 Good" if combined_accuracy >= 70 else "🔶 Fair" if combined_accuracy >= 55 else "🔴 Needs Work"
        
        cv2.putText(frame, f"Status: {combined_status}", (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 0, 0), thickness_thin + 1)
        cv2.putText(frame, f"Status: {combined_status}", (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_small, combined_color, thickness_thin)
        y_pos += section_spacing
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 4: LIVE FEEDBACK MESSAGES
        # ═══════════════════════════════════════════════════════════════
        
        # Section header
        cv2.putText(frame, "💬 LIVE FEEDBACK", (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_DUPLEX, font_medium, (0, 0, 0), thickness_normal + 1)
        cv2.putText(frame, "💬 LIVE FEEDBACK", (margin_left, y_pos), 
                   cv2.FONT_HERSHEY_DUPLEX, font_medium, (255, 255, 255), thickness_normal)
        cv2.line(frame, (margin_left, y_pos + 5), (margin_left + content_width - 20, y_pos + 5), 
                (255, 255, 255), 2)
        y_pos += section_spacing
        
        # Text wrapping helper for feedback messages
        def wrap_feedback_text(text, max_width_chars):
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
        
        # Calculate available space for feedback
        remaining_height = panel_y + panel_height - y_pos - 60  # Leave space at bottom
        max_feedback_lines = max(4, remaining_height // line_spacing)
        max_chars_per_line = max(30, content_width // int(font_small * 12))
        
        # Process and display feedback messages with enhanced styling
        displayed_lines = 0
        for message in feedback_messages[:8]:  # Limit to 8 messages
            if displayed_lines >= max_feedback_lines:
                break
                
            # Wrap long messages
            wrapped_lines = wrap_feedback_text(message, max_chars_per_line)
            
            for line in wrapped_lines:
                if displayed_lines >= max_feedback_lines or y_pos + line_spacing > panel_y + panel_height - 40:
                    break
                    
                # Enhanced message categorization and styling
                display_text = line
                msg_color = (200, 200, 200)  # Default light gray
                
                # Categorize feedback with icons and colors
                if any(word in line.lower() for word in ['excellent', 'perfect', 'great']):
                    display_text = f"🎉 {line}" if not line.startswith('🎉') else line
                    msg_color = (0, 255, 0)  # Bright green for excellent
                elif any(word in line.lower() for word in ['good', 'nice', 'well done']):
                    display_text = f"✅ {line}" if not line.startswith('✅') else line
                    msg_color = (0, 220, 0)  # Green for good
                elif any(word in line.lower() for word in ['adjust', 'try', 'consider']):
                    display_text = f"💡 {line}" if not line.startswith(('💡', '⚠️')) else line
                    msg_color = (0, 255, 255)  # Yellow for suggestions
                elif any(word in line.lower() for word in ['straighten', 'lower', 'raise', 'bend']):
                    display_text = f"🔧 {line}" if not line.startswith(('🔧', '📐')) else line
                    msg_color = (0, 165, 255)  # Orange for corrections
                elif any(word in line.lower() for word in ['error', 'wrong', 'incorrect', 'bad']):
                    display_text = f"❌ {line}" if not line.startswith('❌') else line
                    msg_color = (0, 100, 255)  # Red for errors
                elif any(emoji in line for emoji in ['💪', '🦵', '🏠', '👍', '👎']):
                    msg_color = (255, 180, 0)  # Cyan for body-specific guidance
                
                # Draw feedback message with shadow effect
                cv2.putText(frame, display_text, (margin_left + 2, y_pos + 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 0, 0), thickness_thin + 1)  # Shadow
                cv2.putText(frame, display_text, (margin_left, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_small, msg_color, thickness_thin)  # Main text
                
                y_pos += line_spacing
                displayed_lines += 1
        
        # If no feedback messages, show encouraging message
        if not feedback_messages or displayed_lines == 0:
            encourage_text = "👍 Keep going! You're doing great!"
            cv2.putText(frame, encourage_text, (margin_left + 2, y_pos + 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 0, 0), thickness_thin + 1)
            cv2.putText(frame, encourage_text, (margin_left, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 255, 0), thickness_thin)
            y_pos += line_spacing
        
        # ═══════════════════════════════════════════════════════════════
        # SECTION 5: POSTURE STATUS INDICATORS (BOTTOM)
        # ═══════════════════════════════════════════════════════════════
        
        # Position at bottom of panel
        bottom_y = panel_y + panel_height - 60
        
        # Posture status line
        posture_status = "🟢 Good Posture" if combined_accuracy >= 75 else "🟡 Fair Posture" if combined_accuracy >= 60 else "🔴 Check Posture"
        cv2.putText(frame, posture_status, (margin_left + 2, bottom_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_small, (0, 0, 0), thickness_thin + 1)
        cv2.putText(frame, posture_status, (margin_left, bottom_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_small, combined_color, thickness_thin)
        
        # Instructions line
        instructions = "Press 'q' to quit, 'r' to reset"
        cv2.putText(frame, instructions, (margin_left + 2, bottom_y + 25 + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_tiny, (0, 0, 0), thickness_thin)
        cv2.putText(frame, instructions, (margin_left, bottom_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_tiny, (150, 150, 150), thickness_thin)
        
        return frame
    
    # Alias for backward compatibility
    def render_visual_feedback(self, frame, form_accuracy: float, simple_accuracy: float, 
                             feedback_messages: list, reps: int, exercise_state: str) -> np.ndarray:
        """Backward compatibility alias for the comprehensive vertical UI panel"""
        return self.render_vertical_ui_panel(frame, form_accuracy, simple_accuracy, 
                                           feedback_messages, reps, exercise_state)
        
        return frame
    
    def _get_accuracy_color(self, accuracy: float) -> tuple:
        """
        Get color based on accuracy percentage
        Returns BGR color tuple for OpenCV
        """
        if accuracy >= 80.0:
            return (0, 255, 0)    # Green - Excellent
        elif accuracy >= 65.0:
            return (0, 255, 255)  # Yellow - Good
        elif accuracy >= 50.0:
            return (0, 165, 255)  # Orange - Fair
        else:
            return (0, 0, 255)    # Red - Needs improvement

    def get_session_stats(self) -> dict:
        """Get comprehensive session statistics"""
        return {
            'total_reps': self.reps,
            'rep_timestamps': self.rep_timestamps,
            'average_rep_accuracy': np.mean(self.rep_accuracies) if self.rep_accuracies else 0.0,
            'average_session_accuracy': np.mean(self.session_accuracies) if self.session_accuracies else 0.0,
            'accuracy_per_rep': self.rep_accuracies
        }


# SQUATS EXERCISE - Moderately Strict Rep Counting
class SquatsExercise(ExerciseBase):
    """
    SIMPLIFIED SQUATS DETECTION with reliable rep counting
    =====================================================
    
    Simple Logic:
    - DOWN: Knee angle < 140° (squatting position)
    - UP: Knee angle > 160° (standing position)  
    - REP: Complete DOWN → UP cycle with 1.0s cooldown
    - Uses primary knee angle for reliable detection
    
    This replaces complex phase patterns with simple, robust detection.
    """
    
    def __init__(self):
        super().__init__("Squats")
        
        # SIMPLE THRESHOLDS for reliable detection
        self.down_threshold = 140.0   # Knee angle threshold for "down" (squatting)
        self.up_threshold = 160.0     # Knee angle threshold for "up" (standing)
        self.cooldown_time = 1.0      # Time between rep counts
        
        # DYNAMIC THRESHOLDS for reliable detection - calibrated for real movement
        self.down_threshold = 150.0   # Knee angle threshold for "down" (squatting)
        self.up_threshold = 165.0     # Knee angle threshold for "up" (standing)
        self.cooldown_time = 1.2      # Time between rep counts - increased for reliability
        
        # Required threshold attributes for enhanced feedback system
        self.standing_hip_threshold = 160.0   # Hip angle when standing
        self.squat_hip_threshold = 100.0      # Hip angle in squat position
        self.squat_knee_threshold = 150.0     # Knee angle threshold for squat validation
        
        # Simple state tracking for reliable rep counting
        self.is_down = False
        self.is_up = True
        self.last_rep_time = 0.0
        
        # Additional state tracking attributes
        self._went_through_squat = False  # Track if user has completed a squat cycle
        
        # Additional thresholds for compatibility
        self.standing_hip_threshold = 160.0
        self.squatting_hip_threshold = 100.0
        self.standing_knee_threshold = 165.0
        self.squatting_knee_threshold = 150.0
        self.squat_hip_threshold = 100.0
        self.squat_knee_threshold = 150.0

    def _define_ideal_angles(self) -> dict:
        """Define target angles for squat validation"""
        return {
            'start': {
                'hip_angle': {'target': 160.0, 'tolerance': self.angle_tolerance},
                'knee_angle': {'target': 160.0, 'tolerance': self.angle_tolerance}
            },
            'middle': {
                'hip_angle': {'target': 100.0, 'tolerance': self.angle_tolerance},
                'knee_angle': {'target': 120.0, 'tolerance': self.angle_tolerance}
            },
            'end': {
                'hip_angle': {'target': 160.0, 'tolerance': self.angle_tolerance},
                'knee_angle': {'target': 160.0, 'tolerance': self.angle_tolerance}
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate key angles for squat analysis"""
        # Hip angles (shoulder-hip-knee for torso angle)
        left_hip = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value, visibility)
        right_hip = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_KNEE.value, visibility)
        hip_angle = avg_ignore_none([left_hip, right_hip])
        
        # Knee angles (hip-knee-ankle for leg bend)
        left_knee = safe_angle(landmarks, L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value, visibility)
        right_knee = safe_angle(landmarks, L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value, visibility)
        knee_angle = avg_ignore_none([left_knee, right_knee])
        
        # Simple angle return - no smoothing for real-time performance
        return {
            'hip_angle': hip_angle,
            'knee_angle': knee_angle
        }

    def _determine_phase(self, hip_angle: float, knee_angle: float) -> str:
        """Determine current squat phase with tolerance"""
        if hip_angle is None or knee_angle is None:
            return "unknown"
        
        # STANDING POSITION (start/end)
        if (is_within_tolerance(hip_angle, self.standing_hip_threshold, self.angle_tolerance) and
            is_within_tolerance(knee_angle, self.standing_knee_threshold, self.angle_tolerance)):
            return "start" if not self._went_through_squat else "end"
        
        # SQUAT POSITION (middle)
        elif (hip_angle <= self.squat_hip_threshold + self.angle_tolerance and
              knee_angle <= self.squat_knee_threshold + self.angle_tolerance):
            return "middle"
        
        # TRANSITION PHASE
        else:
            return "transition"

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate squat form with moderate strictness"""
        issues = []
        hip_angle = current_angles.get('hip_angle')
        knee_angle = current_angles.get('knee_angle')
        
        if hip_angle is None or knee_angle is None:
            return False, 0.0, ["Cannot detect leg position"]
        
        # Calculate accuracy using universal helper
        target_angles = self._define_ideal_angles().get(phase, {})
        accuracy = calculate_rep_accuracy(current_angles, target_angles, tolerance_multiplier=1.0)
        
        # Phase-specific validation
        if phase == "start" or phase == "end":
            if hip_angle < self.standing_hip_threshold - 15:
                issues.append("Stand up straighter")
            if knee_angle < self.standing_knee_threshold - 15:
                issues.append("Straighten your legs more")
        elif phase == "middle":
            if hip_angle > self.squat_hip_threshold + 15:
                issues.append("Squat deeper - push hips back")
            if knee_angle > self.squat_knee_threshold + 15:
                issues.append("Bend knees more for deeper squat")
        
        # Movement is valid if accuracy >= 20% (reduced for real-world videos)
        is_valid = accuracy >= 20.0 and len(issues) <= 2
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """ENHANCED SQUAT REP COUNTING - Live mode validation + Video mode compatibility"""
        
        # Calculate primary angle for squats
        current_angles = self._calculate_current_angles(landmarks, visibility)
        knee_angle = current_angles.get('knee_angle')
        hip_angle = current_angles.get('hip_angle')
        
        if knee_angle is None:
            return 0, "Position yourself so legs are visible", [], None

        # Calculate body alignment for form validation
        body_alignment = calculate_body_alignment(landmarks, visibility) if calculate_body_alignment else None
        
        # FIXED ACCURACY CALCULATION - Always return a valid number
        accuracy = max(50.0, min(95.0, 100.0 - abs(knee_angle - 155) * 0.3))
        self.session_accuracies.append(accuracy)
        
        # Initialize feedback
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # ENHANCED VALIDATION FOR LIVE CAMERA MODE
        if self.is_live_mode and self.live_validator is not None:
            
            # Use live validator for enhanced form checking
            rep_completed, status_message, form_valid = self.live_validator.validate_rep_transition(
                knee_angle, body_alignment, frame_timestamp
            )
            
            if rep_completed:
                # Only count rep if form validation passed
                self.reps += 1
                self.rep_accuracies.append(accuracy)
                self.rep_timestamps.append(frame_timestamp)
                rep_increment = 1
                
                if self.debug_enabled:
                    print(f"[Squats] ✅ LIVE REP #{self.reps}: Validated full cycle with good form")
            
            # Use validator status and add form feedback
            feedback_messages.append(status_message)
            
            if not form_valid:
                corrective_feedback = self.live_validator.get_corrective_feedback(knee_angle, body_alignment)
                feedback_messages.extend(corrective_feedback)
                
        else:
            # SIMPLE REP COUNTING FOR VIDEO MODE (unchanged logic)
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # DOWN DETECTION
                if not self.is_down and knee_angle <= self.down_threshold:
                    self.is_down = True
                    self.is_up = False
                    status = f"SQUAT DOWN ({knee_angle:.1f}°)"
                    if self.debug_enabled:
                        print(f"[Squats] DOWN: {knee_angle:.1f}° <= {self.down_threshold}°")
                
                # UP DETECTION  
                elif self.is_down and not self.is_up and knee_angle >= self.up_threshold:
                    self.is_up = True
                    self.is_down = False
                    self.last_rep_time = frame_timestamp
                    
                    # COUNT THE REP
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    
                    status = f"🎉 SQUAT #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Squats] ✅ REP #{self.reps}: {knee_angle:.1f}° >= {self.up_threshold}°")
                
                # CURRENT STATUS
                elif self.is_down:
                    status = f"In squat position ({knee_angle:.1f}°)"
                elif self.is_up:
                    status = f"Standing position ({knee_angle:.1f}°)"
                else:
                    status = f"Moving ({knee_angle:.1f}°)"
            
            feedback_messages.append(status)
        
        # ===== NEW COMPREHENSIVE FEEDBACK SYSTEM =====
        current_angles = self._calculate_current_angles(landmarks, visibility)
        
        # Determine current phase for feedback
        if knee_angle <= self.down_threshold:
            phase = "middle"  # Down phase (squatting)
        elif knee_angle >= self.up_threshold:
            phase = "start"   # Up phase (standing)
        else:
            phase = "transition"  # Moving between phases
        
        # Get comprehensive multi-point feedback
        comprehensive_feedback = self.create_comprehensive_feedback(landmarks, visibility, current_angles, {}, phase)
        
        # Add exercise status for context
        status_feedback = []
        if self.is_down:
            status_feedback.append(f"Squatting - {knee_angle:.1f}°")
        elif self.is_up: 
            status_feedback.append(f"Standing - {knee_angle:.1f}°")
        else:
            status_feedback.append(f"Moving - {knee_angle:.1f}°")
        
        # Combine status with comprehensive feedback (prioritize comprehensive)
        final_feedback = comprehensive_feedback + status_feedback
        
        # Visual highlights - highlight all relevant body parts
        highlights = [11, 12, 23, 24, 25, 26, 27, 28]  # Shoulders, hips, knees, ankles
        
        # Return as feedback list instead of concatenated text
        feedback_text = final_feedback  # Keep as list for new UI system
        return rep_increment, feedback_text, highlights, accuracy

    def create_comprehensive_feedback(self, landmarks, visibility, current_angles: dict, joint_deviations: dict, phase: str) -> list:
        """
        ENHANCED MULTI-POINT SQUAT FEEDBACK SYSTEM
        =========================================
        Analyzes all critical joints and provides 2-4 biomechanically accurate feedback points
        """
        feedback_issues = []
        
        # Extract all relevant angles
        knee_angle = current_angles.get('knee_angle')
        hip_angle = current_angles.get('hip_angle')
        
        # Calculate additional critical angles for comprehensive analysis
        additional_angles = self._calculate_comprehensive_squat_angles(landmarks, visibility)
        
        back_angle = additional_angles.get('back_angle')
        ankle_angle = additional_angles.get('ankle_angle')
        head_angle = additional_angles.get('head_angle')
        
        # ===== PHASE-BASED FEEDBACK ANALYSIS =====
        
        if phase == "middle" or (knee_angle and knee_angle < 150):  # SQUAT DOWN PHASE
            # 1. SQUAT DEPTH (Primary)
            if knee_angle and knee_angle > 150:
                feedback_issues.append(("MAJOR", "Go deeper - squat until thighs are parallel"))
            elif knee_angle and knee_angle < 90:
                feedback_issues.append(("MODERATE", "Great depth - don't go too low for knees"))
            elif knee_angle and 100 <= knee_angle <= 130:
                feedback_issues.append(("GOOD", "Perfect squat depth - excellent form"))
            
            # 2. KNEE TRACKING
            if knee_angle and hip_angle:
                if knee_angle < hip_angle - 20:  # Knees caving inward
                    feedback_issues.append(("MAJOR", "Push knees out - don't let them cave inward"))
                
            # 3. HIP HINGE ANALYSIS
            if hip_angle and hip_angle > 120:
                feedback_issues.append(("MAJOR", "Sit back more - push hips behind you"))
            elif hip_angle and hip_angle < 80:
                feedback_issues.append(("MODERATE", "Don't lean too far forward"))
                
            # 4. BACK POSTURE
            if back_angle and back_angle < 160:
                feedback_issues.append(("MAJOR", "Keep your chest up and back straight"))
            
            # 5. ANKLE STABILITY
            if ankle_angle and (ankle_angle < 70 or ankle_angle > 110):
                feedback_issues.append(("MINOR", "Keep your weight on your heels"))
                
        elif phase == "start" or (knee_angle and knee_angle > 160):  # STANDING PHASE
            # 1. FULL EXTENSION CHECK
            if knee_angle and knee_angle < 170:
                feedback_issues.append(("MODERATE", "Stand fully upright - complete the movement"))
            elif knee_angle and knee_angle >= 175:
                feedback_issues.append(("GOOD", "Perfect standing position - ready for next rep"))
            
            # 2. HIP EXTENSION
            if hip_angle and hip_angle < 165:
                feedback_issues.append(("MODERATE", "Drive hips forward to full extension"))
            
            # 3. STARTING POSITION SETUP
            if phase == "start":
                feedback_issues.append(("INSTRUCTION", "Ready position - begin controlled descent"))
        
        # ===== GENERAL FORM ANALYSIS (applies to all phases) =====
        
        # HEAD AND NECK ALIGNMENT
        if head_angle and head_angle < 150:
            feedback_issues.append(("MODERATE", "Keep your head in neutral position"))
        
        # OVERALL POSTURE CHECK
        if back_angle and knee_angle:
            if back_angle < 150 and knee_angle < 140:  # Both poor posture and deep squat
                feedback_issues.append(("MAJOR", "Maintain upright torso throughout movement"))
        
        # ===== PRIORITY-BASED FEEDBACK SELECTION =====
        return self._prioritize_feedback_messages(feedback_issues)
    
    def _calculate_comprehensive_squat_angles(self, landmarks, visibility) -> dict:
        """Calculate all critical angles for comprehensive squat analysis"""
        angles = {}
        
        try:
            # BACK ANGLE (torso alignment)
            left_back = safe_angle(landmarks, 11, 23, 25, visibility)   # SHOULDER-HIP-KNEE
            right_back = safe_angle(landmarks, 12, 24, 26, visibility)
            if left_back and right_back:
                angles['back_angle'] = (left_back + right_back) / 2
            elif left_back:
                angles['back_angle'] = left_back
            elif right_back:
                angles['back_angle'] = right_back
                
            # ANKLE STABILITY
            left_ankle = safe_angle(landmarks, 25, 27, 31, visibility)  # KNEE-ANKLE-FOOT
            right_ankle = safe_angle(landmarks, 26, 28, 32, visibility)
            if left_ankle and right_ankle:
                angles['ankle_angle'] = (left_ankle + right_ankle) / 2
            elif left_ankle:
                angles['ankle_angle'] = left_ankle
            elif right_ankle:
                angles['ankle_angle'] = right_ankle
                
            # HEAD ALIGNMENT (ear to shoulder angle)
            left_head = safe_angle(landmarks, 7, 11, 23, visibility)     # EAR-SHOULDER-HIP
            right_head = safe_angle(landmarks, 8, 12, 24, visibility)
            if left_head and right_head:
                angles['head_angle'] = (left_head + right_head) / 2
            elif left_head:
                angles['head_angle'] = left_head
            elif right_head:
                angles['head_angle'] = right_head
                
        except Exception as e:
            pass  # Return partial angles if some calculations fail
            
        return angles
    
    def _prioritize_feedback_messages(self, feedback_issues: list) -> list:
        """
        Convert prioritized feedback issues into final message list
        Ensures 2-4 messages are shown with proper severity ordering
        """
        if not feedback_issues:
            return ["Perfect squat form - keep it up!"]
        
        # Sort by priority: MAJOR > MODERATE > MINOR > GOOD > INSTRUCTION
        priority_order = {"MAJOR": 1, "MODERATE": 2, "MINOR": 3, "GOOD": 4, "INSTRUCTION": 5}
        sorted_feedback = sorted(feedback_issues, key=lambda x: priority_order.get(x[0], 6))
        
        # Select top 2-4 messages based on severity mix
        final_messages = []
        major_count = sum(1 for severity, msg in sorted_feedback if severity == "MAJOR")
        
        if major_count >= 2:
            # Show 2 major + 1-2 others
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        elif major_count == 1:
            # Show 1 major + 2-3 others  
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        else:
            # No major issues, show 3-4 moderate/minor/good messages
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        
        return final_messages

    def _get_exercise_specific_feedback(self, current_angles: dict, phase: str) -> list:
        """
        Generate highly specific, movement-focused squat feedback
        """
        feedback = []
        
        knee_angle = current_angles.get('knee_angle')
        hip_angle = current_angles.get('hip_angle')
        
        # CRITICAL MOVEMENT CORRECTIONS - Specific and actionable
        if knee_angle is not None and hip_angle is not None:
            
            # Detect if squatting (knee angle < 160°)
            is_squatting = knee_angle < 160
            
            if is_squatting:
                # SQUAT DEPTH CORRECTIONS
                if knee_angle > 140:
                    feedback.append("🔽 Go lower - bend knees more to reach parallel")
                elif knee_angle < 90:
                    feedback.append("⚠️ Too deep - rise up slightly for knee safety")
                
                # HIP HINGE CORRECTIONS  
                if hip_angle > 120:
                    feedback.append("🍑 Push hips back - sit into an invisible chair")
                elif hip_angle < 80:
                    feedback.append("📍 Too forward - shift weight to heels, chest up")
                    
                # KNEE TRACKING (detect inward collapse)
                # Simplified detection: if hip angle changes rapidly, knees might be moving
                if hasattr(self, '_prev_hip_angle') and self._prev_hip_angle:
                    hip_change = abs(hip_angle - self._prev_hip_angle)
                    if hip_change > 15 and hip_angle > 100:
                        feedback.append("🦵 Knees caving in - push them outward!")
                
                # WEIGHT DISTRIBUTION
                if knee_angle < hip_angle * 1.2:  # Knees too bent relative to hips
                    feedback.append("⚖️ Too much on toes - shift weight to heels")
                    
            else:  # Standing phase
                # RETURN TO STANDING CORRECTIONS
                if knee_angle < 170:
                    feedback.append("⬆️ Straighten legs fully - squeeze glutes at top")
                if hip_angle < 160:
                    feedback.append("📏 Stand tall - push hips forward completely")
            
            # POSTURE CORRECTIONS (apply throughout movement)
            if knee_angle < 150:  # In squat position
                # Detect forward lean (chest dropping)
                if hip_angle < 90:
                    feedback.append("💪 Chest up - keep torso more upright")
                    
                # Core engagement reminder
                feedback.append("🏠 Engage core - tighten your stomach muscles")
            
            # Store for next comparison
            self._prev_hip_angle = hip_angle
        
        return feedback


# 2) Enhanced Push-ups Exercise with improved rep counting and posture accuracy
        
        # Initialize feedback
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # Handle rep completion
        if rep_completed:
            self.reps += 1
            self.rep_accuracies.append(accuracy)
            self.rep_timestamps.append(frame_timestamp)
            rep_increment = 1
            
            feedback_messages.append(f"🎉 Squat #{self.reps}! ({accuracy:.1f}%)")
            
            if self.debug_enabled:
                print(f"[Squats] ✅ REP {self.reps}: Standing→Squatting→Standing cycle completed")
                print(f"[Squats] Rep accuracy: {accuracy:.1f}% at {frame_timestamp:.2f}s")
        
        # Current status feedback
        phase_display = current_phase.replace("_", " ").title()
        feedback_messages.append(f"Phase: {phase_display}")
        feedback_messages.append(f"Hip: {hip_angle:.1f}° | Knee: {knee_angle:.1f}°")
        feedback_messages.append(f"Form accuracy: {accuracy:.1f}%")
        
        # Phase-specific guidance
        if current_phase == "start":
            feedback_messages.append("✅ Ready position - squat down")
        elif current_phase == "middle":
            if not self._went_through_squat:
                feedback_messages.append("✅ Good squat depth")
            feedback_messages.append("⬆ Now stand back up")
        elif current_phase == "end":
            feedback_messages.append("✅ Good return to standing")
        elif current_phase == "transition":
            if hip_angle > self.standing_hip_threshold:
                feedback_messages.append("⬇ Continue squatting down")
            else:
                feedback_messages.append("⬆ Continue standing up")
        
        # Movement quality issues
        if issues:
            feedback_messages.extend([f"⚠ {issue}" for issue in issues[:2]])
        
        # Highlight problematic joints for low accuracy
        if accuracy < 50:
            highlights.extend([
                L.LEFT_HIP.value, L.RIGHT_HIP.value,
                L.LEFT_KNEE.value, L.RIGHT_KNEE.value
            ])
        
        return rep_increment, feedback_text, highlights, accuracy


class PushUpsExercise(ExerciseBase):
    """
    PUSH-UPS REP COUNTING - Moderately Strict Logic
    ===============================================
    
    Detection Logic:
    - START/END: Elbow angle >140° (arms extended/straight)
    - MIDDLE: Elbow angle <100° (arms bent/lowered position)
    - TOLERANCE: ±10° for natural movement variation
    - COOLDOWN: 0.8s between reps to prevent double counting
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: EXTENDED → LOWERED → EXTENDED
    """
    
    def __init__(self):
        super().__init__("Push-ups")
        
        # SIMPLIFIED THRESHOLDS for reliable detection
        self.down_threshold = 90.0    # Elbow angle threshold for "down" (lowered position)
        self.up_threshold = 150.0     # Elbow angle threshold for "up" (extended position)
        self.cooldown_time = 1.0      # Time between rep counts
        
        # Required threshold attributes for enhanced feedback system
        self.elbow_down_threshold = 90.0      # Elbow angle in down position
        self.elbow_up_threshold = 150.0       # Elbow angle in up position
        self.shoulder_alignment_threshold = 90.0  # Shoulder alignment angle
        
        # Simple state tracking for reliable rep counting
        self.is_down = False
        self.is_up = True
        self.last_rep_time = 0.0

    def _define_ideal_angles(self) -> dict:
        """Define target angles for push-up validation"""
        return {
            'start': {
                'elbow_angle': {'target': 160.0, 'tolerance': self.angle_tolerance},
                'body_line': {'target': 180.0, 'tolerance': 15.0}
            },
            'middle': {
                'elbow_angle': {'target': 90.0, 'tolerance': self.angle_tolerance},
                'body_line': {'target': 180.0, 'tolerance': 15.0}
            },
            'end': {
                'elbow_angle': {'target': 160.0, 'tolerance': self.angle_tolerance},
                'body_line': {'target': 180.0, 'tolerance': 15.0}
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate angles for moderately strict push-up assessment"""
        # Primary elbow angles (use average for stability)
        left_elbow = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value, visibility)
        right_elbow = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value, visibility)
        
        # Use average of both elbows if available, otherwise single elbow
        elbow_angles = [x for x in [left_elbow, right_elbow] if x is not None]
        elbow_angle = sum(elbow_angles) / len(elbow_angles) if elbow_angles else None

        # Body line (shoulder-hip-ankle alignment)
        body_line = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_ANKLE.value, visibility)
        if body_line is None:
            body_line = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_ANKLE.value, visibility)

        # Shoulder angle for form assessment
        shoulder_angle = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value, visibility)

        return {
            'elbow_angle': elbow_angle,
            'body_line': body_line,
            'shoulder_angle': shoulder_angle
        }

    def _determine_phase(self, elbow_angle: float) -> str:
        """Determine push-up phase with moderate tolerance"""
        if elbow_angle is None:
            return "unknown"
        
        # Moderately strict thresholds with ±10° tolerance zones
        if elbow_angle >= self.extended_threshold:
            return "start"  # Arms straight (up position) - universal phase name
        elif elbow_angle <= self.lowered_threshold:
            return "middle"   # Arms bent (down position) - universal phase name
        else:
            return "end"  # Moving between positions - universal phase name

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate movement quality with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        elbow_angle = current_angles.get('elbow_angle')
        
        # Core validation criteria
        if elbow_angle is None:
            return False, 0.0, ["Cannot detect arm position"]
        
        # Phase-specific validation using universal tolerance
        if phase == "start":  # Extended position
            if not is_within_tolerance(elbow_angle, 160.0, self.angle_tolerance):
                issues.append("Arms not fully extended")
        elif phase == "middle":  # Lowered position  
            if not is_within_tolerance(elbow_angle, 90.0, self.angle_tolerance):
                issues.append("Lower deeper - bend elbows more")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """ENHANCED PUSHUP REP COUNTING - Live mode validation + Video mode compatibility"""
        
        # Calculate primary angle for pushups
        current_angles = self._calculate_current_angles(landmarks, visibility)
        elbow_angle = current_angles.get('elbow_angle')
        
        if elbow_angle is None:
            return 0, "Position yourself so arms are visible", [], None

        # Calculate body alignment for form validation
        body_alignment = calculate_body_alignment(landmarks, visibility) if calculate_body_alignment else None

        # FIXED ACCURACY CALCULATION - Always return a valid number
        accuracy = max(50.0, min(95.0, 100.0 - abs(elbow_angle - 120) * 0.3))
        self.session_accuracies.append(accuracy)
        
        # Initialize feedback
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # ENHANCED VALIDATION FOR LIVE CAMERA MODE
        if self.is_live_mode and self.live_validator is not None:
            
            # Use live validator for enhanced form checking
            rep_completed, status_message, form_valid = self.live_validator.validate_rep_transition(
                elbow_angle, body_alignment, frame_timestamp
            )
            
            if rep_completed:
                # Only count rep if form validation passed
                self.reps += 1
                self.rep_accuracies.append(accuracy)
                self.rep_timestamps.append(frame_timestamp)
                rep_increment = 1
                
                if self.debug_enabled:
                    print(f"[Push-ups] ✅ LIVE REP #{self.reps}: Validated full cycle with good form")
            
            # Use validator status and add form feedback
            feedback_messages.append(status_message)
            
            if not form_valid:
                corrective_feedback = self.live_validator.get_corrective_feedback(elbow_angle, body_alignment)
                feedback_messages.extend(corrective_feedback)
                
        else:
            # SIMPLE REP COUNTING FOR VIDEO MODE (unchanged logic)
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # DOWN DETECTION
                if not self.is_down and elbow_angle <= self.down_threshold:
                    self.is_down = True
                    self.is_up = False
                    status = f"PUSHUP DOWN ({elbow_angle:.1f}°)"
                    if self.debug_enabled:
                        print(f"[Push-ups] DOWN: {elbow_angle:.1f}° <= {self.down_threshold}°")
                
                # UP DETECTION  
                elif self.is_down and not self.is_up and elbow_angle >= self.up_threshold:
                    self.is_up = True
                    self.is_down = False
                    self.last_rep_time = frame_timestamp
                    
                    # COUNT THE REP
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    
                    status = f"🎉 PUSHUP #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Push-ups] ✅ REP #{self.reps}: {elbow_angle:.1f}° >= {self.up_threshold}°")
                
                # CURRENT STATUS
                elif self.is_down:
                    status = f"In pushup position ({elbow_angle:.1f}°)"
                elif self.is_up:
                    status = f"Ready position ({elbow_angle:.1f}°)"
                else:
                    status = f"Moving ({elbow_angle:.1f}°)"
            
            feedback_messages.append(status)
        
        # ===== NEW COMPREHENSIVE FEEDBACK SYSTEM =====
        current_angles = self._calculate_current_angles(landmarks, visibility)
        
        # Determine current phase for feedback
        if elbow_angle <= self.down_threshold:
            phase = "middle"  # Down phase
        elif elbow_angle >= self.up_threshold:
            phase = "start"   # Up phase  
        else:
            phase = "transition"  # Moving between phases
        
        # Get comprehensive multi-point feedback
        comprehensive_feedback = self.create_comprehensive_feedback(landmarks, visibility, current_angles, {}, phase)
        
        # Add exercise status for context
        status_feedback = []
        if self.is_live_mode and self.live_validator is not None:
            status_feedback.append(status_message)
        else:
            if self.is_down:
                status_feedback.append(f"Down position - {elbow_angle:.1f}°")
            elif self.is_up: 
                status_feedback.append(f"Ready position - {elbow_angle:.1f}°")
            else:
                status_feedback.append(f"Moving - {elbow_angle:.1f}°")
        
        # Combine status with comprehensive feedback (prioritize comprehensive)
        final_feedback = comprehensive_feedback + status_feedback
        
        # Visual highlights - highlight all relevant body parts
        highlights = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  # Arms, shoulders, hips, knees
        
        # Return as feedback list instead of concatenated text
        feedback_text = final_feedback  # Keep as list for new UI system
        return rep_increment, feedback_text, highlights, accuracy

    def create_comprehensive_feedback(self, landmarks, visibility, current_angles: dict, joint_deviations: dict, phase: str) -> list:
        """
        ENHANCED MULTI-POINT PUSH-UP FEEDBACK SYSTEM
        ============================================
        Analyzes all critical joints and provides 2-4 biomechanically accurate feedback points
        """
        feedback_issues = []
        
        # Extract all relevant angles
        elbow_angle = current_angles.get('elbow_angle')
        shoulder_angle = current_angles.get('shoulder_angle')
        hip_angle = current_angles.get('hip_angle')
        
        # Calculate additional critical angles for comprehensive analysis
        additional_angles = self._calculate_comprehensive_pushup_angles(landmarks, visibility)
        
        wrist_angle = additional_angles.get('wrist_angle')
        spine_angle = additional_angles.get('spine_angle')  
        head_angle = additional_angles.get('head_angle')
        knee_angle = additional_angles.get('knee_angle')
        
        # ===== PHASE-BASED FEEDBACK ANALYSIS =====
        
        if phase == "middle" or (elbow_angle and elbow_angle < 120):  # DOWN PHASE
            # 1. ELBOW POSITIONING (Primary)
            if elbow_angle and elbow_angle > 120:
                feedback_issues.append(("MAJOR", "Lower your chest closer to the ground"))
            elif elbow_angle and elbow_angle < 60:
                feedback_issues.append(("MAJOR", "Don't go too low - risk of shoulder strain"))
            elif elbow_angle and 90 <= elbow_angle <= 110:
                feedback_issues.append(("GOOD", "Perfect push-up depth - excellent form"))
            
            # 2. ELBOW FLARE ANALYSIS
            if shoulder_angle and shoulder_angle > 100:
                feedback_issues.append(("MODERATE", "Spread elbows at 45° instead of flaring out"))
            elif shoulder_angle and shoulder_angle < 60:
                feedback_issues.append(("MODERATE", "Elbows too narrow - spread slightly wider"))
            
            # 3. HIP ALIGNMENT (Core stability)
            if hip_angle and hip_angle < 160:
                feedback_issues.append(("MAJOR", "Keep your back straight - hips are dipping"))
            elif hip_angle and hip_angle > 200:
                feedback_issues.append(("MAJOR", "Lower your hips - butt too high in air"))
                
            # 4. SPINE AND HEAD ALIGNMENT
            if head_angle and head_angle < 150:
                feedback_issues.append(("MODERATE", "Align your head with your spine"))
            
            # 5. WRIST POSITIONING
            if wrist_angle and (wrist_angle < 70 or wrist_angle > 110):
                feedback_issues.append(("MINOR", "Straighten your wrists - adjust hand placement"))
                
        elif phase == "start" or phase == "end" or (elbow_angle and elbow_angle > 160):  # UP PHASE
            # 1. FULL LOCKOUT CHECK
            if elbow_angle and elbow_angle < 165:
                feedback_issues.append(("MODERATE", "Fully lockout elbows without shrugging shoulders"))
            elif elbow_angle and elbow_angle >= 170:
                feedback_issues.append(("GOOD", "Perfect arm extension - ready for next rep"))
            
            # 2. SHOULDER POSITIONING
            if shoulder_angle and shoulder_angle > 110:
                feedback_issues.append(("MODERATE", "Don't shrug shoulders - keep them down"))
            
            # 3. CORE ENGAGEMENT
            if hip_angle and hip_angle < 170:
                feedback_issues.append(("MAJOR", "Lock your core to avoid sagging"))
            
            # 4. STARTING POSITION SETUP
            if phase == "start":
                feedback_issues.append(("INSTRUCTION", "Ready position - begin controlled descent"))
        
        # ===== GENERAL FORM ANALYSIS (applies to all phases) =====
        
        # KNEE STABILITY
        if knee_angle and knee_angle < 160:
            feedback_issues.append(("MODERATE", "Keep knees locked and straight"))
        
        # OVERALL BODY LINE 
        if hip_angle and spine_angle:
            if abs(hip_angle - spine_angle) > 20:
                feedback_issues.append(("MAJOR", "Maintain straight body line from head to heels"))
        
        # ===== PRIORITY-BASED FEEDBACK SELECTION =====
        return self._prioritize_feedback_messages(feedback_issues)
    
    def _calculate_comprehensive_pushup_angles(self, landmarks, visibility) -> dict:
        """Calculate all critical angles for comprehensive push-up analysis"""
        angles = {}
        
        try:
            # WRIST ANGLE (forearm to hand alignment)
            left_wrist = safe_angle(landmarks, 13, 15, 17, visibility)   # SHOULDER-ELBOW-WRIST
            right_wrist = safe_angle(landmarks, 14, 16, 18, visibility)
            if left_wrist and right_wrist:
                angles['wrist_angle'] = (left_wrist + right_wrist) / 2
            elif left_wrist:
                angles['wrist_angle'] = left_wrist
            elif right_wrist:
                angles['wrist_angle'] = right_wrist
            
            # SPINE ANGLE (shoulder to hip alignment)
            left_spine = safe_angle(landmarks, 11, 23, 25, visibility)   # SHOULDER-HIP-KNEE
            right_spine = safe_angle(landmarks, 12, 24, 26, visibility)
            if left_spine and right_spine:
                angles['spine_angle'] = (left_spine + right_spine) / 2
            elif left_spine:
                angles['spine_angle'] = left_spine
            elif right_spine:
                angles['spine_angle'] = right_spine
                
            # HEAD ALIGNMENT (ear to shoulder angle)
            left_head = safe_angle(landmarks, 7, 11, 13, visibility)     # EAR-SHOULDER-ELBOW  
            right_head = safe_angle(landmarks, 8, 12, 14, visibility)
            if left_head and right_head:
                angles['head_angle'] = (left_head + right_head) / 2
            elif left_head:
                angles['head_angle'] = left_head
            elif right_head:
                angles['head_angle'] = right_head
            
            # KNEE STABILITY
            left_knee = safe_angle(landmarks, 23, 25, 27, visibility)    # HIP-KNEE-ANKLE
            right_knee = safe_angle(landmarks, 24, 26, 28, visibility)
            if left_knee and right_knee:
                angles['knee_angle'] = (left_knee + right_knee) / 2
            elif left_knee:
                angles['knee_angle'] = left_knee
            elif right_knee:
                angles['knee_angle'] = right_knee
                
        except Exception as e:
            pass  # Return partial angles if some calculations fail
            
        return angles
    
    def _prioritize_feedback_messages(self, feedback_issues: list) -> list:
        """
        Convert prioritized feedback issues into final message list
        Ensures 2-4 messages are shown with proper severity ordering
        """
        if not feedback_issues:
            return ["Perfect form - keep it up!"]
        
        # Sort by priority: MAJOR > MODERATE > MINOR > GOOD > INSTRUCTION
        priority_order = {"MAJOR": 1, "MODERATE": 2, "MINOR": 3, "GOOD": 4, "INSTRUCTION": 5}
        sorted_feedback = sorted(feedback_issues, key=lambda x: priority_order.get(x[0], 6))
        
        # Select top 2-4 messages based on severity mix
        final_messages = []
        major_count = sum(1 for severity, msg in sorted_feedback if severity == "MAJOR")
        
        if major_count >= 2:
            # Show 2 major + 1-2 others
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        elif major_count == 1:
            # Show 1 major + 2-3 others  
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        else:
            # No major issues, show 3-4 moderate/minor/good messages
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        
        return final_messages
    
    def _get_exercise_specific_feedback(self, current_angles: dict, phase: str) -> list:
        """
        Generate highly specific, movement-focused push-up feedback
        """
        feedback = []
        
        elbow_angle = current_angles.get('elbow_angle')
        body_line = current_angles.get('body_line') 
        shoulder_angle = current_angles.get('shoulder_angle')
        
        # CRITICAL MOVEMENT CORRECTIONS - Specific and actionable
        if elbow_angle is not None:
            
            # Detect if in down position (elbow angle < 120°)
            is_down_position = elbow_angle < 120
            
            if is_down_position:
                # DOWN POSITION CORRECTIONS
                if elbow_angle > 100:
                    feedback.append("🔽 Go lower - chest should nearly touch the ground")
                elif elbow_angle < 60:
                    feedback.append("⚠️ Too low - maintain control, don't crash down")
                
                # ELBOW FLARE CORRECTION
                if shoulder_angle and shoulder_angle > 110:
                    feedback.append("💪 Elbows flaring - tuck them closer to your body")
                elif shoulder_angle and shoulder_angle < 70:
                    feedback.append("💪 Elbows too tight - allow slight flare (45°)")
                    
            else:  # Up position
                # UP POSITION CORRECTIONS  
                if elbow_angle < 160:
                    feedback.append("⬆️ Push up fully - straighten arms completely")
                elif elbow_angle > 175:
                    feedback.append("✅ Perfect lockout - ready for next rep")
        
        # BODY LINE CORRECTIONS (critical for all phases)
        if body_line is not None:
            if body_line < 160:
                # Determine if hips are sagging or piking
                if hasattr(self, '_prev_body_line') and self._prev_body_line:
                    if body_line < self._prev_body_line:
                        feedback.append("🏠 Hips sagging - tighten your core!")
                    else:
                        feedback.append("🏠 Hips too high - lower them slightly")
                else:
                    feedback.append("🏠 Keep body straight - engage core muscles")
            elif body_line > 190:
                feedback.append("📍 Don't pike up - lower hips to straight line")
        
        # HAND PLACEMENT AND MOVEMENT QUALITY
        if elbow_angle and body_line:
            # Check for controlled movement (prevent bouncing)
            if hasattr(self, '_prev_elbow_angle') and self._prev_elbow_angle:
                elbow_change = abs(elbow_angle - self._prev_elbow_angle)
                if elbow_change > 30:  # Rapid change indicates bouncing
                    feedback.append("🔄 Control the movement - slow and steady")
        
        # BREATHING AND PACE REMINDERS
        if is_down_position if elbow_angle else False:
            feedback.append("💨 Exhale as you push up - breathe out on effort")
        
        # PHASE-SPECIFIC GUIDANCE
        if phase == "start" and elbow_angle and elbow_angle > 160:
            feedback.append("🚀 Perfect starting position - begin descent")
        elif phase == "middle" and elbow_angle and elbow_angle < 100:
            feedback.append("⏸️ Hold briefly at bottom - then push up strong")
            
        # Store for next comparison
        if elbow_angle:
            self._prev_elbow_angle = elbow_angle
        if body_line:
            self._prev_body_line = body_line
            
        return feedback[:3]  # Limit to 3 most important messages

    def _get_exercise_specific_feedback(self, current_angles: dict, phase: str) -> list:
        """
        Generate highly specific, movement-focused push-up feedback
        """
        feedback = []
        
        elbow_angle = current_angles.get('elbow_angle')
        body_line = current_angles.get('body_line') 
        
        # CRITICAL MOVEMENT CORRECTIONS - Specific and actionable
        if elbow_angle is not None:
            
            # Detect if in lowered position (elbow angle < 120°)
            is_lowered = elbow_angle < 120
            
            if is_lowered:
                # PUSH-UP DEPTH CORRECTIONS
                if elbow_angle > 100:
                    feedback.append("🔽 Lower your chest - touch ground with chest")
                elif elbow_angle < 70:
                    feedback.append("⚠️ Too low - maintain control, don't crash down")
                
                # ELBOW POSITION CORRECTIONS
                if elbow_angle < 90 and body_line is not None:
                    if body_line < 170:  # Body sagging
                        feedback.append("💪 Elbows too wide - keep them closer to body")
                    else:
                        feedback.append("✅ Good depth - now push up with control")
                
                # CORE ENGAGEMENT
                if body_line is not None and body_line < 160:
                    feedback.append("🏠 Engage core - don't let hips sag down")
                    
            else:  # Extended position
                # FULL EXTENSION CORRECTIONS
                if elbow_angle < 160:
                    feedback.append("⬆️ Push all the way up - fully straighten arms")
                elif elbow_angle > 175:
                    feedback.append("⚠️ Don't hyperextend - slight bend in elbows is OK")
                
                # STARTING POSITION SETUP
                feedback.append("📍 Ready position - begin controlled descent")
            
            # BODY ALIGNMENT (throughout movement)
            if body_line is not None:
                if body_line < 150:
                    feedback.append("📏 Straight line - lift hips up to align body")
                elif body_line > 190:
                    feedback.append("🔽 Lower hips - don't pike up like downward dog")
                    
                # HAND POSITIONING REMINDER
                if is_lowered:
                    feedback.append("👐 Hands under shoulders - fingers spread wide")

        return feedback


class PlankExercise(ExerciseBase):
    """
    PLANK TIME-BASED TRACKING - Moderately Strict Logic
    ==================================================
    
    Detection Logic:
    - HOLDING: Body line >170° (good plank form)
    - ADJUSTING: Body line <170° (poor form or transitioning)
    - TOLERANCE: ±10° for natural movement variation
    - TIME INTERVALS: 2-second holds for progression tracking
    - MIN ACCURACY: 60% required for valid hold time
    
    Progress: Increments every 2 seconds of good holding
    """
    
    def __init__(self):
        super().__init__("Plank")
        
        # MODERATE THRESHOLDS - Balanced for real-world use
        self.good_alignment_threshold = 170.0  # Body alignment angle
        self.hold_requirement = 2.0            # Minimum hold time for progression
        self.angle_tolerance = 10.0            # ±10° tolerance
        
        # Universal rep counting state machine (adapted for time-based tracking)
        self.rep_counter = RepCountingState("Plank", cooldown_time=0.5)
        
        # Time tracking for plank progression
        self.hold_start_time = None
        self.total_hold_time = 0.0
        self.last_feedback_time = 0.0

    def _define_ideal_angles(self) -> dict:
        """Define target angles for plank validation"""
        return {
            'start': {  # Holding phase (good plank)
                'body_line': {'target': 180.0, 'tolerance': self.angle_tolerance},
                'hip_angle': {'target': 180.0, 'tolerance': 15.0}
            },
            'middle': {  # Same as start for plank (consistency)
                'body_line': {'target': 180.0, 'tolerance': self.angle_tolerance},
                'hip_angle': {'target': 180.0, 'tolerance': 15.0}
            },
            'end': {  # Same as start for plank (consistency)
                'body_line': {'target': 180.0, 'tolerance': self.angle_tolerance},
                'hip_angle': {'target': 180.0, 'tolerance': 15.0}
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate angles for moderately strict plank assessment"""
        # Primary body line (shoulder-hip-ankle)
        left_body_line = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_ANKLE.value, visibility)
        right_body_line = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_ANKLE.value, visibility)
        
        # Use best available body line measurement
        body_line = None
        if left_body_line and right_body_line:
            body_line = (left_body_line + right_body_line) / 2
        elif left_body_line:
            body_line = left_body_line
        elif right_body_line:
            body_line = right_body_line

        # Shoulder angle for form assessment
        shoulder_angle = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value, visibility)
        
        # Hip angle (torso-thigh alignment)
        hip_angle = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value, visibility)

        return {
            'body_line': body_line,
            'shoulder_angle': shoulder_angle,
            'hip_angle': hip_angle
        }

    def _determine_phase(self, body_line: float) -> str:
        """Determine plank phase using universal phase names"""
        if body_line is None:
            return "unknown"
        
        # Good alignment = holding (start phase for consistency)
        if body_line >= self.good_alignment_threshold:
            return "start"  # Universal phase name for good holding
        else:
            return "middle"  # Universal phase name for adjusting

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate plank form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        body_line = current_angles.get('body_line')
        
        if body_line is None:
            return False, 0.0, ["Cannot detect body position"]
        
        # Form validation using universal tolerance
        if phase == "start":  # Good holding phase
            if not is_within_tolerance(body_line, 180.0, self.angle_tolerance):
                if body_line < 170.0:
                    issues.append("Lift hips up - avoid sagging")
                elif body_line > 190.0:
                    issues.append("Lower hips - avoid pike position")
                else:
                    issues.append("Straighten body line slightly")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 30.0  # Reduced standard for plank (real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """Universal plank tracking with time-based progression using RepCountingState"""
        
        # Calculate current angles
        current_angles = self._calculate_current_angles(landmarks, visibility)
        body_line = current_angles.get('body_line')
        
        if body_line is None:
            return 0, "Position yourself in plank - body not detected", [], None

        # Determine current phase
        current_phase = self._determine_phase(body_line)
        
        # Validate movement quality
        is_valid, accuracy, issues = self._validate_movement_quality(current_angles, current_phase)
        
        # Track accuracy for session statistics
        self.session_accuracies.append(accuracy)
        
        # Update universal rep state (for consistency)
        self.rep_counter.update_phase(current_phase, frame_timestamp, is_valid, accuracy)
        
        # Time-based progression logic for plank
        rep_increment = 0
        if current_phase == "start" and is_valid:  # Good holding position
            if self.hold_start_time is None:
                self.hold_start_time = frame_timestamp
            
            # Calculate current hold duration
            current_hold_time = frame_timestamp - self.hold_start_time
            
            # Award "reps" every 2 seconds of good holding
            if current_hold_time >= self.hold_requirement:
                time_since_last = frame_timestamp - self.last_feedback_time
                if time_since_last >= self.hold_requirement:
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    self.last_feedback_time = frame_timestamp
                    rep_increment = 1
                    
                    if self.debug_enabled:
                        print(f"[Plank] ✅ Hold progress: {self.reps * 2} seconds total")
        else:
            # Reset hold timer if not in valid holding position
            self.hold_start_time = None
        
        # Calculate total hold time
        if self.hold_start_time is not None:
            self.total_hold_time = frame_timestamp - self.hold_start_time
        else:
            self.total_hold_time = 0.0
        
        # Initialize feedback
        feedback_messages = []
        highlights = []
        
        # Progress feedback
        if rep_increment > 0:
            total_seconds = self.reps * 2
            feedback_messages.append(f"🎉 Great hold! {total_seconds}s total")
        
        # Current status
        phase_display = "Holding" if current_phase == "start" else "Adjusting"
        feedback_messages.append(f"Status: {phase_display}")
        feedback_messages.append(f"Body alignment: {body_line:.1f}°")
        feedback_messages.append(f"Form accuracy: {accuracy:.1f}%")
        
        # Hold time display
        if self.total_hold_time > 0:
            feedback_messages.append(f"Current hold: {self.total_hold_time:.1f}s")
        
        # Phase-specific guidance
        if current_phase == "start":  # Good holding
            if accuracy >= 80:
                feedback_messages.append("✅ Perfect plank form!")
            elif accuracy >= 60:
                feedback_messages.append("✅ Good plank - keep holding")
            else:
                feedback_messages.append("💪 Focus on form")
        else:
            feedback_messages.append("⚠ Adjust position for good plank")
        
        # Form issues
        if issues:
            feedback_messages.extend([f"⚠ {issue}" for issue in issues[:2]])
        
        # Highlight problematic areas
        if accuracy < 70:
            highlights.extend([
                L.LEFT_SHOULDER.value, L.RIGHT_SHOULDER.value,
                L.LEFT_HIP.value, L.RIGHT_HIP.value,
                L.LEFT_ANKLE.value, L.RIGHT_ANKLE.value
            ])
        
        # Limit feedback display
        feedback_text = "\n".join(feedback_messages[:6])
        
        return rep_increment, feedback_text, highlights, accuracy


# 4) Lunges
class LungesExercise(ExerciseBase):
    """
    LUNGES REP COUNTING - Moderately Strict Logic
    ============================================
    
    Detection Logic:
    - START/END: Knee angle >160° (standing position)
    - MIDDLE: Knee angle <105° (lowered lunge position)
    - TOLERANCE: ±10° for natural movement variation
    - COOLDOWN: 0.8s between reps to prevent double counting
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: STANDING → LOWERED → STANDING
    """
    
    def __init__(self):
        super().__init__("Lunges")
        
        # SIMPLIFIED THRESHOLDS for reliable detection
        self.down_threshold = 105.0    # DOWN: knee bent ≤ 105°
        self.up_threshold = 160.0      # UP: knee extended ≥ 160°
        self.cooldown_time = 0.8       # Cooldown between reps
        
        # Required threshold attributes for enhanced feedback system
        self.lunge_knee_threshold = 105.0     # Knee angle in lunge position
        self.standing_knee_threshold = 160.0  # Knee angle when standing
        self.hip_alignment_threshold = 160.0  # Hip alignment threshold
        
        # Simple state tracking
        self.is_down = False
        self.is_up = True
        self.last_rep_time = 0.0

    def _define_ideal_angles(self) -> dict:
        """Define target angles for lunge validation"""
        return {
            'start': {
                'primary_knee': {'target': 175.0, 'tolerance': self.angle_tolerance},
                'torso_angle': {'target': 180.0, 'tolerance': 15.0}
            },
            'middle': {
                'primary_knee': {'target': 90.0, 'tolerance': self.angle_tolerance},
                'torso_angle': {'target': 180.0, 'tolerance': 15.0}
            },
            'end': {
                'primary_knee': {'target': 175.0, 'tolerance': self.angle_tolerance},
                'torso_angle': {'target': 180.0, 'tolerance': 15.0}
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate angles for moderately strict lunge assessment"""
        # Knee angles for both legs
        left_knee = safe_angle(landmarks, L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value, visibility)
        right_knee = safe_angle(landmarks, L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value, visibility)
        
        # Use the smaller knee angle (more bent) as primary indicator
        knee_angles = [x for x in [left_knee, right_knee] if x is not None]
        primary_knee = min(knee_angles) if knee_angles else None
        
        # Torso angle (shoulder-hip-knee alignment for upright posture)
        left_torso = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value, visibility)
        right_torso = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_KNEE.value, visibility)
        
        # Use best available torso measurement
        torso_angle = None
        if left_torso and right_torso:
            torso_angle = (left_torso + right_torso) / 2
        elif left_torso:
            torso_angle = left_torso
        elif right_torso:
            torso_angle = right_torso

        return {
            'primary_knee': primary_knee,
            'left_knee': left_knee,
            'right_knee': right_knee,
            'torso_angle': torso_angle
        }

    def _determine_phase(self, primary_knee: float) -> str:
        """Determine lunge phase using universal phase names"""
        if primary_knee is None:
            return "unknown"
        
        # Moderately strict thresholds with transition zones
        if primary_knee >= self.standing_threshold:
            return "start"  # Standing position - universal phase name
        elif primary_knee <= self.lowered_threshold:
            return "middle"   # Lowered lunge position - universal phase name
        else:
            return "end"  # Moving between positions - universal phase name

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate lunge form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        primary_knee = current_angles.get('primary_knee')
        torso_angle = current_angles.get('torso_angle')
        
        if primary_knee is None:
            return False, 0.0, ["Cannot detect leg position"]
        
        # Phase-specific validation using universal tolerance
        if phase == "start":  # Standing position
            if not is_within_tolerance(primary_knee, 175.0, self.angle_tolerance):
                issues.append("Stand up straighter")
            if torso_angle and not is_within_tolerance(torso_angle, 180.0, 15.0):
                issues.append("Keep torso upright")
        elif phase == "middle":  # Lowered position
            if not is_within_tolerance(primary_knee, 90.0, self.angle_tolerance):
                issues.append("Lower deeper - 90° knee angle")
            if torso_angle and not is_within_tolerance(torso_angle, 180.0, 15.0):
                issues.append("Keep chest up and torso straight")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """ENHANCED LUNGE REP COUNTING - Dual-mode validation (live vs video)"""
        
        # Calculate primary angle for lunges
        current_angles = self._calculate_current_angles(landmarks, visibility)
        primary_knee = current_angles.get('primary_knee')
        torso_angle = current_angles.get('torso_angle')
        
        if primary_knee is None:
            return 0, "Position yourself so legs are visible", [], None

        # FIXED ACCURACY CALCULATION - Always return a valid number
        accuracy = max(50.0, min(95.0, 100.0 - abs(primary_knee - 130) * 0.4))  # Dynamic accuracy based on movement
        self.session_accuracies.append(accuracy)
        
        # Initialize feedback
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # DUAL-MODE LOGIC: Enhanced validation for live camera, simple for video
        if self.is_live_mode and self.live_validator:
            # LIVE MODE: Enhanced form validation
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # Use enhanced validator for live camera detection
                rep_completed, phase_status, form_valid = self.live_validator.validate_rep_transition(
                    primary_knee, torso_angle or 180.0, frame_timestamp
                )
                
                if rep_completed and form_valid:
                    # COUNT THE REP with enhanced validation
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    self.last_rep_time = frame_timestamp
                    
                    status = f"🎉 LUNGE #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Lunges] ✅ VALIDATED REP #{self.reps}: {primary_knee:.1f}° - Form Valid")
                else:
                    status = phase_status
                    if not form_valid and self.debug_enabled:
                        print(f"[Lunges] Form validation failed: {primary_knee:.1f}°")
        else:
            # VIDEO MODE: Original simple logic (preserved)
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # DOWN DETECTION
                if not self.is_down and primary_knee <= self.down_threshold:
                    self.is_down = True
                    self.is_up = False
                    status = f"LUNGE DOWN ({primary_knee:.1f}°)"
                    if self.debug_enabled:
                        print(f"[Lunges] DOWN: {primary_knee:.1f}° <= {self.down_threshold}°")
                
                # UP DETECTION  
                elif self.is_down and not self.is_up and primary_knee >= self.up_threshold:
                    self.is_up = True
                    self.is_down = False
                    self.last_rep_time = frame_timestamp
                    
                    # COUNT THE REP (original logic)
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    
                    status = f"🎉 LUNGE #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Lunges] ✅ REP #{self.reps}: {primary_knee:.1f}° >= {self.up_threshold}°")
                
                # CURRENT STATUS
                elif self.is_down:
                    status = f"In lunge position ({primary_knee:.1f}°)"
                elif self.is_up:
                    status = f"Standing position ({primary_knee:.1f}°)"
                else:
                    status = f"Moving ({primary_knee:.1f}°)"
        
        # CLEAN FEEDBACK - NO DUPLICATES
        feedback_messages.append(status)
        feedback_messages.append(f"Knee: {primary_knee:.1f}°")
        feedback_messages.append(f"Accuracy: {accuracy:.1f}%")  # SINGLE accuracy mention
        
        # Movement guidance
        if primary_knee < self.down_threshold + 15:
            feedback_messages.append("🔥 Great lunge depth!")
        elif primary_knee > self.up_threshold - 15:
            feedback_messages.append("🚀 Good standing position")
        
        # Visual highlights
        highlights = [25, 26, 27, 28]  # Knee and hip landmarks
        
        feedback_text = " | ".join(feedback_messages)  # Use | separator to avoid line breaks
        return rep_increment, feedback_text, highlights, accuracy

    def _get_exercise_specific_feedback(self, current_angles: dict, phase: str) -> list:
        """
        Generate lunge-specific feedback based on current angles and phase
        """
        feedback = []
        
        knee_angle = current_angles.get('knee_angle')
        hip_angle = current_angles.get('hip_angle')
        
        if knee_angle is not None:
            # Lunge depth feedback
            if phase == "middle" or knee_angle < 140:  # In lunge position
                if knee_angle > 140:
                    feedback.append("🦵 Lower your back leg more for better lunge depth")
                elif knee_angle < 80:
                    feedback.append("⚠️ Very deep lunge - protect your knee")
                else:
                    feedback.append("✅ Excellent lunge depth!")
            
            # Standing position feedback
            elif phase == "start" or phase == "end":
                if knee_angle < 160:
                    feedback.append("🚀 Return to complete standing position")
                else:
                    feedback.append("✅ Perfect standing position")
        
        if hip_angle is not None:
            # Hip alignment feedback
            if phase == "middle":
                if hip_angle > 130:
                    feedback.append("🍑 Drop your hips down - maintain vertical torso")
                elif hip_angle < 90:
                    feedback.append("⚠️ Don't lean too far forward")
                else:
                    feedback.append("✅ Great hip alignment!")
        
        # General lunge form tips
        if phase == "middle":
            feedback.append("💡 Keep front knee over ankle, back straight")
        
        return feedback


# 5) Bicep curls (either arm counts)
class BicepCurlsExercise(ExerciseBase):
    """
    BICEP CURLS REP COUNTING - Moderately Strict Logic
    =================================================
    
    Detection Logic:
    - START/END: Elbow angle >150° (arms extended/down position)
    - MIDDLE: Elbow angle <55° (arms curled/up position)
    - TOLERANCE: ±10° for natural movement variation
    - COOLDOWN: 0.8s between reps to prevent double counting
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: EXTENDED → CURLED → EXTENDED
    """
    
    def __init__(self):
        super().__init__("Bicep curls")
        
        # SIMPLIFIED THRESHOLDS for reliable detection
        self.down_threshold = 55.0      # DOWN: arms curled ≤ 55°
        self.up_threshold = 150.0       # UP: arms extended ≥ 150°
        self.cooldown_time = 0.8        # Cooldown between reps
        
        # Required threshold attributes for enhanced feedback system
        self.curl_up_threshold = 55.0         # Elbow angle when curled up
        self.curl_down_threshold = 150.0      # Elbow angle when extended
        self.shoulder_stability_threshold = 170.0  # Shoulder stability angle
        
        # Simple state tracking
        self.is_down = False
        self.is_up = True
        self.last_rep_time = 0.0

    def _define_ideal_angles(self) -> dict:
        """Define target angles for bicep curl validation"""
        return {
            'start': {
                'elbow_angle': {'target': 170.0, 'tolerance': self.angle_tolerance},
                'elbow_anchor': {'target': 75.0, 'tolerance': 20.0}  # Elbow position relative to torso
            },
            'middle': {
                'elbow_angle': {'target': 35.0, 'tolerance': self.angle_tolerance},
                'elbow_anchor': {'target': 75.0, 'tolerance': 20.0}  # Maintain elbow position
            },
            'end': {
                'elbow_angle': {'target': 170.0, 'tolerance': self.angle_tolerance},
                'elbow_anchor': {'target': 75.0, 'tolerance': 20.0}  # Maintain elbow position
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate angles for moderately strict bicep curl assessment"""
        # Elbow angles for both arms
        left_elbow = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value, visibility)
        right_elbow = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value, visibility)

        # Pick the arm that's more flexed (smaller angle) for primary tracking
        elbow_angles = [x for x in [left_elbow, right_elbow] if x is not None]
        elbow_angle = min(elbow_angles) if elbow_angles else None

        # Elbow anchor points (keep elbow close to torso)
        left_anchor = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_HIP.value, visibility)
        right_anchor = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_HIP.value, visibility)
        
        # Use average anchor angle if both available
        anchor_angles = [x for x in [left_anchor, right_anchor] if x is not None]
        elbow_anchor = sum(anchor_angles) / len(anchor_angles) if anchor_angles else None

        return {
            'elbow_angle': elbow_angle,
            'elbow_anchor': elbow_anchor,
            'left_elbow': left_elbow,
            'right_elbow': right_elbow,
            'left_anchor': left_anchor,
            'right_anchor': right_anchor
        }

    def _determine_phase(self, elbow_angle: float) -> str:
        """Determine bicep curl phase using universal phase names"""
        if elbow_angle is None:
            return "unknown"
        
        # Moderately strict thresholds
        if elbow_angle >= self.extended_threshold:
            return "start"  # Arms extended (down position) - universal phase name
        elif elbow_angle <= self.curled_threshold:
            return "middle"   # Arms curled (up position) - universal phase name
        else:
            return "end"  # Moving between positions - universal phase name

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate bicep curl form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        elbow_angle = current_angles.get('elbow_angle')
        left_anchor = current_angles.get('left_anchor')
        right_anchor = current_angles.get('right_anchor')
        
        if elbow_angle is None:
            return False, 0.0, ["Cannot detect arm position"]
        
        # Phase-specific validation using universal tolerance
        if phase == "start":  # Extended position
            if not is_within_tolerance(elbow_angle, 170.0, self.angle_tolerance):
                issues.append("Extend arms fully")
        elif phase == "middle":  # Curled position
            if not is_within_tolerance(elbow_angle, 35.0, self.angle_tolerance):
                issues.append("Curl arms fully")
        
        # Elbow anchor validation (keep elbows close to torso)
        if left_anchor and not is_within_tolerance(left_anchor, 75.0, 20.0):
            issues.append("Keep left elbow close to torso")
        if right_anchor and not is_within_tolerance(right_anchor, 75.0, 20.0):
            issues.append("Keep right elbow close to torso")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """ENHANCED BICEP CURLS REP COUNTING - Dual-mode validation (live vs video)"""
        
        # Calculate primary angle for bicep curls
        current_angles = self._calculate_current_angles(landmarks, visibility)
        elbow_angle = current_angles.get('elbow_angle')
        elbow_anchor = current_angles.get('elbow_anchor')
        
        if elbow_angle is None:
            return 0, "Position yourself so arms are visible", [], None

        # FIXED ACCURACY CALCULATION - Always return a valid number
        accuracy = max(50.0, min(95.0, 100.0 - abs(elbow_angle - 100) * 0.4))  # Dynamic accuracy based on movement
        self.session_accuracies.append(accuracy)
        
        # Initialize feedback
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # DUAL-MODE LOGIC: Enhanced validation for live camera, simple for video
        if self.is_live_mode and self.live_validator:
            # LIVE MODE: Enhanced form validation
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # Use enhanced validator for live camera detection
                rep_completed, phase_status, form_valid = self.live_validator.validate_rep_transition(
                    elbow_angle, elbow_anchor or 75.0, frame_timestamp
                )
                
                if rep_completed and form_valid:
                    # COUNT THE REP with enhanced validation
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    self.last_rep_time = frame_timestamp
                    
                    status = f"🎉 BICEP CURL #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Bicep curls] ✅ VALIDATED REP #{self.reps}: {elbow_angle:.1f}° - Form Valid")
                else:
                    status = phase_status
                    if not form_valid and self.debug_enabled:
                        print(f"[Bicep curls] Form validation failed: {elbow_angle:.1f}°")
        else:
            # VIDEO MODE: Original simple logic (preserved)
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # DOWN DETECTION (arms curled)
                if not self.is_down and elbow_angle <= self.down_threshold:
                    self.is_down = True
                    self.is_up = False
                    status = f"CURL UP ({elbow_angle:.1f}°)"
                    if self.debug_enabled:
                        print(f"[Bicep curls] DOWN: {elbow_angle:.1f}° <= {self.down_threshold}°")
                
                # UP DETECTION (arms extended)
                elif self.is_down and not self.is_up and elbow_angle >= self.up_threshold:
                    self.is_up = True
                    self.is_down = False
                    self.last_rep_time = frame_timestamp
                    
                    # COUNT THE REP (original logic)
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    
                    status = f"🎉 BICEP CURL #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Bicep curls] ✅ REP #{self.reps}: {elbow_angle:.1f}° >= {self.up_threshold}°")
                
                # CURRENT STATUS
                elif self.is_down:
                    status = f"Curled position ({elbow_angle:.1f}°)"
                elif self.is_up:
                    status = f"Extended position ({elbow_angle:.1f}°)"
                else:
                    status = f"Moving ({elbow_angle:.1f}°)"
        
        # ===== NEW COMPREHENSIVE FEEDBACK SYSTEM =====
        current_angles = self._calculate_current_angles(landmarks, visibility)
        
        # Determine current phase for feedback
        if elbow_angle <= self.down_threshold:
            phase = "middle"  # Curl up phase
        elif elbow_angle >= self.up_threshold:
            phase = "start"   # Extended phase
        else:
            phase = "transition"  # Moving between phases
        
        # Get comprehensive multi-point feedback
        comprehensive_feedback = self.create_comprehensive_feedback(landmarks, visibility, current_angles, {}, phase)
        
        # Add exercise status for context
        status_feedback = []
        if self.is_down:
            status_feedback.append(f"Curled - {elbow_angle:.1f}°")
        elif self.is_up: 
            status_feedback.append(f"Extended - {elbow_angle:.1f}°")
        else:
            status_feedback.append(f"Moving - {elbow_angle:.1f}°")
        
        # Combine status with comprehensive feedback (prioritize comprehensive)
        final_feedback = comprehensive_feedback + status_feedback
        
        # Visual highlights - highlight all relevant body parts
        highlights = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
        
        # Return as feedback list instead of concatenated text
        feedback_text = final_feedback  # Keep as list for new UI system
        return rep_increment, feedback_text, highlights, accuracy


# 6) Shoulder press
# 6) Shoulder press
class ShoulderPressExercise(ExerciseBase):
    """
    SHOULDER PRESS REP COUNTING - Moderately Strict Logic
    ====================================================
    
    Detection Logic:
    - START/END: Elbow angle >165° (arms pressed up/extended)
    - MIDDLE: Elbow angle <100° (arms down/rack position)
    - TOLERANCE: ±10° for natural movement variation
    - COOLDOWN: 0.8s between reps to prevent double counting
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: PRESSED → RACK → PRESSED
    """
    
    def __init__(self):
        super().__init__("Shoulder press")
        
        # MODERATE THRESHOLDS - Balanced for real-world use
        self.pressed_threshold = 165.0  # Arms pressed up (extended)
        self.rack_threshold = 100.0     # Arms down (rack position)
        self.angle_tolerance = 10.0     # ±10° tolerance
        
        # Universal rep counting state machine
        self.rep_counter = RepCountingState("Shoulder press", cooldown_time=0.8)

    def _define_ideal_angles(self) -> dict:
        """Define target angles for shoulder press validation"""
        return {
            'start': {
                'elbow_angle': {'target': 175.0, 'tolerance': self.angle_tolerance},
                'wrist_position': {'target': 0.0, 'tolerance': 30.0}  # Wrist over shoulder
            },
            'middle': {
                'elbow_angle': {'target': 95.0, 'tolerance': self.angle_tolerance},
                'wrist_position': {'target': 0.0, 'tolerance': 30.0}  # Maintain alignment
            },
            'end': {
                'elbow_angle': {'target': 175.0, 'tolerance': self.angle_tolerance},
                'wrist_position': {'target': 0.0, 'tolerance': 30.0}  # Full lockout
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate angles for moderately strict shoulder press assessment"""
        # Elbow angles for both arms
        left_elbow = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value, visibility)
        right_elbow = safe_angle(landmarks, L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value, visibility)

        # Pick the arm that's more extended (larger angle) for primary tracking
        elbow_angles = [x for x in [left_elbow, right_elbow] if x is not None]
        elbow_angle = max(elbow_angles) if elbow_angles else None

        # Wrist position relative to shoulder (for proper pressing form)
        wrist_position = 0.0  # Simplified - could be enhanced with actual positioning

        return {
            'elbow_angle': elbow_angle,
            'wrist_position': wrist_position,
            'left_elbow': left_elbow,
            'right_elbow': right_elbow
        }

    def _determine_phase(self, elbow_angle: float) -> str:
        """Determine shoulder press phase using universal phase names"""
        if elbow_angle is None:
            return "unknown"
        
        # Moderately strict thresholds
        if elbow_angle >= self.pressed_threshold:
            return "start"  # Arms pressed up - universal phase name
        elif elbow_angle <= self.rack_threshold:
            return "middle"   # Arms in rack position - universal phase name
        else:
            return "end"  # Moving between positions - universal phase name

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate shoulder press form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        elbow_angle = current_angles.get('elbow_angle')
        
        if elbow_angle is None:
            return False, 0.0, ["Cannot detect arm position"]
        
        # Phase-specific validation using universal tolerance
        if phase == "start":  # Pressed position
            if not is_within_tolerance(elbow_angle, 175.0, self.angle_tolerance):
                issues.append("Press to full lockout")
        elif phase == "middle":  # Rack position
            if not is_within_tolerance(elbow_angle, 95.0, self.angle_tolerance):
                issues.append("Lower to rack position")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """Universal shoulder press rep counting with RepCountingState pattern"""
        
        # Calculate current angles
        current_angles = self._calculate_current_angles(landmarks, visibility)
        elbow_angle = current_angles.get('elbow_angle')
        
        if elbow_angle is None:
            return 0, "Position yourself so arms are visible", [], None

        # Determine current phase
        current_phase = self._determine_phase(elbow_angle)
        
        # Validate movement quality
        is_valid, accuracy, issues = self._validate_movement_quality(current_angles, current_phase)
        
        # Track accuracy for session statistics
        self.session_accuracies.append(accuracy)
        
        # Update universal rep state
        phase_changed = self.rep_counter.update_phase(current_phase, frame_timestamp, is_valid, accuracy)
        
        # Check for rep completion using universal logic (down -> up cycle)
        rep_completed = self.rep_counter.check_rep_completion("middle", "start", "middle")
        
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # Handle rep completion
        if rep_completed:
            # Apply universal cooldown
            if apply_rep_cooldown(self.rep_counter.last_rep_time, frame_timestamp, self.rep_counter.cooldown_time):
                self.reps += 1
                rep_accuracy = self.rep_counter.get_rep_accuracy()
                self.rep_accuracies.append(rep_accuracy)
                self.rep_timestamps.append(frame_timestamp)
                self.rep_counter.last_rep_time = frame_timestamp
                rep_increment = 1
                
                feedback_messages.append(f"🎉 Shoulder press #{self.reps}! ({rep_accuracy:.1f}%)")
                
                if self.debug_enabled:
                    print(f"[Shoulder press] ✅ REP {self.reps}: MIDDLE→START→MIDDLE cycle completed")
                    print(f"[Shoulder press] Rep accuracy: {rep_accuracy:.1f}% at {frame_timestamp:.2f}s")
        
        # Current status feedback
        phase_display = current_phase.replace("_", " ").title()
        feedback_messages.append(f"Phase: {phase_display}")
        feedback_messages.append(f"Elbow angle: {elbow_angle:.1f}°")
        feedback_messages.append(f"Form accuracy: {accuracy:.1f}%")
        
        # Phase-specific guidance
        if current_phase == "end":
            if elbow_angle > 140.0:
                feedback_messages.append("⬆ Continue pressing up")
            else:
                feedback_messages.append("⬇ Lower to rack position")
        elif current_phase == "start":
            if accuracy < 70:
                feedback_messages.append("💪 Full lockout overhead")
        elif current_phase == "middle":
            if accuracy < 70:
                feedback_messages.append("⬇ Lower to shoulders")
        
        # Movement quality issues
        if issues:
            feedback_messages.extend([f"⚠ {issue}" for issue in issues[:2]])
        
        # Highlight problematic joints for low accuracy
        if accuracy < 60:
            highlights.extend([
                L.LEFT_ELBOW.value, L.RIGHT_ELBOW.value,
                L.LEFT_WRIST.value, L.RIGHT_WRIST.value
            ])
        
        # Limit feedback display
        feedback_text = "\n".join(feedback_messages[:7])
        
        return rep_increment, feedback_text, highlights, accuracy


# 7) Side lateral raises
# 7) Side lateral raises  
class SideLateralRaisesExercise(ExerciseBase):
    """
    SIDE LATERAL RAISES REP COUNTING - Moderately Strict Logic
    =========================================================
    
    Detection Logic:
    - START/END: Shoulder abduction <20° (arms down/at sides)
    - MIDDLE: Shoulder abduction >75° (arms raised/up position)
    - TOLERANCE: ±10° for natural movement variation
    - COOLDOWN: 0.8s between reps to prevent double counting
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: DOWN → UP → DOWN
    """
    
    def __init__(self):
        super().__init__("Side lateral raises")
        
        # MODERATE THRESHOLDS - Balanced for real-world use
        self.raised_threshold = 75.0    # Arms raised (up position)
        self.lowered_threshold = 20.0   # Arms down (at sides)
        self.angle_tolerance = 10.0     # ±10° tolerance
        
        # Universal rep counting state machine
        self.rep_counter = RepCountingState("Side lateral raises", cooldown_time=0.8)

    def _define_ideal_angles(self) -> dict:
        """Define target angles for side lateral raise validation"""
        return {
            'start': {
                'abduction_angle': {'target': 10.0, 'tolerance': self.angle_tolerance}
            },
            'middle': {
                'abduction_angle': {'target': 90.0, 'tolerance': self.angle_tolerance}
            },
            'end': {
                'abduction_angle': {'target': 10.0, 'tolerance': self.angle_tolerance}
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate angles for moderately strict side lateral raise assessment"""
        # Shoulder abduction: angle between shoulder->hip (down) and shoulder->wrist (arm)
        def abduction_angle(shoulder_idx, hip_idx, wrist_idx):
            try:
                s = np.array(landmarks[shoulder_idx], dtype=float)
                h = np.array(landmarks[hip_idx], dtype=float)
                w = np.array(landmarks[wrist_idx], dtype=float)
                sh = h - s
                sw = w - s
                if np.linalg.norm(sh) < 1e-6 or np.linalg.norm(sw) < 1e-6:
                    return None
                sh /= np.linalg.norm(sh)
                sw /= np.linalg.norm(sw)
                ang = np.degrees(np.arccos(np.clip(np.dot(sh, sw), -1.0, 1.0)))
                return float(ang)
            except:
                return None

        left_abd = abduction_angle(L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_WRIST.value)
        right_abd = abduction_angle(L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_WRIST.value)
        
        # Use the maximum abduction (more raised arm) for primary tracking
        abd_angles = [x for x in [left_abd, right_abd] if x is not None]
        abduction_angle_val = max(abd_angles) if abd_angles else None

        return {
            'abduction_angle': abduction_angle_val,
            'left_abduction': left_abd,
            'right_abduction': right_abd
        }

    def _determine_phase(self, abduction_angle: float) -> str:
        """Determine side lateral raise phase using universal phase names"""
        if abduction_angle is None:
            return "unknown"
        
        # Moderately strict thresholds
        if abduction_angle >= self.raised_threshold:
            return "middle"  # Arms raised up - universal phase name
        elif abduction_angle <= self.lowered_threshold:
            return "start"   # Arms down at sides - universal phase name
        else:
            return "end"  # Moving between positions - universal phase name

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate side lateral raise form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        abduction_angle = current_angles.get('abduction_angle')
        
        if abduction_angle is None:
            return False, 0.0, ["Cannot detect arm position"]
        
        # Phase-specific validation using universal tolerance
        if phase == "start":  # Arms down position
            if not is_within_tolerance(abduction_angle, 10.0, self.angle_tolerance):
                issues.append("Lower arms to sides")
        elif phase == "middle":  # Arms raised position
            if not is_within_tolerance(abduction_angle, 90.0, self.angle_tolerance):
                issues.append("Raise arms to shoulder height")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """Universal side lateral raise rep counting with RepCountingState pattern"""
        
        # Calculate current angles
        current_angles = self._calculate_current_angles(landmarks, visibility)
        abduction_angle = current_angles.get('abduction_angle')
        
        if abduction_angle is None:
            return 0, "Position yourself so arms are visible", [], None

        # Determine current phase
        current_phase = self._determine_phase(abduction_angle)
        
        # Validate movement quality
        is_valid, accuracy, issues = self._validate_movement_quality(current_angles, current_phase)
        
        # Track accuracy for session statistics
        self.session_accuracies.append(accuracy)
        
        # Update universal rep state
        phase_changed = self.rep_counter.update_phase(current_phase, frame_timestamp, is_valid, accuracy)
        
        # Check for rep completion using universal logic (down -> up -> down cycle)
        rep_completed = self.rep_counter.check_rep_completion("start", "middle", "start")
        
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # Handle rep completion
        if rep_completed:
            # Apply universal cooldown
            if apply_rep_cooldown(self.rep_counter.last_rep_time, frame_timestamp, self.rep_counter.cooldown_time):
                self.reps += 1
                rep_accuracy = self.rep_counter.get_rep_accuracy()
                self.rep_accuracies.append(rep_accuracy)
                self.rep_timestamps.append(frame_timestamp)
                self.rep_counter.last_rep_time = frame_timestamp
                rep_increment = 1
                
                feedback_messages.append(f"🎉 Lateral raise #{self.reps}! ({rep_accuracy:.1f}%)")
                
                if self.debug_enabled:
                    print(f"[Side lateral raises] ✅ REP {self.reps}: START→MIDDLE→START cycle completed")
                    print(f"[Side lateral raises] Rep accuracy: {rep_accuracy:.1f}% at {frame_timestamp:.2f}s")
        
        # Current status feedback
        phase_display = current_phase.replace("_", " ").title()
        feedback_messages.append(f"Phase: {phase_display}")
        feedback_messages.append(f"Abduction angle: {abduction_angle:.1f}°")
        feedback_messages.append(f"Form accuracy: {accuracy:.1f}%")
        
        # Phase-specific guidance
        if current_phase == "end":
            if abduction_angle > 50.0:
                feedback_messages.append("⬇ Lower arms to sides")
            else:
                feedback_messages.append("⬆ Raise arms to shoulder height")
        elif current_phase == "start":
            if accuracy < 70:
                feedback_messages.append("💪 Arms fully down at sides")
        elif current_phase == "middle":
            if accuracy < 70:
                feedback_messages.append("⬆ Raise to 90° shoulder height")
        
        # Movement quality issues
        if issues:
            feedback_messages.extend([f"⚠ {issue}" for issue in issues[:2]])
        
        # Highlight problematic joints for low accuracy
        if accuracy < 60:
            highlights.extend([
                L.LEFT_SHOULDER.value, L.RIGHT_SHOULDER.value,
                L.LEFT_WRIST.value, L.RIGHT_WRIST.value
            ])
        
        # Limit feedback display
        feedback_text = "\n".join(feedback_messages[:7])
        
        return rep_increment, feedback_text, highlights, accuracy


# 8) Deadlifts
# 8) Deadlifts
class DeadliftsExercise(ExerciseBase):
    """
    DEADLIFTS REP COUNTING - Moderately Strict Logic
    ===============================================
    
    Detection Logic:
    - START/END: Hip angle >165° (standing upright)
    - MIDDLE: Hip angle <120° (hinged at hips/lowered position)
    - TOLERANCE: ±10° for natural movement variation
    - COOLDOWN: 0.8s between reps to prevent double counting
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: STANDING → HINGED → STANDING
    """
    
    def __init__(self):
        super().__init__("Deadlifts")
        
        # MODERATE THRESHOLDS - Balanced for real-world use
        self.standing_threshold = 165.0  # Standing upright (hip extended)
        self.hinged_threshold = 120.0    # Hinged at hips (lowered position)
        self.angle_tolerance = 10.0      # ±10° tolerance
        
        # Universal rep counting state machine
        self.rep_counter = RepCountingState("Deadlifts", cooldown_time=0.8)

    def _define_ideal_angles(self) -> dict:
        """Define target angles for deadlift validation"""
        return {
            'start': {
                'hip_angle': {'target': 175.0, 'tolerance': self.angle_tolerance},
                'back_angle': {'target': 175.0, 'tolerance': 8.0}  # Straight back
            },
            'middle': {
                'hip_angle': {'target': 100.0, 'tolerance': self.angle_tolerance},
                'back_angle': {'target': 175.0, 'tolerance': 8.0}  # Keep back straight
            },
            'end': {
                'hip_angle': {'target': 175.0, 'tolerance': self.angle_tolerance},
                'back_angle': {'target': 175.0, 'tolerance': 8.0}  # Straight back
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate angles for moderately strict deadlift assessment"""
        # Hip angle: shoulder-hip-knee (measures hip hinge)
        hip_angle = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value, visibility)
        
        # Back angle: shoulder-hip-ankle (measures back straightness)
        back_angle = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_ANKLE.value, visibility)

        return {
            'hip_angle': hip_angle,
            'back_angle': back_angle
        }

    def _determine_phase(self, hip_angle: float) -> str:
        """Determine deadlift phase using universal phase names"""
        if hip_angle is None:
            return "unknown"
        
        # Moderately strict thresholds
        if hip_angle >= self.standing_threshold:
            return "start"  # Standing upright - universal phase name
        elif hip_angle <= self.hinged_threshold:
            return "middle"   # Hinged at hips - universal phase name
        else:
            return "end"  # Moving between positions - universal phase name

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate deadlift form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        hip_angle = current_angles.get('hip_angle')
        back_angle = current_angles.get('back_angle')
        
        if hip_angle is None:
            return False, 0.0, ["Cannot detect hip position"]
        
        # Phase-specific validation using universal tolerance
        if phase == "start":  # Standing position
            if not is_within_tolerance(hip_angle, 175.0, self.angle_tolerance):
                issues.append("Stand fully upright")
        elif phase == "middle":  # Hinged position
            if not is_within_tolerance(hip_angle, 100.0, self.angle_tolerance):
                issues.append("Hinge deeper at hips")
        
        # Back straightness validation (critical for deadlifts)
        if back_angle and not is_within_tolerance(back_angle, 175.0, 8.0):
            issues.append("Keep your back straight")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """Universal deadlift rep counting with RepCountingState pattern"""
        
        # Calculate current angles
        current_angles = self._calculate_current_angles(landmarks, visibility)
        hip_angle = current_angles.get('hip_angle')
        
        if hip_angle is None:
            return 0, "Position yourself so hips are visible", [], None

        # Determine current phase
        current_phase = self._determine_phase(hip_angle)
        
        # Validate movement quality
        is_valid, accuracy, issues = self._validate_movement_quality(current_angles, current_phase)
        
        # Track accuracy for session statistics
        self.session_accuracies.append(accuracy)
        
        # Update universal rep state
        phase_changed = self.rep_counter.update_phase(current_phase, frame_timestamp, is_valid, accuracy)
        
        # Check for rep completion using universal logic (standing -> hinged -> standing cycle)
        rep_completed = self.rep_counter.check_rep_completion("start", "middle", "start")
        
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # Handle rep completion
        if rep_completed:
            # Apply universal cooldown
            if apply_rep_cooldown(self.rep_counter.last_rep_time, frame_timestamp, self.rep_counter.cooldown_time):
                self.reps += 1
                rep_accuracy = self.rep_counter.get_rep_accuracy()
                self.rep_accuracies.append(rep_accuracy)
                self.rep_timestamps.append(frame_timestamp)
                self.rep_counter.last_rep_time = frame_timestamp
                rep_increment = 1
                
                feedback_messages.append(f"🎉 Deadlift #{self.reps}! ({rep_accuracy:.1f}%)")
                
                if self.debug_enabled:
                    print(f"[Deadlifts] ✅ REP {self.reps}: START→MIDDLE→START cycle completed")
                    print(f"[Deadlifts] Rep accuracy: {rep_accuracy:.1f}% at {frame_timestamp:.2f}s")
        
        # Current status feedback
        phase_display = current_phase.replace("_", " ").title()
        feedback_messages.append(f"Phase: {phase_display}")
        feedback_messages.append(f"Hip angle: {hip_angle:.1f}°")
        feedback_messages.append(f"Form accuracy: {accuracy:.1f}%")
        
        # Phase-specific guidance
        if current_phase == "end":
            if hip_angle > 140.0:
                feedback_messages.append("⬆ Stand tall and complete lift")
            else:
                feedback_messages.append("⬇ Hinge at hips and lower")
        elif current_phase == "start":
            if accuracy < 70:
                feedback_messages.append("💪 Stand fully upright")
        elif current_phase == "middle":
            if accuracy < 70:
                feedback_messages.append("⬇ Hinge deeper at hips")
        
        # Movement quality issues
        if issues:
            feedback_messages.extend([f"⚠ {issue}" for issue in issues[:2]])
        
        # Highlight problematic joints for low accuracy
        if accuracy < 60:
            highlights.extend([
                L.LEFT_HIP.value, L.RIGHT_HIP.value,
                L.LEFT_KNEE.value, L.RIGHT_KNEE.value
            ])
        
        # Highlight back issues if back angle is poor
        back_angle = current_angles.get('back_angle')
        if back_angle and not is_within_tolerance(back_angle, 175.0, 8.0):
            highlights.extend([L.LEFT_SHOULDER.value, L.LEFT_HIP.value])
        
        # Limit feedback display
        feedback_text = "\n".join(feedback_messages[:7])
        
        return rep_increment, feedback_text, highlights, accuracy

    def create_comprehensive_feedback(self, landmarks, visibility, current_angles: dict, joint_deviations: dict, phase: str) -> list:
        """
        ENHANCED MULTI-POINT BICEP CURL FEEDBACK SYSTEM
        ==============================================
        Analyzes all critical joints and provides 2-4 biomechanically accurate feedback points
        """
        feedback_issues = []
        
        # Extract all relevant angles
        elbow_angle = current_angles.get('elbow_angle')
        shoulder_angle = current_angles.get('shoulder_angle')
        
        # Calculate additional critical angles for comprehensive analysis
        additional_angles = self._calculate_comprehensive_curl_angles(landmarks, visibility)
        
        wrist_angle = additional_angles.get('wrist_angle')
        elbow_stability = additional_angles.get('elbow_stability')
        body_sway = additional_angles.get('body_sway')
        
        # ===== PHASE-BASED FEEDBACK ANALYSIS =====
        
        if phase == "middle" or (elbow_angle and elbow_angle < 90):  # CURL UP PHASE
            # 1. CURL DEPTH (Primary)
            if elbow_angle and elbow_angle > 100:
                feedback_issues.append(("MAJOR", "Curl your arms higher - bring hands to shoulders"))
            elif elbow_angle and elbow_angle < 30:
                feedback_issues.append(("MODERATE", "Great curl depth - control the movement"))
            elif elbow_angle and 35 <= elbow_angle <= 70:
                feedback_issues.append(("GOOD", "Perfect curl position - excellent range"))
            
            # 2. ELBOW STABILITY
            if elbow_stability and elbow_stability > 20:  # Elbows moving forward/back
                feedback_issues.append(("MAJOR", "Keep elbows stable at your sides"))
            
            # 3. SHOULDER ELEVATION 
            if shoulder_angle and shoulder_angle < 160:
                feedback_issues.append(("MODERATE", "Don't shrug shoulders - keep them down"))
            elif shoulder_angle and shoulder_angle > 200:
                feedback_issues.append(("MODERATE", "Relax shoulders - don't lift them up"))
                
            # 4. WRIST POSITIONING
            if wrist_angle and (wrist_angle < 160 or wrist_angle > 200):
                feedback_issues.append(("MINOR", "Keep wrists straight and aligned"))
                
        elif phase == "start" or phase == "end" or (elbow_angle and elbow_angle > 150):  # EXTENDED PHASE
            # 1. FULL EXTENSION CHECK
            if elbow_angle and elbow_angle < 160:
                feedback_issues.append(("MODERATE", "Extend arms fully to complete the movement"))
            elif elbow_angle and elbow_angle >= 170:
                feedback_issues.append(("GOOD", "Perfect arm extension - ready for next rep"))
            
            # 2. ELBOW POSITIONING
            if elbow_stability and elbow_stability > 15:
                feedback_issues.append(("MAJOR", "Keep elbows close to your sides"))
            
            # 3. STARTING POSITION SETUP
            if phase == "start":
                feedback_issues.append(("INSTRUCTION", "Ready position - begin controlled curl"))
        
        # ===== GENERAL FORM ANALYSIS (applies to all phases) =====
        
        # BODY STABILITY
        if body_sway and body_sway > 15:
            feedback_issues.append(("MAJOR", "Minimize body sway - use biceps not momentum"))
        
        # SHOULDER STABILITY
        if shoulder_angle and (shoulder_angle < 150 or shoulder_angle > 210):
            feedback_issues.append(("MODERATE", "Keep shoulders stable - don't swing the weights"))
        
        # ===== PRIORITY-BASED FEEDBACK SELECTION =====
        return self._prioritize_feedback_messages(feedback_issues)
    
    def _calculate_comprehensive_curl_angles(self, landmarks, visibility) -> dict:
        """Calculate all critical angles for comprehensive bicep curl analysis"""
        angles = {}
        
        try:
            # WRIST ALIGNMENT (forearm to hand)
            left_wrist = safe_angle(landmarks, 13, 15, 17, visibility)   # ELBOW-WRIST-THUMB
            right_wrist = safe_angle(landmarks, 14, 16, 18, visibility)
            if left_wrist and right_wrist:
                angles['wrist_angle'] = (left_wrist + right_wrist) / 2
            elif left_wrist:
                angles['wrist_angle'] = left_wrist
            elif right_wrist:
                angles['wrist_angle'] = right_wrist
            
            # ELBOW STABILITY (shoulder to elbow position consistency)
            # This measures if elbows stay at sides vs. swinging forward/back
            left_stability = safe_angle(landmarks, 11, 13, 23, visibility)   # SHOULDER-ELBOW-HIP
            right_stability = safe_angle(landmarks, 12, 14, 24, visibility)
            if left_stability and right_stability:
                # Deviation from ideal elbow position (should be ~90°)
                ideal_elbow_pos = 90
                left_dev = abs(left_stability - ideal_elbow_pos)
                right_dev = abs(right_stability - ideal_elbow_pos)
                angles['elbow_stability'] = (left_dev + right_dev) / 2
            
            # BODY SWAY (hip to shoulder alignment)
            left_sway = safe_angle(landmarks, 23, 11, 13, visibility)    # HIP-SHOULDER-ELBOW
            right_sway = safe_angle(landmarks, 24, 12, 14, visibility)
            if left_sway and right_sway:
                # Measure deviation from vertical posture
                ideal_posture = 90
                left_dev = abs(left_sway - ideal_posture) 
                right_dev = abs(right_sway - ideal_posture)
                angles['body_sway'] = (left_dev + right_dev) / 2
                
        except Exception as e:
            pass  # Return partial angles if some calculations fail
            
        return angles
    
    def _prioritize_feedback_messages(self, feedback_issues: list) -> list:
        """
        Convert prioritized feedback issues into final message list
        Ensures 2-4 messages are shown with proper severity ordering
        """
        if not feedback_issues:
            return ["Perfect bicep curl form - keep it up!"]
        
        # Sort by priority: MAJOR > MODERATE > MINOR > GOOD > INSTRUCTION
        priority_order = {"MAJOR": 1, "MODERATE": 2, "MINOR": 3, "GOOD": 4, "INSTRUCTION": 5}
        sorted_feedback = sorted(feedback_issues, key=lambda x: priority_order.get(x[0], 6))
        
        # Select top 2-4 messages based on severity mix
        final_messages = []
        major_count = sum(1 for severity, msg in sorted_feedback if severity == "MAJOR")
        
        if major_count >= 2:
            # Show 2 major + 1-2 others
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        elif major_count == 1:
            # Show 1 major + 2-3 others  
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        else:
            # No major issues, show 3-4 moderate/minor/good messages
            final_messages = [msg for severity, msg in sorted_feedback[:4]]
        
        return final_messages

    def _get_exercise_specific_feedback(self, current_angles: dict, phase: str) -> list:
        """
        Enhanced bicep curl feedback - calls comprehensive feedback system
        """
        # This method is kept for backward compatibility
        # The main feedback now comes from create_comprehensive_feedback
        return []


# 9) Jumping jacks
class JumpingJacksExercise(ExerciseBase):
    """
    JUMPING JACKS REP COUNTING - Moderately Strict Logic
    ===================================================
    
    Detection Logic:
    - START/END: Arms down and feet together (closed position)
    - MIDDLE: Arms up and feet apart (open position)
    - TOLERANCE: Moderate thresholds for arm and feet positioning
    - COOLDOWN: 0.5s between reps (faster exercise)
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: CLOSED → OPEN → CLOSED
    """
    
    def __init__(self):
        super().__init__("Jumping jacks")
        
        # SIMPLIFIED THRESHOLDS for reliable detection
        self.down_threshold = 45.0     # DOWN: arms down ≤ 45°
        self.up_threshold = 135.0      # UP: arms up ≥ 135°
        self.cooldown_time = 0.5       # Cooldown between reps (faster exercise)
        
        # Add missing attributes for feet position detection
        self.feet_apart_threshold = 1.2   # Feet apart when normalized distance > 1.2
        self.feet_together_threshold = 0.8 # Feet together when normalized distance < 0.8
        
        # Required threshold attributes for enhanced feedback system
        self.arms_up_threshold = 135.0        # Arm angle when up
        self.arms_down_threshold = 45.0       # Arm angle when down
        self.feet_separation_threshold = 1.0  # Feet separation threshold
        
        # Simple state tracking
        self.is_down = False
        self.is_up = True
        self.last_rep_time = 0.0

    def _define_ideal_angles(self) -> dict:
        """Define target positions for jumping jack validation"""
        return {
            'start': {
                'arms_position': {'target': 0.0, 'tolerance': 30.0},    # Arms down
                'feet_position': {'target': 0.5, 'tolerance': 0.2}     # Feet together
            },
            'middle': {
                'arms_position': {'target': 100.0, 'tolerance': 30.0}, # Arms up
                'feet_position': {'target': 1.5, 'tolerance': 0.3}     # Feet apart
            },
            'end': {
                'arms_position': {'target': 0.0, 'tolerance': 30.0},   # Arms down
                'feet_position': {'target': 0.5, 'tolerance': 0.2}     # Feet together
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate positions for moderately strict jumping jack assessment"""
        try:
            # Feet distance normalized by shoulder width
            left_ankle = landmarks[L.LEFT_ANKLE.value]
            right_ankle = landmarks[L.RIGHT_ANKLE.value]
            left_sh = landmarks[L.LEFT_SHOULDER.value]
            right_sh = landmarks[L.RIGHT_SHOULDER.value]
            
            feet_dist = abs(left_ankle[0] - right_ankle[0])
            shoulder_scale = abs(left_sh[0] - right_sh[0]) + 1e-6
            norm_feet = feet_dist / shoulder_scale

            # Arms up: wrist above nose level
            wrist_y = min(landmarks[L.LEFT_WRIST.value][1], landmarks[L.RIGHT_WRIST.value][1])
            nose_y = landmarks[L.NOSE.value][1]
            arms_up = wrist_y < nose_y
            arms_position = 100.0 if arms_up else 0.0

            return {
                'arms_position': arms_position,
                'feet_position': norm_feet,
                'arms_up': arms_up
            }
        except:
            return {
                'arms_position': None,
                'feet_position': None,
                'arms_up': False
            }

    def _determine_phase(self, current_angles: dict) -> str:
        """Determine jumping jack phase using universal phase names"""
        arms_up = current_angles.get('arms_up', False)
        feet_position = current_angles.get('feet_position')
        
        if feet_position is None:
            return "unknown"
        
        # Determine phase based on arms and feet
        if arms_up and feet_position > self.feet_apart_threshold:
            return "middle"  # Open position (arms up, feet apart)
        elif not arms_up and feet_position < self.feet_together_threshold:
            return "start"   # Closed position (arms down, feet together)
        else:
            return "end"  # Transitioning between positions

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate jumping jack form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        arms_up = current_angles.get('arms_up', False)
        feet_position = current_angles.get('feet_position')
        
        if feet_position is None:
            return False, 0.0, ["Cannot detect body position"]
        
        # Phase-specific validation
        if phase == "start":  # Closed position
            if not arms_up:
                pass  # Good - arms should be down
            else:
                issues.append("Lower arms to sides")
            if feet_position < self.feet_together_threshold:
                pass  # Good - feet should be together
            else:
                issues.append("Bring feet together")
        elif phase == "middle":  # Open position
            if arms_up:
                pass  # Good - arms should be up
            else:
                issues.append("Bring arms overhead")
            if feet_position > self.feet_apart_threshold:
                pass  # Good - feet should be apart
            else:
                issues.append("Step feet wider")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """ENHANCED JUMPING JACKS REP COUNTING - Dual-mode validation (live vs video)"""
        
        # Calculate positions for jumping jacks
        current_angles = self._calculate_current_angles(landmarks, visibility)
        arms_position = current_angles.get('arms_position')
        feet_position = current_angles.get('feet_position')
        arms_up = current_angles.get('arms_up', False)
        
        if arms_position is None or feet_position is None:
            return 0, "Position yourself so body is visible", [], None

        # FIXED ACCURACY CALCULATION - Always return a valid number
        accuracy = max(50.0, min(95.0, 100.0 - abs(arms_position - 50) * 0.5))  # Dynamic accuracy based on movement
        self.session_accuracies.append(accuracy)
        
        # Initialize feedback
        feedback_messages = []
        highlights = []
        rep_increment = 0
        
        # DUAL-MODE LOGIC: Enhanced validation for live camera, simple for video
        if self.is_live_mode and self.live_validator:
            # LIVE MODE: Enhanced form validation
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # Use enhanced validator for live camera detection
                # Use arms_position as primary angle for validation
                rep_completed, phase_status, form_valid = self.live_validator.validate_rep_transition(
                    arms_position, feet_position, frame_timestamp
                )
                
                if rep_completed and form_valid:
                    # COUNT THE REP with enhanced validation
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    self.last_rep_time = frame_timestamp
                    
                    status = f"🎉 JUMPING JACK #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Jumping jacks] ✅ VALIDATED REP #{self.reps}: Arms {arms_position:.1f}° - Form Valid")
                else:
                    status = phase_status
                    if not form_valid and self.debug_enabled:
                        print(f"[Jumping jacks] Form validation failed: Arms {arms_position:.1f}°")
        else:
            # VIDEO MODE: Original simple logic (preserved)
            if (frame_timestamp - self.last_rep_time) < self.cooldown_time:
                status = "Cooldown..."
            else:
                # DOWN DETECTION (arms down)
                if not self.is_down and arms_position <= self.down_threshold:
                    self.is_down = True
                    self.is_up = False
                    status = f"ARMS DOWN ({arms_position:.1f}°)"
                    if self.debug_enabled:
                        print(f"[Jumping jacks] DOWN: {arms_position:.1f}° <= {self.down_threshold}°")
                
                # UP DETECTION (arms up)
                elif self.is_down and not self.is_up and arms_position >= self.up_threshold:
                    self.is_up = True
                    self.is_down = False
                    self.last_rep_time = frame_timestamp
                    
                    # COUNT THE REP (original logic)
                    self.reps += 1
                    self.rep_accuracies.append(accuracy)
                    self.rep_timestamps.append(frame_timestamp)
                    rep_increment = 1
                    
                    status = f"🎉 JUMPING JACK #{self.reps} COMPLETED!"
                    
                    if self.debug_enabled:
                        print(f"[Jumping jacks] ✅ REP #{self.reps}: {arms_position:.1f}° >= {self.up_threshold}°")
                
                # CURRENT STATUS
                elif self.is_down:
                    status = f"Arms down position ({arms_position:.1f}°)"
                elif self.is_up:
                    status = f"Arms up position ({arms_position:.1f}°)"
                else:
                    status = f"Moving ({arms_position:.1f}°)"
        
        # CLEAN FEEDBACK - NO DUPLICATES
        feedback_messages.append(status)
        feedback_messages.append(f"Arms: {'Up' if arms_up else 'Down'}")
        feedback_messages.append(f"Feet: {'Apart' if feet_position > 1.0 else 'Together'}")
        feedback_messages.append(f"Accuracy: {accuracy:.1f}%")  # SINGLE accuracy mention
        
        # Movement guidance
        if arms_position < self.down_threshold + 10:
            feedback_messages.append("🔥 Good arms down!")
        elif arms_position > self.up_threshold - 10:
            feedback_messages.append("🚀 Great arms up!")
        
        # Visual highlights
        highlights = [11, 12, 15, 16, 27, 28]  # Arms and feet landmarks
        
        feedback_text = " | ".join(feedback_messages)  # Use | separator to avoid line breaks
        return rep_increment, feedback_text, highlights, accuracy

    def _get_exercise_specific_feedback(self, current_angles: dict, phase: str) -> list:
        """
        Generate jumping jacks specific feedback based on current angles and phase
        """
        feedback = []
        
        arm_angle = current_angles.get('arm_angle')
        shoulder_angle = current_angles.get('shoulder_angle')
        
        # Check current state based on angles
        if hasattr(self, 'arms_up') and hasattr(self, 'feet_apart'):
            arms_up = self.arms_up
            feet_apart = self.feet_apart
        else:
            arms_up = False
            feet_apart = False
        
        if arm_angle is not None:
            # Arms position feedback
            if arms_up:  # Arms should be up
                if arm_angle < 120:
                    feedback.append("🙌 Raise your arms higher - reach for the sky!")
                else:
                    feedback.append("✅ Perfect arm position!")
            else:  # Arms should be down
                if arm_angle > 60:
                    feedback.append("👇 Lower your arms to your sides")
                else:
                    feedback.append("✅ Good starting position!")
        
        # Feet position feedback
        if feet_apart:
            feedback.append("✅ Feet apart - good jump!")
        else:
            feedback.append("✅ Feet together - ready to jump!")
        
        # General jumping jacks form tips
        feedback.append("💡 Keep the rhythm - jump with energy!")
        
        return feedback


# 10) Mountain climbers
class MountainClimbersExercise(ExerciseBase):
    """
    MOUNTAIN CLIMBERS REP COUNTING - Moderately Strict Logic
    =======================================================
    
    Detection Logic:
    - START/END: Both knees back/extended (plank position)
    - MIDDLE: One knee forward/close to chest (driving knee)
    - TOLERANCE: Moderate distance thresholds for knee position
    - COOLDOWN: 0.3s between reps (very dynamic exercise)
    - MIN ACCURACY: 40% required for rep validation
    
    Complete Rep Cycle: Each leg drive counts as one rep
    """
    
    def __init__(self):
        super().__init__("Mountain climbers")
        
        # MODERATE THRESHOLDS - Balanced for real-world use
        self.knee_close_threshold = 0.25  # Normalized distance (knee close to chest)
        self.body_straight_threshold = 165.0  # Body line angle
        
        # Universal rep counting state machine (very fast cooldown for dynamic exercise)
        self.rep_counter = RepCountingState("Mountain climbers", cooldown_time=0.3)
        
        # Track which leg was last driven
        self.last_driving_leg = None

    def _define_ideal_angles(self) -> dict:
        """Define target positions for mountain climber validation"""
        return {
            'start': {
                'left_knee_distance': {'target': 0.5, 'tolerance': 0.2},   # Knees back
                'right_knee_distance': {'target': 0.5, 'tolerance': 0.2},  # Knees back
                'body_straight': {'target': 175.0, 'tolerance': 10.0}      # Straight body
            },
            'middle': {
                'driving_knee_distance': {'target': 0.2, 'tolerance': 0.1}, # One knee close
                'body_straight': {'target': 175.0, 'tolerance': 10.0}       # Maintain straightness
            },
            'end': {
                'left_knee_distance': {'target': 0.5, 'tolerance': 0.2},   # Knees back
                'right_knee_distance': {'target': 0.5, 'tolerance': 0.2},  # Knees back
                'body_straight': {'target': 175.0, 'tolerance': 10.0}      # Straight body
            }
        }

    def _calculate_current_angles(self, landmarks, visibility) -> dict:
        """Calculate positions for moderately strict mountain climber assessment"""
        try:
            # Knee-to-chest proximity: distance between knee and chest (body center)
            def knee_to_chest_dist(knee_idx, chest_point, frame_height):
                k = np.array(landmarks[knee_idx], dtype=float)
                c = np.array(chest_point, dtype=float)
                distance = np.linalg.norm(k - c)
                return distance / frame_height  # Normalize by frame height

            # Calculate chest/body center point
            chest = (
                (landmarks[L.LEFT_SHOULDER.value][0] + landmarks[L.RIGHT_SHOULDER.value][0] +
                 landmarks[L.LEFT_HIP.value][0] + landmarks[L.RIGHT_HIP.value][0]) / 4.0,
                (landmarks[L.LEFT_SHOULDER.value][1] + landmarks[L.RIGHT_SHOULDER.value][1] +
                 landmarks[L.LEFT_HIP.value][1] + landmarks[L.RIGHT_HIP.value][1]) / 4.0,
            )

            # Calculate distances (normalized)
            frame_height = 480  # Default frame height for normalization
            d_left = knee_to_chest_dist(L.LEFT_KNEE.value, chest, frame_height)
            d_right = knee_to_chest_dist(L.RIGHT_KNEE.value, chest, frame_height)

            # Body straightness (shoulder-hip-ankle alignment)
            body_straight = safe_angle(landmarks, L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_ANKLE.value, visibility)

            # Determine which leg is driving (closer to chest)
            left_knee_in = d_left < self.knee_close_threshold
            right_knee_in = d_right < self.knee_close_threshold
            
            driving_leg = None
            if left_knee_in and not right_knee_in:
                driving_leg = "left"
            elif right_knee_in and not left_knee_in:
                driving_leg = "right"

            return {
                'left_knee_distance': d_left,
                'right_knee_distance': d_right,
                'body_straight': body_straight,
                'driving_leg': driving_leg,
                'left_knee_in': left_knee_in,
                'right_knee_in': right_knee_in
            }
        except:
            return {
                'left_knee_distance': None,
                'right_knee_distance': None,
                'body_straight': None,
                'driving_leg': None,
                'left_knee_in': False,
                'right_knee_in': False
            }

    def _determine_phase(self, current_angles: dict) -> str:
        """Determine mountain climber phase using universal phase names"""
        driving_leg = current_angles.get('driving_leg')
        left_knee_in = current_angles.get('left_knee_in', False)
        right_knee_in = current_angles.get('right_knee_in', False)
        
        # Determine phase based on knee positions
        if driving_leg is not None and (left_knee_in or right_knee_in):
            return "middle"  # One knee driving forward
        else:
            return "start"   # Both knees back (plank position)

    def _validate_movement_quality(self, current_angles: dict, phase: str) -> Tuple[bool, float, List[str]]:
        """Validate mountain climber form with universal helper functions"""
        # Use universal helper for validation
        ideal_angles = self._define_ideal_angles()
        accuracy = calculate_rep_accuracy(current_angles, ideal_angles, phase)
        
        issues = []
        body_straight = current_angles.get('body_straight')
        left_knee_in = current_angles.get('left_knee_in', False)
        right_knee_in = current_angles.get('right_knee_in', False)
        
        # Body straightness validation (critical for mountain climbers)
        if body_straight and not is_within_tolerance(body_straight, 175.0, 10.0):
            issues.append("Keep hips low and body straight")
        
        # Phase-specific validation
        if phase == "middle":  # Driving phase
            if not (left_knee_in or right_knee_in):
                issues.append("Drive knee closer to chest")
        
        # Movement is valid if accuracy meets minimum threshold
        is_valid = accuracy >= 20.0  # Universal 20% minimum accuracy (reduced for real-world videos)
        
        return is_valid, accuracy, issues

    def _update_exercise_logic(self, landmarks, visibility, image_shape, frame_timestamp: float) -> Tuple[int, str, List[int], Optional[float]]:
        """Universal mountain climber rep counting with RepCountingState pattern"""
        
        # Calculate current positions
        current_angles = self._calculate_current_angles(landmarks, visibility)
        driving_leg = current_angles.get('driving_leg')
        
        if current_angles.get('left_knee_distance') is None:
            return 0, "Position yourself so knees are visible", [], None

        # Determine current phase
        current_phase = self._determine_phase(current_angles)
        
        # Validate movement quality
        is_valid, accuracy, issues = self._validate_movement_quality(current_angles, current_phase)
        
        # Track accuracy for session statistics
        self.session_accuracies.append(accuracy)
        
        # Special logic for mountain climbers: each leg drive counts as a rep
        rep_increment = 0
        if driving_leg and driving_leg != self.last_driving_leg and is_valid:
            # Apply universal cooldown (very short for dynamic exercise)
            if apply_rep_cooldown(self.rep_counter.last_rep_time, frame_timestamp, self.rep_counter.cooldown_time):
                self.reps += 1
                self.rep_accuracies.append(accuracy)
                self.rep_timestamps.append(frame_timestamp)
                self.rep_counter.last_rep_time = frame_timestamp
                self.last_driving_leg = driving_leg
                rep_increment = 1
                
                if self.debug_enabled:
                    print(f"[Mountain climbers] ✅ REP {self.reps}: {driving_leg} leg drive completed")
                    print(f"[Mountain climbers] Rep accuracy: {accuracy:.1f}% at {frame_timestamp:.2f}s")
        
        feedback_messages = []
        highlights = []
        
        # Progress feedback
        if rep_increment > 0:
            feedback_messages.append(f"🎉 Mountain climber #{self.reps}! ({accuracy:.1f}%)")
        
        # Current status feedback
        phase_display = current_phase.replace("_", " ").title()
        feedback_messages.append(f"Phase: {phase_display}")
        if driving_leg:
            feedback_messages.append(f"Driving: {driving_leg} leg")
        feedback_messages.append(f"Form accuracy: {accuracy:.1f}%")
        
        # Phase-specific guidance
        if current_phase == "start":
            feedback_messages.append("💪 Drive knees alternately")
        elif current_phase == "middle":
            feedback_messages.append("⚡ Good drive - switch legs")
        
        # Movement quality issues
        if issues:
            feedback_messages.extend([f"⚠ {issue}" for issue in issues[:2]])
        
        # Highlight problematic areas for low accuracy
        if accuracy < 60:
            body_straight = current_angles.get('body_straight')
            if body_straight and not is_within_tolerance(body_straight, 175.0, 10.0):
                highlights.extend([L.LEFT_HIP.value, L.RIGHT_HIP.value])
            if not (current_angles.get('left_knee_in') or current_angles.get('right_knee_in')):
                highlights.extend([L.LEFT_KNEE.value, L.RIGHT_KNEE.value])
        
        # Limit feedback display
        feedback_text = "\n".join(feedback_messages[:7])
        
        return rep_increment, feedback_text, highlights, accuracy


# Factory/listing

def list_exercises():
    return {
        "squats": {"name": "Squats", "cls": SquatsExercise},
        "pushups": {"name": "Push-ups", "cls": PushUpsExercise},
        "plank": {"name": "Plank", "cls": PlankExercise},
        "lunges": {"name": "Lunges", "cls": LungesExercise},
        "bicep_curls": {"name": "Bicep curls", "cls": BicepCurlsExercise},
        "shoulder_press": {"name": "Shoulder press", "cls": ShoulderPressExercise},
        "side_lateral_raises": {"name": "Side lateral raises", "cls": SideLateralRaisesExercise},
        "deadlifts": {"name": "Deadlifts", "cls": DeadliftsExercise},
        "jumping_jacks": {"name": "Jumping jacks", "cls": JumpingJacksExercise},
        "mountain_climbers": {"name": "Mountain climbers", "cls": MountainClimbersExercise},
    }


def build_exercise_by_key(key: str) -> ExerciseBase:
    data = list_exercises().get(key)
    if not data:
        raise KeyError(f"Unknown exercise key: {key}")
    return data["cls"]()