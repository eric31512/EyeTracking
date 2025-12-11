"""
MediaPipe Face Mesh landmark definitions and gaze calculation.

MediaPipe Face Mesh with refine_landmarks=True provides 478 landmarks including:
- 468-472: Left iris landmarks (center + 4 points on iris edge)
- 473-477: Right iris landmarks (center + 4 points on iris edge)

For gaze estimation, we calculate the position of the iris center relative
to the eye corners, which gives us a gaze ratio independent of head position.
"""

import numpy as np


# =============================================================================
# MEDIAPIPE FACE MESH LANDMARKS
# =============================================================================

# Iris landmarks (MediaPipe Face Mesh with refine_landmarks=True)
# Each iris has 5 landmarks: center + 4 edge points
LEFT_IRIS = [468, 469, 470, 471, 472]  # 468 is center
RIGHT_IRIS = [473, 474, 475, 476, 477]  # 473 is center
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Eye contour landmarks for accurate eye region detection
# These define the complete eye outline for calculating eye center
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Eye landmarks for EAR (Eye Aspect Ratio) calculation
# Left eye: vertical points (159, 145), (158, 153) and horizontal points (33, 133)
# Right eye: vertical points (386, 374), (385, 380) and horizontal points (362, 263)
LEFT_EYE_LANDMARKS = {
    'top': [159, 158],
    'bottom': [145, 153],
    'left': 33,
    'right': 133
}

RIGHT_EYE_LANDMARKS = {
    'top': [386, 385],
    'bottom': [374, 380],
    'left': 362,
    'right': 263
}

def calculate_ear(landmarks, eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    Where p1-p6 are the 6 eye landmarks defining the eye contour.
    When the eye closes, the vertical distances decrease, lowering the EAR.
    
    Args:
        landmarks: MediaPipe face landmarks
        eye_landmarks: Dictionary with eye landmark indices
        
    Returns:
        float: Eye Aspect Ratio value
    """
    # Get landmark coordinates
    top1 = np.array([landmarks[eye_landmarks['top'][0]].x, 
                     landmarks[eye_landmarks['top'][0]].y])
    top2 = np.array([landmarks[eye_landmarks['top'][1]].x, 
                     landmarks[eye_landmarks['top'][1]].y])
    bottom1 = np.array([landmarks[eye_landmarks['bottom'][0]].x, 
                        landmarks[eye_landmarks['bottom'][0]].y])
    bottom2 = np.array([landmarks[eye_landmarks['bottom'][1]].x, 
                        landmarks[eye_landmarks['bottom'][1]].y])
    left = np.array([landmarks[eye_landmarks['left']].x, 
                     landmarks[eye_landmarks['left']].y])
    right = np.array([landmarks[eye_landmarks['right']].x, 
                      landmarks[eye_landmarks['right']].y])
    
    # Calculate vertical distances
    vertical1 = np.linalg.norm(top1 - bottom1)
    vertical2 = np.linalg.norm(top2 - bottom2)
    
    # Calculate horizontal distance
    horizontal = np.linalg.norm(left - right)
    
    # Calculate EAR
    if horizontal == 0:
        return 0.0
    
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


def _get_eye_center(landmarks, contour_indices):
    """
    Calculate the center of an eye using all contour landmarks.
    This gives a more stable reference point than just using corner landmarks.
    """
    x_sum = y_sum = 0
    for idx in contour_indices:
        x_sum += landmarks[idx].x
        y_sum += landmarks[idx].y
    return x_sum / len(contour_indices), y_sum / len(contour_indices)


def _get_iris_center_averaged(landmarks, iris_indices):
    """
    Get iris center by averaging all 5 iris landmarks.
    This is more stable than using just the center point.
    """
    x_sum = y_sum = 0
    for idx in iris_indices:
        x_sum += landmarks[idx].x
        y_sum += landmarks[idx].y
    return x_sum / len(iris_indices), y_sum / len(iris_indices)


def _calculate_gaze_ratio(landmarks, eye_contour, iris_indices, eye_landmarks):
    """
    Calculate gaze ratio for one eye.
    
    Gaze ratio = (iris_position - eye_left) / (eye_right - eye_left)
    This gives 0.0 when looking left, 1.0 when looking right.
    
    For vertical, we use iris position relative to eye top/bottom.
    """
    try:
        # Get eye corners
        left_corner = np.array([landmarks[eye_landmarks['left']].x, 
                                landmarks[eye_landmarks['left']].y])
        right_corner = np.array([landmarks[eye_landmarks['right']].x, 
                                 landmarks[eye_landmarks['right']].y])
        
        # Get iris center (averaged from all iris points for stability)
        iris_x, iris_y = _get_iris_center_averaged(landmarks, iris_indices)
        iris = np.array([iris_x, iris_y])
        
        # Calculate horizontal gaze ratio
        eye_width = np.linalg.norm(right_corner - left_corner)
        if eye_width < 0.001:
            return None
        
        # Project iris onto the eye width axis for horizontal ratio
        eye_vec = right_corner - left_corner
        iris_vec = iris - left_corner
        horizontal_ratio = np.dot(iris_vec, eye_vec) / (eye_width ** 2)
        
        # Calculate vertical gaze ratio
        top_y = (landmarks[eye_landmarks['top'][0]].y + landmarks[eye_landmarks['top'][1]].y) / 2
        bottom_y = (landmarks[eye_landmarks['bottom'][0]].y + landmarks[eye_landmarks['bottom'][1]].y) / 2
        eye_height = bottom_y - top_y
        
        if abs(eye_height) < 0.001:
            vertical_ratio = 0.5  # Default to center if can't calculate
        else:
            vertical_ratio = (iris_y - top_y) / eye_height
        
        return horizontal_ratio, vertical_ratio, (iris_x, iris_y)
        
    except (IndexError, AttributeError):
        return None


def get_iris_position(landmarks):
    """
    Get the iris position optimized for gaze tracking.
    
    Uses all iris landmarks (5 per eye) for stability and calculates
    gaze ratio based on iris position within the eye socket.
    
    Returns:
        tuple: ((gaze_x, gaze_y), (abs_x, abs_y)) for tracking and visualization
    """
    try:
        # Calculate gaze ratios for both eyes
        left_result = _calculate_gaze_ratio(landmarks, LEFT_EYE_CONTOUR, LEFT_IRIS, LEFT_EYE_LANDMARKS)
        right_result = _calculate_gaze_ratio(landmarks, RIGHT_EYE_CONTOUR, RIGHT_IRIS, RIGHT_EYE_LANDMARKS)
        
        if left_result is None or right_result is None:
            return None
        
        left_h, left_v, left_iris = left_result
        right_h, right_v, right_iris = right_result
        
        # Average the gaze ratios from both eyes for stability
        gaze_x = (left_h + right_h) / 2
        gaze_y = (left_v + right_v) / 2
        
        # Average iris positions for visualization
        abs_x = (left_iris[0] + right_iris[0]) / 2
        abs_y = (left_iris[1] + right_iris[1]) / 2
        
        return (gaze_x, gaze_y), (abs_x, abs_y)
        
    except (IndexError, AttributeError):
        return None

