"""
MediaPipe Face Mesh landmark definitions and EAR calculation.
"""

import numpy as np


# =============================================================================
# MEDIAPIPE FACE MESH LANDMARKS
# =============================================================================

# Iris landmarks (MediaPipe Face Mesh with refine_landmarks=True)
# Left iris center: 468, Right iris center: 473
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

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


def get_iris_position(landmarks):
    """
    Get the average iris position from both eyes.
    
    The iris landmarks provide normalized coordinates (0-1) that we use
    to determine where the user is looking.
    
    Args:
        landmarks: MediaPipe face landmarks
        
    Returns:
        tuple: (x, y) normalized iris position, or None if not detected
    """
    try:
        left_iris = landmarks[LEFT_IRIS_CENTER]
        right_iris = landmarks[RIGHT_IRIS_CENTER]
        
        # Average both iris positions for stability
        avg_x = (left_iris.x + right_iris.x) / 2
        avg_y = (left_iris.y + right_iris.y) / 2
        
        return (avg_x, avg_y)
    except (IndexError, AttributeError):
        return None
