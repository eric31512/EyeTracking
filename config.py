"""
Configuration settings for the eye-controlled mouse system.
"""

import pyautogui


class Config:
    """Configuration parameters for the eye tracking system."""
    
    # Screen dimensions
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
    
    # Camera settings
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Smoothing parameters
    # EMA (Exponential Moving Average) alpha: lower = smoother but more lag
    EMA_ALPHA = 0.15  # Reduced from 0.3 for more stability
    
    # Moving average window size for additional smoothing
    SMOOTHING_WINDOW = 10  # Increased from 5 for more stability
    
    # Blink detection parameters
    # Eye Aspect Ratio threshold - below this is considered a blink
    EAR_THRESHOLD = 0.21
    # Consecutive frames below threshold to register as blink
    EAR_CONSEC_FRAMES = 2
    # Double blink detection parameters
    DOUBLE_BLINK_REQUIRED = 2  # Number of blinks required to trigger click
    DOUBLE_BLINK_TIME_WINDOW = 0.8  # Max time (seconds) between blinks to count as double blink
    # Cooldown time between clicks (seconds)
    CLICK_COOLDOWN = 0.5
    
    # Visual feedback colors (BGR format)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_CYAN = (255, 255, 0)
