"""
Utility classes for smoothing and calibration.
"""

import numpy as np
from collections import deque
import math


class OneEuroFilter:
    """
    One Euro Filter for adaptive smoothing.
    
    This filter adapts its smoothing based on the speed of movement:
    - When movement is slow (noise), it smooths aggressively
    - When movement is fast (intentional), it responds quickly
    
    Reference: http://cristal.univ-lille.fr/~casiez/1euro/
    """
    
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        Args:
            freq: Frequency of updates (Hz) - approximate FPS
            min_cutoff: Minimum cutoff frequency. Lower = more smoothing when still.
            beta: Speed coefficient. Higher = more responsive to fast movements.
            d_cutoff: Cutoff for derivative filtering.
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = 0.0
        
    def _alpha(self, cutoff):
        """Compute alpha for exponential smoothing."""
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, x):
        """Apply the One Euro Filter to a value."""
        if self.x_prev is None:
            self.x_prev = x
            return x
        
        # Compute derivative
        dx = (x - self.x_prev) * self.freq
        
        # Smooth derivative
        alpha_d = self._alpha(self.d_cutoff)
        dx_smooth = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        self.dx_prev = dx_smooth
        
        # Compute adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        
        # Smooth the value
        alpha = self._alpha(cutoff)
        x_smooth = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_smooth
        
        return x_smooth
    
    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = 0.0


class Smoother:
    """
    Applies smoothing to reduce jitter in gaze tracking.
    
    Uses the One Euro Filter for adaptive smoothing that is:
    - Smooth when gaze is stationary (reduces jitter)
    - Responsive when gaze moves quickly (reduces lag)
    """
    
    def __init__(self, alpha=0.3, window_size=5):
        # Keep original params for compatibility but use One Euro Filter
        self.alpha = alpha
        self.window_size = window_size
        
        # One Euro filters for x and y
        # min_cutoff=0.5 -> very smooth when still
        # beta=0.5 -> responds well to fast movement  
        self.filter_x = OneEuroFilter(freq=30.0, min_cutoff=0.5, beta=0.5)
        self.filter_y = OneEuroFilter(freq=30.0, min_cutoff=0.5, beta=0.5)
        
        # Additional moving average buffer for extra stability
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
    
    def smooth(self, x, y):
        """
        Apply One Euro Filter + Moving Average smoothing.
        
        Args:
            x, y: Raw coordinates
            
        Returns:
            tuple: Smoothed (x, y) coordinates
        """
        # Apply One Euro Filter first
        filtered_x = self.filter_x.filter(x)
        filtered_y = self.filter_y.filter(y)
        
        # Then apply Moving Average for additional stability
        self.buffer_x.append(filtered_x)
        self.buffer_y.append(filtered_y)
        
        smooth_x = sum(self.buffer_x) / len(self.buffer_x)
        smooth_y = sum(self.buffer_y) / len(self.buffer_y)
        
        return smooth_x, smooth_y
    
    def reset(self):
        """Reset smoother state (useful when recalibrating)."""
        self.filter_x.reset()
        self.filter_y.reset()
        self.buffer_x.clear()
        self.buffer_y.clear()


class CalibrationData:
    """
    Stores calibration data for mapping eye coordinates to screen coordinates.
    
    9-POINT CALIBRATION:
    ====================
    Uses a 3x3 grid of calibration points for more accurate mapping.
    Grid layout:
        1  2  3    (top row)
        4  5  6    (middle row)
        7  8  9    (bottom row)
        
    Each point captures the eye position when looking at that screen location.
    Mapping uses grid-based interpolation for better local accuracy.
    """
    
    # 9 calibration points in a 3x3 grid (x, y as screen ratios)
    CALIBRATION_POINTS = [
        (0.1, 0.1),  # 1: Top-Left
        (0.5, 0.1),  # 2: Top-Center
        (0.9, 0.1),  # 3: Top-Right
        (0.1, 0.5),  # 4: Middle-Left
        (0.5, 0.5),  # 5: Center
        (0.9, 0.5),  # 6: Middle-Right
        (0.1, 0.9),  # 7: Bottom-Left
        (0.5, 0.9),  # 8: Bottom-Center
        (0.9, 0.9),  # 9: Bottom-Right
    ]
    
    POINT_NAMES = [
        "Top-Left", "Top-Center", "Top-Right",
        "Middle-Left", "Center", "Middle-Right",
        "Bottom-Left", "Bottom-Center", "Bottom-Right"
    ]
    
    def __init__(self):
        self.eye_points = []  # List of (eye_x, eye_y) for each calibration point
        self.screen_points = []  # List of (screen_x, screen_y) for each calibration point
        self.is_calibrated = False
        self.calibration_step = 0
        
        # Store calibration bounds for simple mapping
        self.min_x = self.max_x = self.min_y = self.max_y = None
    
    def get_current_target(self, screen_width, screen_height):
        """Get the current calibration target position in pixels."""
        if self.calibration_step >= len(self.CALIBRATION_POINTS):
            return None
        ratio_x, ratio_y = self.CALIBRATION_POINTS[self.calibration_step]
        return (int(ratio_x * screen_width), int(ratio_y * screen_height))
    
    def get_current_step_name(self):
        """Get human-readable name for current calibration step."""
        if self.calibration_step < len(self.POINT_NAMES):
            return self.POINT_NAMES[self.calibration_step]
        return "Complete"
    
    def add_calibration_point(self, eye_x, eye_y, screen_width, screen_height):
        """
        Add a calibration point from current eye position.
        
        Returns:
            bool: True if calibration is complete, False if more points needed
        """
        if self.calibration_step >= len(self.CALIBRATION_POINTS):
            return True
        
        # Get the screen position for this calibration point
        ratio_x, ratio_y = self.CALIBRATION_POINTS[self.calibration_step]
        screen_x = ratio_x * screen_width
        screen_y = ratio_y * screen_height
        
        # Store the eye position and corresponding screen position
        self.eye_points.append((eye_x, eye_y))
        self.screen_points.append((screen_x, screen_y))
        
        step_name = self.get_current_step_name()
        step_num = self.calibration_step + 1
        total = len(self.CALIBRATION_POINTS)
        print(f"[Calibration] {step_num}/{total} {step_name}: eye({eye_x:.4f}, {eye_y:.4f})")
        
        self.calibration_step += 1
        
        # Check if all points are captured
        if self.calibration_step >= len(self.CALIBRATION_POINTS):
            self._compute_mapping()
            return True
        
        return False
    
    def _compute_mapping(self):
        """Compute mapping parameters from calibration points."""
        if len(self.eye_points) < 9:
            print("[Calibration] Error: Not enough calibration points")
            return
        
        # Extract eye coordinate bounds with margin
        eye_x_vals = [p[0] for p in self.eye_points]
        eye_y_vals = [p[1] for p in self.eye_points]
        
        self.min_x = min(eye_x_vals)
        self.max_x = max(eye_x_vals)
        self.min_y = min(eye_y_vals)
        self.max_y = max(eye_y_vals)
        
        # Add margins
        range_x = self.max_x - self.min_x
        range_y = self.max_y - self.min_y
        margin_x = max(range_x * 0.1, 0.01)
        margin_y = max(range_y * 0.1, 0.02)
        
        self.min_x -= margin_x
        self.max_x += margin_x
        self.min_y -= margin_y
        self.max_y += margin_y
        
        self.is_calibrated = True
        print(f"[Calibration] Complete! 9-point calibration finished.")
        print(f"[Calibration] Eye bounds: X[{self.min_x:.4f}, {self.max_x:.4f}], Y[{self.min_y:.4f}, {self.max_y:.4f}]")
    
    def map_to_screen(self, raw_x, raw_y, screen_width, screen_height):
        """
        Map raw eye coordinates to screen coordinates.
        
        Uses linear interpolation based on the calibrated bounds.
        With 9 points, the bounds are more accurate than 2-point calibration.
        """
        if not self.is_calibrated:
            return None, None
        
        # Linear interpolation using calibrated bounds
        screen_x = np.interp(raw_x, [self.min_x, self.max_x], [0, screen_width])
        screen_y = np.interp(raw_y, [self.min_y, self.max_y], [0, screen_height])
        
        return screen_x, screen_y

