"""
Utility classes for smoothing and calibration.
"""

import numpy as np
from collections import deque


class Smoother:
    """
    Applies smoothing to reduce jitter in gaze tracking.
    
    Combines Exponential Moving Average (EMA) with a Moving Average buffer
    for smooth cursor movement.
    
    EMA Formula: 
        smoothed = alpha * current + (1 - alpha) * previous
        
    Where alpha (0-1) controls responsiveness:
        - Higher alpha = more responsive, less smooth
        - Lower alpha = smoother, more lag
    """
    
    def __init__(self, alpha=0.3, window_size=5):
        self.alpha = alpha
        self.window_size = window_size
        self.ema_x = None
        self.ema_y = None
        self.buffer_x = deque(maxlen=window_size)
        self.buffer_y = deque(maxlen=window_size)
    
    def smooth(self, x, y):
        """
        Apply combined EMA and Moving Average smoothing.
        
        Args:
            x, y: Raw coordinates
            
        Returns:
            tuple: Smoothed (x, y) coordinates
        """
        # First apply EMA
        if self.ema_x is None:
            self.ema_x = x
            self.ema_y = y
        else:
            self.ema_x = self.alpha * x + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * y + (1 - self.alpha) * self.ema_y
        
        # Then apply Moving Average on top
        self.buffer_x.append(self.ema_x)
        self.buffer_y.append(self.ema_y)
        
        smooth_x = sum(self.buffer_x) / len(self.buffer_x)
        smooth_y = sum(self.buffer_y) / len(self.buffer_y)
        
        return smooth_x, smooth_y
    
    def reset(self):
        """Reset smoother state (useful when recalibrating)."""
        self.ema_x = None
        self.ema_y = None
        self.buffer_x.clear()
        self.buffer_y.clear()


class CalibrationData:
    """
    Stores calibration data for mapping eye coordinates to screen coordinates.
    
    CALIBRATION MATH EXPLANATION:
    =============================
    1. During calibration, user looks at screen corners while we capture
       the raw iris coordinates from MediaPipe (normalized 0-1).
       
    2. Looking at Top-Left gives us (min_x, min_y) - the iris position
       when looking at upper-left of their gaze range.
       
    3. Looking at Bottom-Right gives us (max_x, max_y) - the iris position
       when looking at lower-right of their gaze range.
       
    4. These define the "Gaze Bounds" - the range of eye movement we map
       to the full screen.
       
    5. To convert raw iris coords to screen position, we use linear interpolation:
       
       screen_x = numpy.interp(raw_x, [cal_min_x, cal_max_x], [0, screen_width])
       screen_y = numpy.interp(raw_y, [cal_min_y, cal_max_y], [0, screen_height])
       
    This maps the calibrated eye movement range linearly to the screen resolution.
    
    NOTE: We add a small margin (5%) to the calibrated bounds to give users
    a bit more range of motion.
    """
    
    def __init__(self):
        self.min_x = None  # Looking at top-left
        self.min_y = None
        self.max_x = None  # Looking at bottom-right  
        self.max_y = None
        self.is_calibrated = False
        self.calibration_step = 0  # 0 = top-left, 1 = bottom-right
    
    def set_top_left(self, x, y):
        """Store calibration point for top-left corner."""
        self.min_x = x
        self.min_y = y
        self.calibration_step = 1
        print(f"[Calibration] Top-Left captured: ({x:.4f}, {y:.4f})")
    
    def set_bottom_right(self, x, y):
        """Store calibration point for bottom-right corner."""
        self.max_x = x
        self.max_y = y
        self.calibration_step = 2
        self._finalize_calibration()
    
    def _finalize_calibration(self):
        """
        Finalize calibration and apply margin adjustments.
        
        We add a 5% margin to extend the usable range slightly beyond
        the exact calibration points.
        """
        if self.min_x is None or self.max_x is None:
            return
        
        # Calculate range
        range_x = abs(self.max_x - self.min_x)
        range_y = abs(self.max_y - self.min_y)
        
        # Add 5% margin on each side
        margin_x = range_x * 0.05
        margin_y = range_y * 0.05
        
        # Ensure min < max and apply margins
        if self.min_x > self.max_x:
            self.min_x, self.max_x = self.max_x, self.min_x
        if self.min_y > self.max_y:
            self.min_y, self.max_y = self.max_y, self.min_y
            
        self.min_x -= margin_x
        self.max_x += margin_x
        self.min_y -= margin_y
        self.max_y += margin_y
        
        self.is_calibrated = True
        print(f"[Calibration] Complete!")
        print(f"[Calibration] Gaze Bounds: X[{self.min_x:.4f}, {self.max_x:.4f}], "
              f"Y[{self.min_y:.4f}, {self.max_y:.4f}]")
    
    def map_to_screen(self, raw_x, raw_y, screen_width, screen_height):
        """
        Map raw eye coordinates to screen coordinates using calibration data.
        
        Uses numpy.interp for linear interpolation:
        - Maps raw_x from [min_x, max_x] to [0, screen_width]
        - Maps raw_y from [min_y, max_y] to [0, screen_height]
        
        Args:
            raw_x, raw_y: Raw iris coordinates from MediaPipe
            screen_width, screen_height: Target screen dimensions
            
        Returns:
            tuple: (screen_x, screen_y) pixel coordinates
        """
        if not self.is_calibrated:
            return None, None
        
        # Linear interpolation from calibrated bounds to screen coordinates
        # np.interp automatically clamps values to the output range
        screen_x = np.interp(raw_x, [self.min_x, self.max_x], [0, screen_width])
        screen_y = np.interp(raw_y, [self.min_y, self.max_y], [0, screen_height])
        
        return screen_x, screen_y
