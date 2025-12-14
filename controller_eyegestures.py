"""
Eye tracking controller using EyeGestures library.

Based on official EyeGestures example from:
https://github.com/NativeSensors/EyeGestures
"""

import cv2
import pyautogui
import time
import warnings

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

from config import Config


# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class EyeGestureController:
    """
    Eye tracking controller using EyeGestures library.
    Based on official example from EyeGestures GitHub.
    """
    
    def __init__(self):
        # Initialize EyeGestures engine
        self.gestures = EyeGestures_v3()
        
        # Use EyeGestures VideoCapture (as per official example)
        self.cap = VideoCapture(Config.CAMERA_INDEX)
        
        # State
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_duration = 8.0  # Longer calibration for better results
        
        # Dwell click settings
        self.last_fixation = False
        self.fixation_start_time = 0
        self.dwell_threshold = 1.2
        self.last_click_time = 0
        self.click_cooldown = 1.5
        self.dwell_progress = 0.0
        
        # Smoothing
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_factor = 0.3
    
    def run(self):
        """Main application loop."""
        print("=" * 60)
        print("EYE TRACKING WITH EYEGESTURES V3")
        print("=" * 60)
        print("\nInstructions:")
        print("  - Look around ALL corners during calibration (8 seconds)")
        print("  - After calibration, your gaze will control the mouse")
        print("  - Fix your gaze on a point for 1.2s to click")
        print("  - Press 'Q' to quit")
        print("  - Press 'R' to recalibrate")
        print()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[Error] Failed to read camera frame")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Check calibration status
                elapsed = time.time() - self.calibration_start_time
                if self.is_calibrating and elapsed >= self.calibration_duration:
                    self.is_calibrating = False
                    print("\n[Info] Calibration complete! Mouse control active.\n")
                
                # Process frame with EyeGestures (positional args as per official example)
                event, cevent = self.gestures.step(
                    frame,
                    self.is_calibrating,  # calibration flag
                    Config.SCREEN_WIDTH,  # screen_width
                    Config.SCREEN_HEIGHT, # screen_height
                    "main"                # context
                )
                
                # Process gaze event
                if event:
                    self._process_gaze(event, frame)
                
                # Draw UI
                self._draw_ui(frame, event)
                
                # Display frame
                cv2.imshow("EyeGestures Tracker", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Info] Quitting...")
                    break
                elif key == ord('r'):
                    self._restart_calibration()
                    
        finally:
            self.cleanup()
    
    def _process_gaze(self, event, frame):
        """Process gaze event and move mouse."""
        cursor_x, cursor_y = event.point[0], event.point[1]
        fixation = event.fixation
        
        # Apply smoothing
        if self.smooth_x is None:
            self.smooth_x = cursor_x
            self.smooth_y = cursor_y
        else:
            self.smooth_x = self.smooth_factor * cursor_x + (1 - self.smooth_factor) * self.smooth_x
            self.smooth_y = self.smooth_factor * cursor_y + (1 - self.smooth_factor) * self.smooth_y
        
        # Clamp to screen bounds
        screen_x = max(0, min(Config.SCREEN_WIDTH - 1, int(self.smooth_x)))
        screen_y = max(0, min(Config.SCREEN_HEIGHT - 1, int(self.smooth_y)))
        
        # Move mouse if not calibrating
        if not self.is_calibrating:
            pyautogui.moveTo(screen_x, screen_y)
        
        # Dwell click detection (alternative to blink)
        current_time = time.time()
        if fixation and not self.is_calibrating:
            if not self.last_fixation:
                # Fixation just started
                self.fixation_start_time = current_time
            
            # Calculate dwell progress
            dwell_time = current_time - self.fixation_start_time
            self.dwell_progress = min(1.0, dwell_time / self.dwell_threshold)
            
            if dwell_time >= self.dwell_threshold:
                # Fixation held long enough - click!
                if current_time - self.last_click_time > self.click_cooldown:
                    pyautogui.click()
                    self.last_click_time = current_time
                    print(f"[Click] Dwell click at ({screen_x}, {screen_y})")
                    self.fixation_start_time = current_time  # Reset
                    self.dwell_progress = 0.0
        else:
            self.dwell_progress = 0.0
        
        self.last_fixation = fixation
    
    def _draw_ui(self, frame, event):
        """Draw UI elements on frame."""
        height, width = frame.shape[:2]
        
        # Draw header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        if self.is_calibrating:
            elapsed = time.time() - self.calibration_start_time
            remaining = max(0, self.calibration_duration - elapsed)
            
            cv2.putText(frame, "CALIBRATING...", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Look around the screen edges ({remaining:.1f}s)", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw calibration progress bar
            progress = elapsed / self.calibration_duration
            bar_width = int((width - 40) * progress)
            cv2.rectangle(frame, (20, height - 30), (width - 20, height - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, height - 30), (20 + bar_width, height - 10), (0, 255, 0), -1)
        else:
            cv2.putText(frame, "ACTIVE - EyeGestures V3", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Fix gaze 0.8s to click | 'R' recalibrate | 'Q' quit", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw gaze point if available
        if event and not self.is_calibrating:
            # Scale gaze to frame coordinates
            gaze_x = int(self.smooth_x / Config.SCREEN_WIDTH * width)
            gaze_y = int(self.smooth_y / Config.SCREEN_HEIGHT * height)
            
            # Draw gaze indicator
            color = (0, 255, 0) if event.fixation else (0, 255, 255)
            cv2.circle(frame, (gaze_x, gaze_y), 15, color, -1)
            cv2.circle(frame, (gaze_x, gaze_y), 20, color, 2)
            
            # Draw dwell progress ring
            if self.dwell_progress > 0:
                # Draw arc showing dwell progress
                angle = int(360 * self.dwell_progress)
                cv2.ellipse(frame, (gaze_x, gaze_y), (25, 25), -90, 0, angle, (255, 0, 255), 3)
            
            # Show coordinates
            pos_text = f"Gaze: ({int(self.smooth_x)}, {int(self.smooth_y)})"
            if event.fixation:
                pos_text += f" [DWELL {self.dwell_progress*100:.0f}%]"
            cv2.putText(frame, pos_text, (width - 320, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _restart_calibration(self):
        """Restart calibration."""
        print("\n[Info] Restarting calibration...")
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.gestures = EyeGestures_v3()  # Reset the engine
    
    def cleanup(self):
        """Release resources."""
        # EyeGestures VideoCapture may not have release(), so try-except
        try:
            if hasattr(self.cap, 'release'):
                self.cap.release()
        except:
            pass
        cv2.destroyAllWindows()
        print("[Info] Cleanup complete. Goodbye!")


def main():
    """Entry point."""
    controller = EyeGestureController()
    controller.run()


if __name__ == "__main__":
    main()
