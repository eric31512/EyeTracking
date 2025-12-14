"""
Eye tracking controller using dlib directly.

This implementation uses:
- dlib's 68-point facial landmark detector for face/eye detection
- Direct pupil position calculation from eye landmarks
- Iris center estimation using eye contour

No external gaze tracking library - pure dlib + OpenCV.
"""

import cv2
import dlib
import numpy as np
import pyautogui
import time
import os

from config import Config


# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Path to shape predictor model
SHAPE_PREDICTOR_PATH = os.path.join(
    os.path.dirname(__file__), 
    'gaze_tracking_lib', 
    'gaze_tracking', 
    'trained_models', 
    'shape_predictor_68_face_landmarks.dat'
)

# Eye landmark indices (from 68-point model)
LEFT_EYE_INDICES = list(range(36, 42))   # Points 36-41
RIGHT_EYE_INDICES = list(range(42, 48))  # Points 42-47


class DlibEyeTracker:
    """
    Eye tracking controller using dlib directly.
    
    Uses dlib's 68-point facial landmark model to:
    1. Detect face
    2. Extract eye regions
    3. Find pupil/iris center
    4. Calculate gaze direction
    """
    
    def __init__(self):
        # Initialize dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        # Calibration
        self.is_calibrating = True
        self.calibration_step = 0
        self.calibration_points = []
        self.calibration_targets = [
            ("Top-Left", (100, 100)),
            ("Top-Right", (Config.SCREEN_WIDTH - 100, 100)),
            ("Bottom-Left", (100, Config.SCREEN_HEIGHT - 100)),
            ("Bottom-Right", (Config.SCREEN_WIDTH - 100, Config.SCREEN_HEIGHT - 100)),
            ("Center", (Config.SCREEN_WIDTH // 2, Config.SCREEN_HEIGHT // 2)),
        ]
        
        # Gaze bounds (set during calibration)
        self.x_min, self.x_max = 0.3, 0.7
        self.y_min, self.y_max = 0.3, 0.7
        
        # Blink detection
        self.blink_threshold = 0.2
        self.blink_start_time = None
        self.last_click_time = 0
        self.click_cooldown = 1.0
        
        # Smoothing
        self.smooth_x = None
        self.smooth_y = None
        self.smooth_factor = 0.2
    
    def run(self):
        """Main application loop."""
        print("=" * 60)
        print("EYE TRACKING WITH DLIB")
        print("=" * 60)
        print("\nInstructions:")
        print("  - Press 'C' to capture calibration point")
        print("  - After calibration, gaze controls cursor")
        print("  - Long blink (0.5s) to click")
        print("  - Press 'R' to recalibrate")
        print("  - Press 'Q' to quit")
        print()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Error] Failed to read camera frame")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detector(gray)
                
                gaze_data = None
                is_blinking = False
                
                if len(faces) > 0:
                    # Use first face
                    face = faces[0]
                    landmarks = self.predictor(gray, face)
                    
                    # Extract eye data
                    left_eye = self._get_eye_data(landmarks, LEFT_EYE_INDICES, gray)
                    right_eye = self._get_eye_data(landmarks, RIGHT_EYE_INDICES, gray)
                    
                    # Check for blink
                    left_ear = self._calculate_ear(landmarks, LEFT_EYE_INDICES)
                    right_ear = self._calculate_ear(landmarks, RIGHT_EYE_INDICES)
                    avg_ear = (left_ear + right_ear) / 2
                    is_blinking = avg_ear < self.blink_threshold
                    
                    # Calculate average gaze
                    if left_eye and right_eye:
                        gaze_x = (left_eye['gaze_x'] + right_eye['gaze_x']) / 2
                        gaze_y = (left_eye['gaze_y'] + right_eye['gaze_y']) / 2
                        gaze_data = {'x': gaze_x, 'y': gaze_y}
                    
                    # Draw eye landmarks
                    self._draw_eyes(frame, landmarks)
                
                # Process based on mode
                if self.is_calibrating:
                    self._process_calibration(frame, gaze_data)
                else:
                    self._process_control(frame, gaze_data, is_blinking)
                
                # Display
                cv2.imshow("Dlib Eye Tracker", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and self.is_calibrating:
                    self._capture_calibration(gaze_data)
                elif key == ord('r'):
                    self._restart_calibration()
                    
        finally:
            self.cleanup()
    
    def _get_eye_data(self, landmarks, indices, gray):
        """Extract eye region and calculate gaze direction."""
        # Get eye points
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
        points = np.array(points, dtype=np.int32)
        
        # Get bounding box
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add margin
        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(gray.shape[1], x_max + margin)
        y_max = min(gray.shape[0], y_max + margin)
        
        # Extract eye region
        eye_region = gray[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return None
        
        # Find pupil using thresholding
        pupil_pos = self._find_pupil(eye_region)
        
        if pupil_pos is None:
            return None
        
        # Calculate gaze ratio (0-1)
        eye_width = x_max - x_min
        eye_height = y_max - y_min
        
        gaze_x = pupil_pos[0] / eye_width if eye_width > 0 else 0.5
        gaze_y = pupil_pos[1] / eye_height if eye_height > 0 else 0.5
        
        return {
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'pupil': (x_min + pupil_pos[0], y_min + pupil_pos[1])
        }
    
    def _find_pupil(self, eye_region):
        """Find pupil center in eye region using threshold and contour detection."""
        if eye_region.size == 0:
            return None
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(eye_region, (7, 7), 0)
        
        # Threshold to find dark pupil region
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours, use darkest point
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
            return min_loc
        
        # Find largest contour (likely the pupil)
        largest = max(contours, key=cv2.contourArea)
        
        # Get center of contour
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        
        return None
    
    def _calculate_ear(self, landmarks, indices):
        """Calculate Eye Aspect Ratio for blink detection."""
        # Get eye points
        p1 = np.array([landmarks.part(indices[0]).x, landmarks.part(indices[0]).y])
        p2 = np.array([landmarks.part(indices[1]).x, landmarks.part(indices[1]).y])
        p3 = np.array([landmarks.part(indices[2]).x, landmarks.part(indices[2]).y])
        p4 = np.array([landmarks.part(indices[3]).x, landmarks.part(indices[3]).y])
        p5 = np.array([landmarks.part(indices[4]).x, landmarks.part(indices[4]).y])
        p6 = np.array([landmarks.part(indices[5]).x, landmarks.part(indices[5]).y])
        
        # EAR formula
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal) if horizontal > 0 else 0
        return ear
    
    def _draw_eyes(self, frame, landmarks):
        """Draw eye landmarks on frame."""
        # Draw eye contours
        for indices in [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]:
            points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
            points = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 0), 1)
    
    def _process_calibration(self, frame, gaze_data):
        """Process calibration phase."""
        height, width = frame.shape[:2]
        
        # Draw header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        if self.calibration_step < len(self.calibration_targets):
            target_name, target_pos = self.calibration_targets[self.calibration_step]
            
            cv2.putText(frame, f"CALIBRATION: {self.calibration_step + 1}/{len(self.calibration_targets)}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Look at {target_name} and press 'C'", 
                       (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw target
            target_frame_x = int(target_pos[0] / Config.SCREEN_WIDTH * width)
            target_frame_y = int(target_pos[1] / Config.SCREEN_HEIGHT * height)
            cv2.circle(frame, (target_frame_x, target_frame_y), 15, (0, 0, 255), -1)
            
            # Show current gaze
            if gaze_data:
                cv2.putText(frame, f"Gaze: X={gaze_data['x']:.2f} Y={gaze_data['y']:.2f}", 
                           (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _capture_calibration(self, gaze_data):
        """Capture calibration point."""
        if gaze_data is None:
            print("[Warning] No gaze detected")
            return
        
        target_name, _ = self.calibration_targets[self.calibration_step]
        self.calibration_points.append(gaze_data.copy())
        
        print(f"[Calibration] {self.calibration_step + 1}/{len(self.calibration_targets)} {target_name}: X={gaze_data['x']:.3f} Y={gaze_data['y']:.3f}")
        
        self.calibration_step += 1
        
        if self.calibration_step >= len(self.calibration_targets):
            self._finalize_calibration()
    
    def _finalize_calibration(self):
        """Compute calibration bounds."""
        x_vals = [p['x'] for p in self.calibration_points]
        y_vals = [p['y'] for p in self.calibration_points]
        
        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)
        
        self.x_min = min(x_vals) - x_range * 0.1
        self.x_max = max(x_vals) + x_range * 0.1
        self.y_min = min(y_vals) - y_range * 0.1
        self.y_max = max(y_vals) + y_range * 0.1
        
        print(f"[Calibration] Complete! X[{self.x_min:.3f}, {self.x_max:.3f}] Y[{self.y_min:.3f}, {self.y_max:.3f}]")
        print("\n[Info] Mouse control active!\n")
        
        self.is_calibrating = False
    
    def _process_control(self, frame, gaze_data, is_blinking):
        """Process control phase."""
        height, width = frame.shape[:2]
        
        # Draw header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, "ACTIVE - dlib Eye Tracker", (20, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Long blink to click | 'R' recalibrate | 'Q' quit", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Handle blink click
        if is_blinking:
            if self.blink_start_time is None:
                self.blink_start_time = time.time()
            elif time.time() - self.blink_start_time >= 0.5:
                if time.time() - self.last_click_time > self.click_cooldown:
                    pyautogui.click()
                    self.last_click_time = time.time()
                    print("[Click] Blink click!")
                    self.blink_start_time = None
            
            cv2.putText(frame, "BLINKING", (width // 2 - 60, height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            self.blink_start_time = None
        
        if gaze_data and not is_blinking:
            # Map gaze to screen
            screen_x = self._map_value(gaze_data['x'], self.x_min, self.x_max, 0, Config.SCREEN_WIDTH)
            screen_y = self._map_value(gaze_data['y'], self.y_min, self.y_max, 0, Config.SCREEN_HEIGHT)
            
            # Invert X (looking left moves cursor right in mirror view)
            screen_x = Config.SCREEN_WIDTH - screen_x
            
            # Smooth
            if self.smooth_x is None:
                self.smooth_x = screen_x
                self.smooth_y = screen_y
            else:
                self.smooth_x = self.smooth_factor * screen_x + (1 - self.smooth_factor) * self.smooth_x
                self.smooth_y = self.smooth_factor * screen_y + (1 - self.smooth_factor) * self.smooth_y
            
            # Clamp
            sx = max(0, min(Config.SCREEN_WIDTH - 1, int(self.smooth_x)))
            sy = max(0, min(Config.SCREEN_HEIGHT - 1, int(self.smooth_y)))
            
            pyautogui.moveTo(sx, sy)
            
            # Draw gaze indicator
            gx = int(self.smooth_x / Config.SCREEN_WIDTH * width)
            gy = int(self.smooth_y / Config.SCREEN_HEIGHT * height)
            cv2.circle(frame, (gx, gy), 15, (0, 255, 0), -1)
            cv2.circle(frame, (gx, gy), 20, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Cursor: ({sx}, {sy})", (width - 220, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _map_value(self, value, in_min, in_max, out_min, out_max):
        """Map value from one range to another."""
        if in_max == in_min:
            return (out_min + out_max) / 2
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    def _restart_calibration(self):
        """Restart calibration."""
        print("\n[Info] Restarting calibration...")
        self.is_calibrating = True
        self.calibration_step = 0
        self.calibration_points = []
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("[Info] Cleanup complete!")


def main():
    controller = DlibEyeTracker()
    controller.run()


if __name__ == "__main__":
    main()
