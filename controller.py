"""
Main controller class for eye-based mouse interaction.
"""

import cv2
import time
import mediapipe as mp
import pyautogui

from config import Config
from landmarks import (
    LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS,
    calculate_ear, get_iris_position
)
from utils import Smoother, CalibrationData


# Disable PyAutoGUI fail-safe for smoother control (move mouse to corner to exit)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # Remove delay between PyAutoGUI calls


class EyeMouseController:
    """
    Main controller for eye-based mouse interaction.
    
    Phases:
        1. Calibration: Capture gaze bounds by looking at screen corners
        2. Active Control: Move mouse based on gaze, click on double blink
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Required for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        # State management
        self.calibration = CalibrationData()
        self.smoother = Smoother(Config.EMA_ALPHA, Config.SMOOTHING_WINDOW)
        
        # Blink detection state
        self.blink_counter = 0  # Counts consecutive frames with eyes closed
        self.blink_count = 0  # Counts number of completed blinks for double-blink detection
        self.last_blink_time = 0  # Time of last completed blink
        self.last_click_time = 0  # Time of last click (for cooldown)
        self.was_blinking = False  # Track if eyes were closed in previous frame
        
        # Current gaze position (for visualization)
        self.current_gaze = None
        
    def run(self):
        """Main application loop."""
        print("=" * 60)
        print("EYE-CONTROLLED MOUSE - Starting...")
        print("=" * 60)
        print("\nControls:")
        print("  - Press 'C' during calibration to capture points")
        print("  - Press 'Q' to quit at any time")
        print("  - Double blink (blink twice quickly) to left-click")
        print()
        
        # Create window
        cv2.namedWindow("Eye Mouse Control", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Error] Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                # Process based on current phase
                if not self.calibration.is_calibrated:
                    frame = self._process_calibration_phase(frame, results)
                else:
                    frame = self._process_control_phase(frame, results)
                
                # Show frame
                cv2.imshow("Eye Mouse Control", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n[Info] Quitting...")
                    break
                elif key == ord('c') or key == ord('C'):
                    self._handle_calibration_keypress(results)
                    
        finally:
            self.cleanup()
    
    def _process_calibration_phase(self, frame, results):
        """
        Process frame during calibration phase.
        
        Displays instructions and visualizes detected face/iris.
        Does NOT move the mouse.
        """
        height, width = frame.shape[:2]
        
        # Draw semi-transparent overlay for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Draw calibration instructions
        if self.calibration.calibration_step == 0:
            text1 = "CALIBRATION: Step 1/2"
            text2 = "Look at the TOP-LEFT corner of your screen"
            text3 = "Press 'C' to capture"
            corner_pos = (50, 150)
            cv2.circle(frame, corner_pos, 20, Config.COLOR_GREEN, -1)
            cv2.circle(frame, corner_pos, 25, Config.COLOR_GREEN, 2)
        else:
            text1 = "CALIBRATION: Step 2/2"
            text2 = "Look at the BOTTOM-RIGHT corner of your screen"
            text3 = "Press 'C' to capture"
            corner_pos = (width - 50, height - 50)
            cv2.circle(frame, corner_pos, 20, Config.COLOR_GREEN, -1)
            cv2.circle(frame, corner_pos, 25, Config.COLOR_GREEN, 2)
        
        # Draw text
        cv2.putText(frame, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, Config.COLOR_YELLOW, 2)
        cv2.putText(frame, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, Config.COLOR_WHITE, 1)
        cv2.putText(frame, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, Config.COLOR_CYAN, 1)
        
        # Draw detected iris position if face is detected
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            iris_pos = get_iris_position(landmarks)
            
            if iris_pos:
                # Convert to pixel coordinates for visualization
                iris_x = int(iris_pos[0] * width)
                iris_y = int(iris_pos[1] * height)
                
                # Draw iris indicator
                cv2.circle(frame, (iris_x, iris_y), 10, Config.COLOR_RED, -1)
                cv2.circle(frame, (iris_x, iris_y), 15, Config.COLOR_RED, 2)
                
                # Show raw coordinates
                coord_text = f"Iris: ({iris_pos[0]:.3f}, {iris_pos[1]:.3f})"
                cv2.putText(frame, coord_text, (width - 250, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_GREEN, 1)
        else:
            # No face detected warning
            cv2.putText(frame, "No face detected!", (width//2 - 100, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.COLOR_RED, 2)
        
        return frame
    
    def _process_control_phase(self, frame, results):
        """
        Process frame during active control phase.
        
        Moves the mouse based on gaze and detects blinks for clicking.
        """
        height, width = frame.shape[:2]
        
        # Draw header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        cv2.putText(frame, "ACTIVE CONTROL MODE", (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_GREEN, 2)
        cv2.putText(frame, "Double blink to click | Press 'Q' to quit", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WHITE, 1)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Get iris position
            iris_pos = get_iris_position(landmarks)
            
            if iris_pos:
                raw_x, raw_y = iris_pos
                
                # Map to screen coordinates using calibration
                screen_x, screen_y = self.calibration.map_to_screen(
                    raw_x, raw_y, 
                    Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
                )
                
                if screen_x is not None:
                    # Apply smoothing
                    smooth_x, smooth_y = self.smoother.smooth(screen_x, screen_y)
                    
                    # Clamp to screen bounds
                    smooth_x = max(0, min(Config.SCREEN_WIDTH - 1, smooth_x))
                    smooth_y = max(0, min(Config.SCREEN_HEIGHT - 1, smooth_y))
                    
                    # Move mouse
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                    
                    # Store for visualization
                    self.current_gaze = (int(smooth_x), int(smooth_y))
                    
                    # Draw gaze point on frame (scaled to frame size)
                    gaze_frame_x = int(smooth_x / Config.SCREEN_WIDTH * width)
                    gaze_frame_y = int(smooth_y / Config.SCREEN_HEIGHT * height)
                    cv2.circle(frame, (gaze_frame_x, gaze_frame_y), 15, Config.COLOR_GREEN, -1)
                    cv2.circle(frame, (gaze_frame_x, gaze_frame_y), 20, Config.COLOR_GREEN, 2)
                    
                    # Show coordinates
                    pos_text = f"Screen: ({int(smooth_x)}, {int(smooth_y)})"
                    cv2.putText(frame, pos_text, (width - 200, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_GREEN, 1)
            
            # Blink detection
            self._detect_blink(landmarks, frame)
        else:
            cv2.putText(frame, "Face lost - please center yourself", 
                       (width//2 - 150, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_RED, 2)
        
        return frame
    
    def _detect_blink(self, landmarks, frame):
        """
        Detect DOUBLE blinks using Eye Aspect Ratio (EAR).
        
        Requires two blinks within a short time window to trigger a click.
        This reduces accidental clicks from natural blinking.
        
        Logic:
        1. Track when eyes close (EAR drops below threshold)
        2. When eyes open again after being closed, count as one blink
        3. If two blinks occur within DOUBLE_BLINK_TIME_WINDOW, trigger click
        4. Reset blink count if too much time passes between blinks
        """
        # Calculate EAR for both eyes
        left_ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_LANDMARKS)
        avg_ear = (left_ear + right_ear) / 2
        
        current_time = time.time()
        
        # Display EAR value and blink count
        ear_text = f"EAR: {avg_ear:.3f} | Blinks: {self.blink_count}/{Config.DOUBLE_BLINK_REQUIRED}"
        cv2.putText(frame, ear_text, (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_YELLOW, 1)
        
        # Check if currently blinking (eyes closed)
        is_blinking = avg_ear < Config.EAR_THRESHOLD
        
        if is_blinking:
            self.blink_counter += 1
        else:
            # Eyes just opened - check if this was a valid blink
            if self.was_blinking and self.blink_counter >= Config.EAR_CONSEC_FRAMES:
                # Valid blink completed!
                
                # Check if this blink is within time window of previous blink
                if current_time - self.last_blink_time > Config.DOUBLE_BLINK_TIME_WINDOW:
                    # Too much time passed, reset count
                    self.blink_count = 0
                
                # Increment blink count
                self.blink_count += 1
                self.last_blink_time = current_time
                print(f"[Blink] Detected blink {self.blink_count}/{Config.DOUBLE_BLINK_REQUIRED}")
                
                # Check if we have enough blinks for a click
                if self.blink_count >= Config.DOUBLE_BLINK_REQUIRED:
                    # Check cooldown
                    if current_time - self.last_click_time > Config.CLICK_COOLDOWN:
                        pyautogui.click()
                        self.last_click_time = current_time
                        print("[Click] Double blink detected - Left Click!")
                        
                        # Visual feedback
                        cv2.putText(frame, "CLICK!", (frame.shape[1]//2 - 50, frame.shape[0]//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, Config.COLOR_RED, 3)
                    
                    # Reset blink count after click
                    self.blink_count = 0
            
            self.blink_counter = 0
        
        # Update blinking state for next frame
        self.was_blinking = is_blinking
        
        # Reset blink count if too much time has passed since last blink
        if self.blink_count > 0 and current_time - self.last_blink_time > Config.DOUBLE_BLINK_TIME_WINDOW:
            self.blink_count = 0
        
        # Blink indicator - shows current state
        if is_blinking:
            cv2.circle(frame, (frame.shape[1] - 30, 90), 15, Config.COLOR_RED, -1)
        elif self.blink_count > 0:
            # Show yellow when waiting for second blink
            cv2.circle(frame, (frame.shape[1] - 30, 90), 15, Config.COLOR_YELLOW, -1)
        else:
            cv2.circle(frame, (frame.shape[1] - 30, 90), 15, Config.COLOR_GREEN, -1)
    
    def _handle_calibration_keypress(self, results):
        """Handle 'C' key press during calibration."""
        if self.calibration.is_calibrated:
            return
        
        if not results.multi_face_landmarks:
            print("[Warning] No face detected - cannot capture calibration point")
            return
        
        landmarks = results.multi_face_landmarks[0].landmark
        iris_pos = get_iris_position(landmarks)
        
        if iris_pos is None:
            print("[Warning] Iris not detected - cannot capture calibration point")
            return
        
        if self.calibration.calibration_step == 0:
            self.calibration.set_top_left(iris_pos[0], iris_pos[1])
        elif self.calibration.calibration_step == 1:
            self.calibration.set_bottom_right(iris_pos[0], iris_pos[1])
            print("\n[Info] Calibration complete! Entering control mode...\n")
    
    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        print("[Info] Cleanup complete. Goodbye!")
