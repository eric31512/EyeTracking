"""
Eye-Controlled Mouse Interaction Script with Calibration
=========================================================
Main entry point for the eye-controlled mouse application.

This script enables mouse control using eye gaze tracking with MediaPipe Face Mesh.
It implements a two-phase system: calibration followed by active control.

Dependencies:
    pip install opencv-python mediapipe pyautogui numpy

Controls:
    - Press 'C' during calibration to capture calibration points
    - Press 'Q' at any time to quit
    - Double blink (blink twice quickly) to left-click (during active control phase)

Usage:
    python main.py

Project Structure:
    - main.py       : Entry point (this file)
    - config.py     : Configuration settings
    - landmarks.py  : MediaPipe landmark definitions and EAR calculation
    - utils.py      : Smoother and CalibrationData classes
    - controller.py : Main EyeMouseController class
"""

from controller import EyeMouseController


def main():
    """Main entry point for the application."""
    print("\n" + "=" * 60)
    print("   EYE-CONTROLLED MOUSE WITH CALIBRATION")
    print("=" * 60 + "\n")
    
    controller = EyeMouseController()
    controller.run()


if __name__ == "__main__":
    main()
