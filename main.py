"""
Eye-Controlled Mouse with dlib
==============================
Main entry point for the eye-controlled mouse application.

Uses dlib directly for facial landmark detection and pupil tracking.

Controls:
    - Press 'C' during calibration to capture points
    - After calibration, gaze controls mouse
    - Long blink (0.5s) to click
    - Press 'R' to recalibrate
    - Press 'Q' to quit

Usage:
    python main.py
"""

from controller_dlib import DlibEyeTracker


def main():
    """Main entry point for the application."""
    print("\n" + "=" * 60)
    print("   EYE TRACKING WITH DLIB")
    print("=" * 60 + "\n")
    
    controller = DlibEyeTracker()
    controller.run()


if __name__ == "__main__":
    main()
