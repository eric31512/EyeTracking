"""
Head Pose Estimation and Gaze Disentanglement.

This module estimates head pose (pitch, yaw, roll) from MediaPipe facial landmarks
and provides functions to normalize gaze direction independent of head rotation.

Head Pose Disentanglement separates:
1. Head direction (where the head is pointing)
2. Eye-in-head gaze (where the eyes are looking relative to the head)

Final gaze = head direction + eye-in-head gaze

This allows accurate gaze tracking even when the user moves their head.
"""

import numpy as np
import cv2


# 3D model points for head pose estimation
# These are standard facial landmark positions in a canonical 3D coordinate system
# Based on a generic face model (units in mm, but scale doesn't matter for rotation)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices for the 3D model points
# These correspond to the MODEL_POINTS_3D positions
POSE_LANDMARK_INDICES = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    291,  # Right mouth corner
]


class HeadPoseEstimator:
    """
    Estimates head pose from MediaPipe facial landmarks using solvePnP.
    
    The head pose is represented as rotation (pitch, yaw, roll) and translation.
    This is used to normalize gaze direction independent of head orientation.
    """
    
    def __init__(self, frame_width, frame_height):
        """
        Initialize the head pose estimator.
        
        Args:
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Approximate camera matrix (assumes camera at center of image)
        # Focal length is approximated as frame width (common approximation)
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume no lens distortion
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Previous pose for smoothing
        self.prev_rvec = None
        self.prev_tvec = None
        self.smoothing_factor = 0.7  # Higher = more smoothing
    
    def get_2d_points(self, landmarks):
        """
        Extract 2D image points from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            numpy array of 2D points in image coordinates
        """
        points_2d = []
        for idx in POSE_LANDMARK_INDICES:
            x = landmarks[idx].x * self.frame_width
            y = landmarks[idx].y * self.frame_height
            points_2d.append((x, y))
        return np.array(points_2d, dtype=np.float64)
    
    def estimate_pose(self, landmarks):
        """
        Estimate head pose from facial landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            tuple: (rotation_vector, translation_vector, euler_angles)
                   euler_angles = (pitch, yaw, roll) in degrees
                   Returns None if estimation fails
        """
        try:
            # Get 2D points from landmarks
            points_2d = self.get_2d_points(landmarks)
            
            # Solve PnP to get rotation and translation
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D,
                points_2d,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None
            
            # Apply smoothing
            if self.prev_rvec is not None:
                rvec = self.smoothing_factor * self.prev_rvec + (1 - self.smoothing_factor) * rvec
                tvec = self.smoothing_factor * self.prev_tvec + (1 - self.smoothing_factor) * tvec
            
            self.prev_rvec = rvec
            self.prev_tvec = tvec
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Extract Euler angles from rotation matrix
            euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
            
            return rvec, tvec, euler_angles
            
        except Exception as e:
            return None
    
    def _rotation_matrix_to_euler(self, rotation_matrix):
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            tuple: (pitch, yaw, roll) in degrees
        """
        # Decompose rotation matrix to get Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return (pitch, yaw, roll)
    
    def normalize_gaze(self, iris_pos, landmarks, euler_angles):
        """
        Normalize gaze direction by removing head pose influence.
        
        This transforms the iris position from camera coordinates to 
        head-normalized coordinates, making gaze tracking robust to head movement.
        
        Args:
            iris_pos: (x, y) iris position in normalized coordinates (0-1)
            landmarks: MediaPipe face landmarks
            euler_angles: (pitch, yaw, roll) from estimate_pose
            
        Returns:
            tuple: (normalized_x, normalized_y) gaze direction independent of head pose
        """
        if euler_angles is None:
            return iris_pos
        
        pitch, yaw, roll = euler_angles
        
        # Get eye center for reference
        # Using the midpoint between inner eye corners as reference
        left_inner = landmarks[133]   # Left eye inner corner
        right_inner = landmarks[362]  # Right eye inner corner
        eye_center_x = (left_inner.x + right_inner.x) / 2
        eye_center_y = (left_inner.y + right_inner.y) / 2
        
        # Calculate iris offset from eye center
        iris_x, iris_y = iris_pos
        offset_x = iris_x - eye_center_x
        offset_y = iris_y - eye_center_y
        
        # Compensate for yaw (left-right head rotation)
        # When head turns right (positive yaw), iris appears to move left in camera
        # We need to add back the apparent movement
        yaw_rad = np.radians(yaw)
        yaw_compensation = np.tan(yaw_rad) * 0.02  # Scale factor for compensation
        normalized_x = iris_x + yaw_compensation
        
        # Compensate for pitch (up-down head rotation)
        # When head tilts down (positive pitch), iris appears to move up in camera
        pitch_rad = np.radians(pitch)
        pitch_compensation = np.tan(pitch_rad) * 0.02  # Scale factor for compensation
        normalized_y = iris_y - pitch_compensation
        
        # Compensate for roll (head tilt)
        # Roll causes rotation of the apparent gaze position
        roll_rad = np.radians(roll)
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)
        
        # Rotate the offset back to compensate for roll
        rotated_offset_x = offset_x * cos_roll + offset_y * sin_roll
        rotated_offset_y = -offset_x * sin_roll + offset_y * cos_roll
        
        # Apply rotated offset to normalized position
        normalized_x = eye_center_x + rotated_offset_x + yaw_compensation
        normalized_y = eye_center_y + rotated_offset_y - pitch_compensation
        
        return (normalized_x, normalized_y)
    
    def draw_pose_axes(self, frame, rvec, tvec, length=100):
        """
        Draw head pose axes on the frame for visualization.
        
        Args:
            frame: OpenCV image frame
            rvec: Rotation vector from estimate_pose
            tvec: Translation vector from estimate_pose
            length: Length of axes to draw
        """
        if rvec is None or tvec is None:
            return
        
        # Define axes endpoints in 3D
        axes_3d = np.array([
            [0, 0, 0],        # Origin
            [length, 0, 0],   # X axis (red)
            [0, length, 0],   # Y axis (green)
            [0, 0, length],   # Z axis (blue)
        ], dtype=np.float64)
        
        # Project to 2D
        axes_2d, _ = cv2.projectPoints(
            axes_3d, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        
        origin = tuple(axes_2d[0].ravel().astype(int))
        x_end = tuple(axes_2d[1].ravel().astype(int))
        y_end = tuple(axes_2d[2].ravel().astype(int))
        z_end = tuple(axes_2d[3].ravel().astype(int))
        
        # Draw axes
        cv2.line(frame, origin, x_end, (0, 0, 255), 2)  # X - Red
        cv2.line(frame, origin, y_end, (0, 255, 0), 2)  # Y - Green
        cv2.line(frame, origin, z_end, (255, 0, 0), 2)  # Z - Blue
