
import unittest
import numpy as np
from landmarks import _get_relative_eye_position, LEFT_EYE_LANDMARKS, LEFT_IRIS_CENTER

class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TestRelativeIrisTracking(unittest.TestCase):
    def test_center_iris_horizontal(self):
        """Test iris exactly in the horizontal center of the eye."""
        landmarks = {}
        self._setup_eye(landmarks, 0.0, 0.0, iris_x=5.0, iris_y=0.0)
        
        rel_pos = _get_relative_eye_position(landmarks, LEFT_EYE_LANDMARKS, LEFT_IRIS_CENTER)
        
        # Expected: X ratio = 0.5 (midpoint of 0 and 10)
        self.assertAlmostEqual(rel_pos[0], 0.5)

    def test_horizontal_head_movement_invariance(self):
        """Test that horizontal head movement doesn't change horizontal tracking ratio."""
        landmarks1 = {}
        self._setup_eye(landmarks1, 0.0, 0.0, iris_x=2.0, iris_y=1.0) # Looking left
        
        landmarks2 = {}
        self._setup_eye(landmarks2, 100.0, 50.0, iris_x=2.0, iris_y=1.0) # Same look, moved head
        
        rel_pos1 = _get_relative_eye_position(landmarks1, LEFT_EYE_LANDMARKS, LEFT_IRIS_CENTER)
        rel_pos2 = _get_relative_eye_position(landmarks2, LEFT_EYE_LANDMARKS, LEFT_IRIS_CENTER)
        
        # Horizontal ratio should remain the same
        self.assertAlmostEqual(rel_pos1[0], rel_pos2[0])
        
    def _setup_eye(self, landmarks, off_x, off_y, iris_x, iris_y):
        # Helper to setup landmarks with an offset
        # Eye corners: 0 to 10 width
        # Eye height: -2 to 2 height
        
        # Map to actual indices used in code
        # left=33 (outer), right=133 (inner)
        landmarks[33] = MockLandmark(off_x + 0.0, off_y + 0.0)
        landmarks[133] = MockLandmark(off_x + 10.0, off_y + 0.0)
        
        # Top: 159, 158
        landmarks[159] = MockLandmark(off_x + 5.0, off_y - 2.0)
        landmarks[158] = MockLandmark(off_x + 5.0, off_y - 2.0)
        
        # Bottom: 145, 153
        landmarks[145] = MockLandmark(off_x + 5.0, off_y + 2.0)
        landmarks[153] = MockLandmark(off_x + 5.0, off_y + 2.0)
        
        # Iris
        landmarks[LEFT_IRIS_CENTER] = MockLandmark(off_x + iris_x, off_y + iris_y)

if __name__ == '__main__':
    unittest.main()
