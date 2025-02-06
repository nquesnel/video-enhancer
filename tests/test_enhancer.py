import unittest
import cv2
import numpy as np
import os
from src.enhancer import VideoEnhancer

class TestVideoEnhancer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.create_test_video()
        cls.enhancer = VideoEnhancer()

    @classmethod
    def create_test_video(cls):
        output_path = 'test_input.mp4'
        height, width = 64, 64
        fps = 24
        duration = 2  # seconds

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Create noisy frames
        for _ in range(fps * duration):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add some structure to the noise
            frame = cv2.GaussianBlur(frame, (7, 7), 0)
            out.write(frame)

        out.release()

    def test_enhancement(self):
        input_path = 'test_input.mp4'
        output_path = 'test_output.mp4'
        
        self.enhancer.enhance_video(input_path, output_path)
        
        # Verify output exists and is larger
        self.assertTrue(os.path.exists(output_path))
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        self.assertGreater(output_size, input_size)

        # Verify resolution increase
        cap_in = cv2.VideoCapture(input_path)
        cap_out = cv2.VideoCapture(output_path)
        
        in_width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_width = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.assertEqual(out_width, in_width * 4)
        self.assertEqual(out_height, in_height * 4)
        
        cap_in.release()
        cap_out.release()

    def tearDown(self):
        # Clean up test files
        for file in ['test_input.mp4', 'test_output.mp4']:
            if os.path.exists(file):
                os.remove(file)

if __name__ == '__main__':
    unittest.main()