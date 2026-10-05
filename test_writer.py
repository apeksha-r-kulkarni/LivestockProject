import cv2
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_out.mp4', fourcc, 0, (640, 480))
for i in range(30):
    out.write(np.zeros((480, 640, 3), dtype=np.uint8))
out.release()
