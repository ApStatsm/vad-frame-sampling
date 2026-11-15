"""Keep the highest adjacent-frame brightness difference in each window."""

import cv2
import numpy as np


class LumaWindowSelector:
    def __init__(self, window_frames, stop, *, yuv_pixel_count=None):
        self.window_frames = window_frames
        self.stop = stop
        self.yuv_pixel_count = yuv_pixel_count
        self.previous_luma = None
        self.best = None

    def observe(self, index, frame, decoder_pts):
        if index >= self.stop:
            return None
        luma = (np.frombuffer(frame, dtype=np.uint8, count=self.yuv_pixel_count)
                if self.yuv_pixel_count is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # At constant resolution the L1 sum ranks frames exactly like its mean.
        difference = (0.0 if self.previous_luma is None else
                      cv2.norm(luma, self.previous_luma, cv2.NORM_L1))
        self.previous_luma = luma
        if self.best is None or difference > self.best[2]:
            self.best = (index, frame, difference, decoder_pts)
        end = index + 1
        if end == self.stop or end % self.window_frames == 0:
            best_index, best_frame, total, best_pts = self.best
            self.best = None
            return best_index, best_frame, total / luma.size, best_pts
        return None
