import numpy as np
import mediapy as media
import matplotlib.pyplot as plt

class Recorder():
    def __init__(self, duration, fps, n_cameras, folder="./../img/"):
        self.n_frame   = 1
        self.duration  = duration
        self.fps       = fps
        self.videos    = [[] for _ in range(n_cameras)]
        self.folder    = folder

    def add_frame(self, readings):
        for i in range(len(readings)):
            self.videos[i].append(readings[i][:, :, :3])

    def save_videos(self):
        print(self.videos[0][0].shape)
        for i in range(len(self.videos)):
            plt.imsave(f"photo_camera_{i}.png", np.divide(self.videos[i][0], 255))
            media.write_video(f"video_camera_{i}.mp4", self.videos[i], fps=self.fps, qp=18)