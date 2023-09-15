import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import cv2

class Recorder():
    def __init__(self, duration, fps, n_cameras, folder="./../img/"):
        self.n_frame   = 1
        self.duration  = duration
        self.fps       = fps
        self.videos    = [[] for _ in range(n_cameras)]
        self.depth    = [[] for _ in range(n_cameras)]
        self.segement    = [[] for _ in range(n_cameras)]
        self.folder    = folder



    def add_frame(self, readings):
        for i in range(len(readings)):
            self.videos[i].append(readings[i][:, :, :3])
            self.depth[i].append(readings[i][:, :, 3])
            self.segement[i].append(readings[i][:, :, 4])



    def save_videos(self):
        print(self.videos[0][0].shape)
        for i in range(len(self.videos)):
            # plt.imsave(f"photo_camera_{i}.png", np.divide(self.videos[i][0], 255))
            # self.write_video_cv2(self.videos[i], f"video_camera_{i}.mp4")
            media.write_video(f"video_camera_rgb_{i}.mp4", 
                              self.videos[i], 
                              fps=self.fps)
            media.write_video(f"video_camera_depth_{i}.mp4", 
                              self.depth[i], 
                              fps=self.fps)
            media.write_video(f"video_camera_seg_{i}.mp4", 
                              self.segement[i], 
                              fps=self.fps)


    
    def show_videos(self):
        print(self.videos[0][0].shape)
        for i in range(len(self.videos)):
            # plt.imshow(np.divide(self.videos[i][0], 255))
            # self.write_video_cv2(self.videos[i], f"video_camera_{i}.mp4")
            media.show_video( self.videos[i], 
                              fps=self.fps)
            media.show_video( self.depth[i], 
                              fps=self.fps)
            media.show_video( self.segement[i], 
                              fps=self.fps)
            


    def write_video_cv2(self, frames, name):
        size = 720*16//9, 720
        duration = 2
        fps = 25
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (size[1], size[0]), False)
        for _ in range(fps * duration):
            data = np.random.randint(0, 256, size, dtype='uint8')
            out.write(data)
        out.release()