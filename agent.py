import numpy as np

class Agent:
    def __init__(self, name, shape_feature_vect, pos, vel=None, battery=0, time_step=1):
        self.name = name
        self.shape_feature_vect = shape_feature_vect
        self.prev_pos = None
        self.curr_pos = pos
        self.vel = vel
        self.battery = battery
        self.time_step = time_step

    def update_pos(self, pos):
        self.prev_pos = self.curr_pos
        self.curr_pos = pos
        self.update_vel()

    def update_vel(self):
        self.vel = (self.curr_pos - self.prev_pos) / self.time_step

    def to_string(self):
        class_desc = ""

        class_desc += f"Agent: {self.name}\n"
        class_desc += f"Shape Feature Vector: {self.shape_feature_vect}\n"
        class_desc += f"Previous Position: {self.prev_pos}\n"
        class_desc += f"Current Position: {self.curr_pos}\n"
        class_desc += f"Velocity vector: {self.vel}\n"
        try:
            class_desc += f"Velocity magnitude: {np.linalg.norm(self.vel)}\n"
        except:
            pass
        class_desc += f"Battery: {self.battery}\n"
        class_desc += f"Time Step: {self.time_step}\n"

        return class_desc

    def vector(self):
        vector = np.concatenate((self.shape_feature_vect,
                                 self.curr_pos,
                                 self.vel,
                                 self.battery))
        return vector