import mujoco
import numpy as np

class Sensors:

    def __init__(self) -> None:
        pass
    
    def get_rgb_image_matrices(self, m:mujoco.MjModel, 
                                     d:mujoco.MjData,
                                     r:mujoco.Renderer,
                                     cameras=None|list[str]|list[int]) -> dict[any, np.ndarray]:
        """
        Returns dictionary of RGB image matrices for each camera in the model.

        Parameters
        ----------
        m : The mujoco model.
        d : The mujoco data.
        r : The mujoco renderer.
        cameras : The list of cameras to render. If None, all cameras are rendered. The default is None.
        """

        result = {}

        if type(cameras) == list:
            cams = cameras
        else:
            cams = range(m.cam_user.shape[0])

        for cam in cams:
                r.update_scene(d, cam)
                # Can add out = np.zeros((m.vis.map[cam].height, m.vis.map[cam].width, 3), 
                # dtype=np.uint8) to render() to save memory
                result[cam] = r.render()
                
        return result