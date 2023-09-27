import mujoco
import numpy as np

class Sensors:

    def __init__(self) -> None:
        pass

    def get_depth_image_matrices(self,
                                  m:mujoco.MjModel,
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
                render = r.render()
                result[cam] = render
                
        return result
    
    def get_segment_image_matrices(self,
                                  m:mujoco.MjModel,
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
                render = r.render()[:, :, 0]
                result[cam] = render
                
        return result
    
    def get_rgb_image_matrices(self,
                               m:mujoco.MjModel,
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
    
    def get_rgbd_image_matrices(self, m:mujoco.MjModel, 
                                     d:mujoco.MjData,
                                     r_rgb:mujoco.Renderer,
                                     r_depth:mujoco.Renderer,
                                     cameras=None|list[str]|list[int]) -> dict[any, np.ndarray]:
        """
        Returns dictionary of RGBD image matrices for each camera in the model.

        Parameters
        ----------
        m : The mujoco model.
        d : The mujoco data.
        r_rgb : The mujoco renderer for RGB images.
        r_depth : The mujoco renderer for depth images.
        cameras : The list of cameras to render. If None, all cameras are rendered. The default is None.
        """

        result = {}

        if type(cameras) == list:
            cams = cameras
        else:
            cams = range(m.cam_user.shape[0])

        for cam in cams:
                r_rgb.update_scene(d, cam)
                r_depth.update_scene(d, cam)
                # Can add out = np.zeros((m.vis.map[cam].height, m.vis.map[cam].width, 3), 
                # dtype=np.uint8) to render() to save memory
                try:
                    # Add a dimension to the depth image to make it 3D
                    depth = np.expand_dims(r_depth.render(), axis=2)
                    # Limit the depth values to be between 0.3 and 4.0
                    depth[depth < 0.3] = np.nan
                    depth[depth > 4.0] = np.nan
                    
                    # Join the RGB and depth images
                    rgb = r_rgb.render()
                    result[cam] = np.append(rgb, depth, axis=2)
                except:
                    raise Exception("Error joining depth and rgb. Reading have different sizes.")
                
        return result

    def get_rgbd_seg_matrices(self, m:mujoco.MjModel, 
                                     d:mujoco.MjData,
                                     r_rgb:mujoco.Renderer,
                                     r_depth:mujoco.Renderer,
                                     r_seg:mujoco.Renderer,
                                     cameras=None|list[str]|list[int]) -> dict[any, np.ndarray]:
        """
        Returns dictionary of RGBD image matrices for each camera in the model.

        Parameters
        ----------
        m : The mujoco model.
        d : The mujoco data.
        r_rgb : The mujoco renderer for RGB images.
        r_depth : The mujoco renderer for depth images.
        r_seg : The mujoco renderer for segmentation images.
        cameras : The list of cameras to render. If None, all cameras are rendered. The default is None.
        """

        result = {}

        if type(cameras) == list:
            cams = cameras
        else:
            cams = range(m.cam_user.shape[0])

        for cam in cams:
                r_rgb.update_scene(d, cam)
                r_depth.update_scene(d, cam)
                r_seg.update_scene(d, cam)
                # Can add out = np.zeros((m.vis.map[cam].height, m.vis.map[cam].width, 3), 
                # dtype=np.uint8) to render() to save memory
                try:
                    # Add a dimension to the depth image to make it 3D
                    depth = np.expand_dims(r_depth.render(), axis=2)
                    # Limit the depth values to be between 0.3 and 4.0
                    depth[depth < 0.01] = np.nan
                    depth[depth > 4.0] = np.nan

                    # Add a dimension to the segmentation image to make it 3D
                    seg = np.expand_dims(r_seg.render()[:, :, 0], axis=2)

                    # Join the RGB and depth images
                    rgb = r_rgb.render()
                    result[cam] = np.append(rgb, depth, axis=2)
                    result[cam] = np.append(result[cam], seg, axis=2)
                except:
                    raise Exception("Error joining depth and rgb. Reading have different sizes.")
                
        return result