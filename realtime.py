#!/usr/bin/env python 
import torch 
import torch.nn as nn 
import numpy  as np 
import struct 
import open3d as o3d 
import time
import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil 
import threading

from modules.user import User 

import rospy 

from std_msgs.msg import String 
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2 


from network.rangeret import RangeRet 

from dataloader.laserscan import LaserScan



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model_path = "/media/s186/One_Touch/RangeARNet_weights/rangeret-best_kitti.pt"   

ARCH_PATH = "/media/s186/One_Touch/RangeRet/config/RangeRet-semantickitti.yaml"
DATA_PATH ="/media/s186/One_Touch/RangeRet/config/labels/semantic-kitti.yaml" 

FP16 = False 

with open(ARCH_PATH, "r") as f:
    ARCH = yaml.safe_load(f)

with open(DATA_PATH, "r") as f:
    DATA = yaml.safe_load(f)


# ============================================================
# 2. Load RangeRet
# ============================================================

H = ARCH["dataset"]["sensor"]["img_prop"]["height"]
W = ARCH["dataset"]["sensor"]["img_prop"]["width"]

sensor = ARCH["dataset"]["sensor"] 

N_CLASSES = len(DATA["learning_map_inv"])



# Load checkpoint safely onto the selected device.
loaded_checkpoint = torch.load(
    model_path,
    map_location=device
)

# Support the checkpoint formats used by this project.
if isinstance(loaded_checkpoint, dict):
    if "model_state_dict" in loaded_checkpoint:
        model_parameters = loaded_checkpoint["model_state_dict"]
    elif "state_dict" in loaded_checkpoint:
        model_parameters = loaded_checkpoint["state_dict"]
    elif "model_state" in loaded_checkpoint:
        model_parameters = loaded_checkpoint["model_state"]
    else:
        # The checkpoint may itself be a state_dict.
        model_parameters = loaded_checkpoint
else:
    raise TypeError(
        f"Unexpected checkpoint type: {type(loaded_checkpoint)}"
    )


model = RangeRet(ARCH['model_params'], (H, W), N_CLASSES) 
model.load_state_dict(model_parameters, strict=True)
model.to(device)
model.eval() 



means = torch.tensor(
    sensor["img_means"],
    dtype=torch.float32,
    device=device
).view(-1, 1, 1)


stds = torch.tensor(
    sensor["img_stds"],
    dtype=torch.float32,
    device=device
).view(-1, 1, 1)

laser = LaserScan(
    project=True,
    H=H,
    W=W,
    fov_up=sensor["fov_up"],
    fov_down=sensor["fov_down"]
)


# ============================================================
# 4. Store the latest ROS point cloud
# ============================================================

lock = threading.Lock()

latest_cloud = {
    "xyz": None,
    "intensity": None,
    "sequence": 0
}





class NonBlockVisualizer:
    def __init__(self, point_size=5, background_color=[1, 1, 1]):
        self.__visualizer = o3d.visualization.Visualizer()
        self.__visualizer.create_window()
        opt = self.__visualizer.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt = self.__visualizer.get_render_option()
        opt.point_size = point_size

        self.__pcd_vis = o3d.geometry.PointCloud()
        self.__initialized = False
        
        self.view_control = self.__visualizer.get_view_control()
        
       
        
        
        
        
        

    def update_renderer(self, pcd, wait_time=0):
        self.__pcd_vis.points = pcd.points
        self.__pcd_vis.colors = pcd.colors

        if not self.__initialized:
            self.__initialized = True
            self.__visualizer.add_geometry(self.__pcd_vis)
        else:
            self.__visualizer.update_geometry(self.__pcd_vis)
            
            
        
        #self.view_control.set_up(np.array([0, -1, 1]))
        #self.view_control.set_front(np.array([0, -0.5, -1]))
        #self.view_control.set_lookat([0,0,5])
        #self.view_control.set_zoom(0.1)
        
        self.__visualizer.poll_events()
        self.__visualizer.update_renderer()
        
        

 


color_dict = {'0' : [0, 0, 0], 
  '1' : [0, 0, 1],
  '10': [0.8, 0.1, 0.5],
  '11': [0.01, 0.4, 0.8],
  '13': [1, 0.3, 0.4],
  '15': [0.6, 0, 0.01],
  '16': [1, 0, 0],
  '18': [0.7, 0.1, 0.4],
  '20': [1, 0, 0],
  '30': [0.8, 0.5, 0],
  '31': [0.8, 0.02, 1],
  '32': [0.45, 0.01, 0.6],
  '40': [1, 0, 1],
  '44': [0.01, 0.001, 0.015],
  '48': [0.35, 0, 0.35],
  '49': [0.35, 0, 0.75],
  '50': [0, 0.8, 1],
  '51': [0.2, 0.51, 1],
  '52': [0, 0.65, 1],
  '60': [0.7, 1, 0.5],
  '70': [0, 1, 0],
  '71': [0.5, 0.23, 0.45],
  '72': [0.3, 0.9, 0.6],
  '80': [0.6, 0.9, 1],
  '81': [0, 0.5, 1],
  '99': [0.01, 1, 0.2],
  '252': [0.91, 0.62, 0.001],
  '256': [1, 0, 0],
  '253': [0.8, 0.3, 0.15],
  '254': [0.16, 0.16, 1],
  '255': [0.38, 0.16, 0.6],
  '257': [0.9, 0.35, 0.45],
  '258': [0.7, 0.2, 0.2],
  '259': [1, 0, 0]}


       



def callback(msg):
       
    """ROS callback: extract XYZ and intensity."""

    fields = {field.name for field in msg.fields}

    if not {"x", "y", "z"}.issubset(fields):
        rospy.logwarn_throttle(
            5.0,
            "PointCloud2 does not contain x, y, z fields"
        )
        return

    # Velodyne PointCloud2 usually contains intensity.
    intensity_field = (
        "intensity" if "intensity" in fields else None
    )

    field_names = ("x", "y", "z")
    if intensity_field is not None:
        field_names = ("x", "y", "z", intensity_field)

    raw = np.asarray(
        list(
            pc2.read_points(
                msg,
                field_names=field_names,
                skip_nans=True
            )
        ),
        dtype=np.float32
    )

    if raw.size == 0:
        return

    raw = raw.reshape(-1, len(field_names))

    xyz = raw[:, :3]

    if intensity_field is not None:
        intensity = raw[:, 3]
    else:
        # Fallback only; this is not equivalent to real remission.
        intensity = np.zeros(len(xyz), dtype=np.float32)

    # Remove non-finite values and zero-range points.
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(intensity)
        & (np.linalg.norm(xyz, axis=1) > 1e-6)
    )

    xyz = np.ascontiguousarray(xyz[valid], dtype=np.float32)
    intensity = np.ascontiguousarray(
        intensity[valid], dtype=np.float32
    )

    if len(xyz) == 0:
        return

    with lock:
        latest_cloud["xyz"] = xyz
        latest_cloud["intensity"] = intensity
        latest_cloud["sequence"] += 1



# ============================================================
# 5. Convert a cloud into the RangeRet input tensor
# ============================================================

def make_range_input(xyz, intensity):
    """
    Input:
        xyz:       [N, 3]
        intensity: [N]

    Returns:
        proj_in: [1, 5, H, W]
        px, py:   original point -> range-image indices
    """

    # The same projection class used by the RangeRet parser.
    laser.set_points(xyz, intensity)

    # Construct [5, H, W]:
    # channel 0: range
    # channels 1-3: x, y, z
    # channel 4: intensity/remission
    proj = np.concatenate(
        [
            laser.proj_range[None, :, :],
            laser.proj_xyz.transpose(2, 0, 1),
            laser.proj_remission[None, :, :]
        ],
        axis=0
    ).astype(np.float32)

    mask = laser.proj_mask.astype(np.float32)

    proj = torch.from_numpy(proj).to(device)

    # Match the normalization in the training parser.
    proj = (proj - means) / stds
    proj = proj * torch.from_numpy(mask).to(device)[None, :, :]

    # Add batch dimension: [1, 5, H, W]
    proj_in = proj.unsqueeze(0)

    # These indices preserve the original order of the points.
    px = torch.from_numpy(
        laser.proj_x.copy()
    ).long().to(device)

    py = torch.from_numpy(
        laser.proj_y.copy()
    ).long().to(device) 
    

    return proj_in, px, py


# ============================================================
# 6. Point-wise semantic inference
# ============================================================

def predict_points(xyz, intensity):
    """
    Returns:
        point_labels: [N], model learning-class IDs
        elapsed:      inference time in seconds
    """

    proj_in, px, py = make_range_input(xyz, intensity)

    start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(FP16 and device.type == "cuda")
        ):  
               
            start_time = time.time()   
            output = model(proj_in) 
            print("model time = ", time.time() - start_time)

        # Match the output handling in your infer_subset().
        # Model output: [B, H, W, C]
        predictions = output.permute(0, 3, 1, 2)

        # [H, W]: predicted class at each range-image pixel
        proj_argmax = predictions[0].argmax(dim=0)

        # Back-project to the original point order: [N]
        point_labels = proj_argmax[py, px]

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return point_labels.cpu().numpy().astype(np.int32), elapsed




# ============================================================
# 7. Map predictions to visualization colors
# ============================================================

# Keep your existing color_dict here.
# It maps ORIGINAL SemanticKITTI label IDs (as strings)
# to RGB colors in the range [0, 1].
#
# Example:
# color_dict = {
#     "0": [0, 0, 0],
#     "10": [0.8, 0.1, 0.5],
#     ...
# }


def labels_to_colors(predictions):
    """
    predictions contains learning-class IDs (0 ... C-1).
    Convert them to original dataset IDs before color lookup.
    """

    original_ids = np.asarray(
        [
            DATA["learning_map_inv"].get(int(label), 0)
            for label in predictions
        ],
        dtype=np.int32
    )

    colors = np.asarray(
        [
            color_dict.get(str(int(label)), [0.5, 0.5, 0.5])
            for label in original_ids
        ],
        dtype=np.float32
    )

    return np.clip(colors, 0.0, 1.0)


     
def listener():
 
       # In ROS, nodes are uniquely named. If two nodes with the same
       # name are launched, the previous one is kicked off. The
       # anonymous=True flag means that rospy will choose a unique
       # name for our 'listener' node so that multiple listeners can
       # run simultaneously.
       rospy.init_node('lidar subscriber', anonymous=True)
   
       rospy.Subscriber("/velodyne_points", PointCloud2, callback, queue_size=1)
       
       visualizer = NonBlockVisualizer(point_size=2.0)
       
       rospy.loginfo("Subscribed to /velodyne_points")
       rospy.loginfo("Waiting for LiDAR data...")

       last_sequence = 0
       frame_count = 0
       total_time = 0.0 
       
       try:
          while not rospy.is_shutdown():

            # Read the latest available frame.
            with lock:
                sequence = latest_cloud["sequence"]

                if sequence == last_sequence:
                    xyz = None
                    intensity = None
                else:
                    xyz = latest_cloud["xyz"]
                    intensity = latest_cloud["intensity"]
                    last_sequence = sequence

            if xyz is not None:
               
                labels, elapsed = predict_points(xyz, intensity) 
           
                colors = labels_to_colors(labels)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                                    xyz.astype(np.float64)
                                        )
                pcd.colors = o3d.utility.Vector3dVector(
                                            colors.astype(np.float64)
                                             )

                visualizer.update_renderer(pcd)

                frame_count += 1
                total_time += elapsed

                rospy.loginfo(
                    "Points: %d | Inference: %.2f ms | FPS: %.2f",
                    len(xyz),
                    elapsed * 1000.0,
                    1.0 / max(elapsed, 1e-9)
                )

            # Keep the main thread responsive for the GUI.
            time.sleep(0.001)

       finally:
          visualizer.close()

          if frame_count > 0:
            print(
                "Average inference time: {:.2f} ms".format(
                    1000.0 * total_time / frame_count
                )
            )
   
if __name__ == '__main__': 
      
       listener()  
       
       
       