from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class Image2PCLOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Image2PCL options")

        # INPUT DATA options
        self.parser.add_argument("--image_path",
                                 type=str,
                                 help="path to the image data. Can be folder or file",
                                 default=os.path.join(file_dir, "data/kitti_test_images/"))
        self.parser.add_argument("--model_path",
                                 type=str,
                                 help="path to the trained model",
                                 default=os.path.join(file_dir, "models/kitti_mono_640x192/"))
        self.parser.add_argument("--nusc_camera_parameters",
                                  type=str,
                                  help="path to nuscenes camera parameters json file",
                                  default=os.path.join(__file__, "data/nusc_cam_params.json"))

        # DEPTH PREDICTION options
        self.parser.add_argument("--data_type",
                                 type=str,
                                 help="dataset to test on",
                                 default="kitti_raw",
                                 choices=["kitti_raw", "nuscenes"])
        self.parser.add_argument("--ext",
                                 type=str,
                                 help="set the image extension",
                                 default="jpg",
                                 choices=["jpg", "png"])
        self.parser.add_argument("--compare_gt",
                                 help="if set, compares predicted point cloud with lidar gt for kitti data",
                                 action="store_true")
        

        # SEGMENTATION options
        self.parser.add_argument("--segmentor_config_path",
                                 help="path to the segmentation model config file",
                                 default=os.path.join(file_dir, "mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"))
        self.parser.add_argument("--segmentor_ckpt_path",
                                 help="path to the trained segementation model checkpoint",
                                 default=os.path.join(file_dir, "mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"))

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
