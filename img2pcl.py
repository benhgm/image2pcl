import os
import math
import glob
import json
import torch
import skimage.transform

import numpy as np
import open3d as o3d
import PIL.Image as pil

from torchvision import transforms
from mmseg.apis import inference_segmentor, init_segmentor

import networks
from options import Image2PCLOptions
from kitti_utils import generate_depth_map

options = Image2PCLOptions()
opts = options.parse()

class Image2PCL:
    def __init__(self, options):
        self.opt = options

        self.device=torch.device("cuda")
        self.model_path = self.opt.model_path

        self.data_path = self.opt.image_path

        # Predicting only one image
        if os.path.isfile(self.data_path):
            self.paths = [self.data_path]
            self.output_directory = os.path.dirname(self.data_path)
        elif os.path.isdir(self.data_path):
            # Perform prediction for a folder of images
            print("Is folder")
            self.paths = glob.glob(os.path.join(self.data_path, '*.{}'.format(self.opt.ext)))
            print(self.paths)
            self.output_directory = self.data_path
        else:
            raise Exception("Cannot find image path: {}".format(self.data_path))

        # Define path to trained encoder and decoder models
        encoder_path = os.path.join(self.model_path, "encoder.pth")
        depth_decoder_path = os.path.join(self.model_path, "depth.pth")

        # Initialize encoder
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Initialize decoder
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict =torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()
    
    def perform_prediction(self, idx, image_path):
        outputs = {}

        with torch.no_grad():
            outputs[('image_path', idx)] = image_path

            # Get sky mask
            sky_mask= self.get_sky_mask(image_path)

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')

            # Save the image as a numpy array to project to a point cloud
            outputs[('color_image', idx)] = np.array(input_image)

            # Get image dimensions
            img_w, img_h = input_image.size

            # Perform transforms
            input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # Perform prediction
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            disp = self.depth_decoder(features)[('disp', 0)]
            disp = torch.nn.functional.interpolate(
                    disp, (img_h, img_w), mode="bilinear", align_corners=False)
            disp = disp.squeeze().cpu().numpy()
            outputs[('disp', idx)] = disp
            outputs[('disp_no_sky', idx)] = np.where(sky_mask, outputs[('disp', idx)], math.nan)

        return outputs
    
    def get_sky_mask(self, image_path):
        model = init_segmentor(config=self.opt.segmentor_config_path,
                               checkpoint=self.opt.segmentor_ckpt_path,
                               device=self.device)
        
        result = np.asarray(inference_segmentor(model, image_path))
        shape = result.shape

        mask = np.ones((shape[1], shape[2]), dtype=bool)
        mask = np.logical_and(mask, result != 10)
        mask = mask.squeeze()

        return mask
    
    def ProjectDispto3D(self, disp, color, data_type):
        """
        Converts a HxWxD image to XYZRGB points
        Conversion is based on https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
        """
        height, width = disp.shape
        v = []
        u = []
        inv_z = []
        R = []
        G = []
        B = []

        for i in range(height):
            for j in range(width):
                if math.isnan(disp[i][j]) or disp[i][j] < 0.0333 or np.isinf(disp[i][j]):
                    continue
                v.append(i)
                u.append(j)
                inv_z.append(disp[i][j])
                R.append(color[i][j][0])
                G.append(color[i][j][1])
                B.append(color[i][j][2])
        
        rgb = np.vstack((R,G,B))
        ones = np.ones((1, len(inv_z)))
        cam_coords = np.vstack((u, v, ones, inv_z))

        if data_type=="nuscenes":
            inv_K = np.array([[1/1260, 0, -800/1260, 0],
						[0, 1/1260, -450/1260, 0],
						[0, 0, 1, 0],
						[0, 0, 0, 1]], dtype=np.float32)

        elif data_type=="kitti_raw":
            inv_K = np.array([[1/721.5377, 0, -621/721.5377, 0],
		        			  [0, 1/721.5377, -187/721.5377, 0],
						      [0, 0, 1, 0],
						      [0, 0, 0, 1]], dtype=np.float32)		

        xyz = np.dot(inv_K, cam_coords)

        return xyz, rgb/255

    def ProjectToWorld(self, xyz, image_path, data_type):
        """
        Project the 3D points from the camera coordinates to world coordinates
        """
        if data_type == "nuscenes":
            camera_position = os.path.splitext(image_path.split("/")[5])[0]
		    # print("camera position: " + camera_position)

            with open(self.opt.nusc_camera_parameters, "r") as read_file:
                camera_parameters = json.load(read_file)
                intrinsics = camera_parameters[camera_position]["intrinsics"]
                extrinsics = camera_parameters[camera_position]["extrinsics"]

            R = o3d.geometry.get_rotation_matrix_from_quaternion(extrinsics["rotation"])
            T = np.array(extrinsics["translation"])

        elif data_type == "kitti_raw":
		    ### use R and T from cam_to_cam.txt file
            R = np.array([[9.999758e-01, -5.267463e-03, -4.552439e-03],
		 				  [5.251945e-03, 9.999804e-01, -3.413835e-03],
		 				  [4.570332e-03, 3.389843e-03, 9.999838e-01]], dtype=np.float32)

            T = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03], dtype=np.float32)	

            P = np.eye(4)
            P[:3, :3] = R
            P[:3, -1] = T

            xyz = np.dot(P, xyz)
            xyz = xyz/xyz[3,:]
            xyz = np.delete(xyz, (3), axis=0)

            rotate_y = np.array([[0, 0, 1],
						         [0, 1, 0],
						         [-1, 0, 0]], dtype=np.float32)

            rotate_z = np.array([[0, -1, 0],
						         [1, 0, 0],
						         [0, 0, 1]])

            xyz = np.dot(rotate_y, xyz)
            xyz = np.dot(rotate_z, xyz)

            return xyz

    def get_kitti_gt(self, image_path):
        dirs = image_path.split("/")
        date = dirs[5]
        frame_index = os.path.splitext(dirs[-1])[0]
        cam = int(dirs[7].split("_")[1])
        calib_path = os.path.join("/data", date)
        velo_filename = os.path.join(
            calib_path,
            dirs[6],
            "velodyne_points/data/{:010d}.bin".format(int(frame_index))
        )
        depth_gt = generate_depth_map(calib_path, velo_filename, cam)
        depth_gt = skimage.transform.resize(
            depth_gt, (375, 1242), order=0, preserve_range=True, mode='constant'
        )

        return 1.0 / depth_gt

    def plot_pointclouds(self):
        pcds = []
        pcd = o3d.geometry.PointCloud()

        for idx, image_path in enumerate(self.paths):
            if image_path.endswith("_disp.jpg"):
                continue
            outputs = self.perform_prediction(idx, image_path)
            xyz, rgb = self.ProjectDispto3D(outputs[('disp_no_sky', idx)], outputs[('color_image', idx)], data_type=self.opt.data_type)
            xyz = self.ProjectToWorld(xyz, image_path, self.opt.data_type)
            
            pcd.points = o3d.utility.Vector3dVector(xyz.T)
            pcd.colors = o3d.utility.Vector3dVector(rgb.T)
            pcds.append(pcd)
        
            if self.opt.data_type == "kitti" and self.opt.compare_gt:
                pcd_gt = []

                disp_gt = self.get_kitti_gt(image_path)
                img = np.array(pil.open(image_path).convert('RGB'))

                xyz_gt, rgb_gt = self.ProjectDispto3D(disp_gt, img, data_type=self.opt.data_type )
                xyz_gt = self.ProjectToWorld(xyz_gt, image_path, data_type=self.opt.data_type)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(xyz_gt.T)
                pcd_gt.colors = o3d.utility.Vector3dVector(rgb_gt.T)
                pcds.append(pcd_gt)
        
        o3d.visualization.draw_geometries(pcds)


if __name__ == "__main__":
    plotter = Image2PCL(opts)
    plotter.plot_pointclouds()





			    