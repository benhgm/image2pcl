# Image2PCL
Enter the metaverse with 2D image to 3D projections!
<br>
This is an implementation of an algorithm to project 2D images into the 3D space. See below for a visual summary of the project
<br><br>
<p align="center">
  <img src="misc/chart.png" width="600"/>
</p>
<br><br>
The published code is inspired by the following works:
<br>
Monodepth2: https://www.github.com/nianticlabs/monodepth2
<br>
MMSegmentation: https://www.github.com/open-mmlab/mmsegmentation

## Setup
Assuming you have already set up an [Anaconda](https://www.anaconda.com/download/) environment with PyTorch, CUDA and Python, install additional dependencies with:
```shell
pip install open3d
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
Clone the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) repository to your working directory
```shell
git clone https://github.com/open-mmlab/mmsegmentation
```

## Test
To get a pointcloud for an image, run:
```shell
python img2pcl.py
```
