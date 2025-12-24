# GPGV-FusionLCD

## GPGV-FusionLCD: Visual-Lidar Fusion Loop Closure Detection for Multi-Sensor SLAM

## Introduction

GPGV-FusionLCD is a cross-modal loop closure detection framework for multi-sensor SLAM that integrates codebook-free global visual prefiltering with efficient LiDAR geometric verification: a codebook-free global visual prefilter quickly narrows the candidate set, and a LiDAR geometric verifier is invoked only for the most promising candidates to produce high-confidence relative pose constraints for pose-graph optimization in Back End.

<div align="center">
    <img src="pics/Overview.png" width = 100% >
</div>

## Requirements
ROS Noetic (Ubuntu 20.04) 
GTSAM>=4.1
OpenCV>=4.0
Eigen>=3.3.4
PCL>=1.8

### ROS
ROS Noetic (Ubuntu 20.04) 
http://wiki.ros.org/ROS/Installation

### python
pip install torch torchvision numpy opencv-python scipy
pip install faiss-cpu  # or faiss-gpu

### GTSAM
wget -O gtsam.zip https://github.com/borglab/gtsam/archive/refs/tags/4.1.1.zip
unzip gtsam.zip
cd gtsam-4.1.1/
mkdir build && cd build
cmake -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON ..
sudo make install -j8
sudo ldconfig

### Fast-LIVO2-SAM
Coming soon

## Dataset
The KITTI dataset is publicly available from the official website https://www.cvlibs.net/datasets/kitti/.
