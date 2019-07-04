<!--
    DO NOT EDIT THIS FILE BY HAND.

    This file is automatically generated by /home/shingo/ros/kinetic/src/jsk-ros-pkg/jsk_recognition/generate_readme.py at 2019-07-04T19:36:06.025122.
-->

jsk\_recognition
===============

[![GitHub version](https://badge.fury.io/gh/jsk-ros-pkg%2Fjsk_recognition.svg)](https://badge.fury.io/gh/jsk-ros-pkg%2Fjsk_recognition)
[![Build Status](https://travis-ci.org/jsk-ros-pkg/jsk_recognition.svg)](https://travis-ci.org/jsk-ros-pkg/jsk_recognition)
[![Read the Docs](https://readthedocs.org/projects/jsk-docs/badge/?version=latest)](http://jsk-docs.readthedocs.org/en/latest/jsk_recognition/doc/index.html)

jsk_recognition is a stack for the perception packages which are used in JSK lab.


ROS packages
------------

| Package                 | Description                                           | Documentation                                                                                                                                       | Code                                                                                                                                             |
|:------------------------|:------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| jsk_recognition_msgs    | ROS messages for jsk_pcl_ros and jsk_perception.      | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://wiki.ros.org/jsk_recognition_msgs)                                             | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/jsk_recognition_msgs)    |
| jsk_perception          | ROS nodes and nodelets for 2-D image perception.      | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_perception)          | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/jsk_perception)          |
| jsk_pcl_ros             | ROS nodelets for pointcloud perception.               | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_pcl_ros)             | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/jsk_pcl_ros)             |
| jsk_pcl_ros_utils       | ROS utility nodelets for pointcloud perception.       | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_pcl_ros_utils)       | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/jsk_pcl_ros_utils)       |
| resized_image_transport | ROS nodes to publish resized images.                  | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/resized_image_transport) | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/resized_image_transport) |
| jsk_recognition_utils   | C++ library about sensor model, geometrical modeli... | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/jsk_recognition_utils)       | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/jsk_recognition_utils)   |
| checkerboard_detector   | Uses opencv to find checkboards and compute their ... | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/checkerboard_detector)   | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/checkerboard_detector)   |
| imagesift               | For every image, computes its sift features and se... | [![](https://img.shields.io/badge/docs-here-brightgreen.svg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/imagesift)               | [![](https://img.shields.io/badge/code-here-brightgreen.svg)](http://github.com/jsk-ros-pkg/jsk_recognition/tree/master/imagesift)               |


Gallery
-------

### [jsk_perception](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_perception)

[![](.readme/gallery_jsk_perception.jpg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_perception)

### [jsk_pcl_ros](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_pcl_ros)

[![](.readme/gallery_jsk_pcl_ros.jpg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_pcl_ros)

### [jsk_pcl_ros_utils](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_pcl_ros_utils)

[![](.readme/gallery_jsk_pcl_ros_utils.jpg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/jsk_pcl_ros_utils)

### [imagesift](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/imagesift)

[![](.readme/gallery_imagesift.jpg)](http://jsk-docs.readthedocs.io/en/latest/jsk_recognition/doc/imagesift)



Deb build status
----------------

[//]: # (!!DO NOT EDIT !!)

[//]: # (THIS SECTION IS AUTOMATICALLY GENERATED BY)

[//]: # (rosrun jsk_tools generate_deb_status_table.py jsk_recognition)


| Package                 | Kinetic (Xenial)                                                                                                                                                                                           | Melodic (Bionic)                                                                                                                                                                                           |
|:------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| jsk_recognition (arm64) | [![Build Status](http://build.ros.org/job/Kbin_uxv8_uXv8__jsk_recognition__ubuntu_xenial_arm64__binary/badge/icon)](http://build.ros.org/job/Kbin_uxv8_uXv8__jsk_recognition__ubuntu_xenial_arm64__binary) | [![Build Status](http://build.ros.org/job/Mbin_ubv8_uBv8__jsk_recognition__ubuntu_bionic_arm64__binary/badge/icon)](http://build.ros.org/job/Mbin_ubv8_uBv8__jsk_recognition__ubuntu_bionic_arm64__binary) |
| jsk_recognition (armhf) | [![Build Status](http://build.ros.org/job/Kbin_uxhf_uXhf__jsk_recognition__ubuntu_xenial_armhf__binary/badge/icon)](http://build.ros.org/job/Kbin_uxhf_uXhf__jsk_recognition__ubuntu_xenial_armhf__binary) | [![Build Status](http://build.ros.org/job/Mbin_ubhf_uBhf__jsk_recognition__ubuntu_bionic_armhf__binary/badge/icon)](http://build.ros.org/job/Mbin_ubhf_uBhf__jsk_recognition__ubuntu_bionic_armhf__binary) |
| jsk_recognition (i386)  | [![Build Status](http://build.ros.org/job/Kbin_uX32__jsk_recognition__ubuntu_xenial_i386__binary/badge/icon)](http://build.ros.org/job/Kbin_uX32__jsk_recognition__ubuntu_xenial_i386__binary)             | ---                                                                                                                                                                                                        |
| jsk_recognition (amd64) | [![Build Status](http://build.ros.org/job/Kbin_uX64__jsk_recognition__ubuntu_xenial_amd64__binary/badge/icon)](http://build.ros.org/job/Kbin_uX64__jsk_recognition__ubuntu_xenial_amd64__binary)           | [![Build Status](http://build.ros.org/job/Mbin_uB64__jsk_recognition__ubuntu_bionic_amd64__binary/badge/icon)](http://build.ros.org/job/Mbin_uB64__jsk_recognition__ubuntu_bionic_amd64__binary)           |

[//]: #


Deprecated packages
-------------------
* [cr\_calibration](https://github.com/jsk-ros-pkg/jsk_recognition/tree/master/cr_calibration)
* [cr\_capture](https://github.com/jsk-ros-pkg/jsk_recognition/tree/master/cr_capture)
* [orbit\_pantilt](https://github.com/jsk-ros-pkg/jsk_recognition/tree/master/orbit_pantilt)
* [posedetectiondb](https://github.com/jsk-ros-pkg/jsk_recognition/tree/master/posedetectiondb)
