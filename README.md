# EE367: Estimating the Camera Response Function of an iPhone 15 Pro to Create New HDR Images

Authors: Audrey Lee and Lycia Tran
<!-- CONTACT -->
## Team Members
To contact any of the team members, please check out our Github.
* [Audrey Lee](https://github.com/Audrey-Lee88)
* [Lycia Tran](https://github.com/lyciat)

<!-- TABLE OF CONTENTS -->
<details>
  <summary> Table of Contents</summary>
  <ol>
    <li><a href="#team-members">Team Members</a></li>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#overview-of-project-files">Overview of Project Files</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
As technology continues to improve, the want for images to better replicate the human vision grows. High Dynamic Range (HDR) images are images that try to replicate what the human eyes should see by balancing lighting within an image. In recent years, the average person is able to use their mobile phone to capture images, and phones themselves, have been able to create their own HDR images. However, these HDR images do not seem to be significantly better or more balanced. For this project, we propose to estimate the camera response function of the most accessible digital camera – your smartphone – and create our own HDR image pipeline to produce balanced images that can be more interchangeable and modifiable to suit different people’s tastes easier to make a more appealing final image. 

## Prerequisites
Our code relies on numpy, matplotlib, pandas, and OpenCv. To install all the necessary dependencies, please use 

  ```sh
  pip install numpy
  ```
  for all packages.

<!-- USAGE -->
## Usage
To run the our HDR pipeline, please run

  ```sh
  python main.py
  ```

This will get the camera response function, apply it to a different set of images, retrieve the RAW images back from the new set of images, obtain a fused HDR image, and apply tone mapping to the fused HDR image to get a final tone mapped HDR image that looks appealing to the eye.

## Overview of Project Files
Below is an overview of what the different files in the repository do:

* ```camera_resp.py``` contains the camera response class. It will take in the path of the set of images with their corresponding exposure times and obtain the camera response function in the ```get_camera_resp()``` method. This will obtain the inverse of the camera response function for all three color channels (RGB)
* ```get_hdr_img.py``` contains the functions to retrieve the fused HDR image. The ```get_ldr()``` function takes in the camera response function and the set of images that we want to get an HDR image of. This function returns the RAW image values for these set of images which are then sent to the ```get_hdr()``` function where we are able to obtain our HDR image.
* ```tonemapping.py``` contains different functions for tonemapping. We compare our methodology for tonemapping (in the ```my_tonemap()``` function) with OpenCv's Drago and Mantiuk tonemapping.

<!-- Poster and Report -->
## Poster and Report
* [Report](EE367_Computational_Imaging_Report.pdf)
* [Poster](EE367_Computational_Imaging_Poster.pdf)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments
Special thanks to the EE367 teaching team and especially to our mentor Professor Gordon Wetzstein. Also special thanks to Apple for letting us find out your camera response function.

## References
Here are some of our references that helped us with developing the code:
* https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4587655
* https://pages.cs.wisc.edu/~lizhang/courses/cs766-2007f/projects/hdr/mitsunaga_CVPR_1999.pdf
