# VISION

This repository contains practical works (PW) I completed, along with slides from a presentation I co-delivered with another student, as part of my master's course in computer vision (https://perso.ensta-paris.fr/~manzaner/Cours/IMA/VISION/).
## Table of Contents
- [PW n°1: GC Disparity](#pw-n1-gc-disparity)
- [PW n°2: Face Alignment](#pw-n2-face-alignment)
- [PW n°3: Fundamental Matrix](#pw-n3-fundamental-matrix)
- [PW n°4: Object Tracking](#pw-n4-object-tracking)
- [PW n°5: Optical Flow](#pw-n5-optical-flow)
- [PW n°6: Panorama Stitching](#pw-n6-panorama-stitching)
- [PW n°7: Seeds](#pw-n7-seeds)
- [Oral Presentation](#oral-presentation)

## PW n°1: GC Disparity
This section contains C++ code and instructions for computing disparity using Graph Cuts.
- **Code**: [GCDisparity.cpp](PW_GCDisparity/GCDisparity.cpp), [exampleGC.cpp](PW_GCDisparity/exampleGC.cpp)
- **Instructions**: [PDF](PW_GCDisparity/instructions_PW_GCDisparity.pdf)

## PW n°2: Face Alignment
This section includes Python code (Jupyter Notebook) and instructions for face alignment.
- **Code**: [PW_face_alignment.ipynb](PW_face_alignment/PW_face_alignment.ipynb)
- **Instructions**: [PDF](PW_face_alignment/instructions_PW_face_alignment.pdf)

## PW n°3: Fundamental Matrix
This section contains C++ code and instructions for computing the fundamental matrix.
- **Code**: [Fundamental.cpp](PW_fundamental/Fundamental.cpp)
- **Instructions**: [PDF](PW_fundamental/instructions_PW_fundamental.pdf)

## PW n°4: Object Tracking
This section includes Python code and instructions for object tracking using MeanShift.
- **Code**: [code](PW_object_tracking/code)
- **Instructions**: [PDF](PW_object_tracking/instructions_PW_tracking.pdf)

## PW n°5: Optical Flow
This section contains Python code (Jupyter notebook) and instructions for computing the optical flow of images, and a report presenting the results.
- **Instructions**: [PDF](https://github.com/pictoune/VISION/blob/main/PW_optical_flow/instructions_PW_optical_flow.pdf)
- **Code**: [code](https://github.com/pictoune/VISION/tree/main/PW_optical_flow/code)
- **Report**: [PDF](https://github.com/pictoune/VISION/blob/main/PW_optical_flow/report_PW_optical_flow.pdf)

## PW n°6: Panorama Stitching
This section contains C++ code and instructions for stitching images to create a panorama.
- **Code**: [Panorama.cpp](PW_panorama/Panorama.cpp)
- **Instructions**: [PDF](PW_panorama/instructions_PW_panorama.pdf)

## PW n°7: Seeds
This section contains C++ code and instructions related to the Seeds algorithm.
- **Code**: [Seeds.cpp](PW_seeds/Seeds.cpp)
- **Instructions**: [PDF](PW_seeds/instructions_PW_seeds.pdf)

## Oral Presentation
This presentation, entitled 'Particle Tracking with Multiple Event Cameras', I did with another student (Nils Aurdal), focused on using event-based cameras and the Kalman filter for 3D tracking of particles in a wind tunnel. It highlighted the advantages of event-based cameras, such as their high dynamic range and temporal resolution. The presentation also discussed various filtering methods for data processing and different techniques for reconstructing particle tracks in 3D space.
- [Presentation Slides](presentation_slides.pdf)

## Usage 
### Python projects (PW n°2, 4 & 5)
#### Step 1: Clone the Repository
Clone the VISION repository to your local machine using the following commands:
```bash
git clone https://github.com/pictoune/VISION.git
cd VISION
```
#### Step 2: Create the required Conda Environment
Set up the required environment using Conda:
  ```bash
  conda env create -f environment.yml -n VISION_env
  ```
#### Step 3: Running the code
- If the practical work you are interested in is written in *.py* files, you must first activate the conda environment: 
  ```bash
  conda activate TADI_env
  ```
  then you can run it:
  ```bash
  python <script_name>.py
  ```
- Otherwise if it is written in a *jupyter notebook*, you need to initiate the notebook first:
  ```bash
    jupyter-notebook
  ```
  Once Jupyter Notebook is open, navigate to the notebook you want to run. Then, change the kernel to the VISION environment:
  Go to `Kernel` -> `Change kernel` -> `Python [conda env:VISION_env]`.
  Then you can run the cells.
### C++ projects (PW n°1, 3, 6 & 7)
To test the code of the practical work done in C++, you just need to make sure that the Imagine++ library is installed (see https://imagine.enpc.fr/~monasse/Imagine++/).

## License
This project is open source and available under the [MIT License](LICENSE).

Feel free to explore the projects and reach out if you have any questions or suggestions.
