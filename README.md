# wound_sensor

This project is focused on analyzing the results generated from a wound sensor to help patients assess the healing progress of their wounds with minimal assistance from doctors. The research for this project was conducted during a research internship at Mayo Clinic, ASU.

## Project Description
The project involves image registration and analysis techniques to compare reference images of wounds with sensed images captured over time. The goal is to identify and measure the differences between the reference and sensed images, enabling the assessment of wound healing progress.

The file heatmap.py contains the code for generating a heatmap in order to analyze the results of the Wound Sensor. The result generated from this code is shown below.

![result](https://github.com/meghasn/wound_sensor/assets/41339621/b14cb2e5-ffbc-4668-b497-ade036968912)

Dependencies
The code requires the following dependencies:

OpenCV (cv2)
NumPy (numpy)
Matplotlib (matplotlib.pyplot)
Please make sure these dependencies are installed before running the code.

Data Organization
The code assumes that the reference and sensed images are organized in specific directories. The structure should be as follows:

Copy code
stand images/
  ├── 0mM/
  │    ├── Dry/
  │    │    ├── reference_image_1.jpg
  │    │    ├── reference_image_2.jpg
  │    │    └── ...
  │    ├── Android/
  │    │    ├── sensed_image_1.jpg
  │    │    ├── sensed_image_2.jpg
  │    │    └── ...
  │    └── ...
  ├── 10mM/
  │    ├── Dry/
  │    ├── Android/
  │    └── ...
  └── ...
The code assumes that the reference images are stored in the Dry subdirectory for each concentration (0mM, 10mM, etc.), and the sensed images are stored in the corresponding subdirectories (Android, etc.).

The output images will be saved in the specified output path with appropriate directories and names.

Contact Information
For any questions or inquiries, please contact:

Your Name: Megha Sudhakaran Nair
Email: mnair5@asu.edu

