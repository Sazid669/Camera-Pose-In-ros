# Hands on Perceptron

## Group Members
1. Mir Mohibullah Sazid
2. Syma Afsha

### How to Run
To  execute the algorithm, at first install the following libraries:
```bash
pip install opencv-contrib-python
```
```bash
pip install scipy
```
```bash
pip install networkx
```

To run the each file, run the following command:
1. board_generation.py: This file will generate CHARUCOBOARD which can used for camera calibration. In this project, the calibration mainly done using single aruco. 
2. Generate.py: Here aruco is generated with the DICT 5x5_100
3. calibration_ChAruco.py: This file will calibrate the camera with the ChAruco board. From the video it will take 80 pictures to calibrate the camera. It can be cahnged according to our need. It will save distortion_cofficients and camera matrix as a output to save in npy file. To run the calibration:
```bash
python calibration_ChAruco.py -v callibration.mp4 -c 80
```
4. callibrate.py: This file is used to callibrate the camera with single Aruco. In the result analysis of the report, this callibration process is used. From the video it will take 80 pictures to calibrate the camera. It can be cahnged according to our need. It will save distortion_cofficients and camera matrix as a output to save in npy file. To run the file:
```bash
python callibrate.py -v arucoCallibrate.mp4 -c 80
```
5. detect_aruco_video.py: This file is used to detect the Aruco makers from videos or from the laptop camera. To run it from laptop camera:
```bash
python detect_aruco_video.py --type DICT_5X5_100 --camera True
```
To run it with the video:
```bash
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video pose.MOV
```
6. cameraPose.py: This will calculate the camera pose from the aruco marker and generate a tree of Aruco's showing the marker ID and distance between the graph. To run the file:
```bash
python cameraPose.py --K_Matrix K_matrix.npy --D_Coeff distortion.npy --type DICT_5X5_100
```
7. bundleadjustment.py: It will calculate the reproject error and do bundle adjsutment with a scipy library using least sqaure method. It will also generate a graph of reprojections error before and after bundle adjsutment. The error graph is plotted taking the 1st 50 errors. To run the file:
```bash
python bundleadjustment.py --K_Matrix K_matrix.npy --D_Coeff distortion.npy --type DICT_5X5_100
```
8. ROS Launch: Complie the ros file hands_on_precption in catkin work sapce and run the launch file:
```bash
roslaunch hands_on_precption hop.launch
```
stonefish simulator need to be installed in pc. The file have a dead reckoning node which allows us to know the position of robot. CameraPose node which calculate the camera pose in real time. Here for running in the robot and visulalizing the marker in rviz the aruco marker is transformed in the world frame. markers.py is used for plotting in rviz. To run the robot and controll it run the following commands.  
```bash
rosrun hands_on_precption controller.py
```



