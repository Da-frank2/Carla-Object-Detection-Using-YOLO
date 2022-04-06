**Carla Object Detection Using YOLO**
![Screenshot from 2022-04-01 16-36-00](https://user-images.githubusercontent.com/85341949/161832011-831f0b98-e9ca-4658-b9bf-e8a0e2b3c916.png)

This repository aims to provide YOLO object detection within the carla simulation environment. The YOLOv3 detects objects such as car, bike, person,etc. The operating system used for this implementation is Ubuntu 18.04 with the gtx 1070 GPU. The car is controlled in the pygame window using keyboard keys (left, right,up,down) or (W,A,S,D), the detections are run in a separate window. The detections window is a rear-view camera facing the back of the car.

**Requirements**

1.Python 3.7

2.OpenCV 4.5.0 (build cuda and cudnn) to effectively utilize GPU

3.Pygame numpy

**Installation**

1. Install the linux quick start installation package [CARLA simulator](https://carla.readthedocs.io/en/latest/start_quickstart/) (version 0.9.13) 

2. Clone the repository  `git clone https://github.com/Da-frank2/Carla-Object-Detection-Using-YOLO.git
`
3. Copy the files from the repository (`Yolo_object_detection_carla.py,coco.names and yolov3.cfg`) into CARLA_0.9.13/PythonAPI/example folder

4. Download the `YOLOv3 weights` file [yolov3.weights](https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view)

5. Copy the `yolov3.weights` file into the directory CARLA_0.9.13/PythonAPI/example

6. Open the python script `Yolo_object_detection_carla.py` using any python editor and modify lines` 75,76 and 79` to point to your relevent directory of the `coco.names, yolov3.cfg and yolov3.weights` files    

7. Run `./CarlaUE4.sh` to connect to the CARLA server

8. Issue the command `python3 Yolo_object_detection_carla.py` within the CARLA_0.9.13/PythonAPI/example folder to start pygame window as on the image above 

9. The python script to generate traffic within the carla simulator is generate_traffic.py located in same folder as number 8, issue command `python3 generate_traffic.py -n 50 -w 50 --safe` to spawn 50 vehicles and 50 pedastrian.

![Screenshot from 2022-04-01 16-38-05](https://user-images.githubusercontent.com/85341949/161938758-1c5fb53b-63d4-4d82-9cd3-efb9cfa53746.png)
![Screenshot from 2022-04-01 16-38-09](https://user-images.githubusercontent.com/85341949/161939728-439a9874-826a-4c93-83be-d1600fd8b5d6.png)

**Note:** if you do not have a GPU available then use `pip3 install opencv-python` for CPU version of opencv

Credit to the repository https://github.com/shayantaherian/Object_Detection_Carla, follow instructions on the mentioned repository to implement the same system on **Windows operating system**
