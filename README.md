#####**Carla Object Detection Using YOLO**
![Screenshot from 2022-04-01 16-36-00](https://user-images.githubusercontent.com/85341949/161832011-831f0b98-e9ca-4658-b9bf-e8a0e2b3c916.png)

This repository aims to provide YOLO object detection within the carla simulation environment. The YOLOv3 detects objects such as car, bike, person,etc. The operating system used for this implementation is Ubuntu 18.04 with the gtx 1070 GPU.

**Requirements**

1.Python 3.7

2.OpenCV 4.5.0 (build cuda and cudnn) to effectively utilize GPU

3.Numpy

**Installation**

1. Install the linux quick start installation package [CARLA simulator](https://carla.readthedocs.io/en/latest/start_quickstart/) (version 0.9.13) 

2. Clone the repository https://github.com/Da-frank2/Carla-Object-Detection-Using-YOLO.git

3. Copy the files from the repository (Yolo_object_detection_carla.py,coco.names and yolov3.cfg) into CARLA_0.9.13/PythonAPI/example folder

4. Download the YOLOv3 weights file [yolov3.weights](https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view)

5. Copy the yolo.weights file into the directory CARLA_0.9.13/PythonAPI/example

6. Open the python script Yolo_object_detection_carla.py using any python editor and modify lines 75,76 and 79 to point to your relevent directory of the coco.names, yolov3.cfg and yolov3.weights files    

4. Run ./CarlaUE4.sh to connect to the CARLA server

5. lastly run the Yolo_object_detection_carla.py python script 
