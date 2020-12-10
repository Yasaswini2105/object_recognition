# object_recognition
Computer vision is an interdisciplinary field that has been gaining huge amounts of traction in recent years (since CNN) and self-driving cars have taken center stage. Another integral part of computer vision is object detection. Object detection aids in pose estimation, vehicle detection, surveillance etc. The difference between object detection algorithms and classification algorithms is that in detection algorithms, we try to draw a bounding box around the object of interest to locate it within the image. Also, you might not necessarily draw just one bounding box in an object detection case, there could be many bounding boxes representing different objects of interest within the image and you would not know how many beforehand. 

For pascal_ds.py, Follow the instructions below:

Install the following packages 

pip install tensorflow==1.15.3
pip install keras==2.2.4

Clone the git hub repository using the following command: 
git clone https://github.com/matterport/Mask_RCNN.git

or you can go to the link 
https://github.com/matterport/Mask_RCNN.git and download the repository in your local machine

If downloaded locally, cd into the Mask_RCNN folder and run the following
python setup.py install

for MacOS, run it with sudo

Once, installed make sure you have the following packages:
os, xml.etree, numpy, mrcnn

While running if you got module not found, install the approriate packages in the system.

Download the pascal_Voc dataset using the link given below,

https://pjreddie.com/projects/pascal-voc-dataset-mirror/

2007 version of pascal VOC is what we are using in this dataset. 
Change the interior dirctory name to voc from VOC 2007.

Move the location of the downloaded dataset to the object_recognition directory.


If everything is good, pascal_ds.py file will run perfectly.


You can also run the object detection algorithm using pytorch. 

Install the necessary dependancies, make sure the python_model.py is in the same directory as the dataset.

Run the code using 
python python_model.py

If the system is cuda compatible, code will run without any errors. If your system doesn't have a gpu, remove the functions with cuda commands on it and execute the program again. 



