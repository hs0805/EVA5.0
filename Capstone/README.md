# Session 15 - Construction PPE Kit detection and Depth Estimation

This project aims to create a network that can perform 3 tasks simultaneously:

    Predict the boots, PPE, hardhat, and mask if there is an image
    Predict the depth map of the image
    Predict the Planar Surfaces in the region

Right now model is not able to perform all the three tasks as planerCNN is not integrated in it. I'm working on it. Currently it can detect hardhat, vest, mask and boots in the images and predict the depth map for the image.

As it is very difficult to train model from scratch and handling huge dataset can be very diffcult task so I have used pretrained models.

    1.Midas for Depth images
    2.YoloV3 for object detection

In this project, I have used Encoder-Decoder architecture which takes in a image with and outputs the depth image and image with PPE kit detection(hardhat, vest, mask, boots). 

### Model Architecture

  As network is performing two tasks simultaneously so for this I have used encoder decoder architecture. It has an encoder and two decoders for the depth estimation and PPE kit detection.<br>
  For Encoder I have used the exact copy of MiDaS. The backbone for this whole architecture is imported from MiDaS as it is based on the resnet.<br>
  For depth detection decoder I have used the copy of MiDaS decoder.<br>
  For PPE Kit detection I have used the Yolo Layers over the MiDas backbone. I have removed the darknet53 part of it and used only 3 layers of yolo on the top of MiDaS.<br>
  The code for the Integrated Midas and YoloV3 can be found [here](EncoderDecoder.py)<br>
  The summary of the model can be found [here](ModelSummary.md)

### Dataset
  The dataset is taken from [here](https://drive.google.com/drive/folders/1vk43e_U43_vN-QrxDq82XUjt1nOOyhZ6?usp=sharing). This dataset contains the bounding box for the classes.

  - This Dataset includes 4 classes :<br>
    - hardhat<br>
    - vest<br>
    - mask<br>
    - boots<br><br>

  This dataset contains two folders which are <b>images</b> and <b>labels</b>. <b>images</b> folder contains input images and <b>labels</b> folder contains bounding boxes for the images in <b>images</b> folder

  For more info about the dataset you can refer this [link](https://github.com/Radion-AI/EVA5-Phase1/blob/master/14%20-%20Segmentation%20and%20Depth%20Estimation/README.md)


  ### Training
  For training the yolo branch is trained on the top of MiDaS backbone. Training was planned for 200 epochs for yolo branch. But it could be trained only for 170 epochs due to colab usage limitations. Each epoch took around 4-5 minutes for training.<br>
  After training the yolo branch I was able to reduce the loss by a greater extent but loss got stuck on 33 after 130 epochs. The backbone was freezed till 130 epochs. After 130 epochs i unfreezed the 2 layers of backbone and able to reduce the loss. The model was finetuned over very small learning rate, 1e-05. <br>
  For the training i have used [Yolo_training_on_midas](Yolo_training_on_midas.ipynb)
  The code for the Yolo layers over MiDaS backbone can be found [here](EncoderDecoder.py).<br>
  During training phase i faced many problems as colab has usage limitations. I also faced problems in the dataset also as there were few images whose bounding boxes were not correct and eventully code was crashing. and there were few files where labels were not there for those images. So i corrected those in order to train the model. The issue was on the drive also due to storage issue.
  The summary of the model can be found [here](ModelSummary.md)

  ### Setup 

  1) Download the model weights [model-f6b98070.pt](https://drive.google.com/file/d/14-oWX2l_jCx-xuBl9FqkIJvLCqpEvK8N/view?usp=sharingt) 
 and place the file in the root folder.

  2) Download the trained yolo weights [trained_yolo_model](https://drive.google.com/file/d/11coL58QJDkxYm0yCYBc2TfpoBz8DCrv6/view?usp=sharing) on PPE dataset and place the file in the root folder.
    
### Usage

1) Place one or more input images in the folder `input`.

2) Run the [Midas_Yolo_Pipe](https://colab.research.google.com/drive/1S5-ydgxePgjjUv3CJmzME_JcTu4dRnZL?usp=sharing) colab file:

3) The resulting inverse depth maps and the PPE Kit detected images are written to the `output` folder.

    



 