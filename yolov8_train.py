from ultralytics import YOLO
from lib.lib_utils import Utils 
from pathlib import Path
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torch
import numpy as np
import math

#trasformazione .xyz to .png 
# Utils.from_xyz_to_png(
#     Path('/home/gabro/GrapheDetectProject/data.xyz/subset_xyz'), 
#     Path('/home/gabro/GrapheDetectProject/dataset_imm_300'), 
#     100)

#divisione del dataset in train/ test/
#inserire percorso dataset da dividere e percentuale del test 
# Utils.split_dataset('/home/gabro/GrapheDetectProject/data_yolo_new/', 0.2)   

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/gabro/GrapheDetectProject/best_100_campioni_new.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# model.train(data="config_yolov8.yaml", epochs=100)  # train the model

# metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/gabro/GrapheDetectProject/b74c9ce2-graphene_218641_bonds.png")  # predict on an image

boxes = results[0].boxes


#devo trasformare xy (alto sx), xy (alto dx) in [start_row:end_row, start_col:end_col]
print(boxes[0].xyxy)    #posizione angolo in alto a sx e in basso a dx in pixel
array = torch.Tensor.numpy(boxes[0].xyxy)
# print(array.size)
x1 = math.ceil(array[0,0])
y1 = math.ceil(array[0,1])
x2 = math.floor(array[0,2])
y2 = math.floor(array[0,3])
# print(x1)
# print(y1)
# print(x2)
# print(y2)

# res_plotted = results[0].plot(labels = False, line_width = 1)
# plt.imshow(res_plotted)
# plt.show()
# cv2.imshow("result", res_plotted)

#crop difetto
img = cv2.imread('/home/gabro/GrapheDetectProject/b74c9ce2-graphene_218641_bonds.png')
print(img.shape) # Print image shape
cv2.imshow("original", img)
 
# Cropping an image
cropped_image = img[y1:y2, x1:x2] #img[start_row:end_row, start_col:end_col]

# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# Save the cropped image
cv2.imwrite("Cropped Image.jpg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()


# success = model.export(format="onnx")  # export the model to ONNX format

