from ultralytics import YOLO
from lib.lib_utils import Utils 
from pathlib import Path

#trasformazione .xyz to .png 
Utils.from_xyz_to_png(
    Path('/home/gabro/GrapheDetectProject/data.xyz/subset_xyz'), 
    Path('/home/gabro/GrapheDetectProject/immagini_prova_from_xyz_to_png'), 
    10)

#divisione del dataset in train/ test/
#inserire percorso dataset da dividere e percentuale del test 
Utils.split_dataset('/home/gabro/GrapheDetectProject/data_copy_test_split_dataset/', 0.5)   

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# model.train(data="config_yolov8.yaml", epochs=100)  # train the model

# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format