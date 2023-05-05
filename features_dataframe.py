from ultralytics import YOLO
from lib.lib_utils import Utils 
from lib_fetaure_analysis import Features
from defect_analysis import Test
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
import torch as Torch
import math
import os
import pandas as pd

pathCartellaTresh = Path('/home/gabro/GrapheDetectProject/cartellaTrash')
pathCartellaContours = Path('/home/gabro/GrapheDetectProject/contours')
shapes = Features.from_thresh_to_contours_print_features(pathCartellaTresh,pathCartellaContours)
for shape in shapes:
    print("Shape features:")
    for key in shape:
        print(key, ' : ', shape[key])
    print()

df = pd.DataFrame.from_records(shapes)
df.to_csv('/home/gabro/GrapheDetectProject/dataframe.csv')
print(df)


