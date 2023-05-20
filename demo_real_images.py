from lib_fetaure_analysis import Features
from defect_analysis import Test
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import torch as Torch
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ultralytics import YOLO
from lib.lib_utils import Utils 
import seaborn as sns
import seaborn.objects as so
from PIL import Image

def trasforma_in_rosso_nero(image_path, output_path):
    # Apri l'immagine utilizzando PIL
    image = Image.open(image_path)
    
    # Converte l'immagine in scala di grigi
    image_gray = image.convert("L")
    
    # Converte l'immagine in formato RGBA
    image_rgba = image.convert("RGBA")
    
    # Ottieni i dati dei pixel dell'immagine
    pixel_data_gray = image_gray.load()
    pixel_data_rgba = image_rgba.load()
    
    # Dimensioni dell'immagine
    width, height = image.size
    
    # Itera su tutti i pixel dell'immagine
    for x in range(width):
        for y in range(height):
            # Ottieni il valore del pixel in scala di grigi
            gray_value = pixel_data_gray[x, y]
            
            # Imposta il pixel su rosso o nero in base al valore di grigio
            if gray_value > 40:
                # Se il pixel è chiaro, imposta il valore rosso
                pixel_data_rgba[x, y] = (255, 0, 0, 255)
            else:
                # Altrimenti, imposta il valore nero
                pixel_data_rgba[x, y] = (0, 0, 0, 255)
    
    # Salva l'immagine modificata
    image_rgba.save(output_path, "PNG")

trasforma_in_rosso_nero("/home/gabro/GrapheDetectProject/real_images_demo/real_images/graphene_real_1.png",
                         "/home/gabro/GrapheDetectProject/real_images_demo/real_images/graphene_real_1_red.png")
trasforma_in_rosso_nero("/home/gabro/GrapheDetectProject/real_images_demo/real_images/graphene_real_0.png",
                         "/home/gabro/GrapheDetectProject/real_images_demo/real_images/graphene_real_0_red.png")

model = YOLO("/home/gabro/GrapheDetectProject/best_100_campioni_new.pt")  # load a pretrained model (recommended for training)


# work in progress - ancora 0 detect 
# results = model("/home/gabro/GrapheDetectProject/real_images_demo/real_images")  # predict on an image

Utils.crop_from_folder("/home/gabro/GrapheDetectProject/real_images_demo/real_scaled", "/home/gabro/GrapheDetectProject/real_images_demo/scaled_crop", model)

Utils.from_crops_to_thresh("/home/gabro/GrapheDetectProject/real_images_demo/real_scaled", "/home/gabro/GrapheDetectProject/real_images_demo/tresh")


# pathCartellaTresh = Path('/home/gabro/GrapheDetectProject/real_images_demo/tresh')
# pathCartellaContours = Path('/home/gabro/GrapheDetectProject/real_images_demo/contours')
# shapes = Features.from_thresh_to_contours_print_features(pathCartellaTresh,pathCartellaContours)
# #stampa le feature estratte per ogni difetto analizzato 
# for shape in shapes:
#     print("Shape features:")
#     for key in shape:
#         print(key, ' : ', shape[key])
#     print()






