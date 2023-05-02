#imports 
try:
    from datetime import datetime
    #import seaborn as sns
    import numpy as np
    import pandas as pd
    import shutil
    #from tqdm import tqdm
    import random
    from PIL import Image
    #import cv2
    from time import time
    import multiprocessing as mp
    #from torch.utils.data import DataLoader
    from pathlib import Path
    import matplotlib.pyplot as plt
    import math
    #from scipy.stats import pearsonr, spearmanr, kendalltau, boxcox
    #import yaml
    #from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    #from scipy.spatial.distance import pdist

    #import per generate_bonds_png
    from chemfiles import Trajectory    
    from PIL import Image, ImageDraw, ImageFilter
    import os
    import cv2
    import skimage.exposure
    
    #import per split_dataset
    import glob
except Exception as e:
    print("Some module are missing {}".format(e))

class Utils:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

    def funzione_prova():
        print("ok!")
    
    #trasforma file .xyz in immagine .png 
    def generate_bonds_png(spath, dpath, resolution=320):

        #prima di fare lo split per estrare il nome trasformo l'oggetto Path in stringa, in questo modo non .split non da errori
        spath = str(spath)
        name = (spath.split("/")[-1])[:-4]  #rimuovo l'estensione del 

        print("generando .png del file %s.xyz..." % name)

        with Trajectory(spath) as trajectory:
            mol = trajectory.read()

        B = Image.new("RGB", (resolution, resolution))
        B_ = ImageDraw.Draw(B)

        mol.guess_bonds()
        if mol.topology.bonds_count() == 0:
            print(f"No bonds guessed for {name}\n")
        bonds = mol.topology.bonds

        for i in range(len(bonds)):
            x_1 = round(mol.positions[bonds[i][0]][0] * 2) + resolution / 2
            y_1 = round(mol.positions[bonds[i][0]][1] * 2) + resolution / 2
            x_2 = round(mol.positions[bonds[i][1]][0] * 2) + resolution / 2
            y_2 = round(mol.positions[bonds[i][1]][1] * 2) + resolution / 2
            line = [(x_1, y_1), (x_2, y_2)]
            first_atom = mol.atoms[bonds[i][0]].name
            second_atom = mol.atoms[bonds[i][1]].name
            color = Utils.find_bound_type(first_atom, second_atom)
            #color = "red"
            B_.line(line, fill=color, width=0)

        #chiamo la funzione crop_image per scontornare l'immagine
        B = Utils.crop_image(B)
        B.save(os.path.join(dpath, name + "_bonds.png"))

        # create blurred image copy 
        #img = cv2.imread(os.path.join(dpath, name + "_bonds.png"))
        #blur = cv2.GaussianBlur(img, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
        #blur = cv2.GaussianBlur(img,(5,5),0)
        #blur = cv2.blur(img,(2,2),0)
        #result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
        # save output
        #cv2.imwrite(os.path.join(dpath, name + "_bonds_blurred.png"), result)

    #funzione che da' il colore all'immagine nel mio caso monocromatica essendoci solo atomi di carbonio
    def find_bound_type(first_atom, second_atom):
        if (first_atom == "C" and second_atom == "C") or (
            second_atom == "C" and first_atom == "C"
        ):
            return "red"
        elif (first_atom == "C" and second_atom == "O") or (
            second_atom == "O" and first_atom == "C"
        ):
            return "blue"
        elif (first_atom == "O" and second_atom == "H") or (
            second_atom == "H" and first_atom == "O"
        ):
            return "white"
        elif (first_atom == "C" and second_atom == "H") or (
            second_atom == "H" and first_atom == "C"
        ):
            return "yellow"


    @staticmethod
    def crop_image(image: Image, name: str = None, dpath: Path = None) -> Image:

        image_data = np.asarray(image)
        if len(image_data.shape) == 2:
            image_data_bw = image_data
        else:
            image_data_bw = image_data.max(axis=2)
        non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
        cropBox = (
            min(non_empty_rows),
            max(non_empty_rows),
            min(non_empty_columns),
            max(non_empty_columns),
        )

        if len(image_data.shape) == 2:
            image_data_new = image_data[
                cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1
            ]
        else:
            image_data_new = image_data[
                cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :
            ]

        new_image = Image.fromarray(image_data_new)
        if dpath is not None:
            new_image.save(dpath.joinpath(name))

        return new_image
    

    #prende la posizione del dataset dove ci sono tutte le immagini e i labels e la percentuale di divisione,
    #crea due cartelle train/ e test/ all'interno e le divide secondo la percentuale data 
    @staticmethod
    def split_dataset(pathData: str, testPerc: int = 0.2):

        print(pathData)
        filelist = glob.glob(pathData+'*.txt') #prendo la lista di tutti i path delle labels nel dataset
        test = random.sample(filelist, int(len(filelist)*testPerc))    #prendo il 20% dei path dei labels
        output_path_test = pathData+"test/"
        output_path_train = pathData+"train/"
        if not os.path.exists(output_path_test): #creo la cartella test se non esiste ancora
            os.makedirs(output_path_test)
        if not os.path.exists(output_path_train): #creo la cartella train se non esiste ancora
            os.makedirs(output_path_train)

        for file in test:  #scorro tutte le labels di test
            txtpath = file
            impath = file[:-4] + '.png' #costruisco i rispettivi path delle immagini 
            out_text = os.path.join(output_path_test, os.path.basename(txtpath)) #questi ultime due liste di path sono i path finali nelle due cartelle nuove 
            out_image = os.path.join(output_path_test, os.path.basename(impath))
            print(txtpath,impath,out_text,out_image)
            os.system('mv ' + txtpath + ' ' + out_text)
            os.system('mv ' + impath + ' ' + out_image)

        #sposto tutto il rimanente in una cartella train 
        for filename in os.listdir(pathData):
            f = os.path.join(pathData, filename)
            # checking if it is a file
            if os.path.isfile(f):
                os.system('mv ' + f + ' ' + output_path_train)


            #dichiarazione path utili
            # cartellaImmaginiPath = Path('/home/gabro/GrapheDetectProject/data/images')
            cartellaImmaginiPath = Path('/home/gabro/GrapheDetectProject/data300')
            rootPath = Path('/home/gabro/GrapheDetectProject')
            cartellaXYZPath = Path('/home/gabro/GrapheDetectProject/data.xyz/subset_xyz')
            cartellaDataSet = Path('/home/gabro/GrapheDetectProject/data_s')

            # #generazione immagine singola
            # nomeFileProva = "graphene_67.xyz"   #inserire file .xyz di esempio 
            # path_xyz_prova = Path.joinpath(cartellaXYZPath, nomeFileProva)
            # Utils.generate_bonds_png(path_xyz_prova, rootPath)

    #prende il path 
    @staticmethod
    def from_xyz_to_png(pathXYZ: Path, pathPNG: Path, NUM_IMM: int):
        
        #generazione di tutte le immagini
        i=0
        for nome_file_xyz in os.listdir(pathXYZ):
            if(i>=NUM_IMM):
                break
            #recupero il path di ogni immagine 
            path_file_xyz = Path.joinpath(pathXYZ, nome_file_xyz)
            #print(path_file_xyz)
            #print(nome_file_xyz)
            #chiamo la funzione che tarsforma da .xyz a .png specificando la risoluzione
            Utils.generate_bonds_png(path_file_xyz, pathPNG, 320)
            i = i+1


