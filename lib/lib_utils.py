#imports 
try:

    from datetime import datetime
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import shutil
    from tqdm import tqdm
    import random
    from PIL import Image
    import cv2
    from time import time
    import multiprocessing as mp
    from torch.utils.data import DataLoader
    from pathlib import Path
    import matplotlib.pyplot as plt
    import math
    from scipy.stats import pearsonr, spearmanr, kendalltau, boxcox
    import yaml
    #from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from scipy.spatial.distance import pdist

except Exception as e:

    print("Some module are missing {}".format(e))


class Utils:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

    @staticmethod
    def read_from_xyz_file(spath: Path):
        """Read xyz files and return lists of x,y,z coordinates and atoms"""

        X = []
        Y = []
        Z = []
        atoms = []

        with open(str(spath), "r") as f:    #apro il file .xyz

            for line in f:      #itero le righe del file, splittandole 
                l = line.split()
                if len(l) == 4:
                    X.append(float(l[1]))
                    Y.append(float(l[2]))
                    Z.append(float(l[3]))
                    atoms.append(str(l[0]))

        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        return X, Y, Z, atoms   #ritorna 4 aray rappresentanti X,Y,Z e atom
    
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
    
    @staticmethod
    def generate_png(
        spath: Path,
        dpath: Path,
        resolution=320,
        z_relative=False,
        single_channel_images=False,
    ):
        """Generate a .npy matrix starting from lists of x,y,z coordinates"""

        X, Y, Z, atoms = Utils.read_from_xyz_file(spath)    #recupero le coordinate dalla funzione read_from_xyz_file

        if z_relative: #se sto lavorando con la coordinata z relativa, i relativi massimi e minimi saranno i max e min dell'array di valori
            z_max = np.max(Z)
            z_min = np.min(Z)

            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            #calcolo la risoluzione dell'immagine partendo dai valori massimi ottenuti prima 
            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )
        else:
            path = spath.parent.joinpath("max_min_coordinates.txt")

            x = np.loadtxt(str(path))   #prendo dal file max_min_coordinates.txt i massimi e minimi valori.

            x_max = x[0][0]
            x_min = x[1][0]

            y_max = x[0][1]
            y_min = x[1][1]

            z_max = x[0][2]
            z_min = x[1][2]

            resolution = round(
                4
                * (
                    5
                    + np.max(
                        [np.abs(x_max), np.abs(x_min), np.abs(y_max), np.abs(y_min)]
                    )
                )
            )
        #ho 3 matrici, una sopra l'altra che rappresentano le posizioni dei 3 atomi di interesse.
        C = np.zeros((resolution, resolution))
        O = np.zeros((resolution, resolution))
        H = np.zeros((resolution, resolution))

        z_norm = lambda x: (x - z_min) / (z_max - z_min)

        C_only = True

        for i in range(len(X)):
            if atoms[i] == "C":
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if C[y_coord, x_coord] < z_norm(Z[i]):
                    C[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "O":
                C_only = False
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if O[y_coord, x_coord] < z_norm(Z[i]):
                    O[y_coord, x_coord] = z_norm(Z[i])
            elif atoms[i] == "H":
                C_only = False
                x_coord = int(round(X[i] * 2) + resolution / 2)
                y_coord = int(round(Y[i] * 2) + resolution / 2)
                if H[y_coord, x_coord] < z_norm(Z[i]):
                    H[y_coord, x_coord] = z_norm(Z[i])

        name = spath.stem

        if single_channel_images:
            C = (C * 255.0).astype(np.uint8)
            O = (O * 255.0).astype(np.uint8)
            H = (H * 255.0).astype(np.uint8)

            image_C = Image.fromarray(C)
            Utils.crop_image(image_C, name + "_C.png", dpath)
            image_O = Image.fromarray(O)
            Utils.crop_image(image_O, name + "_O.png", dpath)
            image_H = Image.fromarray(H)
            Utils.crop_image(image_H, name + "_H.png", dpath)

        else:
            if C_only:
                Matrix = C.copy()
            else:
                Matrix = np.stack((C, O, H), axis=2)
            Matrix = (Matrix * 255.0).astype(np.uint8)
            # Matrix = np.flip(Matrix, 0)

            image = Image.fromarray(Matrix)
            Utils.crop_image(image, name + ".png", dpath)


    @staticmethod
    def dataset_max_and_min(spath: Path, dpath: Path = None) -> list:
        """
        This static method returns a list of the maximum and minimum values for each coordinate given a folder of .xyz files. It takes two parameters,
        spath (Path) and dpath (Path), with dpath being optional. It creates two lists, MAX and MIN, which are initialized to [0, 0, 0].
        It then iterates through the files in the spath directory and checks if each file is an .xyz file. If it is, it calls the find_max_and_min() method
        from Utils to get the max and min values for that file. It then compares these values to MAX and MIN respectively to update them if necessary.
        Finally, if dpath is not None, it saves the MAX and MIN lists as a text file in the dpath directory. Otherwise it just prints out MAX and MIN.
        The method returns both MAX and MIN as a list.
        """
        """Return a list of Max and Min for each coordinate, given a folder of .xyz files"""

        MAX = [0, 0, 0]
        MIN = [0, 0, 0]

        max = []
        min = []

        for file in spath.iterdir():
            if file.suffix == ".xyz":
                max, min = Utils.find_max_and_min(file)
                if max[0] > MAX[0]:
                    MAX[0] = max[0]
                if min[0] < MIN[0]:
                    MIN[0] = min[0]
                if max[1] > MAX[1]:
                    MAX[1] = max[1]
                if min[1] < MIN[1]:
                    MIN[1] = min[1]
                if max[2] > MAX[2]:
                    MAX[2] = max[2]
                if min[2] < MIN[2]:
                    MIN[2] = min[2]

        if dpath is not None:
            np.savetxt(dpath.joinpath("max_min_coordinates.txt"), [MAX, MIN])
        else:
            print([MAX, MIN])

        return MAX, MIN

    @staticmethod
    def find_max_and_min(spath: Path):
        """Return a list of Max and Min for each coordinate, given a single .xyz file"""

        X = []
        Y = []
        Z = []

        with open(str(spath), "r") as f:
            for line in f:
                l = line.split()
                if len(l) == 4:
                    X.append(float(l[1]))
                    Y.append(float(l[2]))
                    Z.append(float(l[3]))

        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        max = [np.max(X), np.max(Y), np.max(Z)]
        min = [np.min(X), np.min(Y), np.min(Z)]

        return max, min