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
    import torch as Torch 
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
        print(NUM_IMM)

    #prende una cartella di immagini e una cartella di uscita dove deposita tutti i difetti croppati 
    def crop_from_folder(cartellaImm: str, cartellaCrop: str, model):
        #prende una cartella di immagini su cui fare detection dei difetti,
        #deposita su una cartella i difetti croppati 

        #versione con for al posto di .predict
        #cartellaImm = "/home/gabro/GrapheDetectProject/immProvaInferenza/"

        #cartellaCrop = "/home/gabro/GrapheDetectProject/cartellaCrop/"  #cartella dove andranno messi i difetti croppati 

        for nomeImm in os.listdir(cartellaImm): #ietro i nomi delle immagini all'interno della cartella

            pathImm = os.path.join(cartellaImm, nomeImm)    #estraggo il path dell'immagine 
            results = model(pathImm)  # predict on an image
            boxes = results[0].boxes
            index = 0
            # print(boxes.data.size)
            # for box in boxes.data:
            #     print(box.xyxy)
            for box in boxes:
                #devo trasformare xy (alto sx), xy (alto dx) in [start_row:end_row, start_col:end_col]
                # print(boxes[index].xyxy)    #posizione angolo in alto a sx e in basso a dx in pixel
                array = Torch.Tensor.numpy(boxes[index].xyxy)
                # print(array.size)
                x1 = math.floor(array[0,0])
                y1 = math.floor(array[0,1])
                x2 = math.ceil(array[0,2])
                y2 = math.ceil(array[0,3])

                # res_plotted = results[0].plot(labels = False, line_width = 1)
                # plt.imshow(res_plotted)
                # plt.show()
                # cv2.imshow("result", res_plotted)

                #crop difetto
                img = cv2.imread(pathImm)
                # print(img.shape) # Print image shape
                # cv2.imshow("original", img)
                
                # Cropping an image
                cropped_image = img[y1:y2, x1:x2] #img[start_row:end_row, start_col:end_col]

                # Display cropped image
                # cv2.imshow("cropped", cropped_image)

                nomeCropped = os.path.basename(pathImm).removesuffix('.png') + "_cropped_box_" + str(index)+ ".png" #creo nome dell'immagine croppata
                pathCropped = os.path.join(cartellaCrop, nomeCropped)   #compongo il path dell'immagine croppata aggiungendo il giusto percorso di output
                # Save the cropped image
                cv2.imwrite(pathCropped, cropped_image)
                
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                index = index+1
    
    # #dalla cartella cropped 
    # def from_crops_to_threshOld(pathImmagini: str):

    #     pathImmagini = "/home/gabro/GrapheDetectProject/cartellaCrop/"
    #     for nomeImm in os.listdir(pathImmagini):

    #         pathImm = os.path.join(pathImmagini, nomeImm)    #estraggo il path dell'immagine 
    #         defect1 = cv2.imread(pathImm)

    #         # cv2.namedWindow('defect1',cv2.WINDOW_NORMAL)
    #         # cv2.resizeWindow('defect1', 600,600)
    #         # cv2.imshow("defect1", defect1)

    #         # cv2.namedWindow('defect2',cv2.WINDOW_NORMAL)
    #         # cv2.resizeWindow('defect2', 600,600)
    #         # cv2.imshow("defect2", defect2)

    #         #1) trasformo l'immagine in scala di grigi
    #         defect1_grey = cv2.cvtColor(defect1, cv2.COLOR_BGR2GRAY)
    #         # cv2.namedWindow('defect1_grey',cv2.WINDOW_NORMAL)
    #         # cv2.resizeWindow('defect1_grey', 600,600)
    #         # cv2.imshow("defect1_grey", defect1_grey)


    #         #2) apply binary thresholding, porto tutti i pixel significativi a 255 di bianco 
    #         ret, thresh = cv2.threshold(defect1_grey, 0, 255, cv2.THRESH_BINARY)
    #         # visualize the binary image
    #         # cv2.namedWindow('Binary image',cv2.WINDOW_NORMAL)
    #         # cv2.resizeWindow('Binary image', 600,600)
    #         # cv2.imshow('Binary image', thresh)
    #         # cv2.waitKey(0)
    #         # cv2.imwrite('image_thres1.jpg', thresh)
    #         # cv2.destroyAllWindows()


    #         #3) detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    #         contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                                
    #         # draw contours on the original image
    #         image_copy = defect1.copy()
    #         cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
                            
    #         # see the results
    #         # cv2.namedWindow('None approximation',cv2.WINDOW_NORMAL)
    #         # cv2.resizeWindow('None approximation', 600,600)
    #         # cv2.imshow('None approximation', image_copy)
    #         # cv2.waitKey(0)
    #         # cv2.imwrite('contours_none_image1.jpg', image_copy)
    #         # cv2.destroyAllWindows()

    #         # Applica la soglia
    #         ret,thresh1 = cv2.threshold(image_copy,245,255,cv2.THRESH_BINARY_INV)
    #         # ret,thresh2 = cv2.threshold(image_copy,127,255,cv2.THRESH_BINARY_INV)

    #         # Salva l'immagine
    #         #cv2.imwrite('nome_file_immagine_thresh.jpg', thresh)
    #         nameThresh = pathImm.removesuffix('.png') + "_thresh_" + ".png"

    #         cv2.imwrite(nameThresh, thresh1)



    #         #devo trasformare tutti i pixel non completaemnte verdi in nero e tutti gli altri in bianco


    #         #to-do calcolare area delle due cavità 
            
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows() 


    #dalla cartella cropped alla cartella trash 
    def from_crops_to_thresh(spath: str, dpath: str):

        # spath = "/home/gabro/GrapheDetectProject/cartellaCrop/"
        for nomeImm in os.listdir(spath):

            pathImm = os.path.join(spath, nomeImm)    #estraggo il path dell'immagine 
            defect1 = cv2.imread(pathImm)

            # cv2.namedWindow('defect1',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('defect1', 600,600)
            # cv2.imshow("defect1", defect1)

            # cv2.namedWindow('defect2',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('defect2', 600,600)
            # cv2.imshow("defect2", defect2)

            #1) trasformo l'immagine in scala di grigi
            defect1_grey = cv2.cvtColor(defect1, cv2.COLOR_BGR2GRAY)
            # cv2.namedWindow('defect1_grey',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('defect1_grey', 600,600)
            # cv2.imshow("defect1_grey", defect1_grey)


            #2) apply binary thresholding, porto tutti i pixel significativi a 255 di bianco 
            ret, thresh = cv2.threshold(defect1_grey, 0, 255, cv2.THRESH_BINARY)
            # visualize the binary image
            # cv2.namedWindow('Binary image',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Binary image', 600,600)
            # cv2.imshow('Binary image', thresh)
            # cv2.waitKey(0)
            # cv2.imwrite('image_thres1.jpg', thresh)
            # cv2.destroyAllWindows()


            #3) detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
            contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                                
            # draw contours on the original image
            image_copy = defect1.copy()
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
                            
            # see the results
            # cv2.namedWindow('None approximation',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('None approximation', 600,600)
            # cv2.imshow('None approximation', image_copy)
            # cv2.waitKey(0)
            # cv2.imwrite('contours_none_image1.jpg', image_copy)
            # cv2.destroyAllWindows()

            # Applica la soglia
            ret,thresh1 = cv2.threshold(image_copy,245,255,cv2.THRESH_BINARY_INV)
            # ret,thresh2 = cv2.threshold(image_copy,127,255,cv2.THRESH_BINARY_INV)

            # Salva l'immagine
            #cv2.imwrite('nome_file_immagine_thresh.jpg', thresh)


            # nameThresh = pathImm.removesuffix('.png') + "_thresh_" + ".png"

            newName = os.path.basename(pathImm) #recuepero il base name dell'immagine di partenza 
            newName = newName.removesuffix('.png') + "_thresh_" + ".png"    #creo il nuovo nome a partire da quello 
            finalPath = os.path.join(dpath, newName)    #compong il path finale

            cv2.imwrite(finalPath, thresh1) #salvo la nuova immagine del dpath 



            #devo trasformare tutti i pixel non completaemnte verdi in nero e tutti gli altri in bianco


            #to-do calcolare area delle due cavità 
            
            cv2.waitKey(0)
            cv2.destroyAllWindows() 


