try:
    import cv2
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm
    from skimage.feature import hog
    from scipy.fftpack import fft2, fftshift
    from mahotas.features import haralick
    from defect_analysis import Test  
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
    import importlib
    import os
except Exception as e:
    print(f"Some module are missing for {__file__}: {e}\n")


class Features:
    #prende la cartella dove ci sono tresh 
    def from_thresh_to_contours_print_features(spath: Path, dpath: Path):

        shapes = []
        images = []
        nomiImages = os.listdir(spath)
        for nomeImm in nomiImages:
            pathImm=os.path.join(spath, nomeImm)
            images.append(pathImm)
        for imm in images:  #itero sui apth delle immagini
            # print(images)
            # print("Imm:" + str(imm))
            imm = Path(imm)
            nameImm = os.path.basename(imm)
            newName = str(nameImm).removesuffix('.png') + "countour_" + ".png"    #name of countour image produced
            dpathName = os.path.join(dpath, newName)
            shape = Test.extract_shape_features_edited(imm, dest_path=dpathName)   #computate shape features 
            # edge = Test.extract_edge_features(imm)
            # texture = Test.extract_texture_features(imm)
            # fourier = Test.extract_fourier_features(imm)
            # haralick = Test.extract_haralick_features(imm)
            # hog = Test.extract_hog_features(imm)
            # lbp = Test.extract_lbp_features(imm)

            # print()
            # print("Imm: "+str(imm))
            # print("Shape features:")
            # for key in shape:
            #     print(key, ' : ', shape[key])
            # print()

            # #plot contourn image 
            # plotImm = mpimg.imread(str(dpathName))
            # imgplot = plt.imshow(plotImm)
            # plt.show()

            # print()
            # print('-------------------------------------------------------------------------------------------------------------------------')
            shapes.append(shape)    #aggiungo il dizionario dell'imm corrente a shapes
        return shapes
    