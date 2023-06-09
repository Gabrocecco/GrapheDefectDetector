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


IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

spath = Path('/home/gabro/GrapheDetectProject/cartellaTrashProva')   #cartella dove sono contenuti tutti i trash 
dpath = Path('/home/gabro/GrapheDetectProject/contoursProva')   #cartella dove sono contenuti tutti i trash 


#defect analysis selected images
# image1 = Path('/home/gabro/GrapheDetectProject/cartellaCrop/0a5c3409-graphene_301236_bonds_cropped_box_0_thresh_.png')
# img1 = mpimg.imread(str(image1))
# image2 = Path('/home/gabro/GrapheDetectProject/cartellaTrash/0a288a23-graphene_227444_bonds_cropped_box_1_thresh_.png')
# img2 = mpimg.imread(str(image2))
# image3 = Path('/home/gabro/GrapheDetectProject/cartellaTrash/2d81c496-graphene_122634_bonds_cropped_box_1_thresh_.png')    
# img3 = mpimg.imread(str(image3))
# image4 = Path('/home/gabro/GrapheDetectProject/cartellaTrash/2d81c496-graphene_122634_bonds_cropped_box_1_thresh_.png')    
# img4 = mpimg.imread(str(image4))
# images = [image1, image2, image3, image4]

images = []
nomiImages = os.listdir(spath)
for nomeImm in nomiImages:
    pathImm=os.path.join(spath, nomeImm)
    images.append(pathImm)
for imm in images:  #itero sui apth delle immagini 
    print("Imm:" + str(imm))
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

    print()
    print("Imm: "+str(imm))
    print("Shape features:")
    for key in shape:
        print(key, ' : ', shape[key])
    print()

    #plot contourn image 
    # plotImm = mpimg.imread(str(dpathName))
    # imgplot = plt.imshow(plotImm)
    # plt.show()

    # print more propierties 
    # print("Edge features:")
    # for key in edge:
    #     print(key, ' : ', edge[key])
    # print()
    # print("Texture features:")
    # for key in texture:
    #     print(key, ' : ', texture[key])


    # print()
    # print("Fourier features:")
    # for key in fourier:
    #     print(key, ' : ', fourier[key])
    # print()

    # print("Heralik features:")
    # print("heralik")
    # print("hog features:")
    # print(hog)
    # print("lbp features:")
    # print(lbp)
    print()
    print('-------------------------------------------------------------------------------------------------------------------------')

    #comment test