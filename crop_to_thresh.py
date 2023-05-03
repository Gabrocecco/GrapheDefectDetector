# import the cv2 library
import cv2
import numpy as np
import os 

#prende una cartella contenente solo imamgini croppate di difetti e ne crea per ognuna la copia thresh

pathImmagini = "/home/gabro/GrapheDetectProject/cartellaCrop/"

for nomeImm in os.listdir(pathImmagini):

    pathImm = os.path.join(pathImmagini, nomeImm)    #estraggo il path dell'immagine 
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
    nameThresh = pathImm.removesuffix('.png') + "_thresh_" + ".png"

    cv2.imwrite(nameThresh, thresh1)



    #devo trasformare tutti i pixel non completaemnte verdi in nero e tutti gli altri in bianco


    #to-do calcolare area delle due cavità 
    
    cv2.waitKey(0)
    cv2.destroyAllWindows() 