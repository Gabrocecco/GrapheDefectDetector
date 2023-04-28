import os
import glob
import random
from pathlib import Path 
import os 

#PERCENTUALE TEST:
TEST_perc  = 0.20

#path cartella dati, img e labels insieme
pathData = '/home/gabro/GrapheDetectProject/data_copy3/'


filelist = glob.glob(pathData+'*.txt') #prendo la lista di tutti i path delle labels nel dataset
test = random.sample(filelist, int(len(filelist)*TEST_perc))    #prendo il 20% dei path dei labels
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
