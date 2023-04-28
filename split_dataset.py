import os
import glob
import random
from pathlib import Path 
import os 

filelist = glob.glob('/home/gabro/GrapheDetectProject/data_copy2/*.txt') #prendo la lista di tutti i path delle labels nel dataset
test = random.sample(filelist, int(len(filelist)*0.20))    #prendo il 20% dei path dei labels
output_path = '/home/gabro/GrapheDetectProject/data_copy2/test/'
if not os.path.exists(output_path): #creo la cartella test se non esiste ancora
    os.makedirs(output_path)

for file in test:  #scorro tutte le labels di test
    txtpath = file
    impath = file[:-4] + '.png' #costruisco i rispettivi path delle immagini 
    out_text = os.path.join(output_path, os.path.basename(txtpath)) #questi ultime due liste di path sono i path finali nelle due cartelle nuove 
    out_image = os.path.join(output_path, os.path.basename(impath))
    print(txtpath,impath,out_text,out_image)
    os.system('mv ' + txtpath + ' ' + out_text)
    os.system('mv ' + impath + ' ' + out_image)



# #TO-DO splittare dataset in trining e val 

# #recupero la lista di tutti i path delle immagini e dei label del dataset 
# img_paths = glob.glob(cartellaDataSet+'*.png')
# txt_paths = glob.glob(cartellaDataSet+'*.txt')
# # print(img_paths)
# # print(txt_paths)

# # Calculate number of files for training, validation
# data_size = len(img_paths)
# r = 0.8 
# train_size = int(data_size * 0.8)

# # Shuffle two list
# img_txt = list(zip(img_paths, txt_paths))
# print(img_txt)
# random.seed(43)
# random.shuffle(img_txt)
# img_paths, txt_paths = zip(*img_txt)

# # Now split them
# train_img_paths = img_paths[:train_size]
# train_txt_paths = txt_paths[:train_size]

# valid_img_paths = img_paths[train_size:]
# valid_txt_paths = txt_paths[train_size:]

# # Move them to train, valid folders
# train_folder = cartellaDataSet+'train/' 
# valid_folder = cartellaDataSet+'valid/'
# os.mkdir(train_folder)
# os.mkdir(valid_folder)

# def move(paths, folder):
#     for p in paths:
#         shutil.move(p, folder)

# move(train_img_paths, train_folder)
# move(train_txt_paths, train_folder)
# move(valid_img_paths, valid_folder)
# move(valid_txt_paths, valid_folder)