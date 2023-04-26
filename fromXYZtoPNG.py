#imports 
from pathlib import Path
#from rdkit import Chem
from lib.lib_utils import Utils
import os

file_xyz = Path('/home/gabro/grapheneDetectProject/data.xyz/subset_xyz/graphene_67.xyz')
cartellaImmaginiPath = Path('/home/gabro/grapheneDetectProject/images')
dpath = Path('/home/gabro/grapheneDetectProject')
cartellaXYZ = Path('/home/gabro/grapheneDetectProject/data.xyz/subset_xyz/')

#print("spath: %s" % file_xyz)
#print("dpath: %s" % cartellaImmaginiPath)

#lancio la funzione che genera le imamgini 
Utils.generate_bonds_png(file_xyz, cartellaImmaginiPath)

Utils.generate_bonds_png('/home/gabro/grapheneDetectProject/data.xyz/subset_xyz/graphene_238951.xyz', cartellaImmaginiPath)

i=0
#ciclo delle prime 10 immagini 
for nome_file_xyz in os.listdir(cartellaXYZ):
    if(i>=10):
        break
    #recupero il path di ogni immagine 
    path_file_xyz = Path.joinpath(cartellaXYZ, nome_file_xyz)
    #print(path_file_xyz)
    #print(nome_file_xyz)
    #chiamo la funzione che tarsforma da .xyz a .png specificando la risoluzione
    Utils.generate_bonds_png(path_file_xyz, cartellaImmaginiPath, 320)
    i = i+1



#prova con altri file non di grafene per vedere i colori di più atomi
# file_xyz = Path('/home/gabro/grapheneDetectProject/data.xyz/subset_xyz/graphene_67.xyz')
# cartellaImmaginiPath = Path('/home/gabro/grapheneDetectProject/images2')
# dpath = Path('/home/gabro/grapheneDetectProject')
# cartellaXYZ = Path('/home/gabro/grapheneDetectProject/xyz_files')


# i=0
# #ciclo delle prime 10 immagini 
# for nome_file_xyz in os.listdir(cartellaXYZ):
#     if(i>=10):
#         break
#     #recupero il path di ogni immagine 
#     path_file_xyz = Path.joinpath(cartellaXYZ, nome_file_xyz)
#     #print(path_file_xyz)
#     #print(nome_file_xyz)
#     #chiamo la funzione che tarsforma da .xyz a .png specificando la risoluzione
#     Utils.generate_bonds_png(path_file_xyz, cartellaImmaginiPath, 320)
#     i = i+1




