#imports 
from pathlib import Path
#from rdkit import Chem
from lib.lib_utils import Utils
import os

#dichiarazione path utili
cartellaImmaginiPath = Path('/home/gabro/grapheneDetectProject/images')
projectPath = Path('/home/gabro/grapheneDetectProject')
cartellaXYZ = Path('/home/gabro/grapheneDetectProject/data.xyz/subset_xyz/')

#generazione immagine singola
nomeFileProva = "graphene_67.xyz"   #inserire file .xyz di esempio 
path_xyz_prova = Path.joinpath(cartellaXYZ, nomeFileProva)
Utils.generate_bonds_png(path_xyz_prova, projectPath)


#ciclo delle prime 10 immagini 
i=0
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




