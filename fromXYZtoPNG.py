#imports 
try:
    from pathlib import Path
    from lib.lib_utils import Utils 
    # from sklearn.model_selection import train_test_split
    import os
except Exception as e:

    print("Some module are missing {}".format(e))

#dichiarazione path utili
cartellaImmaginiPath = Path('/home/gabro/grapheneDetectProject/data/images')
rootPath = Path('/home/gabro/grapheneDetectProject')
cartellaXYZPath = Path('/home/gabro/grapheneDetectProject/data.xyz/subset_xyz')
cartellaDataSet = Path('/home/gabro/GrapheDetectProject/data_s')

# #generazione immagine singola
# nomeFileProva = "graphene_67.xyz"   #inserire file .xyz di esempio 
# path_xyz_prova = Path.joinpath(cartellaXYZPath, nomeFileProva)
# Utils.generate_bonds_png(path_xyz_prova, rootPath)


#generazione di tutte le immagini
i=0
for nome_file_xyz in os.listdir(cartellaXYZPath):
    if(i>=100):
        break
    #recupero il path di ogni immagine 
    path_file_xyz = Path.joinpath(cartellaXYZPath, nome_file_xyz)
    #print(path_file_xyz)
    #print(nome_file_xyz)
    #chiamo la funzione che tarsforma da .xyz a .png specificando la risoluzione
    Utils.generate_bonds_png(path_file_xyz, cartellaImmaginiPath, 320)
    i = i+1


# training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)

# print(f"No. of training examples: {training_data.shape[0]}")
# print(f"No. of testing examples: {testing_data.shape[0]}")

# # No. of training examples: 120
# # No. of testing examples: 30





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




