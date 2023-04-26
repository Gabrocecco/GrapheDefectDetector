#imports 
try:
    from lib.lib_utils import Utils
    from pathlib import Path
    from rdkit import Chem

except Exception as e:
    print("Some module are missing {}".format(e))

filePath = Path('/home/gabro/grapheneDetectProject/data.xyz/subset_xyz/graphene_67.xyz')
immPath = Path('/home/gabro/grapheneDetectProject/images')


print(filePath)
#generate_bonds_png(pathFile, immPath)

