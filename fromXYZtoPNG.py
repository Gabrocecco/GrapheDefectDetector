try:
    from lib.lib_utils import Utils

    from pathlib import Path

except Exception as e:
    print("Some module are missing {}".format(e))

pathLocal = Path("/home/gabro/grapheneDetectProject")
pathXYZ = Path("/home/gabro/grapheneDetectProject/dataXYZ")
pathFile = Path("/home/gabro/grapheneDetectProject/dataXYZ/sphere_spiral_700.xyz")

#Utils.dataset_max_and_min(pathXYZ)
#pathXYZ = Path("/home/gabro/grapheneDetectProject/dataXYZ/sphere_spiral_700.xyz")
#Utils.find_max_and_min(pathXYZ)

#prima devo usare read_from_xyz_file per trasformare i file .xyz in liste di valori x,y,z e atomi 

listsXYZ = Utils.read_from_xyz_file(pathFile) 


#Utils.generate_png(pathFile, pathLocal)



#spath = Path("/home/gabro/grapheneDetectProject/dataXYZ/cube.xyz")
#dpath = Path("/home/gabro/grapheneDetectProject/imm.png")


#png = Utils.generate_png(spath, dpath)

