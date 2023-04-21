try:
    from lib.lib_utils import Utils

    from pathlib import Path

except Exception as e:
    print("Some module are missing {}".format(e))

pathLocal = Path("/home/gabro/grapheneDetectProject")
pathXYZ = Path("/home/gabro/grapheneDetectProject/dataXYZ")

#Utils.dataset_max_and_min(pathXYZ)
pathXYZ = Path("/home/gabro/grapheneDetectProject/dataXYZ/cube.xyz")
Utils.find_max_and_min(pathXYZ)

#spath = Path("/home/gabro/grapheneDetectProject/dataXYZ/cube.xyz")
#dpath = Path("/home/gabro/grapheneDetectProject/imm.png")


#png = Utils.generate_png(spath, dpath)

