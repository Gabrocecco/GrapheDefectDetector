#imports 
try:
    from lib.lib_utils import Utils
    from pathlib import Path
    from rdkit import Chem

except Exception as e:
    print("Some module are missing {}".format(e))

