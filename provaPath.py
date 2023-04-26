from pathlib import Path
from lib.lib_utils import Utils

p = Path('.')

[x for x in p.iterdir() if x.is_dir()]

Utils.funzione_prova()



