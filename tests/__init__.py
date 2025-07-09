import importlib
import os, sys
from traceback import print_exc

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)

__all__ = []
glob = globals()

for file in os.listdir(root):
    if file.endswith("Tests.py"):
        try:
            mod = importlib.import_module(os.path.splitext(os.path.basename(file))[0])
        except:
            print_exc()
        else:
            __all__ += mod.__all__
            for var in mod.__all__:
                glob[var] = getattr(mod, var)