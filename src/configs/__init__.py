from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        cur_f = f.split('\\')[-1][:-3]
        exec(f"from src.configs.{cur_f} import *")

