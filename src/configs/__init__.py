# from src.configs.atacseq import atacseq_config
# from src.configs.atacseq_classify_only import atacseq_clonly_config
# from src.configs.atacseq_no_alternating import atacseq_all_config
#
# from src.configs.atacseq import atacseq_config
# from src.configs.default import default_config
# from src.configs.dlpfc import dlpfc_config
# from src.configs.patchseq import patchseq_config
# from src.configs.dbitseq import dbitseq_config
from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        cur_f = f.split('\\')[-1][:-3]
        exec(f"from src.configs.{cur_f} import *")

