import joblib
import numpy as np
import sklearn
import traceback
import sys
from pathlib import Path
p = Path(r"D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training\models\GradientBoosting_processed_20260731_165209_20260820_120212.pkl")
print('numpy', np.__version__)
print('joblib', joblib.__version__)
print('sklearn', sklearn.__version__)
print('attempting load:', p)
try:
    joblib.load(str(p))
    print('Loaded OK')
except Exception:
    traceback.print_exc()
    sys.exit(2)
