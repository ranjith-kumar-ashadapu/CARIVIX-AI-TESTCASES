import joblib
import traceback
from pathlib import Path
import numpy as np
import numpy.random._pickle as nrp
import numpy.random._mt19937 as mt
p = Path(r"D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training\models\GradientBoosting_processed_20260731_165209_20260820_120212.pkl")
print('np', np.__version__)
print('patching numpy.random._pickle.__bit_generator_ctor to handle class objects')
orig = getattr(nrp, '__bit_generator_ctor', None)

def patched(bit_generator_name):
    try:
        return orig(bit_generator_name)
    except Exception as e:
        # Try to resolve class objects by name
        try:
            # If a class object was pickled directly, map by known class names
            name = getattr(bit_generator_name, '__name__', None)
            if name == 'MT19937':
                return mt.MT19937
        except Exception:
            pass
        raise

nrp.__bit_generator_ctor = patched

print('attempting to load with patched ctor...')
try:
    model = joblib.load(str(p))
    print('Loaded OK, model type:', type(model).__name__)
except Exception:
    traceback.print_exc()
    raise
