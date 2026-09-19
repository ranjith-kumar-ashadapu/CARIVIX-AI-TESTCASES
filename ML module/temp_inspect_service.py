import sys
sys.path.insert(0, r'D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training')
from src.model_service import ModelService, PROJECT_ROOT, DEFAULT_MODELS_DIR
s = ModelService()
print('PROJECT_ROOT=', PROJECT_ROOT)
print('DEFAULT_MODELS_DIR=', DEFAULT_MODELS_DIR)
print('models_dir=', s.models_dir)
print('loaded_models_count=', len(s.loaded_models))
print('loaded_models_keys=', list(s.loaded_models.keys()))
