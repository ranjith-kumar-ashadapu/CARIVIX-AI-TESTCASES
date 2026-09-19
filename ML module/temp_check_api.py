import sys
sys.path.insert(0, r'D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training')
import api
app = api.create_app()
print('models:', app.state.model_service.list_models())
if hasattr(app.state.model_service, 'get_load_errors'):
    print('load_errors:', app.state.model_service.get_load_errors())
else:
    print('no get_load_errors')
