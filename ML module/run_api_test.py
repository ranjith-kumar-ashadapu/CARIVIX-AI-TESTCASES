from fastapi.testclient import TestClient
import sys
sys.path.insert(0, r'D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training')
from api import create_app

app = create_app()
client = TestClient(app)

# RAG query
resp = client.post('/api/v1/ai/query', json={'query': 'What is CARIVIX?', 'k': 3})
print('RAG status:', resp.status_code)
print(resp.json())

# ML query
resp2 = client.post('/api/v1/ai/query', json={'query': 'Predict next quarter GDP, income 50000, age 35'})
print('\nML status (no auto-fill):', resp2.status_code)
print(resp2.json())

# ML query with auto_fill_missing=True
resp3 = client.post('/api/v1/ai/query', json={'query': 'Predict next quarter GDP, income 50000, age 35', 'auto_fill_missing': True})
print('\nML status (auto-fill):', resp3.status_code)
print(resp3.json())
