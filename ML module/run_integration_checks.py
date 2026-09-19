"""
Run quick integration checks:
 - Verify nlp_module loads classifier and routes queries
 - Verify RAG pipeline can load index and answer a sample question
 - Verify ModelService has models loaded
"""
import sys
from pathlib import Path
sys.path.insert(0, r'D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training')

import nlp_module
from rag.pipeline import RAGPipeline
from src.model_service import service as model_service

print("NLP checks:")
for q in ["What is CARIVIX?", "Predict next quarter GDP"]:
    print(f"Query: {q} ->", nlp_module.analyze(q))

print("\nModelService loaded models:")
try:
    print(model_service.list_models())
except Exception as e:
    print("ModelService error:", e)

print("\nRAG pipeline check:")
pipeline = RAGPipeline(documents_dir='data/documents', vector_store_dir='data/vector_store')
loaded = False
try:
    loaded = pipeline.load_index()
    print("Index loaded:", loaded)
except Exception as e:
    print("Error loading index:", e)

if loaded:
    try:
        res = pipeline.query('What is CARIVIX?', verbose=False)
        print('Retrieved chunks count:', len(res.get('retrieved_chunks', [])))
        print('Response (truncated):', str(res.get('response'))[:400])
    except Exception as e:
        print('Query failed:', e)
else:
    print('Index not loaded; skipping query test.')
