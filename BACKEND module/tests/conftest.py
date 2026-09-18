import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["data_processing", "data_acquisition", "predictive_analytics", "nlp", "database", "visualization"]:
    sys.path.insert(0, os.path.join(ROOT, sub))
    