import joblib,sys
p = r'D:\CARIVIX\CARIVIX AI\CARIVIX_AI_Model_Training\models\LogisticRegression_processed_20260731_165209_20260820_120147.pkl'
try:
    m = joblib.load(p)
    print('LOADED', type(m))
    print('HAS_PREDICT', hasattr(m, 'predict'))
except Exception as e:
    print('ERROR', repr(e))
    sys.exit(1)
