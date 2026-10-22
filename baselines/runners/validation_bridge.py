"""Validation score of a candidate output under the protocol of baselines/runners/common.py."""
import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common
z=np.load(sys.argv[2]);score=common.validation_score(sys.argv[1],z['X'],z['y'],z['Xv'])
print('CANONICAL_VALIDATION '+json.dumps(float(score)))
