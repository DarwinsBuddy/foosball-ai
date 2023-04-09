import os
import sys
# necessary for pytest-cov to measure coverage
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')