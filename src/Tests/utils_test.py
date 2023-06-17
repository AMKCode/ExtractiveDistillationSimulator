import unittest
import os, sys
import numpy as np
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from utils.AntoineEquation import *


if __name__ == '__main__':
    unittest.main()
    
