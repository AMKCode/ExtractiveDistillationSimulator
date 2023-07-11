import numpy as np
import os, sys
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from thermo_models.VLEModelBaseClass import *
from thermo_models.RaoultsLawModel import RaoultsLawModel
from thermo_models.WilsonModel import WilsonModel
from thermo_models.MargulesModel import MargulesModel
from thermo_models.VanLaarModel import VanLaarModel
import utils.AntoineEquation as AE
import matplotlib.pyplot as plt 
import random as rand

#Notes:
#Conditions for a feasible column, profiles match at the feed stage  + no pinch point in between xB and xD
class distillation:
    def __init__(self, thermo_model:VLEModel, xD = None, xB = None, reflux = None, boil_up = None, total_stages = None, feed_stage = None) -> None:
        self.thermo_model = thermo_model
        self.num_comp = thermo_model.num_comp
        #intialize rectifying parameters
        #intialize stripping parameters
    
    def rectifying_surf(self, ):
        # rectifying equations are computed here, 
        # uses convert y to x to compute the corresponding
        pass
    
    def stripping_surf(self):
        pass
    
    def fixed_point(self):
        #find the fixed points instead of simulation? quoting the fidowski paper
        pass
    
    def plot_binary(self):
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
    
    
    
        
        
    
        