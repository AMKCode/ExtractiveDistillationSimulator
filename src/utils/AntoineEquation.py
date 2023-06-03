import numpy as np

class AntoineEquation:
    def __init__(self,A,B,C):
        self.A = A
        self.B = B
        self.C = C
        
    def get_partial_pressure(self,Temp):
        return 10**(self.A - self.B/(Temp + self.C))
    
    def get_temperature(self, partial_pressure):
        return (self.B/(self.A - np.log10(partial_pressure))) - self.C