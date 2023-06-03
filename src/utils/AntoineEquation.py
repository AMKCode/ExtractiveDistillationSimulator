import math

class AntoineEquation:
    def __init__(self,A,B,C):
        self.A = A
        self.B = B
        self.C = C
        
    def partial_pressure(self,Temp):
        return 10**(self.A - self.B/(Temp + self.C))
    
    def temperature(self, partial_pressure):
        return (self.B/(self.A - math.log10(partial_pressure))) - self.C