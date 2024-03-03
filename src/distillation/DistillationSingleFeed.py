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
import matplotlib.pyplot as plt 
import random as rand
from scipy.optimize import fsolve
from scipy.optimize import brentq
from utils.AntoineEquation import *
from thermo_models.RaoultsLawModel import *

#Notes:
#Conditions for a feasible column, profiles match at the feed stage  + no pinch point in between xB and xD
class DistillationModelSingleFeed:
    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        """
        DistillationModel constructor

        Args:
            thermo_model (VLEModel): Vapor-Liquid Equilibrium (VLE) model to be used in the distillation process.
            xF (np.ndarray): Mole fraction of each component in the feed.
            xD (np.ndarray): Mole fraction of each component in the distillate.
            xB (np.ndarray): Mole fraction of each component in the bottom product.
            reflux (Optional): Reflux ratio. If not provided, it will be calculated based on other parameters.
            boil_up (Optional): Boil-up ratio. If not provided, it will be calculated based on other parameters.
            q (float, optional): Feed condition (q) where q = 1 represents saturated liquid feed and q = 0 represents saturated vapor feed. Defaults to 1.
        
        Raises:
            ValueError: If the reflux, boil-up and q are not correctly specified. Only two of these parameters can be independently set.
        """
        self.thermo_model = thermo_model
        self.num_comp = thermo_model.num_comp
        
        self.xF = xF
        self.xD = xD
        self.xB = xB
        self.q = q
        
        if reflux is not None and boil_up is not None and q is None:
            self.boil_up = boil_up
            self.reflux = reflux
            self.q = ((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0])))-(reflux*((xF[0]-xB[0])/(xD[0]-xB[0]))) #this one need 1 component
        elif reflux is None and boil_up is not None and q is not None:
            self.boil_up = boil_up
            self.q = q
            self.reflux = (((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0]))) - self.q)/((xF[0]-xB[0])/(xD[0]-xB[0])) #this one need 1 component
        elif reflux is not None and boil_up is None and q is not None:
            self.reflux = reflux
            self.q = q
            self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1 #this one need 1 component
        else:
            raise ValueError("Underspecification or overspecification: only 2 variables between reflux, boil up, and q can be provided")
        
    def rectifying_step_xtoy(self, x_r_j:np.ndarray):
        """
        Method to calculate y in the rectifying section of the distillation column from given x.

        Args:
            x_r_j (np.ndarray): Mole fraction of each component in the liquid phase in the rectifying section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        r  = self.reflux
        xD = self.xD

        return ((r/(r+1))*x_r_j)+((1/(r+1))*xD)

    def rectifying_step_ytox(self, y_r_j):
        """
        Method to calculate x in the rectifying section of the distillation column from given y.

        Args:
            y_r_j (float): Mole fraction of each component in the vapor phase in the rectifying section.

        Returns:
            float: Mole fraction of each component in the liquid phase in the rectifying section which corresponds to y_r_j.
        """
        r = self.reflux
        xD = self.xD
        return (((r+1)/r)*y_r_j - (xD/r))
    
    def stripping_step_ytox(self, y_s_j):
        """
        Method to calculate x in the stripping section of the distillation column from given y.

        Args:
            y_s_j (float): Mole fraction of each component in the vapor phase in the stripping section.

        Returns:
            float: Mole fraction of each component in the liquid phase in the stripping section that corresponds to y_s_j.
        """
        boil_up = self.boil_up
        xB      = self.xB
        return ((boil_up/(boil_up+1))*y_s_j)+((1/(boil_up+1))*xB)
    
    def stripping_step_xtoy(self, x_s_j):
        """
        Method to calculate y in the stripping section of the distillation column from given x.

        Args:
            x_s_j (float): Mole fraction of each component in the liquid phase in the stripping section.

        Returns:
            float: Mole fraction of each component in the vapor phase in the stripping section.
        """
        boil_up = self.boil_up
        xB = self.xB
        return ((boil_up+1)/boil_up)*x_s_j - (xB/boil_up)
    
    def compute_equib(self):
        
        x1_space = np.linspace(0, 1, 1000)
        y_array = np.zeros((x1_space.size, 2))
        t_array = np.zeros(x1_space.size)
        
        # Initialize numpy arrays
        x_array = np.zeros((x1_space.size, 2))
        for i, x1 in enumerate(x1_space):

            x_array[i] = [x1, 1 - x1]  # Fill the x_array directly
            solution = self.thermo_model.convert_x_to_y(x_array[i])[0]
            y_array[i] = solution[:-1]
            t_array[i] = solution[-1]
            
        return x_array, y_array, t_array
    
    def change_r(self, new_r):

        self.reflux = new_r
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
        return self
        
    def set_xD(self, xD_new):
        self.xD = xD_new
    
    def set_xB(self, xB_new):
        self.xB = xB_new

    def set_xF(self, xF_new):
        self.xF = xF_new
        
    def set_r(self, r_new):

        self.reflux = r_new
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1

    def track_fixed_points_branch(self, op_line, ds, num_steps, x0, l0):

        if op_line == 's':
            return self.track_fixed_points_branch_stripping(ds, num_steps, x0, l0)
        elif op_line != 'r':
            raise ValueError('Please enter either s (for stripping line)  or r (for rectifying line)')
        
        def eqns(uvec, l):

            res = np.empty(self.num_comp+1)
            gammas = self.thermo_model.get_activity_coefficient(uvec[:-1], uvec[-1])

            for i in range(self.num_comp):
                Psat = self.thermo_model.get_Psat_i(i, uvec[-1])
                P = self.thermo_model.get_Psys()
                res[i] = ((Psat*gammas[i]*uvec[i])/P)-((l*uvec[i])/(l+1))-(self.xD[i]/(l+1))
            res[self.num_comp] = np.sum(uvec[:-1]) - 1

            return res

        def get_dFidl(uvec, l):

            res = np.empty(self.num_comp+1)
            for i in range(self.num_comp):
                res[i] = (uvec[i]/(l-1)**2) + (self.xD[i]/(l+1)**2)
            res[self.num_comp] = 0
            return res

        def jac_eqns(uvec, l):
            
            gammas = self.thermo_model.get_activity_coefficient(uvec[:-1], uvec[-1])
            gamma_ders = self.thermo_model.get_gamma_ders(uvec, l)
            res = np.empty((self.num_comp+1,self.num_comp+1))
            P = self.thermo_model.get_Psys()

            for i in range(self.num_comp):
                Psat = self.thermo_model.get_Psat_i(i, uvec[-1])
                dPsatdT = self.thermo_model.get_dPsatdT_i(i, uvec[-1])
                for j in range(self.num_comp):
                    if i == j:
                        res[i,j] = ((Psat/P)*(gammas[i] + uvec[i]*gamma_ders[i,j])) - (l/l-1)
                    else:
                        res[i,j] = ((Psat*uvec[i])/P)*gamma_ders[i,j]
                    res[i,self.num_comp] = (uvec[i]/P)*(Psat*gamma_ders[i,self.num_comp]+gammas[i]*dPsatdT)
            res[self.num_comp,:] = np.concatenate((np.ones(self.num_comp), np.zeros(1)))
            return res

        def eqns_der(xvec, uvec, l):
            res = np.empty(self.num_comp+2)

            jac = jac_eqns(uvec, l)
            dFidl = get_dFidl(uvec, l)

            for i in range(self.num_comp+1):
                res[i] = sum([jac[i,j]*xvec[j] for j in range(self.num_comp+1)])
                res[i] += dFidl[i]*xvec[-1]

            res[self.num_comp+1] = sum([xvec[i]**2 for i in range(self.num_comp+2)]) - 1

            return res
            
        def eqns_aug(uvec, tau, ds, u0):
            l         = uvec[-1]
            res       = np.zeros_like(uvec)
            res[:-1] = eqns(uvec[:-1], l)
            res[-1] = sum([(uvec[i] - u0[i])*tau[i] for i in range(self.num_comp+2)]) - ds

            return res

        def jac_eqns_aug(uvec, der, ds, u0):
            l = uvec[-1]
            res = np.empty((self.num_comp+2,self.num_comp+2))

            jac = jac_eqns(uvec[:-1], l)
            dFidl = get_dFidl(uvec[:-1], l)

            cp1 = self.num_comp+1
            res[:cp1, :cp1] = jac
            res[:cp1, cp1] = dFidl
            res[cp1, :] = der
                
            return res
        
        res = np.empty((num_steps, self.num_comp+2))
        lam_m1 = l0
        old_sol_m1 = fsolve(eqns, x0=x0, fprime=jac_eqns, args=(lam_m1,))
        lam_0 = lam_m1 + ds
        old_sol = fsolve(eqns, x0=old_sol_m1, fprime=jac_eqns, args=(lam_0,))

        for i in range(num_steps):
            if i % 1000 == 0:
                print('sol = ' + str(old_sol) + ', l = ' + str(lam_0))

            # Solve for tangent vector
            # del_s       = math.sqrt(np.linalg.norm(lam_0 - lam_m1)**2)
            del_s = ds

            # Approximation from eqn 8 of Laing
            guess = np.array([(old_sol[i]-old_sol_m1[i])/del_s for i in range(self.num_comp+1)] + [(lam_0 - lam_m1)/del_s])
            # guess = np.array([ (old_sol[0] - old_sol_m1[0])/del_s, (old_sol[1] - old_sol_m1[1])/del_s, (old_sol[2] - old_sol_m1[2])/del_s, (old_sol[3] - old_sol_m1[3])/del_s, (lam_0 - lam_m1)/del_s  ])
            tau = fsolve(eqns_der, guess, args = (old_sol, lam_0))
            
            prev_sol    = np.concatenate((old_sol, np.array([lam_0])))
            new_sol     = fsolve(eqns_aug, x0 = prev_sol + ds*tau, fprime=jac_eqns_aug, args = (tau, ds, prev_sol))
            
            # Edit the variables that hold the two prior solutions
            lam_m1, lam_0 = lam_0, new_sol[-1]
            old_sol_m1 = np.copy(old_sol)
            old_sol    = np.copy(new_sol[:-1])

            res[i, :] = np.concatenate((old_sol, np.array([lam_0])))

            
    def track_fixed_points_branch_stripping(self, ds, num_steps, x0, l0):
        def s(r):
            self.change_r(r)
            return self.boil_up
        
        def eqns(uvec, l):
            res = np.empty(self.num_comp+1)
            gammas = self.thermo_model.get_activity_coefficient(uvec[:-1], uvec[-1])

            for i in range(self.num_comp):
                Psat = self.thermo_model.get_Psat_i(i, uvec[-1])
                P = self.thermo_model.get_Psys()
                res[i] = ((Psat*gammas[i]*uvec[i])/P)-(((s(l)+1)*uvec[i])/s(l))+(self.xB[i]/s(l))
            res[3] = np.sum(uvec[:-1]) - 1

            return res

        def get_dFidl(uvec, l):
            res = np.empty(self.num_comp+1)
            for i in range(self.num_comp):
                res[i] = (uvec[i]-self.xB[i])/s(l)**2
            res[self.num_comp] = 0
            return res

        def jac_eqns(uvec, l):

            gammas     = self.thermo_model.get_activity_coefficient(uvec[:-1], uvec[-1])
            gamma_ders = self.thermo_model.get_gamma_ders(uvec, l)
            res        = np.empty((self.num_comp+1,self.num_comp+1))
            P          = self.thermo_model.get_Psys()

            for i in range(self.num_comp):
                Psat = self.thermo_model.get_Psat_i(i, uvec[-1])
                dPsatdT = self.thermo_model.get_dPsatdT_i(i, uvec[-1])
                for j in range(self.num_comp):
                    if i == j:
                        res[i,j] = ((Psat/P)*(gammas[i] + uvec[i]*gamma_ders[i,j])) - ((s(l)+1)/s(l))
                    else:
                        res[i,j] = ((Psat*uvec[i])/P)*gamma_ders[i,j]
                    res[i,self.num_comp] = (uvec[i]/P)*(Psat*gamma_ders[i,self.num_comp]+gammas[i]*dPsatdT)
            res[self.num_comp,:] = np.concatenate((np.ones(self.num_comp), np.zeros(1)))
            return res

        
        def eqns_der(xvec, uvec, l):
            res = np.empty(self.num_comp+2)

            jac = jac_eqns(uvec, l)
            dFidl = get_dFidl(uvec, l)

            for i in range(self.num_comp+1):
                res[i] = sum([jac[i,j]*xvec[j] for j in range(self.num_comp+1)])
                res[i] += dFidl[i]*xvec[-1]

            res[self.num_comp+1] = sum([xvec[i]**2 for i in range(self.num_comp+2)]) - 1

            return res
            
        def eqns_aug(uvec, tau, ds, u0):
            l         = uvec[-1]
            res       = np.zeros_like(uvec)
            res[0:-1] = eqns(uvec[:-1], l)
            res[-1] = sum([(uvec[i] - u0[i])*tau[i] for i in range(self.num_comp+2)]) - ds

            return res

        def jac_eqns_aug(uvec, der, ds, u0):

            l = uvec[-1]
            res = np.empty((self.num_comp+2,self.num_comp+2))

            jac = jac_eqns(uvec[:-1], l)
            dFidl = get_dFidl(uvec[:-1], l)

            cp1 = self.num_comp+1
            res[:cp1, :cp1] = jac
            res[:cp1, cp1] = dFidl
            res[cp1, :] = der

            return res

        res = np.empty((num_steps, self.num_comp+2))
        lam_m1 = l0
        old_sol_m1 = fsolve(eqns, x0=x0, fprime=jac_eqns, args=(lam_m1,))
        lam_0 = lam_m1 + ds
        old_sol = fsolve(eqns, x0=old_sol_m1, fprime=jac_eqns, args=(lam_0,))

        for i in range(num_steps):
            if i % 1000 == 0:
                print('sol = ' + str(old_sol) + ', l = ' + str(lam_0))

            # Solve for tangent vector
            # del_s       = math.sqrt(np.linalg.norm(lam_0 - lam_m1)**2)
            del_s = ds

            # Approximation from eqn 8 of Laing
            guess = np.array([(old_sol[i]-old_sol_m1[i])/del_s for i in range(self.num_comp+1)] + [(lam_0 - lam_m1)/del_s])
            tau   = fsolve(eqns_der, guess, args = (old_sol, lam_0))
            
            prev_sol    = np.concatenate((old_sol, np.array([lam_0])))
            new_sol     = fsolve(eqns_aug, x0 = prev_sol + ds*tau, fprime=jac_eqns_aug, args = (tau, ds, prev_sol))
            
            # Edit the variables that hold the two prior solutions
            lam_m1, lam_0 = lam_0, new_sol[-1]
            old_sol_m1    = np.copy(old_sol)
            old_sol       = np.copy(new_sol[:-1])

            res[i, :] = np.concatenate((old_sol, np.array([lam_0])))
