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
from utils.AntoineEquation import *
from distillation.DistillationModel import *
from distillation.DistillationDoubleFeed import *
from scipy.interpolate import griddata
   
class phase_portraits():
    def __init__(self, thermo_model:VLEModel, distil_model:DistillationModel = None):
        self.distil_model = distil_model
        self.thermo_model = thermo_model
    
    def plot_phase_vector_fields(self, ax, dxdt, grid_data_points=20):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        if not isinstance(self.distil_model, DistillationModelDoubleFeed):
            raise TypeError("Invalid operation")
        
        x_array = [np.array(point) for point in create_restricted_simplex_grid(3, grid_data_points)]
        vectors = np.zeros((len(x_array), 2))
        valid_points = []


        for i, x in enumerate(x_array):
            try:
                vector = dxdt(None, x)
                vectors[i] = vector[:2]
                # After computing the vector, check if it's valid
                if not (np.isinf(vectors[i]).any() or np.isnan(vectors[i]).any()):
                    valid_points.append(True)
                else:
                    valid_points.append(False)
            except Exception as e:
                # Handle the case where dxdt raises an exception
                print(f"An error occurred at point {x}: {e}")
                vectors[i] = np.nan  # Assign NaN to indicate an error at this point
                valid_points.append(False)

        # Now filter x_array using list comprehension with valid_points
        
        valid_x_array = [x for i, x in enumerate(x_array) if valid_points[i]]
        valid_vectors = np.array([vectors[i] for i in range(len(vectors)) if valid_points[i]])
        

        # Compute the magnitude of vectors for color mapping
        magnitudes = np.linalg.norm(valid_vectors, axis=1)
        norm = plt.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())
        cmap = plt.cm.viridis  # Use a colormap of your choice

        # Plot the vectors with colors based on their magnitude
        for point, vector in zip(valid_x_array, valid_vectors):
            color = cmap(norm(np.linalg.norm(vector)))
            ax.quiver(point[0], point[1], vector[0], vector[1], color=color)

        # Create a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax)

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title('Phase Vector Field with Magnitude Colored Arrows')

    def plot_vector_field_residue(self, ax, grid_data_points=20):
        def dxdt(t, x):
            try:
                return x - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
        self.plot_phase_vector_fields(ax,dxdt,grid_data_points)
        
    def plot_vector_field_middle(self, ax, grid_data_points=20):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        if not isinstance(self.distil_model, DistillationModelDoubleFeed):
            raise TypeError("Invalid operation")
        def dxdt(t, x):
            try:
                return -self.distil_model.middle_step_x_to_y(x) + self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
        self.plot_phase_vector_fields(ax,dxdt,grid_data_points)
             
    def plot_residue_topography_curve_2D(self, ax, grid_data_points = 20, show_grid = True):
        # Generate the simplex grid
        x_array = create_simplex_grid(3, grid_data_points)
        
        # Initialize an array to hold the calculated values
        value_array = []
        
        # For each point in the simplex grid, calculate the value using the thermo_model
        for x_ in x_array:
            value = self.thermo_model.convert_x_to_y(np.array(x_))[0][-1]
            value_array.append(value)
        
        # Prepare for interpolation: We only need the first two dimensions since the third is dependent
        points = np.array(x_array)[:, :2]

        # Scatter plot with enhanced contrast in color
        value_array = np.array(value_array)
        percentile_10 = np.percentile(value_array, 10)
        percentile_90 = np.percentile(value_array, 90)

        if not show_grid:
            plt.scatter(points[:, 0], points[:, 1], c=value_array, cmap='magma', vmin=percentile_10, vmax=percentile_90, marker="x", s=100)

        # Create a fine grid for interpolation
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]

        # Interpolate using linear splines
        grid_z = griddata(points, value_array, (grid_x, grid_y), method='linear')

        # Plotting the heatmap with enhanced color contrast
        cmap = plt.cm.magma
        heatmap = ax.imshow(grid_z.T, extent=(0, 1, 0, 1), origin='lower', cmap=cmap, vmin=percentile_10, vmax=percentile_90)
        plt.colorbar(heatmap, ax=ax)  # Show the color scale for the heatmap
        ax.set_title("Residue Topography Curve")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
    
    def plot_residue_curve(self, ax, t_span, data_points: int = 15, init_comps = None):
        def dxdt(t, x):
            try:
                return x - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
    def plot_strip_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        def dxdt(t, x):
            try:
                return self.distil_model.stripping_step_xtoy(x_s_j=x) - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
            
    def plot_rect_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        def dxdt(t, x):
            try:
                # return self.distil_model.rectifying_step_xtoy(x_r_j=x) - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
                return x - self.distil_model.rectifying_step_ytox(self.thermo_model.convert_x_to_y(x)[0][:-1])
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
    def plot_middle_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):
        if self.distil_model is None:
            raise TypeError("Invalid operation")
        if not isinstance(self.distil_model, DistillationModelDoubleFeed):
            raise TypeError("Invalid operation")
        def dxdt(t, x):
            try:
                return -self.distil_model.middle_step_x_to_y(x) + self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]
            except OverflowError:
                print("Overflow occurred in dxdt.")
                return None
            
        for init_comp in init_comps:
            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)  

    def int_plot_path(self, ax, initial, t_span, num_points, dxdt):
        # Generate the path data using int_path
        path_data = self.int_path(initial, t_span, num_points, dxdt)

        # Plot arrows every 7 points along the path
        for i in range(0, len(path_data)-1, 7):
            dx = path_data[i+1][0] - path_data[i][0]
            dy = path_data[i+1][1] - path_data[i][1]
            ax.arrow(path_data[i][0], path_data[i][1], dx, dy, head_width=0.02, head_length=0.02, fc='k', ec='k')

        # Plot the path as a red line
        ax.plot(path_data[:, 0], path_data[:, 1], color='red')
        
    def int_path(self, initial, t_span, num_points, dxdt):
        x0 = np.array(initial)
        dt = (t_span[1] - t_span[0]) / num_points
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        x_vals = [x0]

        x = x0
        for t in t_eval:
            x = rk4_step(t, x, dt, dxdt)
            
            if x is None or np.isinf(x).any() or np.isnan(x).any() or (x > 1).any() or (x < 0).any():
                print("Integration stopped due to overflow, NaN values, or out-of-bound values.")
                break

            x_vals.append(x)
        
        return np.array(x_vals)

def rk4_step(t, x, dt, dxdt):
    try:
        k1 = dt * dxdt(t, x)
        k2 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k1)
        k3 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k2)
        k4 = dt * dxdt(t + dt, x + k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    except OverflowError:
        print("Overflow occurred during integration.")
        return None
