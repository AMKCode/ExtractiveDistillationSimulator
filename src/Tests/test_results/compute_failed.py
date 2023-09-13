import os
import pandas as pd
import numpy as np

def compute_sys_eq(vlemodel, failed_path):
    df_failed = pd.read_csv(failed_path)
    
    x_columns = [f"x{i+1}" for i in range(vlemodel.num_comp)]
    y_columns = [f"y{i+1}" for i in range(vlemodel.num_comp)]
    
    x_arrays = [np.array(row) for row in df_failed[x_columns].values]
    y_arrays = [np.array(row) for row in df_failed[y_columns].values]
    temp_array = np.array(df_failed["Temp"])

    eqs1_results = []
    eqs2_results = []
    for x, y, Temp in zip(x_arrays, y_arrays, temp_array):
        vars1 = np.append(y, Temp)
        eqs1 = vlemodel.compute_Txy(vars1, x)
        eqs1_results.append(eqs1)

        vars2 = np.append(x, Temp)
        eqs2 = vlemodel.compute_Txy2(vars2, y)
        eqs2_results.append(eqs2)
        
    # Generate the column names for both sets of equations
    eqs1_columns = [f'Eq1_{i+1}' for i in range(len(eqs1))]
    eqs2_columns = [f'Eq2_{i+1}' for i in range(len(eqs2))]
    
    # Convert list of lists to DataFrames
    eqs1_df = pd.DataFrame(eqs1_results, columns=eqs1_columns)
    eqs2_df = pd.DataFrame(eqs2_results, columns=eqs2_columns)
    
    # Concatenate the new DataFrames
    df_new = pd.concat([eqs1_df, eqs2_df], axis=1)
    
    # Generate the new CSV file name and save the DataFrame
    new_csv_path = os.path.join(os.path.dirname(failed_path), f"computed_{os.path.basename(failed_path)}")
    df_new.to_csv(new_csv_path, index=False)
    print(f"Saved computed equations to {new_csv_path}")
