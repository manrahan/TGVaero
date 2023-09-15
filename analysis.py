# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:18:57 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore
from matplotlib.projections import polar
from AeroCalculation import log_wind_profile



def normalize_z(df):
    if df.dropna().empty:  # if all values are NaN
        return df  # return as is
    return pd.Series(zscore(df.dropna().values), index=df.dropna().index)

def correlation_matrix(dfs={}):
    # Normalize each dataframe column and store them in a new dictionary
    normalized_dfs = {key: normalize_z(dataframe) for key, dataframe in dfs.items()}
    
    # Concatenate all dataframes along columns
    all_data = pd.concat([dataframe for key, dataframe in normalized_dfs.items()], axis=1)
    
    # Rename columns to the keys of the original dfs dictionary
    all_data.columns = dfs.keys()
    
    # Calculate the correlation matrix
    corr = all_data.corr()
    
    return corr


# def plot_matrix(Test, correlation_matrix, freq, var, unit):
#     # Create a mask for the lower triangle
#     mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
    
#     # Create a heatmap
#     plt.figure(figsize=(8, 6))
#     ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask)
    
#     # Move x-axis labels to the top
#     ax.xaxis.tick_top()
#     ax.xaxis.set_label_position('top')
    
#     plt.title(f"{Test} Matrix Heatmap: {freq}, {var}, [{unit}]")
#     plt.show()

def plot_matrix(Test, correlation_matrix, freq, var, unit):
    # Create a mask for the lower triangle
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
    
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, center=0)  # Added the 'center' parameter
    
    # Move x-axis labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Set y-axis overall title
    ax.set_ylabel('Predicted ($p$)')
    
    plt.title(f"{Test} Matrix Heatmap: {freq}, {var}, [{unit}]")
    plt.show()
    

def calculate_mae(series1, series2):
    # Calculate mean absolute error
    absolute_error = series1 - series2
    mean_absolute_error = absolute_error.mean()
    return mean_absolute_error


def error_matrix(dfs={}):
    series_list = list(dfs.values())
    num_series = len(series_list)
    
    # Initialize an empty matrix to store MAE values
    mae_matrix = np.zeros((num_series, num_series))
    
    for i in range(num_series):
        for j in range(i, num_series):
            mae = calculate_mae(series_list[i], series_list[j])
            mae_matrix[i, j] = mae
            mae_matrix[j, i] = mae  # Since MAE is symmetric
    
    return pd.DataFrame(mae_matrix, index=dfs.keys(), columns=dfs.keys())



def covariance_matrix(dfs={}):
    series_list = list(dfs.values())
    num_series = len(series_list)
    
    # Initialize an empty matrix to store covariance values
    covariance_matrix = np.zeros((num_series, num_series))
    
    for i in range(num_series):
        for j in range(i, num_series):
            covariance = np.cov(series_list[i], series_list[j])[0, 1]
            covariance_matrix[i, j] = covariance
            covariance_matrix[j, i] = covariance  # Since covariance is symmetric
    
    return pd.DataFrame(covariance_matrix, index=dfs.keys(), columns=dfs.keys()),


            
def read_anisotropic(folder: str, method: str):
    # Check that folder exists
    folder = os.path.join(folder, method)
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return None
    
    out_file = os.path.join(folder, f'{method}_anisotropic_combined.xlsx')
    poi_file = os.path.join(folder, f'{method}_poi_list.txt')
    
    poi_list = []
    dfs = []
    
    if not os.path.exists(out_file):
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder, filename)
                try:
                    with open(file_path, 'r') as file:
                        # Read the first line as the header and split it into individual column names
                        header = next(file).strip().split(' ')
                        
                        # Assuming the header may need cleaning or processing, you can do it here
                        header = [item for item in header if len(item) > 1]
                        
                        # Load the rest of the data
                        text = np.loadtxt(file_path, skiprows=1, dtype=float)
                except ValueError as e:
                    print(f"Error processing file '{filename}': {e}")
                    continue  # Skip this file and proceed to the next one
                
                poi = os.path.splitext(filename)[0].split('_')[-1]
                
                if poi != 'isotropic':
                    poi_list.append(poi)
                
                header = [str(item)+'_'+str(poi) for item in header] 
                
                df = pd.DataFrame(text, columns=header)
                
                
                dfs.append(df)
        
        if not dfs:
            print(f"No '{method}' anisotropic files found in folder '{folder}'.")
            return None
        
        
        # Concatenate the dataframes so they share the index
        out_df = pd.concat(dfs, axis=1)
        
        # Save the combined dataframe to an Excel file
        out_df.to_excel(out_file, index=False)
        print(f"Combined dataframe saved to '{out_file}'.")
        
        
        np.savetxt(poi_file, poi_list, fmt='%s')
    
    else:
        out_df = pd.read_excel(out_file)
        poi_txt = np.loadtxt(poi_file)
        poi_list = list(poi_txt)[:-1]
        
    
    return out_df, poi_list




# def calculate_column_stats(df, target_column_tuple):
#     # Define the bin edges for 5-degree intervals from 0 to 360
#     bins = list(range(0, 365, 5))

#     # Create a new column for the binned wind directions
#     df[('calc', 'v3', 'wind_direction_bin')] = pd.cut(df[('Port5', 'ATMOS 22 Ultrasonic Anemometer', '° Wind Direction')], bins=bins, right=False)

#     # Group by the binned column and calculate mean and standard deviation for the target column
#     grouped = df.groupby(('calc', 'v3', 'wind_direction_bin'))
    
#     # Initialize lists to hold the statistics and bin labels
#     means = []
#     stdevs = []
#     counts = []  # Adding this to hold the counts
#     directions = []
    
#     for group, item in grouped:
#         means.append(item[target_column_tuple].mean())
#         stdevs.append(item[target_column_tuple].std())
#         counts.append(len(item))  # Count the number of items in the group
        
#         # Extract the upper limit of the bin from the group label
#         bin_label = str(group)
#         upper_limit = int(bin_label.split(',')[1].strip().strip(')'))
        
#         directions.append(upper_limit)
        
#     return counts, means, stdevs, directions

def calculate_column_stats(df, target_column_tuple):
    # Define the bin edges for 5-degree intervals from 0 to 360
    bins = list(range(0, 365, 5))

    # Create a new column for the binned wind directions
    df[('calc', 'v3', 'wind_direction_bin')] = pd.cut(df[('Port5', 'ATMOS 22 Ultrasonic Anemometer', '° Wind Direction')], bins=bins, right=False)

    # Group by the binned column and calculate mean and standard deviation for the target column
    grouped = df.groupby(('calc', 'v3', 'wind_direction_bin'))
    
    # Initialize lists to hold the statistics and bin labels
    means = []
    stdevs = []
    counts = []  # Adding this to hold the counts
    directions = []
    
    for group, item in grouped:
        means.append(np.nanmean(item[target_column_tuple]))
        stdevs.append(np.nanstd(item[target_column_tuple]))
        counts.append(len(item))  # Count the number of items in the group
        
        # Extract the upper limit of the bin from the group label
        bin_label = str(group)
        upper_limit = int(bin_label.split(',')[1].strip().strip(')'))
        
        directions.append(upper_limit)
        
    return counts, means, stdevs, directions

def plot_aniso_method_comparison(folder, methods, desired_vars, scale, save, stations, df1, bin_of_interest):
    
    
    
    translate = {'Wd':'Wind Direction', 
                      'pai':'Plan Area Index',
                      'fai':'Frontal Area Index', 
                      'zH':'Average Building Height', 
                      'zHmax':'Max Building Height', 
                      'zHstd':'Building Height Standard Deviation', 
                      'zd':'Displacement Height', 
                      'z0':'Roughness Length', 
                      'noOfPixels':'Number of Pixels'}
        
    unit = {'Wd':'[$\degree$]', 
                  'pai':'[-]',
                  'fai':'[-]', 
                  'zH':'$H_{av}$ [$m$]', 
                  'zHmax':'$H_{max}$ [$m$]', 
                  'zHstd':'$\sigma H$ [$m$]', 
                  'zd':'$z_d$ [$m$]', 
                  'z0':'$z_0$ [$m$]', 
                  'noOfPixels':'[-]'}
    
    
    colors = {'RT':'red',
              'MHO':'blue',
              'KAN':'green',
              'MAC':'magenta',
              }
    
    shape = {'RT':'.',
              'MHO':'+',
              'KAN':'*',
              'MAC':'o'}
    
    proxy_artists= np.nan
    legend_labels = np.nan
    
    res = {}
    
    
    for var in desired_vars:
        # Create a figure for the scatter plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        # Create empty lists to store proxy artists and labels for the legend
        proxy_artists = []
        legend_labels = []

        for method in methods:
            file = os.path.join(folder, method, f'{method}_anisotropic_combined.xlsx')

            # Use the method to check for the file
            if os.path.exists(file):
                df = pd.read_excel(file)

                mat = np.ones((72, len(stations) + 1))

                for i, station in enumerate(stations):
                    if f'Wd_{station}' in df.columns and f'{var}_{station}' in df.columns:
                        # Calculate mean for the current wind direction
                        var_data = df[f'{var}_{station}'] * scale
                        wd_data = df[f'Wd_{station}']
                        
                        # Replace infinite values with nan in var_data and wd_data
                        var_data = var_data.replace([np.inf, -np.inf], np.nan)
                        wd_data = wd_data.replace([np.inf, -np.inf], np.nan)
                
                        mat[:, 0] = wd_data.values
                        mat[:, i + 1] = var_data.values
                
                means = []
                stdevs = []
                wind_directions = mat[:, 0]
                
                
                
                for i in range(len(mat[:, 0])):
                    mean_val = np.nanmean(mat[i, 1:])
                    stdev = np.nanstd(mat[i, 1:])
                    
                    means.append(mean_val)
                    stdevs.append(stdev)
                    
                
                if len(means) > 0:
                    print(f'{method} {var}; {bin_of_interest * 5} degree {means[bin_of_interest]} $\pm$ {stdevs[bin_of_interest]}')
                    # Create a proxy artist (scatter plot) with a label for the legend
                    # scatter = plt.scatter(wind_directions, means, color=colors[method], label=method, marker=shape[method])

                    # Plot the mean line
                    plot = plt.plot(wind_directions, means, color=colors[method], linestyle='-', linewidth=2, label=method)
                    
                    # Plot the shaded region around the mean
                    plt.fill_between(wind_directions, np.array(means) + np.array(stdevs), np.array(means) - np.array(stdevs), alpha=0.2, color=colors[method])
                    
                    res[method] = means
                    # Add the proxy artist and label to the legend lists only if it's not already added
                    # if method not in legend_labels:
                    #     proxy_artists.append(plot)
                    #     legend_labels.append(method)

            else:
                raise FileNotFoundError(f'File: {file} does not exist')
        
            if method == methods[-1] and var == 'z0':
                counts, means, stdevs, directions = calculate_column_stats(df1, ('calc', 'v1', var))
                res['A_log'] = means
                print(f'A_log, n= {counts[bin_of_interest]}, {means[bin_of_interest]}, {stdevs[bin_of_interest]}')
                # Plot the mean line
                plot = plt.plot(directions, means, color='orange', linestyle='-', linewidth=2, label='A_log')
                # Plot the shaded region around the mean
                plt.fill_between(directions, np.array(means) + np.array(stdevs), np.array(means) - np.array(stdevs), alpha=0.2, color='orange')
                
                
        plt.grid()
        plt.xlabel('Wind Direction (Wd)')
        plt.ylabel(f'{translate.get(var, var)} {unit.get(var,var)}')  # Use the translation if available, or use the variable name
        plt.legend()#handles=proxy_artists, labels=legend_labels) 
        plt.title(f'Scatter Plot of {translate[var]} vs. Wind Direction for Different Methods')

        if len(stations) > 50:
            suptitle_string = f'All Stations and Grid Points n={len(stations)}'
            save_string = f'all_grid_{var}'
        elif len(stations) == 5:
            suptitle_string = f'Stations Only n={len(stations)}'
            save_string = f'stations_only_{var}'
        elif len(stations) == 50:
            suptitle_string = f'Stations and Grid Points n={len(stations)}'
            save_string = f'stations_and_grid_{var}'

        plt.suptitle(suptitle_string)
        plt.tight_layout()

        if save:
            plot_path = os.path.join('../outputs/UMEP_outputs_2/Plots/', save_string+'.png')
            plt.savefig(plot_path, dpi=400)

        plt.show()

        fig2 = plt.figure(figsize=(8, 8))
        ax2 = fig2.add_subplot(111, polar=True)
        
        for method in methods:
            file = os.path.join(folder, method, f'{method}_anisotropic_combined.xlsx')
        
            # Use the method to check for the file
            if os.path.exists(file):
                df = pd.read_excel(file)
        
                mat = np.ones((72, len(stations) + 1))
        
                for i, station in enumerate(stations):
                    if f'Wd_{station}' in df.columns and f'{var}_{station}' in df.columns:
                        # Calculate mean for the current wind direction
                        var_data = df[f'{var}_{station}'] * scale
                        wd_data = df[f'Wd_{station}']
                        var_data = var_data.replace([np.inf, -np.inf], np.nan)
                        wd_data = wd_data.replace([np.inf, -np.inf], np.nan)
                
                        mat[:, 0] = wd_data.values
                        mat[:, i + 1] = var_data.values
                
                means = []
                stdevs = []
                wind_directions = mat[:, 0]
                
                for i in range(len(mat[:, 0])):
                    mean_val = np.nanmean(mat[i, 1:])
                    stdev = np.nanstd(mat[i, 1:])
                    
                    means.append(mean_val)
                    stdevs.append(stdev)
                    
                    
            angles = np.array(wind_directions)
            frequencies = np.array(means)
            # Convert angles from degrees to radians
            angles = np.deg2rad(angles)
            # Plot the wind rose lines
            ax2.plot(np.concatenate((angles, [angles[0]])), np.concatenate((frequencies, [frequencies[0]])), color=colors[method], label=method)
        if method == methods[-1] and var == 'z0':
            counts, means, stdevs, directions = calculate_column_stats(df1, ('calc', 'v1', var))
            # Plot the mean line
            angles = np.array(directions)
            frequencies = np.array(means)
            # Convert angles from degrees to radians
            angles = np.deg2rad(angles)
            # Plot the wind rose lines
            ax2.plot(np.concatenate((angles, [angles[0]])), np.concatenate((frequencies, [frequencies[0]])), color='orange', label='A_log')
            
            
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_rlabel_position(90)
        # ax2.set_rticks([10, 20, 30, 40, 50], labels=['10', '20', '30', '40', '50'], fontsize=12)  # Customize the radial labels
        # ax2.set_yticklabels([])
        
        # Set a title for the wind rose plot
        ax2.set_title(f'Polar Plot of {translate[var]}', fontsize=16)
        ax2.legend()
        
        plt.show()
    return res

def plot_wind_profile(df):
    mask = (df[('Port4', 'ATMOS 22 Ultrasonic Anemometer', '° Wind Direction')] >= 320) & \
       (df[('Port4', 'ATMOS 22 Ultrasonic Anemometer', '° Wind Direction')] <= 350)

    filtered_df = df[mask]
    
    methods = ['RT', 'MHO', 'KAN', 'MAC']
    
    fig, ax = plt.subplots(1,1)
    
    for i in range(len(filtered_df)):
        z_0 = filtered_df.iloc[i][('morph', 'RT', 'z0_20596')]
        z_d = filtered_df.iloc[i][('morph', 'RT', 'zd_20596')]
        u_star = filtered_df.iloc[i]['calc', 'v2', 'u_star']
        EWI = filtered_df.iloc[i][('EWI', 'add', 'windSpeed')]
        
        y = np.arange(z_0, 100)
        x = []
        
        for z in y:
            x.append(log_wind_profile(z, np.abs(u_star), z_0, z_d))
            
        ax.plot(x, y)
        print('EWI', EWI)
        ax.scatter(EWI, 100)
        
        if i ==3:
            break
    print(len(filtered_df))
    
                
def rmse_with_nan(list1, list2):
    if len(list1) != len(list2):
        return "Lists must be of the same length"
    
    # Convert the lists to NumPy arrays for easier manipulation
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    
    # Calculate the squared differences
    sq_diffs = (arr1 - arr2)**2
    
    # Remove the pairs where either value is NaN
    sq_diffs = sq_diffs[~np.isnan(arr1) & ~np.isnan(arr2)]
    
    # Calculate the mean of the squared differences
    mean_sq_diff = np.nanmean(sq_diffs)
    
    # Take the square root to get RMSE
    rmse = np.sqrt(mean_sq_diff)
    
    return rmse


