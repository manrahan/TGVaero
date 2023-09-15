# -*- coding: utf-8 -*-
"""
Created on Fri Aug 01 15:14:00 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.plot import show
from datetime import datetime
import numpy as np


class StationData:
    def __init__(self, station_number, output_path):
        self.station_number = station_number
        self.output_path = output_path
        self.metadata = self.read_metadata()
        self.device_name = self.validate_device_name()
        self.coordinates = self.extract_coordinates()
    
    def read_metadata(self):
        """
        Read the metadata from an Excel file for a particular station.

        This function attempts to read metadata from an Excel file whose name is
        formed by concatenating 'metadata_' and the station number. The Excel
        file is expected to reside in the directory specified by self.output_path.

        Returns:
            meta (DataFrame): A Pandas DataFrame containing the metadata.
            Returns None if the metadata file is not found or an error occurs.

        Raises:
            Prints an error message if any exception occurs while reading the Excel file.
        """
        
        meta_path = os.path.join(self.output_path, f'metadata_{self.station_number}.xlsx')
        # print(f"Reading metadata from: {self.station_number}")
        
        try:
            meta = pd.read_excel(meta_path, header=None, index_col=[0, 1], usecols=[1, 2,3])
            # print('meta found')
            return meta
        
        except Exception as e:
            print(f"Error reading Metadata file for station {self.station_number}: {str(e)}")
        
        else:
            print(f"Metadata file not found for station {self.station_number}")
            return None
    
    def validate_device_name(self):
        """
        Validate that the device name in the metadata matches the station number.

        This function checks if the 'Device Name' entry in the metadata DataFrame
        matches the station_number attribute of the class. Raises a ValueError if
        they do not match.

        Returns:
            device_name (str): The validated device name, or None if metadata is None.

        Raises:
            ValueError: If the device name in metadata and station number do not match.
        """
        
        if self.metadata is not None:
            device_name = self.metadata.loc['Configuration', 'Device Name'].values[0]
            
            if device_name != self.station_number:
                raise ValueError(f"Device name '{device_name}' does not match station number '{self.station_number}'")
            
            return device_name
        else:
            return None
            
    def extract_coordinates(self):
        """
        Extract the latitude and longitude from the metadata for a particular station.
    
        This function retrieves the 'Latitude' and 'Longitude' entries from the metadata
        DataFrame and returns them in a dictionary.
    
        Returns:
            coordinates (dict): A dictionary containing the 'Latitude' and 'Longitude'.
                                 Returns None if metadata is not available.
    
        Raises:
            Prints a message if metadata is not available for the station.
        """
        
        if self.metadata is not None:
            latitude = self.metadata.loc['Location', 'Latitude'].values[0]
            longitude = self.metadata.loc['Location', 'Longitude'].values[0]
            # print("Coordinates extracted successfully")
            return {'Latitude': latitude, 'Longitude': longitude}
        else:
            print(f"No Metadata for {self.station_number}")
            return None
    


class PlotLoc:
    def __init__(self, shapefile_path, tiff_path, title, crs):
        self.shapefile_path = shapefile_path
        self.tiff_path = tiff_path
        self.title = title
        self.crs = crs
        
    def plot_location(self):
        """
        Plot the geographical location of a station on top of a georeferenced TIFF image.
    
        This function performs the following steps:
        1. Reads a shapefile containing geographical features, using GeoPandas.
        2. Reads a georeferenced TIFF image using rasterio, if available.
        3. Plots the TIFF image.
        4. Overlays the station location on top of the TIFF, marked in red.
        
        Attributes Used:
            self.shapefile_path (str): The path to the shapefile.
            self.tiff_path (str): The path to the georeferenced TIFF file.
            self.crs (str): Coordinate reference system.
            self.title (str): The title of the plot.
    
        Returns:
            None: The function displays the plot but does not return any values.
    
        Raises:
            Prints messages for debugging and information.
        """
        
        # Load the shapefile using GeoPandas
        gdf = gpd.read_file(self.shapefile_path, crs='EPSG:4326')
        
        if self.tiff_path is not None and os.path.exists(self.tiff_path):
            # Load the georeferenced TIFF using rasterio
            with rasterio.open(self.tiff_path) as src:
                geotiff_crs = src.crs
                # print('src', geotiff_crs)
                fig, ax = plt.subplots(dpi=600)
                
                # Plot the TIFF
                show(src, ax=ax)
                
                # Plot the station locations on top of the TIFF
                gdf.plot(ax=ax, color='red', markersize=20, marker='+')
                
                # Add labels and title
                plt.xlabel(f'Longitude ({self.crs[:]})')
                plt.ylabel('Latitude')
                plt.title(self.title)
                plt.grid(alpha=0.4)
                
                # Show the plot
                plt.show()
        else:
            print('Background TIFF not found')
            
            # Fallback: Plot using GeoPandas without the TIFF background
            gdf.plot(color='blue', markersize=20)
            
            # Add labels to each point
            for idx, row in gdf.iterrows():
                plt.text(row['Longitude'], row['Latitude'], row['Station'], fontsize=10)
            
            # Add labels and title
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(self.title)
            plt.grid(alpha=0.4)
            
            # Show the plot
            plt.show()

    
class CreateShapefile:
    def __init__(self, tup, output_folder, original_crs, desired_crs):
        self.tup = tup
        self.output_folder = output_folder
        self.original_crs = original_crs
        self.desired_crs = desired_crs
    
    def create_shapefile(self):
        """
        Create and save a shapefile based on station coordinates.
    
        This function performs the following steps:
        1. Create a GeoDataFrame using the station coordinates from self.tup.
        2. If available, set the CRS of the GeoDataFrame to self.original_crs.
        3. Convert the CRS to self.desired_crs if it doesn't match the original CRS.
        4. Save the GeoDataFrame as a shapefile to the specified output folder.
    
        Attributes Used:
            self.tup (tuple): Tuple containing station names and coordinates (name, lat, lon).
            self.output_folder (str): The path to the output folder where the shapefile will be saved.
            self.original_crs (str): The original Coordinate Reference System.
            self.desired_crs (str): The desired Coordinate Reference System for the shapefile.
    
        Returns:
            None: The function saves the shapefile and prints its location but does not return any values.
        """
        
        # Create a GeoDataFrame without specifying the CRS
        geometry = [Point(lon, lat) for name, lat, lon in self.tup]
        gdf = gpd.GeoDataFrame(self.tup, geometry=geometry, columns=['Station', 'Latitude', 'Longitude'])
        
        # print(f'\n !!originalcrs!! \n\n {gdf.crs}')
        
        # Set the CRS of the GeoDataFrame to the original CRS if available
        if self.original_crs is not None:
            gdf.crs = self.original_crs
            
        
        # Convert the CRS of the GeoDataFrame to EPSG:4326 if needed
        if gdf.crs != self.desired_crs:
            gdf = gdf.to_crs(self.desired_crs)
        
        # print('gdf.crs = ', gdf.crs)
        # Save the GeoDataFrame as a shapefile
        shapefile_path = os.path.join(self.output_folder, f'station_locations_{str(gdf.crs)[-5:]}.shp')
        gdf.to_file(shapefile_path)
        # print(gdf)
        print(f'Shapefile saved to: {shapefile_path}\n CRS:{gdf.crs}')


class DataSummary:
    def __init__(self, date_start, date_end, station_number, output_folder, summary_path):
        self.start_date_dt = pd.Timestamp(date_start)
        self.end_date_dt = pd.Timestamp(date_end)
        self.station_number = station_number
        self.output_folder = output_folder
        self.summary_path = summary_path
        self.data = self.loader()
    
    def loader(self):
        """
        Load and filter Excel data for a specific station within a date range.
    
        This function reads an Excel file containing data for a specific station
        and filters the rows based on the date range specified in self.start_date_dt 
        and self.end_date_dt.
    
        Attributes Used:
            self.output_folder (str): Directory where the Excel file is located.
            self.station_number (str): Identifier for the station.
            self.start_date_dt (datetime): Start date for data filtering.
            self.end_date_dt (datetime): End date for data filtering.
    
        Returns:
            filtered_df (DataFrame): A DataFrame containing the filtered data.
            None: If the data path does not exist.
        """
        
        data_path = os.path.join(self.output_folder, f'combined_df_{self.station_number}.xlsx')
        
        if os.path.exists(data_path):
            df = pd.read_excel(data_path, header=[0,1], index_col=[0], skiprows=[2], parse_dates=True)
            filtered_df = df[(df.index >= self.start_date_dt) & (df.index <= self.end_date_dt)]
            # print('data loaded!')
            # print(list(filtered_df.keys()))
            return filtered_df
        
        else:
            print(f'datapath does not exist at: {data_path}')
    
    def data_summary(self, instrument, columns_to_summarize, resample_string):
        """
        Generate summary statistics for specified columns and save them to an Excel file.
    
        This function performs the following steps:
        1. Check if the data is loaded.
        2. Resample and aggregate data based on custom and basic statistical functions.
        3. Save the summary statistics to an Excel file.
    
        Attributes Used:
            self.data (DataFrame): Loaded data for the station.
            self.summary_path (str): Directory where the summary Excel file will be saved.
            self.station_number (str): Identifier for the station.
    
        Args:
            instrument (str): The instrument type for which to summarize data.
            columns_to_summarize (list): List of column names to summarize.
            resample_string (str): The time frequency for resampling the data (e.g., 'D' for daily).
    
        Returns:
            None: The function saves the summary Excel file and prints its location but does not return any values.
        """
        
        
        if self.data is None:
            print('No data loaded.')
            return
            
        # Define custom aggregation functions
        def count_zeros(x):
            return (x == 0).sum()
            
        def count_nas(x):
            return x.isna().sum()
            
        
        summary_path = os.path.join(self.summary_path, self.station_number)
        
        # Check if any files with the same resample_string prefix exist in the summary directory
        summary_files = os.listdir(summary_path)
        
        if any(file.startswith(resample_string) for file in summary_files):
            print(f'Summary file(s) with same resample ({resample_string}) already exist in directory (SN: {self.station_number}).')
        
        else:
            # Select columns to summarize
            selected_instrument = self.data[instrument][columns_to_summarize]
            
            # Resample the data and calculate summary statistics
            print(f'resampling and summarising {instrument}')
            
            resampled_data = selected_instrument.resample(resample_string).agg(['min', 'max', 'mean','std', 'var', 'count', count_zeros, count_nas])
            
            # Create the summary Excel filename
            filename = f'{resample_string}_{self.data.index.min().date()}_{self.data.index.max().date()}_resample.xlsx'
            # Create the summary directory if it doesn't exist
            
            d = os.path.dirname(summary_path)
            
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
                
            summary_file_path = os.path.join(d,self.station_number, filename)
            # Save the summary as an Excel file
            resampled_data.to_excel(summary_file_path)
            
            print(f'Summary saved to: {summary_file_path}')




def wind_rose(df, col, saveFig, station, resample_string, start, end):
    """
    Plot and save a wind rose based on wind direction data.

    This function performs the following steps:
    1. Filters the DataFrame based on the provided start and end dates.
    2. Resamples the DataFrame using the specified resampling string.
    3. Creates a new datetime index to match the original DataFrame.
    4. Merges the new and original Series.
    5. Computes a histogram with 36 bins.
    6. Creates a polar plot and plots the histogram bars.
    7. Optionally saves the plot to a specified path.

    Args:
        df (DataFrame): Original DataFrame containing wind direction data.
        col (str): Column name in the DataFrame for the wind speed.
        saveFig (bool): Whether to save the figure.
        station (str): Name or identifier for the station.
        resample_string (str): Time frequency for resampling data (e.g., 'D' for daily).
        start (datetime-like): Start date for data filtering.
        end (datetime-like): End date for data filtering.

    Returns:
        df_ms (DataFrame): A DataFrame containing the resampled data.
    """
    
    # Filter the DataFrame based on the start and end dates
    df = df.loc[start:end]
    df_ws = df[col].resample(resample_string).mean()  # Assuming the 'col' parameter corresponds to the wind speed column
    
    # Create new datetime index
    new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{str(pd.infer_freq(df_ws.index))}')
    
    new_series = pd.Series(index=new_index, name=col)  # Provide a name for the Series
    
    df_ms = pd.concat([df_ws, new_series], axis=1)
    
    # Drop the duplicated column header
    df_ms = df_ms.loc[:, ~df_ms.columns.duplicated()]

    df_ms = df_ms.loc[start:end]

    resample_df = df_ms

    # Compute histogram with 36 bins
    n, bins = np.histogram(df_ms, bins=36, range=(0, 360))
    
    # Convert bin edges to radians
    bin_edges_rad = np.radians(bins)
    
    # Create polar plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Plot the histogram bars
    ax.bar(bin_edges_rad[:-1], n, width=np.radians(360) / 36, edgecolor='black', alpha=0.7)
    
    # Set 0 degrees at the top and rotate clockwise
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    
    fig.suptitle(f'{df_ms.index.date.min()} - {df_ms.index.date.max()}')
    ax.set_title(f'Wind Direction from {station}')
    
    
    if saveFig == True:
        path = r'../Images/'
        plt.savefig(path+f'wind_rose {station} {df_ms.index.date.min()} - {df_ms.index.date.max()}.png', dpi=600)
        
        
    
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        plt.gca().set_facecolor('none')  # Set axes background to transparent
        plt.gcf().set_facecolor('none')  # Set figure background to transparent
        plt.title('')  # Remove title
        plt.suptitle('')  # Remove super title
        filename = f'wind_rose {station} {df_ms.index.date.min()} - {df_ms.index.date.max()}'
        plt.savefig(f"{path}{filename}.svg", dpi=600)
    # print('wd', df_ms)
    return df_ms
    # plt.show()  # Display the plot
    

def wind_speed_plot(df, col, saveFig, station, resample_string, start, end):
    """
    Plot wind speed data.
    
    Parameters:
    df (DataFrame): The input DataFrame containing wind speed data.
    col (str): The column containing the wind speed data.
    saveFig (bool): Whether to save the figure.
    station (str): The station name.
    resample_string (str): The resampling frequency string.
    start (datetime): Start date for the plot.
    end (datetime): End date for the plot.
    
    Returns:
    DataFrame: A DataFrame containing the resampled data.
    """
    
    df = df.loc[start:end]
    
    df_ws = df[col].resample(resample_string).mean()  # Assuming the 'col' parameter corresponds to the wind speed column
    
    # Create new datetime index
    new_index = pd.date_range(start=start, end=df_ws.index.max(), freq=f'{str(pd.infer_freq(df_ws.index))}')
    
    new_series = pd.Series(index=new_index, name=col)  # Provide a name for the Series
    
    df_ms = pd.concat([df_ws, new_series], axis=1)
    
    # Drop the duplicated column header
    df_ms = df_ms.loc[:, ~df_ms.columns.duplicated()]
    
    df_ms = df_ms.loc[start:end]
    
    resample_df = df_ms
    # print(resample_df)

    # Create the wind speed plot
    fig, ax = plt.subplots(figsize=(10, 6.18))
    ax.set_xlabel('Datetime')

    df_ms[col].plot(ax=ax, alpha=0.2, label=f'freq:{str(pd.infer_freq(df_ws.index))}', drawstyle='steps-post')

    # Plot daily stats
    resample_df[col].resample('D').mean().plot(ax=ax, label='Daily Mean', color='black', alpha=0.7)
    resample_df[col].resample('D').min().plot(ax=ax, label='Daily Min', color='blue', alpha=0.7)
    resample_df[col].resample('D').max().plot(ax=ax, label='Daily Max', color='red', alpha=0.7)

    ax.set_ylim(0)
    ax.set_ylabel('Wind Speed [m/s]')
    ax.legend()
    ax.grid()
    ax.set_title(f'Wind Speed from {station}')
    fig.suptitle(f'{df_ms.index.date.min()} - {df_ms.index.date.max()}')

    if saveFig:
        path = '../Images/'
        plt.savefig(f'{path}wind_speed_{station}_{df_ms.index.date.min()}_{df_ms.index.date.max()}.png', dpi=600)
    # print('ws', df_ms)
    return df_ms
    # plt.show()  # Display the plot
    
def temp_plot(df, col, saveFig, station, resample_string, start, end):
    """
    Plot temperature data.
    
    Parameters:
    (same as wind_speed_plot)
    
    Returns:
    DataFrame: A DataFrame containing the resampled data.
    """
    
    
    df_ws = df[col].resample(resample_string).mean()  # Assuming the 'col' parameter corresponds to the wind speed column
    
    # Create new datetime index
    new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{str(pd.infer_freq(df_ws.index))}')
    
    new_series = pd.Series(index=new_index, name=col)  # Provide a name for the Series
    
    df_ms = pd.concat([df_ws, new_series], axis=1)
    
    # Drop the duplicated column header
    df_ms = df_ms.loc[:, ~df_ms.columns.duplicated()]
    
    df_ms = df_ms.loc[start:end]
    
    resample_df = df_ms

    # print(resample_df)

    # Create the wind speed plot
    fig, ax = plt.subplots(figsize=(10, 6.18))
    ax.set_xlabel('Datetime')

    df_ms[col].plot(ax=ax, alpha=0.2, label=f'freq:{str(pd.infer_freq(df_ws.index))}', drawstyle='steps-post')

    # Plot daily stats
    resample_df[col].resample('D').mean().plot(ax=ax, label='Daily Mean', color='black', alpha=0.7)
    resample_df[col].resample('D').min().plot(ax=ax, label='Daily Min', color='blue', alpha=0.7)
    resample_df[col].resample('D').max().plot(ax=ax, label='Daily Max', color='red', alpha=0.7)

    ax.set_ylim(0)
    ax.set_ylabel('Temperature [celcius]')
    ax.legend()
    ax.grid()
    ax.set_title(f'Temperature from {station}')
    fig.suptitle(f'{df_ms.index.date.min()} - {df_ms.index.date.max()}')

    if saveFig:
        path = '../Images/'
        plt.savefig(f'{path}temperature_{station}_{df_ms.index.date.min()}_{df_ms.index.date.max()}.png', dpi=600)
    # print('temp', df_ms)
    return df_ms
    # plt.show()  # Display the plot

def make_path(path, var):
    """
    Generate the path to the data file based on the variable type.
    
    Parameters:
    path (str): The base directory path.
    var (str): The variable type ('ws', 'wd', 't').
    
    Returns:
    str: The complete path to the data file.
    """
    
    # Find the path 
    if var not in ['ws', 't', 'wd']:
        raise ValueError('read_office_station function takes three vars to indicate the desired variable:\n {var} was passed, it should be:\n var: ws , explanation: wind speed\n var: wd , explanation: wind direction\n var: t , temperature')
        
    ls = os.listdir(path)
    
    if var == 'ws':
        ws_file = [file for file in ls if 'Windsnelheid' in file][0]
        return os.path.join(path, ws_file)
    
    if var == 'wd':
        wd_file = [file for file in ls if 'Windrichting' in file][0]
        return os.path.join(path, wd_file)
    
    if var == 't':
        t_file = [file for file in ls if 'Buitentemperatuur' in file][0]
        return os.path.join(path, t_file)


    
def read_office_station(rel_path, var, saveFig, station, resample_string, start_date, end_date):
    """
    Read and plot data from an office station.
    
    Parameters:
    rel_path (str): The relative path to the data directory.
    var (str): The variable type ('ws', 'wd', 't').
    saveFig (bool): Whether to save the figure.
    station (str): The station name.
    resample_string (str): The resampling frequency string.
    
    Returns:
    DataFrame: A DataFrame containing the resampled data.
    """
    
    path = make_path(rel_path, var)
    # print(path)
    
    date_parser = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M')
    df = pd.read_csv(path, parse_dates=True, date_parser=date_parser, index_col='Time') # Assuming the datetime column is the first column
    
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # print(df.index.min())
    
    
    if var == 'ws':
        df_copy['ms'] = df_copy['kmph_mean'] * 5/18
        df = wind_speed_plot(df_copy, 'ms', saveFig, station, resample_string, start_date, end_date)
        return df
        # return complete_df
    
    elif var == 'wd':
        df = wind_rose(df, 'degree_mean', saveFig, station, resample_string, start_date, end_date)
        return df
    
    elif var == 't':
        df = temp_plot(df, 'celcius_mean', saveFig, station, resample_string, start_date, end_date)
        return df

def combine_office_station(office_vars, saveFig, station, resample_string, start_date, end_date):
    """
    Combine multiple variables into a single DataFrame.
    
    Parameters:
    office_vars (list): List of variable types to combine ('ws', 'wd', 't').
    saveFig (bool): Whether to save the figure.
    station (str): The station name.
    resample_string (str): The resampling frequency string.
    
    Returns:
    DataFrame: A DataFrame containing the combined data.
    """
    
    dfs = []
    for var in office_vars:
        df = read_office_station(r"..\AeroData\OFFICE_station", var, saveFig, station, resample_string, start_date, end_date)
        dfs.append(df)
    
    combined = pd.concat(dfs, axis=1)
    return combined

def read_resample(base_folder,station, resample_string):
    """
    Read and resample data from an Excel file.
    
    Parameters:
    base_folder (str): The base directory path.
    station (str): The station name.
    resample_string (str): The resampling frequency string.
    
    Returns:
    DataFrame: A DataFrame containing the resampled data.
    """
    
    data = pd.read_excel(os.path.join(base_folder, station, f'combined_data_{station}.xlsx'))
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    resample = data[['windSpeed', 'outTemp', 'windDir']].resample(resample_string).mean()
    return resample

def plot_TGV_context(shapefile_path, gis_location, TGV_geotiff, TGV_tight_geotiff, desired_crs):
    
    #plot a geotiff with the wider view of The Green Village, for context
    
    geotiff_file_path = os.path.join(gis_location,TGV_geotiff)
    title = 'The Green Village at TU Delft'
    plotting = PlotLoc(shapefile_path, geotiff_file_path, title, desired_crs)
    plotting.plot_location()
    
    #plot a geotiff with a different, for tighter detail
    
    geotiff_file_path = os.path.join(gis_location, TGV_tight_geotiff)
    title = 'The Heat Square at TU Delft'
    plotting = PlotLoc(shapefile_path, geotiff_file_path, title, desired_crs)
    plotting.plot_location()