# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:54:49 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

import os
import re
import pandas as pd
from netCDF4 import Dataset

# Directory containing the .nc files
nc_directory = "H:\\My Drive\\Delft\\TUDELFT\\Additional Thesis\\AeroData\\EWI\\"

# Loop through all .nc files in the directory
for filename in os.listdir(nc_directory):
    if filename.endswith('.nc'):
        
        # Full path to the .nc file
        nc_file_path = os.path.join(nc_directory, filename)

        # Open the NetCDF file
        root = Dataset(nc_file_path, "r")

        # Extract the year and date from the filename
        match = re.search(r"(\d{4})(\d{2})(\d{2})_", filename)
        if match:
            year, month, day = match.groups()
            base_date = f"{year}-{month}-{day}"

        # Read the time variable and convert it to a pandas datetime index
        time_var = root.variables['time']
        time_data = time_var[:]
        
        # Create a time index starting from the base_date, assuming time_data is in seconds from midnight
        time_index = pd.to_datetime(time_data, unit='m', origin=pd.Timestamp(base_date))
        
        # Read wind and temperature variables
        wind_speed = root.variables['wind_speed'][:]
        wind_speed_of_gust = root.variables['wind_speed_of_gust'][:]
        wind_to_direction = root.variables['wind_to_direction'][:]
        wind_gust_from_direction = root.variables['wind_gust_from_direction'][:]
        in_air_temperature = root.variables['in_air_temperature'][:]
        out_air_temperature = root.variables['out_air_temperature'][:]
        
        print(wind_speed)
        
        # Close the NetCDF file
        # root.close()

        # Create a DataFrame
        df = pd.DataFrame({
            'timestamp':time_index,
            'windSpeed': wind_speed,
            'windGust': wind_speed_of_gust,
            'windDir': wind_to_direction,
            'windGustDir': wind_gust_from_direction,
            'inTemp': in_air_temperature,
            'outTemp': out_air_temperature
        })
        
        print(f"Day: {base_date}")
        # print(f"First timestamp: {time_index[0]}")
        # print(f"Mean of each column:")
        # print(df.head())
        print('wind speed non_zero:', np.count_nonzero(root.variables['wind_speed'][:]))

        # Save the DataFrame to a .csv file
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        csv_file_path = os.path.join(nc_directory, csv_filename)
        df.to_csv(csv_file_path)