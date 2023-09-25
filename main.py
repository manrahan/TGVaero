# -*- coding: utf-8 -*-
"""
Created on Fri Aug 01 15:14:00 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

from CSV_reader_module import CSVReader

from station_data import (StationData, 
                          PlotLoc, 
                          CreateShapefile, 
                          DataSummary,
                          wind_rose,
                          read_office_station,
                          wind_speed_plot,
                          temp_plot,
                          combine_office_station,
                          read_resample,
                          plot_TGV_context
                          )

from analysis import (correlation_matrix,
                      error_matrix,
                      covariance_matrix,
                      plot_matrix,
                      read_anisotropic,
                      plot_aniso_method_comparison,
                      plot_wind_profile
                      )

from AeroCalculation import (read_anemom_data,
                             add_aero_cols,
                             calculate_SHF_and_Tpot,
                             calculate_stability,
                             calculate_Ri
                             )
import os
import pandas as pd


# =============================================================================
# Folder structure is kept relative and should work with zipped files
# =============================================================================

base_folder = r"..\AeroData"
output_path = r'../outputs'
summary_path = r'..\outputs\summary'

start_date = '2023-04-12'
end_date = '2023-08-31'

# =============================================================================
# GIS parameters, zipped files should work here
# =============================================================================

gis_location = r'..\GIS\Images'

#TODO: whys is it green
TGV_geotiff = r'AerialScreenshot_modified_32631.tif'

TGV_tight_geotiff = r'TighterAerialScreenshot_32631_modified.tif'

desired_crs = 'EPSG:32631' # UTM31N Netherlands (accurate to 2m)
original_crs = 'EPSG:4326' #from the metadata

instrument = 'ATMOS 22 Ultrasonic Anemometer'

instrument_short = 'anemometer'

instrument_vars = [' m/s Wind Speed', 
                   ' m/s Gust Speed', 
                   '° Wind Direction',
                   ' °C Anemometer Temp',
                   '° X-axis Level',
                    '° Y-axis Level']

station_numbers = ["z6-20594", "z6-20595", "z6-20596", "z6-20597", "z6-21086"]  # List of station numbers

office_vars = ['ws', 'wd', 't'] #wind speed, wind direction, temperature for TGV rooftop

# =============================================================================
# The RESAMPLE STRING is the frequency at which the data is averaged, EWI for
# example with a '15T' will go from 1 min freq, to 15m avg and the 15m ATMOS 
# remains unchanged
# =============================================================================

resample_string = '15T'  #frequency of resample of data, original in 15 mins, 
                            # aggregates to mean min max std var count nas and zeros



# =============================================================================
# Morphometric Analysis Parameters
# =============================================================================

morph_methods = ['RT', 'MHO', 'KAN', 'MAC']     #The column references for the methods calculated
var_list = ['Wd', 'pai', 'fai', 'zH', 'zHmax', 'zHstd', 'zd', 'z0', 'noOfPixels']
len_list = ['zH', 'zHmax', 'zHstd', 'zd', 'z0']     #TODO: remind myself what these were for
length_scale = 0.001    #length scale
# logger_poi = [20594.0,20595.0,20596.0,20597.0,21086.0]  #stations of interest plotted
logger_and_grid =[3.0, 4.0, 11.0, 12.0, 15.0, 20.0, 21.0, 22.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 59.0, 60.0, 61.0, 62.0, 63.0, 70.0, 71.0, 72.0, 20594.0, 20595.0, 20596.0, 20597.0, 21086.0,]
# =============================================================================
# Anemometric Analysis Parameters
# =============================================================================

k = 0.4  # von Karman constant
transfer_coefficient = 0.0012  # as an example
rho = 1.225  # Air density in kg/m^3
cp = 1005  # Specific heat of air at constant pressure in J/(kg*K)
Zu1 = 135 # port 4 in cm
Zu2 = 207 # port 5 in cm
Zs1 = -5  # port 2 in cm
Zs2 = -10 # port 3 in cm

# TODO: code for raw values

raw = False

    
if __name__ == "__main__":
    csv_reader = CSVReader(base_folder, output_path)
    station_data_list = []
    
    for station_number in station_numbers:
        output_file_path = os.path.join(output_path, f'combined_df_{station_number}.xlsx')
          
        if not os.path.exists(output_file_path):
            os.makedirs(output_path, exist_ok=True)
            
            data_frame, meta = csv_reader.read_csv(station_number, raw)
            
            station_summary_file = os.path.join(output_path,'summary', station_number, f'station_summary_{station_number}.xlsx')
            
            if not os.path.exists(station_summary_file):
                summary_folder = os.path.join(output_path, 'summary', station_number)
                
                if not os.path.exists(summary_folder):    
                    os.makedirs(summary_folder)
                
                else:
                    print(summary_folder, 'exists...')
                    None 
                
                data_frame.describe().to_excel(station_summary_file)
                
                print(f'created station summary: {station_summary_file}')
            
            else:
                None
            
            print(f'created: {output_path}')
        
        else:
            None
            
        instrument_summary = DataSummary(start_date, end_date, station_number, output_path, summary_path)
        instrument_summary.data_summary(instrument, instrument_vars, resample_string)
    
    # # Check if the shapefile already exists before creating it
    # shapefile_path = os.path.join(output_path, f'station_locations_{str(desired_crs)[-5:]}.shp')   #The metadata GPS fixes are wrong
    
    shapefile_path = os.path.join(output_path, f'station_locations_{str(desired_crs)[-5:]}_georeferenced.shp')   #station_locations_32631_georeferenced
    
    if not os.path.exists(shapefile_path):
        station_list = []
        
        for station_number in station_numbers:
            print(station_number)
            station_data = StationData(station_number, output_path)
            print('station_data_success', station_number)
            station_list.append((str(station_data.device_name[0]),
                                  float(station_data.coordinates['Latitude'][0]),
                                  float(station_data.coordinates['Longitude'][0])))
        
        shapefile_creator = CreateShapefile(station_list, output_path, original_crs, desired_crs)
        shapefile_creator.create_shapefile()
        
    else:
        print(f'Using .shp from: {shapefile_path}')
    
    
    
    plot_TGV_context(shapefile_path, gis_location, TGV_geotiff, TGV_tight_geotiff, desired_crs)

    
    combined_df = combine_office_station(office_vars, True, 'TGV Office Station', resample_string, start_date, end_date) #True saveFigs saves the figs 
    
    ws_data = {}
    wd_data = {}
    t_data = {}
    
    for sn in station_numbers:
        files = os.listdir(output_path)
    
        # Check if any file contains the station number and 'combined' in the name
        matching_files = [file for file in files if str(sn) in file and file.startswith('combined')]
    
        if matching_files:
            
            
            for file_name in matching_files:
                # Construct the full path to the file
                file_path = os.path.join(output_path, file_name)  # Use output_path here instead of stem
                # print(file_path)
                df_15 = pd.read_excel(file_path, index_col=0, header=[0, 1], parse_dates=True)
                df = df_15.resample('15T').mean()
                
                wind_speed_plot(df, ('ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed'), 
                                True, sn, resample_string, start_date, end_date)
                temp_plot(df, ('ATMOS 22 Ultrasonic Anemometer', ' °C Anemometer Temp'), 
                          True, sn, resample_string, start_date, end_date)
                wind_rose(df, ('ATMOS 22 Ultrasonic Anemometer', '° Wind Direction'), 
                          True, sn, resample_string, start_date, end_date)
                ws_data[sn] = df[('ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed')]#.resample(resample_string).mean()
                wd_data[sn] = df[('ATMOS 22 Ultrasonic Anemometer', '° Wind Direction')]#.resample(resample_string).mean()
                t_data[sn] = df[('ATMOS 22 Ultrasonic Anemometer', ' °C Anemometer Temp')]#.resample(resample_string).mean()
    
    
    # ws_data['TGV_office'] = combined_df['ms']  # Use ['ms'] to access the 'ms' column
    # wd_data['TGV_office'] = combined_df['degree_mean']  
    # t_data['TGV_office'] = combined_df['celcius_mean']  
    
    
    # citg = read_resample(base_folder, 'CITG', resample_string)
    
    ewi = read_resample(base_folder, 'EWI', resample_string)
    
    wind_speed_plot(ewi, 'windSpeed', 
                    True, sn, resample_string, start_date, end_date)
    temp_plot(ewi, 'outTemp',
              True, sn, resample_string, start_date, end_date)
    wind_rose(ewi, 'windDir', 
              True, sn, resample_string, start_date, end_date)
    
    ws_data['EWI'] = ewi['windSpeed']  # Use ['ms'] to access the 'ms' column
    wd_data['EWI'] = ewi['windDir']  
    t_data['EWI'] = ewi['outTemp'] 
    
        
    plot_matrix('Correlation', correlation_matrix(ws_data), resample_string, 'Wind Speed', '$m/s$')
    plot_matrix('Correlation', correlation_matrix(wd_data), resample_string, 'Wind Direction', '$°$')
    plot_matrix('Correlation', correlation_matrix(t_data), resample_string, 'Temperature', '$°C$')
    
    
    plot_matrix('Error', error_matrix(ws_data), resample_string, 'Wind Speed', '$m/s$')
    plot_matrix('Error', error_matrix(wd_data), resample_string, 'Wind Direction', '$°$')
    plot_matrix('Error', error_matrix(t_data), resample_string, 'Temperature', '$°C$')
    
    
    filepath = os.path.join(base_folder, "ATMOS_stations\z6-20596(z6-20596)-1691673946\z6-20596(z6-20596)-1694181960.xlsx")
    
    df, U1, U2, Ta1, Ta2, Ts1, Ts2 = read_anemom_data(filepath)
    
    df = add_aero_cols(df, k, Zu1, Zu2)
    
    method_res = {}
    
    # Example usage:
    for method in morph_methods:
        
        aniso, poi = read_anisotropic(r'../outputs/UMEP_outputs_2/', method)
        method_res[method] = aniso
    
    
    # Iterate through each method to add new columns
    for method, df_method in method_res.items():
        new_column_zd = ('morph', method, 'zd_20596')
        new_column_z0 = ('morph', method, 'z0_20596')
        df[new_column_zd] = np.nan  # Initialize the new column with NaNs
        df[new_column_z0] = np.nan  # Initialize the new column with NaNs
        
        # Loop through each row in the main DataFrame
        for index, row in df.iterrows():
            wind_dir = row[('Port5', 'ATMOS 22 Ultrasonic Anemometer', '° Wind Direction')]
            
            # Find the closest matching wind direction in df_method
            closest_wind_dir = df_method['Wd_20596.0'].iloc[(df_method['Wd_20596.0'] - wind_dir).abs().idxmin()]
            
            # Extract the corresponding zd_20596 and z0_20596 values
            zd_value = df_method.loc[df_method['Wd_20596.0'] == closest_wind_dir, 'zd_20596.0'].values[0]
            z0_value = df_method.loc[df_method['Wd_20596.0'] == closest_wind_dir, 'z0_20596.0'].values[0]
    
            # Check for infinite values and replace them with NaN if necessary
            zd_value = np.nan if np.isinf(zd_value) else zd_value / 1000
            z0_value = np.nan if np.isinf(z0_value) else z0_value / 1000
    
            # Populate the new columns in the main DataFrame
            df.at[index, new_column_zd] = zd_value
            df.at[index, new_column_z0] = z0_value
    
    H, theta_0 = calculate_SHF_and_Tpot(U1, Ta1, Ts1, Zu1, transfer_coefficient)
    
    # # Add these to the DataFrame
    df[('calc', 'v1', 'H')] = H
    df[('calc', 'v1', 'theta_0')] = theta_0
    
    df, stability_param, stability = calculate_stability(Zu1/100, rho, cp, df, zd_key=('morph', 'RT', 'zd_20596'))
    
    df = calculate_Ri(Zu1, Zu2, df)
    
    res = plot_aniso_method_comparison(r'../outputs/UMEP_outputs_2/', ['RT', 'MHO', 'KAN', 'MAC'], ['zd'], length_scale, True,  logger_and_grid, df, 68)
    
    df['EWI', 'add', 'windSpeed'] = ws_data['EWI']
    
    plot_wind_profile(df)
    
    # Calculating potential temperature and sensible heat flux
    
    
   
    
    
    
    
    
    

    



