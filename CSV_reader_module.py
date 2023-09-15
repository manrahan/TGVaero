# -*- coding: utf-8 -*-
"""
Created on Fri Aug 01 15:14:00 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

import os
import pandas as pd

class CSVReader:
    
    def __init__(self, base_folder, output_path):
        """
        Initialize the CSVReader object.

        Parameters:
        base_folder (str): The folder containing the anemometer data.
        output_path (str): The path where output files will be saved.
        """
        self.base_folder = base_folder #the folder with the anemometer data
        self.output_path = output_path

    def get_folder_path(self, station_number):
        """
        Get the path of the folder corresponding to a station number.

        Parameters:
        station_number (str): The station number.

        Returns:
        str: The complete path to the folder.
        """
        
        folders = os.listdir(self.base_folder + '/ATMOS_stations')
        
        matching_folder = None
        
        for folder in folders:
            if station_number in folder:
                matching_folder = folder
                break
        
        matching_folder = os.path.join(self.base_folder, matching_folder)

        return matching_folder


    def list_files(self, station_number):
        """
        List the files in the folder of a given station number.

        Parameters:
        station_number (str): The station number.

        Returns:
        list: List of files in the folder or a message if folder is not found.
        """
        
        station_folder = self.get_folder_path(station_number)
        if os.path.exists(station_folder):
            files = os.listdir(station_folder)
            return files
        else:
            return "Station folder not found."
    
    def read_csv(self, station_number, raw):
        """
        Read and process the CSV files for a given station.

        Parameters:
        station_number (str): The station number.
        raw (bool): Whether to read raw files or not.

        Returns:
        DataFrame, DataFrame: A combined DataFrame containing the station data and metadata.
        """
        
        station_folder = self.get_folder_path(station_number)
        files = self.list_files(station_number)
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        
        
        if raw == False:
            config_files = [file for file in files if "Configuration" in file and "Raw" not in file]
            meta_files = [file for file in files if "Metadata" in file]
            
            data_frames = []
            
            output_file_path = r'..\outputs\combined_df_'+station_number+'.xlsx'
            
            if not os.path.exists(output_file_path):
                
                for config_file in config_files:
                    
                    file_path = os.path.join(station_folder, config_file)
                    
                    
                    if os.path.exists(file_path):
                        
                        try:
                            df = pd.read_csv(file_path, header=[1, 2], index_col=0, parse_dates=[0])
                            
                            if df.shape[1] > 20: #some files are miniscule and are not appended, presumably setup files
                                # print(df.shape)
                                data_frames.append(df)
                                
                        except Exception as e:
                            print(f"Error reading CSV {config_file}: {str(e)}")
                            
                    else:
                        print(f"File not found: {config_file}")
            else:
                None
                
            meta_path = os.path.join(station_folder, meta_files[0])
            meta_file = r'..\outputs\metadata_'+station_number+'.xlsx'
            
            
            if not os.path.exists(meta_file):
                try:
                    meta = pd.read_csv(meta_path)
                    meta.to_excel(r'..\outputs\metadata_'+station_number+'.xlsx')
                    print(f'Created Meta File for SN: {station_number}')
        
                except Exception as e:
                    print(f"Error reading Metadata CSV {meta_files[0]}: {str(e)}")
            
            if len(data_frames)>0:
                # Concatenate DataFrames
                combined_df = pd.concat(data_frames, axis=0, join='outer')
    
                combined_df.to_excel(r'..\outputs\combined_df_'+station_number+'.xlsx')
                
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                print(f"Combined DataFrame {station_number}:")
                
                return combined_df, meta
    
            else:
                print("No DataFrames to combine.")
        else:
            # Implement the case when raw is True
            pass
        
        