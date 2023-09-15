# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 23:08:31 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Initialize station names and corresponding folders
station = ['TUD_CITG', 'TUD_EWI_Roof']
folder = ['CITG', 'EWI']

# Set the base URL for scraping
base_url = f"https://ruisdael.citg.tudelft.nl/davis_weather_station/{station[1]}/"

start_date = datetime.strptime("20230910", "%Y%m%d")

# Set the folder where you want to save the downloaded files
# Uncomment the folder you want to use
# save_folder = "..\\Aerodata\\EWI"
save_folder = "..\\Aerodata\\CITG"

# Create the save folder if it doesn't already exist
os.makedirs(save_folder, exist_ok=True)

# Send a GET request to the base URL
response = requests.get(base_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Loop through all the anchor tags (<a>) in the HTML
    for link in soup.find_all("a"):
        href = link.get("href")
        # If the link ends with '/', it's a folder
        if href.endswith("/"):
            folder_url = urljoin(base_url, href)
            folder_response = requests.get(folder_url)
            
            # Check if the request to access the folder was successful
            if folder_response.status_code == 200:
                folder_soup = BeautifulSoup(folder_response.content, "html.parser")
                
                # Loop through all the anchor tags in the folder
                for file_link in folder_soup.find_all("a"):
                    file_href = file_link.get("href")
                    
                   # Check if href exists
                    if file_href:
                
                        # Check for either '.csv' or '.nc' file extensions
                        if file_href.endswith(".csv") or file_href.endswith(".nc"):
                            # Extract date from the filename (assuming it's formatted like your example)
                            file_date_str = file_href.split("/")[-1][:8]  # This extracts the 'YYYYMMDD' part
                            file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    
                            # Check if the file date is after the start date
                            if file_date >= start_date:
                                file_url = urljoin(folder_url, file_href)
                                filename = os.path.basename(file_href)
                                save_path = os.path.join(save_folder, filename)
                    
                                # Download the file
                                file_response = requests.get(file_url)
                                if file_response.status_code == 200:
                                    # Save the downloaded content to a local file
                                    with open(save_path, "wb") as f:
                                        f.write(file_response.content)
                                    print(f"File '{filename}' downloaded and saved to '{save_path}'.")
                                else:
                                    print(f"Failed to download file '{filename}'.")
                    else:
                        print(f"Encountered a link without a href attribute. This might be the most recent file or an irregular link.")

                        
    print("Download completed.")
else:
    print("Failed to access the base URL.")