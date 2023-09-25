# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:34:33 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

import pandas as pd 
import numpy as np
from math import log as ln
from scipy.optimize import curve_fit
from math import log as ln


def read_anemom_data(filepath):
    """
    Reads anemometer data from an Excel file and extracts relevant time series data.
    
    Parameters:
    - filepath (str): The path to the Excel file to read.
    
    Returns:
    - tuple: A tuple containing the DataFrame and Series objects for the relevant data.
    """
    
    
    # Read the Excel file
    df = pd.read_excel(filepath, header=[0,1,2], sheet_name='Config 1', 
                       index_col=0)
    
    df = df.loc["2023-08-25 16:45:00":] #the date from which two anemometers became available
    
    # Extract relevant time series data
    U1 = df[('Port4', 'ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed')].dropna()
    U2 = df[('Port5', 'ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed')].dropna()
    
    Ta1 = df[('Port4', 'ATMOS 22 Ultrasonic Anemometer', ' °C Anemometer Temp')].dropna()
    Ta2 = df[('Port5', 'ATMOS 22 Ultrasonic Anemometer', ' °C Anemometer Temp')].dropna()
    
    Ts1 = df[('Port2', 'TEROS 12 Moisture/Temp/EC', ' °C Soil Temperature')].dropna()
    Ts2 = df[('Port3', 'TEROS 12 Moisture/Temp/EC', ' °C Soil Temperature')].dropna()
    
    print()
    # Hardcoded anemometer heights for port 4 and 5
    
    
    return df, U1, U2, Ta1, Ta2, Ts1, Ts2

# Calculate Ustar using the von Kármán constant (k), wind speeds U1 and U2 at heights Zu1 and Zu2
def calc_ustar(k: float, U1: pd.Series, U2: pd.Series, Zu1: float, Zu2: float) -> pd.Series:

    """
    Calculate the friction velocity Ustar using the von Kármán constant and wind speeds.
    
    Parameters:
    - k (float): The von Kármán constant.
    - U1, U2 (pd.Series): Wind speeds at two different heights.
    - Zu1, Zu2 (float): Heights of the anemometers.
    
    Returns:
    - pd.Series: A Series containing calculated Ustar value [m/s]]
    """
    
    Ustar = k * (U2 - U1) / ln((Zu2) / (Zu1))
    return np.abs(Ustar)



def log_wind_profile(z, u_star, z_0, z_d):
    kappa = 0.4  # von Kármán constant, you can define this elsewhere
    try:
        return (u_star / kappa) * np.log((z - z_d) / z_0)
    except Exception as e:
        # print(f"Error in log_wind_profile: {e}")
        return None

# Calculate roughness length (z0)
def calc_z0(Zu2: float, k: float, U2: pd.Series, Ustar: pd.Series) -> pd.Series:
    z0 = (Zu2) * np.exp(-k * (U2 / Ustar))
    return z0

def calc_zd(Zu1: float, Zu2: float, k: float, U1: pd.Series, U2: pd.Series, Ustar: pd.Series) -> pd.Series:
    zd = (Zu1) - (Zu2-Zu1) / (np.exp(k * (U2-U1) / Ustar)-1)
    return zd

def fit_from_est(u_star_est, z0_est):
    '''
    function not useful unless there is a third ISL value...
    '''
    # Define the von Kármán constant
    kappa = 0.4

    # Sample data: Replace this DataFrame with your actual data
    data = {'z': [1.35, 2.07, 99], 'U': [0.5, 0.6, 1.5]}
    df = pd.DataFrame(data)

    # Calculate z_0
    z1, z2, z3 = df['z']
    u1, u2, z4 = df['U']

    # Assume u_star is known, replace this value with your actual u_star
    u_star = kappa *(u2-u1)/ln(z2/z1)

    z_0_option1 = z1 * np.exp(-(kappa*u1)/u_star)
    z_0_option2 = z2 * np.exp(-(kappa*u2)/u_star)

    # Use either z_0_option1 or z_0_option2 based on your preference or more reliable U
    print(f"Estimated U* using z1: {u_star}")
    print(f"Estimated z_0 using z1: {z_0_option1}")
    print(f"Estimated ustar using z2: {z_0_option2}")

    # Initial guesses for u_star, z_0, and z_d
    initial_guesses = [u_star_est, z_0_option1, 1]

    # Perform curve fitting
    params, params_covariance = curve_fit(log_wind_profile, df['z'], df['U'], p0=initial_guesses)

    # Extract fitted parameters
    u_star_fitted, z_0_fitted, z_d_fitted = params

    print(f"Fitted u_star: {u_star_fitted}")
    print(f"Fitted z_0: {z_0_fitted}")
    print(f"Fitted z_d: {z_d_fitted}")
    
    return u_star_fitted, z_0_fitted, z_d_fitted

def add_aero_cols(df, k, Zu1, Zu2): #(df, k, U1, U2, Zu1, Zu2)
    # Calculate U1, U2 from df
    U1 = df[('Port4', 'ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed')]
    U2 = df[('Port5', 'ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed')]
    
    # Calculate Ustar, zd, and z0 using the previously defined functions
    Ustar = calc_ustar(k, U1, U2, Zu1/100, Zu2/100)
    z0 = calc_z0(Zu2/100, k, U2, Ustar)
    zd = calc_zd(Zu1/100, Zu2/100, k, U1, U2, Ustar)
    
    # Add calculated columns to the DataFrame
    df[('calc', 'v1', 'USTAR')] = Ustar
    df[('calc', 'v1', 'z0')] = z0
    df[('calc', 'v1', 'zd')] = zd
    
    # df[('calc', 'v1', 'USTAR_fit')] = Ustar
    # df[('calc', 'v1', 'z0_fit')] = z0
    # df[('calc', 'v1', 'zd_fit')] = zd
    return df


def calculate_SHF_and_Tpot(wind_speed, air_temp, soil_temp, anemometer_height, transfer_coefficient):
    #(U1, Ta1, Ts1, Zu1, transfer_coefficient)
    # Constants
    rho = 1.225  # Air density in kg/m^3
    cp = 1005  # Specific heat of air at constant pressure in J/(kg*K)
    g = 9.81  # Acceleration due to gravity in m/s^2
    
    # Convert to Kelvin
    air_temp_k = air_temp + 273.15
    soil_temp_k = soil_temp + 273.15
    
    # Calculate sensible heat flux (H)
    H = rho * cp * transfer_coefficient * wind_speed * (soil_temp_k - air_temp_k)
    
    # Calculate surface potential temperature (theta_0)
    theta_0 = air_temp_k + (g / cp) * (anemometer_height)  # Height is converted from cm to m
    
    return H, theta_0


def calculate_stability(z, rho, cp, df, zd_key=('calc', 'v1', 'zd')): #
    g = 9.81  # acceleration due to gravity (m/s^2)
    k = 0.4  # von Karman constant
    
    zd = df[zd_key]
    theta_0 = df[('calc', 'v1', 'theta_0')]
    H = df[('calc', 'v1', 'H')]
    u_star = df[('calc', 'v1', 'USTAR')]
    
    
    
    # Initialize empty series to store results
    stability_param = pd.Series(index=u_star.index, dtype='float64')
    stability = pd.Series(index=u_star.index, dtype='object')
    
    z_prime = z - zd
    
    L = -((k * (z - zd) * g * theta_0 * H) / (rho * cp * u_star ** 3))
    
    stability_param = z_prime / L
    
    stability[stability_param > 0] = "Stable"
    stability[stability_param < 0] = "Unstable"
    stability[(stability_param >= -0.001) & (stability_param <= 0.001)] = "Neutral"

    # Add to DataFrame
    df[('calc', 'v1', 'stability_param')] = stability_param
    df[('calc', 'v1', 'stability')] = stability
    df[('calc', 'v1', 'L')] = L
    
    return df, stability_param, stability

def potential_temperature(T, P):
    P0 = 1000 #hPa
    Rd = 287  # J/(kg·K)
    Cp = 1005  # J/(kg·K)
    
    return T * (P0 / P) ** (Rd / Cp)

def calculate_RI_methods(Ri, method):
    if method == "RI-Grad":
        if Ri >= 0:
            z_over_L = (Ri / (1 - 5 * Ri))
        else:
            z_over_L = Ri
    elif method == "RI-Bulk":
        if Ri >= 0:
            z_over_L = (10 * Ri / (1 - 5 * Ri))
        else:
            z_over_L = 10 * Ri
    else:
        raise ValueError("Invalid method. Choose either 'RI-Grad' or 'RI-Bulk'")
    return z_over_L

def calculate_psi(z_over_L, variable):
    # For stable conditions (z_over_L >= 0)
    if z_over_L >= 0:
        if variable == 'm':
            return -5 * z_over_L
        elif variable == 'h':
            return -5 * z_over_L
        else:
            raise ValueError("Invalid variable. Choose either 'm' for momentum or 'h' for heat.")
    
    # For unstable conditions (z_over_L < 0)
    else:
        x = (1 - 16 * np.abs(z_over_L)) ** 0.25
        if variable == 'm':
            return 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
        elif variable == 'h':
            return np.log((1 + x ** 2) / 2)
        else:
            raise ValueError("Invalid variable. Choose either 'm' for momentum or 'h' for heat.")

def add_psi_to_dataframe(df, z_over_L_column):
    """
    Add Psi values for momentum and heat to a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to add Psi values to.
    - z_over_L_column (str): The column name that contains z_over_L values.
    
    Returns:
    - pd.DataFrame: The updated DataFrame with added Psi columns.
    """
    
    # Calculate Psi for momentum and heat and add to DataFrame
    df[('calc', 'Psi', 'momentum')] = df[z_over_L_column].apply(lambda x: calculate_psi(x, 'm'))
    df[('calc', 'Psi', 'heat')] = df[z_over_L_column].apply(lambda x: calculate_psi(x, 'h'))
    return df


def iterate_u_theta_star_for_row(row, zu, zt, max_iter=100, tol=1e-3):
    """
    Iteratively calculate u_star and theta_star for a single row in a DataFrame.
    
    Parameters:
    - row (pd.Series): A row from a DataFrame.
    - zu, zt (float): Heights for the momentum and temperature measurements.
    - max_iter (int, optional): Maximum number of iterations for the algorithm. Defaults to 100.
    - tol (float, optional): Tolerance for the convergence. Defaults to 1e-3.
    
    Returns:
    - tuple: A tuple containing u_star, theta_star, tau, and H.
    """
    
    # Extract values from row
    du = row[('calc', 'v2', 'd_U')]
    dt = row[('calc', 'v2', 'dTheta_v')]
    z0 = row[('calc', 'v1', 'z0')]
    z0h = row[('calc', 'v1', 'z0')]

    # Initialize u_star and theta_star with initial estimates
    K = 0.4  # von Karman constant

    # print(f'du: {du}, zu: {zu}, z0: {z0}, K: {K}')

    # If any term is zero, return np.nan for all values
    if du==0 and dt==0 and z0==0 or z0h==0:
        return np.nan, np.nan, np.nan, np.nan

    u_star = du / (np.log(zu / z0) / K)
    theta_star = dt / (np.log(zt / z0h) / K)

    for _ in range(max_iter):
        # Calculate L using the row's data
        Ri = row[('calc', 'v2', 'Ri')]
        L = - (u_star ** 3) / (K * 9.81 * theta_star)

        # Compute Psi-functions
        z_over_L = calculate_RI_methods(Ri, 'RI-Grad')  # or 'RI-Bulk'
        psi_m = calculate_psi(z_over_L, 'm')
        psi_h = calculate_psi(z_over_L, 'h')

        # Compute new u_star and theta_star
        new_u_star = du / (np.log(zu / z0) - psi_m) * K
        new_theta_star = dt / (np.log(zt / z0h) - psi_h) * K

        # Check for convergence
        if np.abs(new_u_star - u_star) < tol and np.abs(new_theta_star - theta_star) < tol:
            break

        u_star, theta_star = new_u_star, new_theta_star

    # Constants
    rho = 1.225  # Air density in kg/m^3 (approximate, may vary with altitude, temperature, etc.)
    c_p = 1005  # Specific heat capacity of air at constant pressure in J/(kg*K)

    # Calculate momentum flux (tau) and sensible heat flux (H)
    tau = rho * (u_star ** 2)
    H = -rho * c_p * theta_star * u_star

    return u_star, theta_star, tau, H



def apply_iteration_to_dataframe(df, zu, zt):
    """
    Apply the iterate_u_theta_star_for_row function to each row of the DataFrame and 
    add calculated u_star, theta_star, tau, and H to new DataFrame columns.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing data to which the iteration will be applied.
    - zu, zt (float): Heights for the momentum and temperature measurements.
    
    Returns:
    - pd.DataFrame: The updated DataFrame with added calculated columns.
    """
    
    results = df.apply(lambda row: iterate_u_theta_star_for_row(row, zu, zt), axis=1)
    u_stars, theta_stars, taus, Hs = zip(*results)
    
    df[('calc', 'v2',  'u_star')] = u_stars
    df[('calc', 'v2', 'theta_star')] = theta_stars
    df[('calc', 'v2', 'tau')] = taus
    df[('calc', 'v2',  'H')] = Hs
    
    return df


def calculate_Ri(Zu1, Zu2, df):
    """
    Calculate Richardson Number (Ri) using heights, wind speeds, and temperatures
    from two anemometers. Add calculated values to new DataFrame columns and save
    the DataFrame to an Excel file.
    
    Parameters:
    - Zu1, Zu2 (float): Heights of the anemometers in centimeters.
    - df (pd.DataFrame): DataFrame containing the raw data.
    
    Returns:
    - pd.DataFrame: The updated DataFrame with added calculated columns, including Ri.
    """
    
    g = 9.81  # Acceleration due to gravity in m/s^2
    d_z = (Zu2 * 1e-2) - (Zu1 * 1e-2)  # Convert cm to m
    P = df[('Port8', 'Barometer', ' kPa Reference Pressure')] * 10  # kPa to hPa
    
    Theta_v_a1 = potential_temperature(df[('Port4', 'ATMOS 22 Ultrasonic Anemometer', ' °C Anemometer Temp')], P)
    Theta_v_a2 = potential_temperature(df[('Port5', 'ATMOS 22 Ultrasonic Anemometer', ' °C Anemometer Temp')], P)
    
    dTheta_v = Theta_v_a2 - Theta_v_a1
    Theta_av = (Theta_v_a1 + Theta_v_a2) / 2
    
    U_1 = df[('Port4', 'ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed')]
    U_2 = df[('Port5', 'ATMOS 22 Ultrasonic Anemometer', ' m/s Wind Speed')]
    
    d_U = U_2 - U_1
    
    Ri = (g * dTheta_v * d_z) / (Theta_av * d_U ** 2)
    
    # Adding calculated values to the DataFrame
    df[('calc', 'v2', 'Theta_v_a1')] = Theta_v_a1
    df[('calc', 'v2', 'Theta_v_a2')] = Theta_v_a2
    df[('calc', 'v2', 'dTheta_v')] = dTheta_v
    df[('calc', 'v2', 'Theta_av')] = Theta_av
    df[('calc', 'v2', 'd_U')] = d_U
    df[('calc', 'v2', 'Ri')] = Ri
    df[('calc', 'v2', 'z_over_L_RI-Grad')] = df[('calc', 'v2', 'Ri')].apply(lambda x: calculate_RI_methods(x, "RI-Grad"))
    df[('calc', 'v2', 'z_over_L_RI-Bulk')] = df[('calc', 'v2', 'Ri')].apply(lambda x: calculate_RI_methods(x, "RI-Bulk"))
    df = apply_iteration_to_dataframe(df, Zu2*1e-2, Zu2*1e-2)
    
    df.to_excel(r'../Anemometer_results_v2.xlsx')
    
    return df








 




