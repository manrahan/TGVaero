# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:12:07 2023

@author: Mike O'Hanrahan (github: manrahan)
"""

import datetime
from osgeo import gdal
import numpy as np
from osgeo.gdalconst import *
from PyQt5.QtCore import QDate, QTime
import sys
import os
from qgis.core import QgsApplication



processing.run("umep:Urban Morphology: Morphometric Calculator (Grid)", {'INPUT_POLYGONLAYER':'H:/My Drive/Delft/TUDELFT/Additional Thesis/GIS/Vectors/Bounding_box_buffers_2.shp','ID_FIELD':'id','SEARCH_METHOD':0,'INPUT_DISTANCE':200,'INPUT_INTERVAL':10,'USE_DSM_BUILD':False,'INPUT_DSM':'H:/My Drive/Delft/TUDELFT/Additional Thesis/GIS/DSM/DSM_0point5_Infill_32631.sdat','INPUT_DEM':'H:/My Drive/Delft/TUDELFT/Additional Thesis/GIS/Images/DEM_AHN_0point5_fillnodataminus1.tif','INPUT_DSMBUILD':None,'ROUGH':0,'FILE_PREFIX':'RT_','IGNORE_NODATA':True,'ATTR_TABLE':False,'OUTPUT_DIR':'H:\\My Drive\\Delft\\TUDELFT\\Additional Thesis\\outputs\\UMEP_outputs\\10_deg_200m\\RT','CALC_SS':False,'SS_HEIGHTS':'','INPUT_CDSM':None})

qgis_process run umep:Urban Morphology: Morphometric Calculator (Grid) --distance_units=meters --area_units=m2 --ellipsoid=EPSG:7030 --INPUT_POLYGONLAYER='H:/My Drive/Delft/TUDELFT/Additional Thesis/GIS/Vectors/Bounding_box_buffers_2.shp' --ID_FIELD=id --SEARCH_METHOD=0 --INPUT_DISTANCE=200 --INPUT_INTERVAL=10 --USE_DSM_BUILD=false --INPUT_DSM='H:/My Drive/Delft/TUDELFT/Additional Thesis/GIS/DSM/DSM_0point5_Infill_32631.sdat' --INPUT_DEM='H:/My Drive/Delft/TUDELFT/Additional Thesis/GIS/Images/DEM_AHN_0point5_fillnodataminus1.tif' --ROUGH=0 --FILE_PREFIX=RT_ --IGNORE_NODATA=true --ATTR_TABLE=false --OUTPUT_DIR='H:\My Drive\Delft\TUDELFT\Additional Thesis\outputs\UMEP_outputs\10_deg_200m\RT' --CALC_SS=false --SS_HEIGHTS=