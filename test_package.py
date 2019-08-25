from model.DeepSatellite import Satellite
from datetime import datetime, timedelta
import pandas as pd
import json
import ee

# Definition of the time window to download the images
now = datetime.now() 
delta = timedelta(weeks=8)
past = now - delta
date_time = now.strftime("%Y-%m-%d")
past_time = past.strftime("%Y-%m-%d")

# Call the Copernicus Sentinel-2 L2A Imagery and set space resolution to 100 m. Alse select the desired bands.
sentinel_esa = Satellite("COPERNICUS/S2_SR", 100, ['B4','B5','B8'])

# Load the shapefile of an area of interest. The function also accept GeoJson format.
x, y = sentinel_esa.load_feature('predio.shp')

# Computation of NDVI index for vegetation analysis
sentinel_esa.ndvi_computation()

# Add the NDVI band to the Band_names property of the object
sentinel_esa.band_names=sentinel_esa.collection.first().bandNames().getInfo()

# Process a collection of images with the given parameters in the time window
collection_object = sentinel_esa.images(past_time, date_time)

# Get the data from the Google Earth Engine repository and convert it to a dictionary, it can also be downloaded to a pandas dataframe 
dic = sentinel_esa.get_data(collection_object)

# Transform the dictionary to numpy batch of images, the size of numpy array: n_image, x_size, y_size, n_bands
images, transform = sentinel_esa.data_to_images(dic)

# Save every image as a GeoTiff Raster format in local folder with prefix sentinel_images, also allows to upload them to google storage system
sentinel_esa.rasterize_images('sentinel_images', images, transform)

# Transform the dictionary of multiple images downloaded into a pix2pix lon/lat/bands Pandas Dataframe
pd_pix=sentinel_esa.data_pix2pix(dic)

# Transform Dataframe to Numpy Arrays and create a binary classification output variable for the Machine Learning Classification Model
x = pd_pix[['B4','B5','B8']].values
y = pd_pix['NDVI'].values
y = y>0.4

# Load Raster parameters
raster = sentinel_esa.load_raster('rasters/')

print(raster.shape)

# Train the Classifier based on the Satellite data, there is Random Forest, Support Vector Machine and KNN available 
sentinel_esa.train_classifier(x, y, 'random-forest-model', model='rf', do_cv=True, n_jobs=1)

