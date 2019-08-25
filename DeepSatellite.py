import numpy as np
import pandas as pd
import geopandas as gpd
import ee
import json
from google.cloud import storage
from osgeo import gdal, ogr, osr
from datetime import datetime, timedelta
import fiona
import os
from google.cloud import storage
import collections
import sklearn
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.externals import joblib
import sqlalchemy
import firebase_admin
from firebase_admin import credentials, firestore


class Satellite:

	def __init__(self, dataset, scale, band_names):
		self.dataset = dataset
		self.scale = scale
		self.band_names = band_names
		ee.Initialize()

	def init_storage(self, file_json, bucket_name):
		self.client = storage.Client.from_service_account_json(file_json)
		self.bucket = self.client.get_bucket(bucket_name)

	def init_cloudsql(self, db_user, db_pass, db_name, cloud_sql_connection_name):
		self.engine = sqlalchemy.create_engine(sqlalchemy.engine.url.URL(drivername='mysql+pymysql', username=db_user, password=db_pass, database=db_name, 
        query={'unix_socket':'/cloudsql/{}'.format(cloud_sql_connection_name)}), pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800,)

	def init_firestore(self, file_json):
		cred = credentials.Certificate(file_json)
		firebase_admin.initialize_app(cred)
		self.db = firestore.client()

	def load_feature(self, geometry_file):

		if type(geometry_file) is not dict:
			df = gpd.read_file(geometry_file)
			df = df['geometry'].to_crs(epsg=4326)
			geometry_file = json.loads(df.to_json())
			geometry_file = geometry_file['features'][0]['geometry']

		geom_type = geometry_file["type"]

		if geom_type == 'Polygon':
			self.geometry = ee.Geometry.Polygon(geometry_file["coordinates"])
		else:
			self.geometry = ee.Geometry.MultiPolygon(geometry_file["coordinates"])

		self.x = self.geometry.centroid().getInfo()['coordinates'][0]
		self.y = self.geometry.centroid().getInfo()['coordinates'][1]

		self.collection = ee.ImageCollection(self.dataset).filterBounds(self.geometry).select(self.band_names)

		return self.x, self.y


	def geofile_to_image(self, file, iterate_field, class_field=None):

		df = gpd.read_file(file)		
		df['geometry'] = df['geometry'].to_crs(epsg=4326)		
		
		if not class_field:
			class_field = iterate_field

		unique_class_field = df[class_field].unique().tolist()
		replace_class_field = range(0,len(unique_class_field))

		df = df.replace(to_replace = { class_field : unique_class_field},  value = replace_class_field) 
		image_list = ee.List([])
			
		for value in df[iterate_field].unique().tolist():

			features = json.loads(df[df[iterate_field]==value].to_json())
			feature_collection = ee.FeatureCollection(features)
			geometry = feature_collection.geometry()
			image = ee.Image().byte()
			image = empty.paint(featureCollection = feature_collection, color = str(class_field)).clip(geometry)
			image.set({'system:time_start': value})
			image_list = ee.List(image_list).add(image)

		return ee.ImageCollection(image_list)

	def load_raster(self, path, output='image'):

		image_arrays = []
		files = [path]

		if os.path.isdir(path):
			files = [path+f for f in os.listdir(path) if f.endswith('.tif')]

		for file in files:

			raster = gdal.Open(file)
			count = raster.RasterCount
			band_arrays = []

			for i in range(1, count+1):
				band_arrays.append(raster.GetRasterBand(i).ReadAsArray().astype(np.float))

			image_arrays.append(np.stack(band_arrays, axis= -1))

		return np.stack(image_arrays, axis=0)

	def ndvi_computation(self, band1='B8', band2='B4'):
		
		def add_ndvi(image):
			
			ndvi = image.normalizedDifference([band1, band2]).rename('NDVI')
			
			return image.addBands(ndvi)

		if 'NDVI' in self.band_names:
			pass
		else:
			self.collection = self.collection.map(add_ndvi)

	def images(self, past_date, now_date, single=False, class_filter=False):

		image_collection = self.collection.filter(ee.Filter.date(past_date, now_date)).sort('system:time_start', False)

		if single:
			image_collection = image_collection.limit(1)

		def image_filter(image):
			
			clase = image.clip(self.geometry).select('SCL')
			clase_filter = clase.gte(3).bitwiseAnd(clase.lte(7))
			ndvi = image.clip(self.geometry).select(self.band_names)
			ndvi = ndvi.updateMask(clase_filter)

			return ndvi

		def image_clip(image):
			return image.clip(self.geometry).select(self.band_names)
		
		if class_filter:
			image_collection = image_collection.map(image_filter)
		else:
			image_collection = image_collection.map(image_clip)

		return image_collection

	def reduction(self, collection, reductor, limit = None, class_filter = False):

		date = collection.first().get('system:time_start')

		if limit:
			collection = collection.limit(limit)
			
		if reductor == 'max':
			image = collection.reduce(ee.Reducer.max()).rename(self.band_names)

		elif reductor == 'median':
			image = collection.reduce(ee.Reducer.median()).rename(self.band_names)

		elif reductor == 'mean':
			image = collection.reduce(ee.Reducer.mean()).rename(self.band_names)

		elif reductor == 'min':
			image = collection.reduce(ee.Reducer.min()).rename(self.band_names)

		image = image.set({'system:time_start': date})
			
		return ee.ImageCollection([image])

	def get_data(self, collection, output='pandas'):
	
		def series(image, serie_list):

			latlon = ee.Image.pixelLonLat().addBands(image.select(self.band_names))
			element = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=self.geometry, maxPixels=1e8, scale=self.scale)
			return ee.List(serie_list).add(element)

		def dates_series(image, serie_list):

			date = ee.Date(image.get('system:time_start')).format()
			return ee.List(serie_list).add(date)

		dates_serie = collection.iterate(dates_series, ee.List([])).getInfo()
		data_serie = collection.iterate(series, ee.List([])).getInfo()

		data_serie, dates_serie =self.same_size_dict(data_serie, dates_serie)

		dataframe = pd.DataFrame(data_serie)
		dataframe['date'] = dates_serie
		
		if output == 'pandas':
			pass

		else:
			dataframe = dataframe.to_dict('index')
		
		return dataframe


	def data_to_images(self, data_object, file_name = None):

		if type(data_object) is not dict:
			data_object = data_object.to_dict('index')
			
		lats = data_object[0]['latitude']
		lons = data_object[0]['longitude']

		images = list(data_object.keys())
		bands = [b for b in list(data_object[0].keys()) if b not in ('latitude','longitude','date')]

		nrows, ncols, transform = self.coords_to_geotransform(lats, lons)
		
		data = np.full((len(images), ncols+1, nrows+1, len(bands)), np.nan)

		index_image, index_band = 0,0
	
		for image in images:

			for band in bands:

				for i in range(0, len(data_object[image]['latitude'])):
					
					col,row = self.world2Pixel(transform, data_object[image]['longitude'][i], data_object[image]['latitude'][i])
					try:
						data[index_image][int(col)][int(row)][index_band] = data_object[image][band][i]
					except:
						pass

				band_index =+1
			index_image =+1

		if file_name:
			np.save(file_name+'_data.npy', data)
			np.save(file_name+'_transform.npy', transform)

		return data, transform

	def rasterize_images(self, prefix, array, transform, dates=None, directory='', epsg_code=4326, google_storage = False):

		projection  = osr.SpatialReference()
		projection.ImportFromEPSG(epsg_code)
		driver = gdal.GetDriverByName('GTiff')

		if not dates:
			dates = range(0, array.shape[0])

		for index_image in range(0, array.shape[0]):
			
			file_name = prefix + '_' + str(dates[index_image]) + '.tif'
			raster = driver.Create(directory + file_name, array.shape[2], array.shape[1], array.shape[3], gdal.GDT_Float32)
			
			for index_band in range(0, array.shape[3]):
				raster.GetRasterBand(index_band+1).WriteArray(array[index_image][:][:][index_band])
			
			raster.SetGeoTransform(transform)
			raster.SetProjection(projection.ExportToWkt())
			raster.FlushCache()
			raster=None

			if google_storage:
				blob = self.bucket.blob(directory + file_name)
				blob.upload_from_filename(directory + file)

	def data_pix2pix(self, data_object):

		data = pd.DataFrame()

		if type(data_object) is not dict:
			data_object = data_object.to_dict('index')

		for key, dictionary in data_object.items():
			data_dic = pd.DataFrame(dictionary)
			data_dic = data_dic.assign(date=key)
			data = data.append(data_dic, ignore_index=True)

		return data

	def image_map(self, collection, palette, bands, min_val=0, max_val=1):
		
		image = collection.first()
		palette_color = ','.join(palette)
		vis = {'min': min_val, 'max': max_val, 'bands': bands, 'palette': palette_color}
		image = image.visualize(**vis)
		mapid = image.getMapId()

		return mapid

	def density_image(self, collection, levels):

		image = collection.first()
		image = image.gt(levels[0])

		for index in range(1,len(levels)):
			image = image.add(image.gt(levels[index]))

		return ee.ImageCollection(image)

	def difference_image(self, collection, rest_collection, percentaje = False):
		
		image = collection.first()
		rest_image = rest_collection.first()

		diff_img =  image.subtract(rest_image)

		if percentaje:
			diff_img = diff_img.divide(rest_img)
		
		return ee.ImageCollection(diff_img)

	def image_threshold(self, collection, threshold):
		
		image = collection.first()
		mask = image.gt(threshold)
		mask_complement = image.lt(threshold)

		return ee.ImageCollection(mask), ee.ImageCollection(mask_complement)

	def extract_pixel(self, data, lon, lat, n_band, transform):
		
		col,row = self.world2Pixel(transform, lon, lat)
		datos = data[:][int(col)][int(row)][n_band]

		return datos

	def world2Pixel(self, geoMatrix, x, y):

		ulX = geoMatrix[0]
		ulY = geoMatrix[3]
		xDist = geoMatrix[1]
		yDist = -geoMatrix[5]
		pixel = int((x - ulX) / xDist)
		line = int((ulY - y) / yDist)

		return (pixel, line)

	def coords_to_geotransform(self, lats, lons):

		unique_lats = np.unique(lats)
		min_lats = np.min(unique_lats)
		max_lats = np.max(unique_lats)

		unique_lons = np.unique(lons)
		min_lons = np.min(unique_lons)
		max_lons = np.max(unique_lons)

		ncols = len(unique_lons)
		nrows = len(unique_lats)

		ys = abs(max_lats-min_lats)/nrows
		xs = abs(max_lons-min_lons)/ncols

		return nrows, ncols, [min_lons, xs, 0, max_lats, 0, -ys]

	def train_model(self, x, y, model='svm', do_cv=True, n_jobs=None):

		param_grid = self.param_grids[model]
		model_pipeline = self.get_pipeline(model)
		best_params = self.default_params[model]

		if do_cv:
			fold_reports, fold_best_params, best_estimators, best_scores = self.nested_cv(x, y, model_pipeline, param_grid, n_jobs=n_jobs)
			counter_params = collections.Counter((tuple(p.items()) for p in fold_best_params))
			best_params = dict(counter_params.most_common(1)[0][0])

		model_pipeline.set_params(**best_params)
		model_pipeline.fit(x, y)

		return model_pipeline

	def train_classifier(self, x, y, output_clf_name, model='svm', do_cv=True, n_jobs=1 ):
		self.classifier_params()
		trained_model = self.train_model(x, y, model=model, do_cv=do_cv, n_jobs=n_jobs)
		joblib.dump(trained_model, output_clf_name+'.pkl')

	def classifier_params(self):
		self.param_grids = {
    	'svm' : {
        'model__kernel' : ['rbf'],
        'model__C' : np.logspace(-3, 3, 10),
        'model__gamma' : np.logspace(-3, 3, 10)},
    	'rf': {
        'model__n_estimators' : np.arange(60,201,20),
        'model__criterion' : ['gini'],
        'model__max_features' : np.arange(0.1, 1.1, 0.2),
        'model__min_samples_split' : np.arange(0.1, 1.1, 0.2),},
    	'knn' : {
        'model__n_neighbors' : np.arange(2, 10),
        'model__weights' : ['uniform', 'distance']}}

		self.default_params = { 'svm' : {
        'model__kernel' : 'rbf',
        'model__C' : 10,
        'model__gamma' : 0.4641688},
    	'rf' : {
        'model__n_estimators' : 180,
        'model__criterion' : 'gini',
        'model__max_features' : 0.5,
        'model__min_samples_split': 0.1,},
    	'knn' : {
        'model__n_neighbors' : 9,
        'model__weights' : 'uniform'}}

	def get_pipeline(self, name='svm'):

		model = None

		if name == 'svm':
			model = sklearn.svm.SVC(probability=True)
		elif name == 'rf':
			model = sklearn.ensemble.RandomForestClassifier()
		elif name == 'knn':
			model = sklearn.neighbors.KNeighborsClassifier()

		pipeline = sklearn.pipeline.Pipeline([('scale', sklearn.preprocessing.StandardScaler()), ('model', model)])

		return pipeline

	def nested_cv(self, X, y, model, param_grid, inner_cv=sklearn.model_selection.StratifiedKFold(10, shuffle=True), outer_cv=sklearn.model_selection.StratifiedKFold(10, shuffle=True), scoring='f1_micro', n_jobs=None):

		fold_reports = []
		best_params = []
		best_estimators = []
		best_scores = []

		for train_indices, test_indices in outer_cv.split(X, y):

			x_train, x_test = X[train_indices], X[test_indices]
			y_train, y_test = y[train_indices], y[test_indices]

			inner_grid = sklearn.model_selection.GridSearchCV(model, param_grid, iid=False, cv=inner_cv, scoring=scoring, n_jobs=n_jobs)
			inner_grid.fit(x_train, y_train)
			best_model = inner_grid.best_estimator_
			best_model.fit(x_train, y_train)

			clf_report = sklearn.metrics.classification_report(y_true=y_test, y_pred=best_model.predict(x_test))
			best_model_score = sklearn.metrics.scorer.SCORERS[scoring](best_model, x_test, y_test)

			fold_reports.append(clf_report)
			best_model_params = best_model.get_params()
			params ={k:best_model_params[k] for k in param_grid.keys()}
			best_params.append(params)
			best_estimators.append(best_model)
			best_scores.append(best_model_score)

		return fold_reports, best_params, best_estimators, best_scores

	def pandas_to_cloudsql(self, df, table_name):

		query = 'select * from' + table_name
		df.to_sql(table_name, con=self.engine, if_exists='append', index=False)
		self.engine.execute(query).fetchall()

	def cloudsql_to_pandas(self, table_name):

		query = 'select * from' + table_name
		df = pd.read_sql(query, self.engine.connect())
		return df

	def dict_to_firestore(self, collection_name, file_name, dictionary):

		doc_ref = self.db.collection(collection_name).document(file_name)
		doc_ref.set(dictionary)

	def firestore_to_dict(self, collection_name):
		collection_ref = self.db.collection(collection_name)
		docs = collection_ref.get()
		return docs

	def same_size_dict(self, data_serie, dates_serie):

		index = 0
		aux_data = []
		aux_dates = []

		for dictionary in data_serie:	
			
			boolean = True
			reference = len(dictionary['latitude'])

			for x in dictionary.values():

				if not len(x)== reference:
					boolean = False
					break

			if boolean:
				aux_dates.append(dates_serie[index])
				aux_data.append(data_serie[index])

			index =+1
		
		return aux_data, aux_dates 