import pandas as pd
import numpy as np
from zipfile import ZipFile
import netCDF4
import gzip
from shutil import copyfile
import matplotlib.pyplot as plt
import shapely.wkt
from shapely.geometry import shape
import math
import warnings
from pathlib import Path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from pathlib import Path
warnings.filterwarnings("ignore")


##########################################################################
##########################################################################
####################	GLOBAL VARIABLES	##############################
##########################################################################
##########################################################################

PATH = '/home/helvecioneto/01_IARA/RADAR/iara_beta_v6/output/S201409070000_E201409100000_VDBZc_T20_L5_SPLTTrue_MERGTrue_TCORTrue_PCORFalse.zip'
DATA_PATH = '/home/helvecioneto/SINAPSE_01/DADOS/sbandradar/'
VAR_NAME = 'DBZc'
LEVEL = 5
THRESHOLD = [20,35,40]
OUTPUT = '../output/'
NC_OUTPUT = '../output/data/'
OUTPUT_FILE = '../output/output_file_tracked'


### READ TRACK FROM PATH
def read_track(path):
	print('Reading files...')
	zip_file = ZipFile(path)
	for f in zip_file.infolist():
		if f.filename.endswith('.csv') and 'FINAL' in f.filename:
			tracking_df = pd.read_csv(zip_file.open(f))
	return tracking_df

### OPEN NETCDF
def open_file(file_path):
	try:
		with gzip.open(file_path) as gz:
			with netCDF4.Dataset('dummy', mode='r', memory=gz.read()) as nc:
				data = nc.variables[VAR_NAME][0][LEVEL][:].filled()
				data[data == -9999.] = np.NAN
		return data
	except:
		pass

### FAMILY GENERATOR
def create_fam(df):
	print('Createing DataFrame...')
	## REORGANIZE FAM BY MULTINDEX
	df = df.groupby('uid').apply(lambda x: x.sort_values(["uid"], ascending = False))
	index_list = df.index.tolist()
	new_list = []
	for i in index_list:
		new_list.append(('Fam_'+str(i[0]),i[1]))
	mux = pd.MultiIndex.from_tuples(new_list)
	df.index = mux
	return df

### GET LAT LONG MATRIX
def get_latlong_matrix(file_path):
	with gzip.open(file_path) as gz:
		with netCDF4.Dataset('dummy', mode='r', memory=gz.read()) as nc:
			data_var = nc.variables.keys()
			for k in data_var:
				if k.lower().startswith('lat'):
					llat = k
				elif k.lower().startswith('lon'):
					llon = k
			lon = nc.variables[llon][:]
			lat = nc.variables[llat][:]
	return lon,lat


### GENERATE PATCH FILES
def generate_path_files(df,DATA_PATH):
	print('Generating Path Files...')
	for i,row in df.iterrows():
		file = str(pd.to_datetime(row.timestamp).strftime(DATA_PATH+'%Y/%m/sbmn_cappi_%Y%m%d_%H%M.nc.gz'))
		df.loc[i,'nc_file'] = file
		cluster = pd.to_datetime(row.timestamp).strftime('clusters/%Y%m%d_%H%M%S_clu.npz')
		df.loc[i,'cluster_file'] = cluster
	return df


## OPEN CLUSTERS BY DBSCAN
def open_cluster(path,file_path):
	try:
		zip_file = ZipFile(path)
		cluster = np.load(zip_file.open(file_path))['arr_0']
		return cluster
	except:
		pass

## Calculate angle of Linestring
def angle(x):
	p1 = list(shape(x).coords)[0]
	p2 = list(shape(x).coords)[1]

	rad_ = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
	direction = math.degrees(rad_)

	if direction < 0:
		direction = direction + 360
	return direction


### CALCULATE VARIEABLES
def calc_dbz(df,threshold,path):
	print('Calculating Clusters Variables...')

	LON,LAT = get_latlong_matrix(df.iloc[0]['nc_file'])

	for i, row in df.iterrows():
		### GET REFLECT AND PIXEL SIZES
		cluster_matrix = open_cluster(path,row.cluster_file)
		dbz_matrix = open_file(row.nc_file)
		if cluster_matrix is not None and dbz_matrix is not None:
			for t in range(len(threshold)):
				clt_matrix = cluster_matrix[:,:,t]
				x,y = np.where(clt_matrix == row.id_t)
				dbz_list = dbz_matrix[x,y]
				if len(dbz_list) > 0:
					### MEAN REFLECT
					mmh6 = np.mean(10**(dbz_list/10))
					dbz_mean = (10*np.log10(mmh6))
					dbz_max = np.max(dbz_list)

					if t == 0:
						df.loc[i,'mean_ref_'+str(threshold[t])] = dbz_mean
						df.loc[i,'max_ref_'+str(threshold[t])] = dbz_max
						df.loc[i,'size_'+str(threshold[t])] = len(x)
					else:
						df.loc[i,'mean_total_ref_'+str(threshold[t])] = dbz_mean
						df.loc[i,'total_size_'+str(threshold[t])] = int(len(x))
		else:
			del cluster_matrix,dbz_matrix
		
		### GET ORIG ANGLE AND VELM
		if row.linestring != '-1':
			for t in range(len(threshold)):
				if t == 0:
					df.loc[i,'angle_'+str(threshold[t])+'_cor'] = row.angle
					df.loc[i,'angle_'+str(threshold[t])+'_orig'] = angle(shapely.wkt.loads(row.linestring))
					
					## VELM
					df.loc[i,'vel_'+str(threshold[t])+'_orig'] = (row['length'] * 2) / 0.2
					df.loc[i,'vel_'+str(threshold[t])+'_cor'] = row.velm
				else:
					df.loc[i,'avg_angle_'+str(threshold[t])] = row['internal_angle_mean_'+str(threshold[t])]
					df.loc[i,'avg_vel_'+str(threshold[t])] = row['internal_velm_'+str(threshold[t])]
					
		### GET LON LAT
		if row.centroid_t != '-1':
			point = list(shape(shapely.wkt.loads(row.centroid_t)).coords)[0]
			p0,p1 = int(np.round(point[0])),int(np.round(point[1]))
			long = LON[p1][p0]
			latg = LAT[p1][p0]
			
			df.loc[i,'lat'] = latg
			df.loc[i,'lon'] = long
			df.loc[i,'p0'] = p0
			df.loc[i,'p1'] = p1  
			
		### GET NUMBER OF CLUSTERS
		for t in threshold[1:]:
			if row['n_poly_'+str(t)] == -1:
				df.loc[i,'n_cluster_'+str(t)] = 0
			else:
				df.loc[i,'n_cluster_'+str(t)] = row['n_poly_'+str(t)]
	return df


### REDEFINE FRAMES
def redefine(df):
	df_copy = df
	#### REPLACE 0 to NAN
	df_copy = df_copy[df_copy.columns[6:]].replace(0,np.nan)
	#### REPLACE -1 to NAN
	df_copy = df_copy.replace(-1,np.nan)
	#### REPLACE '-1' to NAN
	df_copy = df_copy.replace('-1',np.nan)
	df_final = pd.concat([df[df.columns[:6]],df_copy],axis=1)
	return df_final

### TRANSLATE GEOMETRIES
def trans_poly(geom,latitude,longitude):
	x,y = geom.exterior.coords.xy
	points = []
	for v in range(len(x)):
		points.append([longitude[int(np.round(y[v]))][int(np.round(x[v]))],latitude[int(np.round(y[v]))][int(np.round(x[v]))]])
	poly = Polygon(list(zip(np.array(points)[:,0],np.array(points)[:,1])))
	return poly

def trans_multipoly(geom,latitude,longitude):
	polys2 = []
	for gg in geom:
		x,y = gg.exterior.coords.xy
		points = []
		for v in range(len(x)):
			points.append([longitude[int(np.round(y[v]))][int(np.round(x[v]))],latitude[int(np.round(y[v]))][int(np.round(x[v]))]])
		polys2.append(Polygon(list(zip(np.array(points)[:,0],np.array(points)[:,1]))))
	return MultiPolygon(polys2)

def trans_lines(geom,latitude,longitude):
	x,y = geom.xy
	p = []
	for v in range(len(x)):
		p.append(Point([longitude[int(np.round(y[v]))][int(np.round(x[v]))],latitude[int(np.round(y[v]))][int(np.round(x[v]))]]))
	return LineString(p)

def trans_multilines(geom,latitude,longitude):
	lines2 = []
	for gg in geom:
		x,y = gg.xy
		p = []
		for v in range(len(x)):
			p.append(Point([longitude[int(np.round(y[v]))][int(np.round(x[v]))],latitude[int(np.round(y[v]))][int(np.round(x[v]))]]))
		lines2.append(LineString(p))
	return MultiLineString(lines2)


### TRANSLATE GEOMETRIES
def trans_geometries(df_frame,gcols,lat,lon):
	print('Translate geometries....')
	## Replace Null geometries
	df_frame[gcols] = df_frame[gcols].replace(['-1.0', '0',np.nan], 'GEOMETRYCOLLECTION EMPTY')
	
	### Transform to shapely
	for g in gcols:
		df_frame[g] = df_frame[g].astype(str).apply(shapely.wkt.loads)
		
		for geo in df_frame[g].index:
			## Transpolygons
			if type(df_frame.loc[geo][g]) == Polygon:
				geometry = trans_poly(df_frame.loc[geo][g],lat,lon)
				df_frame.loc[geo,g] = geometry.wkt
				
			elif type(df_frame.loc[geo][g]) == MultiPolygon:
				geometry = trans_multipoly(df_frame.loc[geo][g],lat,lon)
				df_frame.loc[geo,g] = geometry.wkt
				
			elif type(df_frame.loc[geo][g]) == LineString:
				geometry = trans_lines(df_frame.loc[geo][g],lat,lon)
				df_frame.loc[geo,g] = geometry.wkt
				
			elif type(df_frame.loc[geo][g]) == MultiLineString:
				geometry = trans_multilines(df_frame.loc[geo][g],lat,lon)
				df_frame.loc[geo,g] = geometry.wkt
				
			else:
				df_frame.loc[geo,g] = 'GEOMETRYCOLLECTION EMPTY'  
	
	return df_frame



def main():

	## TRACK DATA FRAME
	track_df = read_track(PATH)
	track_df = generate_path_files(track_df,DATA_PATH)
	LON,LAT = get_latlong_matrix(track_df.iloc[0]['nc_file'])
	to_fam_df = calc_dbz(track_df,THRESHOLD,PATH)

	## Mount trajectory
	for i,row in to_fam_df.iterrows():
		if '-1' not in row['centroid_tp'] and '-1' not in row['centroid_t']:
			traj = LineString([shapely.wkt.loads(row['centroid_tp']),shapely.wkt.loads(row['centroid_t'])])
			to_fam_df.loc[i,'trajectory'] = traj.wkt
		else:
			to_fam_df.loc[i,'trajectory'] = '-1'

	### USED COLUMNS
	used_columns = ['timestamp','time','uid','id_t','lat','lon','p0','p1']
	for t in range(len(THRESHOLD)):
		if t == 0:
			used_columns.append('size_'+str(THRESHOLD[t]))
			used_columns.append('mean_ref_'+str(THRESHOLD[t]))
			used_columns.append('max_ref_'+str(THRESHOLD[t]))
			used_columns.append('angle_'+str(THRESHOLD[t])+'_orig')
			used_columns.append('angle_'+str(THRESHOLD[t])+'_cor')
			used_columns.append('vel_'+str(THRESHOLD[t])+'_orig')
			used_columns.append('vel_'+str(THRESHOLD[t])+'_cor')
		else:
			used_columns.append('mean_total_ref_'+str(THRESHOLD[t]))
			used_columns.append('total_size_'+str(THRESHOLD[t]))
			used_columns.append('n_cluster_'+str(THRESHOLD[t]))
			used_columns.append('avg_angle_'+str(THRESHOLD[t]))
			used_columns.append('avg_vel_'+str(THRESHOLD[t]))
			
	used_columns.append('status')
	used_columns.append('delta_t')
	used_columns.append('nc_file')
	used_columns.append('cluster_file')

	used_columns.append('geom_intersect')
	used_columns.append('geom_'+str(THRESHOLD[0]))
	used_columns.append('geom_'+str(THRESHOLD[1]))
	used_columns.append('geom_'+str(THRESHOLD[2]))
	used_columns.append('trajectory')
	used_columns.append('vector_'+str(THRESHOLD[0]))
	used_columns.append('vector_'+str(THRESHOLD[1]))
	used_columns.append('vector_'+str(THRESHOLD[2]))

	to_fam_df['delta_t'] = to_fam_df['lifetime']

	## GEOMETRIES
	to_fam_df['geom_intersect'] = to_fam_df['intersect_geom']
	to_fam_df['geom_'+str(THRESHOLD[0])] = to_fam_df['geometry_t']
	to_fam_df['geom_'+str(THRESHOLD[1])] = to_fam_df['geom_'+str(THRESHOLD[1])]
	to_fam_df['geom_'+str(THRESHOLD[2])] = to_fam_df['geom_'+str(THRESHOLD[2])]


	to_fam_df['vector_'+str(THRESHOLD[0])] = to_fam_df['linestring']
	to_fam_df['vector_'+str(THRESHOLD[1])] = to_fam_df['internal_linestring_'+str(THRESHOLD[1])]
	to_fam_df['vector_'+str(THRESHOLD[2])] = to_fam_df['internal_linestring_'+str(THRESHOLD[2])]
			
	to_fam_df = to_fam_df[used_columns]


	### CREATE DIR
	print('Creating output directories...')
	Path(OUTPUT).mkdir(parents=True, exist_ok=True)
	Path(OUTPUT+'clusters').mkdir(parents=True, exist_ok=True)
	Path(OUTPUT+'data').mkdir(parents=True, exist_ok=True)

	### EXTRACT CLUSTERS
	with ZipFile(PATH, 'r') as zipObj:
		listOfFileNames = zipObj.namelist()
		for fileName in listOfFileNames:
			if 'clusters/' in fileName:
				zipObj.extract(fileName, OUTPUT)

	#### COPY NC_FILES
	print('Copy files...')
	for i,row in to_fam_df.iterrows():
		try:
			copyfile(row.nc_file,str(NC_OUTPUT+Path(row.nc_file).name))
			to_fam_df.loc[i,'nc_file'] = str(NC_OUTPUT+Path(row.nc_file).name)
			to_fam_df.loc[i,'cluster_file'] = OUTPUT+row.cluster_file
		except:
			pass

	geo_cols = []
	for c in to_fam_df.columns:
		if 'geom_' in c or 'line' in c or 'traj' in c or 'vect' in c:
			geo_cols.append(c)

    ## REDEFINE FRAMES
	to_fam_df = redefine(to_fam_df)
	## TRANSLATE GEOMETRIES
	to_fam_df = trans_geometries(to_fam_df,geo_cols,LAT,LON)
	## Create fams
	fam_df = create_fam(to_fam_df)

	### FILTER BY SIZE
	sized_fams = []
	for i,g in fam_df.groupby(level=0):
	    if len(g) >= 2:
	        sized_fams.append(i)

	### FINAL FRAME
	fam_df = fam_df.loc[sized_fams]


	### CALCULATE DIFFERENCES
	print('Calculate differences...')
	for i,g in fam_df.groupby(level=0):
	    for t in range(len(THRESHOLD)):
	        if t == 0:
	            fam_df.loc[i,'dsize_'+str(THRESHOLD[t])] = g['size_'+str(THRESHOLD[t])].diff()
	            fam_df.loc[i,'dmean_ref_'+str(THRESHOLD[t])] = g['mean_ref_'+str(THRESHOLD[t])].diff()
	        else:
	            fam_df.loc[i,'dtotal_size_'+str(THRESHOLD[t])] = g['total_size_'+str(THRESHOLD[t])].diff()
	            fam_df.loc[i,'dmean_total_ref_'+str(THRESHOLD[t])] = g['mean_total_ref_'+str(THRESHOLD[t])].diff()

	#### SAVE PICKLE
	print('Saving file...')
	fam_df.to_pickle(OUTPUT_FILE+'.pkl',compression='xz')
	print('Done!!!')


if __name__ == '__main__':
	main()