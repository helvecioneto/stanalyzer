import gzip
import netCDF4
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import datashader as ds
import datashader.transfer_functions as tf
from matplotlib.colors import LinearSegmentedColormap
from colorcet import rainbow
import json
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


## Read NETCDF
def ncdf(nc_file,VAR,LEVEL):
    ## OPEN RADAR DATA COMPRESS
    with gzip.open(nc_file) as gz:
        with netCDF4.Dataset('dummy', mode='r', memory=gz.read()) as nc:
            data = nc.variables[VAR][0][LEVEL][:].filled()
            lon = nc.variables['lon0'][:].filled()
            lat = nc.variables['lat0'][:].filled()
    return data,lon,lat

def geo_img(matrix, lon_, lat_):
    
    ## GENERATE FILE
    img_pts,lons,lats = [],[],[]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] != np.nan:
                img_pts.append(matrix[i][j])
                lons.append(lon_[i][j])
                lats.append(lat_[i][j])
     
    ## IMG FRAME
    img_df = pd.DataFrame(list(zip(lons,lats,img_pts)),
               columns =['lon','lat','metric'])
    
    cvs = ds.Canvas(plot_width=matrix.shape[0], plot_height=matrix.shape[1])
    agg = cvs.points(img_df, x='lon', y='lat')
    coords_lat, coords_lon = agg.coords['lat'].values, agg.coords['lon'].values

    # Corners of the image, which need to be passed to mapbox
    coordinates = [[coords_lon[0], coords_lat[0]],
                   [coords_lon[-1], coords_lat[0]],
                   [coords_lon[-1], coords_lat[-1]],
                   [coords_lon[0], coords_lat[-1]]]
    
    agg.values = matrix
    
    img = tf.shade(agg, cmap=rainbow, alpha=200,how='linear')[::-1].to_pil()
    
    del matrix,lon_,lat_,img_pts,lons,lats,agg,coords_lat, coords_lon
                
    return img,coordinates

def geom_layers(dframe_,time_):
    
    ## Lock for last geometries
    locked_for_geo = dframe_.loc[dframe_['time'] == time_]
    
    ## Lock for get trajectory
    uids = locked_for_geo.uid.to_list()
    locked_for_traj = dframe_.query('time <= @time_  & uid == @uids ')
    
    ## Geo Layers
    layers = []
    colors = ['gray','blue','red','cyan']
    count = 0
    for c in sorted(dframe_.columns):
        if 'traj' in c:
            geo = json.loads(gpd.GeoSeries(locked_for_traj[c].apply(wkt.loads)).to_json())
            layers.append({'source': geo,
                          'type': "line",
                          'color': "black",
                          'name':'Trajectory',
                          'line':dict(width=4)
                          })
        
        if 'geom_' in c and 'geom_intersect' not in c:
            geo = json.loads(gpd.GeoSeries(locked_for_geo[c].apply(wkt.loads)).to_json())
            layers.append({'source': geo,
                           'type': "line",
                           'color': colors[count],
                           'name':c,
                           'line':dict(width=1.5)
                         })
            count += 1
            
    return layers


def track(dframe,VAR,LEVEL,THRESHOLDS,time):

    ### NC_FILE FROM TIME
    nc_path = dframe.loc[dframe['time'] == time]['nc_file'].unique()[0]
    data,lon,lat = ncdf(nc_path,VAR,LEVEL)
    ## NO DATA TO NAN
    data[data == -9999] = np.nan
    
    ## CLUSTER FILE
    cluster_path = dframe.loc[dframe['time'] == time]['cluster_file'].unique()[0]
    cluster_data = np.load(cluster_path)['arr_0']
    cluster_data[cluster_data == 0] = np.nan 

    ## Mount Geo Image
    g_img,extent = geo_img(data, lon, lat)
    
    ## Mount Geo Layers
    layers_ = geom_layers(dframe,time)
    
    ## Mount cluster geo image
    for t in range(len(THRESHOLDS)):
        c_img, extent = geo_img(cluster_data[:,:,t], lon, lat)
        ## Append clusters layer
        layers_.append({"sourcetype": "image",
                       "below": "traces",
                       "source": c_img,
                       "coordinates": extent,
                    })
    
    ## Append img layer
    layers_.append({"sourcetype": "image",
                   "below": "traces",
                   "source": g_img,
                   "coordinates": extent,
                })
    

    mapbox_ = 'pk.eyJ1Ijoic3BsaW50ZXJzdG0iLCJhIjoiY2tydWxjMzdlMTRxYTJwcGZlYmc0aWJyYSJ9.3-nO2w18a1fbjmAXrH7kEA'

    
    ## Templates
    hover_info = dict(bgcolor="white",
                font_size=14,
                font_family="Rockwell")
    
    hover_temp = "<b>DATE: %{customdata[0]} </b><br>" + \
                 "<b>UID:</b> %{customdata[1]}        <b>STATUS:</b> %{customdata[2]}<br>" + \
                 "<b>LIFE: </b>: %{customdata[3]}<br><br>" + \
                 "<b>VELOCITY: </b>       %{customdata[4]} km/h<br>" + \
                 "<b>DIRECTION: </b>     %{customdata[5]} º<br><br>" + \
                 "<b>MAX_REF</b>:          %{customdata[6]} dBZ<br>" + \
                 "<b>MEAN_REF_"+str(THRESHOLDS[0])+"</b>: %{customdata[7]} dBZ<br>" + \
                 "<b>MEAN_REF_"+str(THRESHOLDS[1])+"</b>: %{customdata[8]} dBZ<br>" + \
                 "<b>MEAN_REF_"+str(THRESHOLDS[2])+"</b>: %{customdata[9]} dBZ<br><br>" + \
                 "<b>DMEAN_REF_"+str(THRESHOLDS[0])+"</b>: %{customdata[10]} dBZ<br>" + \
                 "<b>DMEAN_REF_"+str(THRESHOLDS[1])+"</b>: %{customdata[11]} dBZ<br>" + \
                 "<b>DMEAN_REF_"+str(THRESHOLDS[2])+"</b>: %{customdata[12]} dBZ<br><br>" + \
                 "<b>AREA_"+str(THRESHOLDS[0])+"</b>:          %{customdata[13]} km²<br>" + \
                 "<b>AREA_"+str(THRESHOLDS[1])+"</b>:          %{customdata[14]} km²<br>" + \
                 "<b>AREA_"+str(THRESHOLDS[2])+"</b>:          %{customdata[15]} km²<br><br>" + \
                 "<b>SIZE_"+str(THRESHOLDS[0])+"</b>:          %{customdata[16]} pixels<br>" + \
                 "<b>SIZE_"+str(THRESHOLDS[1])+"</b>:          %{customdata[17]} pixels<br>" + \
                 "<b>SIZE_"+str(THRESHOLDS[2])+"</b>:          %{customdata[18]} pixels<br><br>" + \
                 "<b>DSIZE_"+str(THRESHOLDS[0])+"</b>:          %{customdata[19]} pixels<br>" + \
                 "<b>DSIZE_"+str(THRESHOLDS[1])+"</b>:          %{customdata[20]} pixels<br>" + \
                 "<b>DSIZE_"+str(THRESHOLDS[2])+"</b>:          %{customdata[21]} pixels<br><br>" + \
                 "LON: %{lon}   LAT: %{lat}<br><extra></extra>"
    
    ## MARKER AND COLORBAR         
    markers=dict(size=5, color='black',
                                showscale=True,
                                colorscale=rainbow,
                                cmin=-30,cmax=75,
                                colorbar=dict(
                                   title = VAR,
                                   titleside='right',
                                   thicknessmode='pixels',
                                   thickness=20,           
                                   ticklen=3, tickcolor='orange',
                                   tickfont=dict(size=14, color='black')))
    ## Layout
    layer = dict(mapbox=dict(layers=[layers_[-1]],accesstoken=mapbox_,
                          center=dict(lat=lat.mean(), lon=lon.mean()),
                          style='light',
                          zoom=6))
    
    ## Use custom data
    gd_t = dframe.loc[dframe['time'] == time]    
    
    ## Convert polygon areas
    geo_cols = [c for c in gd_t.columns if 'geom_' in c and 'geom_inter' not in c]
    cnt = 0
    for gc in geo_cols:
        converted_series = gpd.GeoSeries(gd_t[gc].apply(wkt.loads),crs='epsg:4326')
        gd_t['area_'+str(THRESHOLDS[cnt])] = converted_series.to_crs({'init': 'epsg:3857'}).area/ 10**6
        cnt +=1

    columns_int = ['timestamp','uid','status','delta_t',
                   'vel_'+str(THRESHOLDS[0])+'_cor','angle_'+str(THRESHOLDS[0])+'_cor',
                   'max_ref_'+str(THRESHOLDS[0]),
                   'mean_ref_'+str(THRESHOLDS[0]),
                   'mean_total_ref_'+str(THRESHOLDS[1]),'mean_total_ref_'+str(THRESHOLDS[2]),
                   'dmean_ref_'+str(THRESHOLDS[0]),'dmean_total_ref_'+str(THRESHOLDS[1]),'dmean_total_ref_'+str(THRESHOLDS[2]),
                   'area_'+str(THRESHOLDS[0]),'area_'+str(THRESHOLDS[1]),'area_'+str(THRESHOLDS[2]),
                   'size_'+str(THRESHOLDS[0]),'total_size_'+str(THRESHOLDS[1]),'total_size_'+str(THRESHOLDS[2]),
                   'dsize_'+str(THRESHOLDS[0]),'dtotal_size_'+str(THRESHOLDS[1]),'dtotal_size_'+str(THRESHOLDS[2]),
                  ]
    
    ## Data
    data = go.Scattermapbox(customdata=gd_t[columns_int].round(2).fillna(0),
                        lat=list(gd_t["lat"]), lon=list(gd_t["lon"]))
    
    ## Mount Figure
    fig = go.Figure(data=data,layout=layer)
    
    ## Update configs
    fig.update_traces(mode="markers", marker=markers,hoverlabel=hover_info, hovertemplate=hover_temp)
    fig.update_layout(height=600,width=900, margin={"r":0,"t":40,"l":100,"b":0,"pad":0})
    fig.update_layout(title_text='Tracking '+str(gd_t['timestamp'].unique()[0]), title_x=0.5)
    
    
    ## Buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="None",
                         method="relayout",
                         args=["mapbox.layers",[layers_[-1]]]),
                    
                    dict(label="All",
                         method="relayout",
                         args=["mapbox.layers",layers_]),
                    
                    dict(label="Clusters "+str(THRESHOLDS[-3])+' dBZ',
                         method="relayout",
                         args=["mapbox.layers",[layers_[-4]]]),
                    
                    dict(label="Clusters "+str(THRESHOLDS[-2])+' dBZ',
                         method="relayout",
                         args=["mapbox.layers",[layers_[-3]]]),
                    
                    dict(label="Clusters "+str(THRESHOLDS[-1])+' dBZ',
                         method="relayout",
                         args=["mapbox.layers",[layers_[-2]]]),
                    
                    dict(label="Geometries "+str(THRESHOLDS[0])+' dBZ',
                         method="relayout",
                         args=["mapbox.layers",[layers_[0],layers_[-1]]]),
                    
                    dict(label="Geometries "+str(THRESHOLDS[1])+' dBZ',
                         method="relayout",
                         args=["mapbox.layers",[layers_[1],layers_[-1]]]),
                    
                    dict(label="Geometries "+str(THRESHOLDS[2])+' dBZ',
                         method="relayout",
                         args=["mapbox.layers",[layers_[2],layers_[-1]]]),
                    
                    dict(label="Trajectory",
                         method="relayout",
                         args=["mapbox.layers",[layers_[3],layers_[-1]]]),
                ],
            )
        ]
    )
    
#     fig.update_layout(
#     annotations=[
#         dict(text="Geometry options:", showarrow=False,
#                              x=-.22, y=1.05, yref="paper", align="left")
#     ]
#     )
    
    fig.show()
