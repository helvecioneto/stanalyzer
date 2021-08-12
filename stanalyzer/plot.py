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
from scipy.stats import circmean,circvar,circstd
import json
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


## Read NETCDF
def ncdf(nc_file,var='DBZc',level=5):
    ## OPEN RADAR DATA COMPRESS
    with gzip.open(nc_file) as gz:
        with netCDF4.Dataset('dummy', mode='r', memory=gz.read()) as nc:
            data = nc.variables[var][0][level][:].filled()
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


## PLOT TRACK
def track(dframe,var='DBZc',level=5):

    ## VERIFY DATA FRAME
    if len(dframe) < 1:
        return print('Invalid DataFrame!')

    ## GET LAST TIME
    time = sorted(dframe.reset_index()['time'])[-1]

    # GET TRHESHOLDS
    THRESHOLDS = [int(c[-2:]) for c in dframe.columns if 'geom_' in c and 'intersect' not in c]

    ### NC_FILE FROM TIME
    nc_path = dframe.loc[dframe['time'] == time]['nc_file'].unique()[0]
    data,lon,lat = ncdf(nc_path,var,level)
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
    
    hover_temp = "<b>%{meta[0]}: %{customdata[0]} </b><br>" + \
                 "<b>%{meta[1]}:</b> %{customdata[1]}        <b>STATUS:</b> %{customdata[2]}<br>" + \
                 "<b>%{meta[3]}: </b>: %{customdata[3]}<br><br>" + \
                 "<b>%{meta[4]}: </b>       %{customdata[4]} km/h<br>" + \
                 "<b>%{meta[5]}: </b>     %{customdata[5]} º<br><br>" + \
                 "<b>%{meta[6]}</b>:          %{customdata[6]} dBZ<br>" + \
                 "<b>%{meta[7]}</b>:  %{customdata[7]} dBZ<br>" + \
                 "<b>%{meta[8]}</b>:  %{customdata[8]} dBZ<br>" + \
                 "<b>%{meta[9]}</b>:  %{customdata[9]} dBZ<br><br>" + \
                 "<b>%{meta[10]}</b>: %{customdata[10]} dBZ<br>" + \
                 "<b>%{meta[11]}</b>: %{customdata[11]} dBZ<br>" + \
                 "<b>%{meta[12]}</b>: %{customdata[12]} dBZ<br><br>" + \
                 "<b>%{meta[13]}</b>: %{customdata[13]} km²<br>" + \
                 "<b>%{meta[14]}</b>: %{customdata[14]} km²<br>" + \
                 "<b>%{meta[15]}</b>: %{customdata[15]} km²<br><br>" + \
                 "<b>%{meta[16]}</b>: %{customdata[16]} pixels<br>" + \
                 "<b>%{meta[17]}</b>: %{customdata[17]} pixels<br>" + \
                 "<b>%{meta[18]}</b>: %{customdata[18]} pixels<br><br>" + \
                 "<b>%{meta[19]}</b>: %{customdata[19]} pixels<br>" + \
                 "<b>%{meta[20]}</b>: %{customdata[20]} pixels<br>" + \
                 "<b>%{meta[21]}</b>: %{customdata[21]} pixels<br><br>" + \
                 "LON: %{lon}   LAT: %{lat}<br><extra></extra>"
    
    ## MARKER AND COLORBAR         
    markers=dict(size=5, color='black',
                                showscale=True,
                                colorscale=rainbow,
                                cmin=-30,cmax=75,
                                colorbar=dict(
                                   title = var,
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
    data = go.Scattermapbox(customdata=gd_t[columns_int].round(2).fillna(0),meta=[each_string.upper() for each_string in columns_int],
                        lat=list(gd_t["lat"]), lon=list(gd_t["lon"]))
    
    ## Mount Figure
    fig = go.Figure(data=data,layout=layer)
    
    ## Update configs
    fig.update_traces(mode="markers", marker=markers,hoverlabel=hover_info, hovertemplate=hover_temp)
    fig.update_layout(height=600,width=900, margin={"r":0,"t":40,"l":100,"b":0,"pad":0})
    fig.update_layout(title_text='Tracking '+str(gd_t['timestamp'].unique()[0]), title_x=0.5)
    
    
    ## Buttons
    fig.update_layout(
        updatemenus=[dict(type="buttons",buttons=[
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
    fig.show()


def degToCompass(d):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((d + 11.25)/22.5)
    return dirs[ix % 16]


## WIND PLOT
def plot_wind(df, style = 'standard'):
    
    if style != 'standard':
        vel_cols = [c for c in df.columns if 'vel_' in c and 'orig' not in c]
        angle_cols = [c for c in df.columns if 'angle_' in c and 'orig' not in c]
        
        series = df.groupby(['uid']).apply(lambda x: [x.uid.unique()[0],x[vel_cols[0]].values.flatten(),x[angle_cols[0]].values.flatten(),x.timestamp.values])
        wind_dir = pd.DataFrame(columns=['uid','timestamp','velocity','direction'])
        for e in series.keys():
            for ee in  range(len(series[e][1])):
                wind_dir = wind_dir.append({'uid':e,
                                            'timestamp':series[e][-1][ee],
                                            'velocity':series[e][1][ee],
                                            'direction':series[e][2][ee]},ignore_index=True)

        wind_frame = pd.DataFrame(columns=['uid','direction','strength','velocity'])
        vel_g = wind_dir.groupby(pd.cut(wind_dir['velocity'], np.arange(0, 100, 10)))
        for i,g in vel_g:
            for value in range(len(g.uid.values)):
                wind_frame = wind_frame.append({'uid':g.uid.values[value],
                                                'timestamp':g.timestamp.values[value],
                                                'strength':str(str(i.left)+'-'+str(i.right))+' km/h',
                                                'direction':g.direction.values[value],
                                                'velocity':g.velocity.values[value],
                                                },ignore_index=True)
        if style == 'bar':
    #        wind_frame['angle'] = wind_frame['direction']
        #     wind_frame['direction'] = wind_frame['direction'].apply(degToCompass)
            fig = px.bar_polar(wind_frame, r="velocity", theta="direction",template="presentation",
                           color="strength", labels={"strength": "Wind Speed:"},direction='counterclockwise',
                           base='strength',hover_data=wind_frame,hover_name='uid',start_angle=0,
                           color_discrete_sequence= px.colors.sequential.Plasma_r)
        if style == 'scatter':
            fig = px.scatter_polar(wind_frame, r="velocity", theta="direction",template="presentation",
                       color="strength", labels={"strength": "Wind Speed:"},direction='counterclockwise',
                       hover_data=wind_frame, hover_name='uid',start_angle=0,
                       color_discrete_sequence= px.colors.sequential.Plasma_r)


        fig.update_layout(width=600, height=600, margin={"r":0,"t":0,"l":100,"b":0,"pad":100})
        fig.update_layout(title_text='Wind Rose: '+style, title_x=0.45,title_y=0.86,font_size=12)
        fig.update_layout(legend=dict(
                orientation="v",
                yanchor="top",
                y=.8,
                x=1.1,
                xanchor="left"),
                polar=dict(radialaxis=dict(showticklabels=True, ticks='', linewidth=0)
                )
            )
        fig.show()
        
        return wind_frame.sort_values(by='timestamp')
    else:
        df = df.reset_index()
        df = df.sort_values(by='time')

        ## Templates
        hover_info = dict(bgcolor="white",
                    font_size=14,
                    font_family="Rockwell")

        hover_temp = "<b>DATE: %{meta} </b><br>" + \
                     "<b>%{hovertext} </b> <br>" + \
                     "<b>MEAN_VEL: %{r} in km/h<br>"+ \
                     "<b>MEAN_ANGLE: %{theta}<br><extra></extra>"

        times = df.timestamp.unique()
        vel_cols = [c for c in df.columns if 'vel_' in c and 'orig' not in c and 'level_' not in c]
        angle_cols = [c for c in df.columns if 'angle_' in c and 'orig' not in c and 'level_' not in c]

        fig = go.Figure()

        for t in times:
            uids = df.query('timestamp == @t').uid.unique().tolist()
            angles = []
            velocities = []
            stds = []
            uid_ = [] 
            
            for ud in uids:
                loocked_df = df.query('uid == @ud and timestamp <= @t')
                if len(loocked_df[vel_cols[0]].dropna().values) > 0:
                    angles.append(circmean(loocked_df[angle_cols[0]],nan_policy='omit',high=360))
                    velocities.append(loocked_df[vel_cols[0]].mean())
                    stds.append(circvar(loocked_df[angle_cols[0]],nan_policy='omit'))
                    uid_.append('UID: '+str(ud))

            if len(angles) > 0:
                fig.add_trace(go.Barpolar(
                    r=velocities,
                    theta=angles,
                    marker_color=uids,
                    width=np.array(stds)+10,
                    marker_line_color="black",
                    marker_line_width=2,
                    hovertext=uid_,
                    opacity=0.8,
                    ids=uids,
                    name=t,
                    meta=t,
                    hoverlabel=hover_info,
                    hovertemplate=hover_temp
                ))     

        fig.update_layout(
            template=None,
            polar = dict(
                radialaxis = dict(range=[0, 120], showticklabels=True),
                angularaxis = dict(showticklabels=True, ticks='')
            )
        )

        fig.update_layout(width=600, height=600, margin={"r":0,"t":0,"l":100,"b":0,"pad":100})
        fig.update_layout(title_text='Wind Rose', title_x=0.44,title_y=0.84,font_size=12)
        fig.update_layout(legend=dict(
                orientation="v",
                yanchor="top",
                y=.8,
                x=1.1,
                xanchor="left"),
                polar=dict(radialaxis=dict(showticklabels=True, ticks='', linewidth=0)
                )
            )

        fig.show()

def plot_lines(df, analyze_columns = None, axis_name = None):
    analize_cols = analyze_columns.copy()
    analyze_columns.insert(0,'timestamp')
    analyze_columns.insert(0,'uid')
    
    orig_df = df.copy()
    df = df[analyze_columns].sort_values(by='timestamp')
#     df = df.round(2).fillna(0)
    dfg = df[analyze_columns].groupby('uid')
    
    dashs = ['solid','dot','dash','dashdot']
    cnt=0
    
    ## Templates
    hover_info = dict(bgcolor="white",
                font_size=14,
                font_family="Rockwell")
    
    hover_temp = "<b>%{meta}:</b> %{y} <br><br>" + \
                 "<b>DATE: %{customdata[0]} </b><br>" + \
                 "<b>UID: %{customdata[2]} </b>                 <b>TIME: %{customdata[1]} </b><br><extra></extra>"

    fig = go.Figure()
    for c in analyze_columns[2:]:
        for i,g in dfg:
            fig.add_trace(go.Scatter(x=g.timestamp.values,y=g[c].values,
                                     name=c+' - UID: '+str(g.uid.unique()[0]),meta=c.upper(),
                                     customdata=orig_df.loc[g.index],mode="markers+lines",
                                     hoverlabel=hover_info,hovertemplate=hover_temp,
                                     visible=True, line=dict(width=2, dash=dashs[cnt])))
        cnt += 1
        
    fig.update_layout(height=400,width=900, margin={"r":0,"t":40,"l":100,"b":0,"pad":0})
    fig.update_layout(title_text='Analized Columns: '+str(analize_cols)[1:-1], title_x=0.2,
                      yaxis_title=axis_name)
    fig.show()
