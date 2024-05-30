import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import dash
from dash import dcc, html
import dash_html_components as html

#import datasets
etas = pd.read_csv('./datasets/ModifiedETAS.csv', sep=',', lineterminator='\n')
usgs = pd.read_csv('./datasets/USGS.csv', sep=',', lineterminator='\n')

#aftershock column
etas_aftershock = etas.copy()
etas_aftershock['aftershock'] = etas_aftershock['aftershock\r']
etas_aftershock = etas_aftershock.drop(columns='aftershock\r')
etas_aftershock['aftershock'] = etas_aftershock['aftershock'].str.replace('\r', '')

start_date = etas['date'].min()
end_date = etas['date'].max()

#defining magnitude cutoff value and constants for binning
magnitude_cutoff = 7
binning_time = '2W'
date_range = 1 #year

#magnitude filtering
etas = etas[etas['mag'] > 3]
usgs = usgs[usgs['mag'] > 3]

#calculating energy from magnitude
formula_constant = (1/1.5)
usgs['energy'] = 10 ** (1.5 * usgs['mag'])
usgs['energy'] = np.log(usgs['energy']) * formula_constant
etas['energy'] = 10 ** (1.5 * etas['mag'])
etas['energy'] = np.log(etas['energy']) * formula_constant

#filtering data points by magnitude cutoff
usgs_large = usgs[usgs['mag'] >= magnitude_cutoff]
etas_large = etas[etas['mag'] >= magnitude_cutoff]

#function to bin data into chunks 
def bin_data(data: pd.DataFrame, start_date: pd.Timestamp, freq: str, range_: int) -> pd.DataFrame:
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(start_date)
    date_range_start = pd.to_datetime(start_date - pd.DateOffset(years=range_))
    date_range_end = pd.to_datetime(start_date + pd.DateOffset(years=range_))
    data_binned = data[(data['date'] >= date_range_start) & (data['date'] <= date_range_end)]
    return pd.DataFrame(data_binned.groupby(pd.Grouper(key='date', freq=freq)).energy.sum())

#graphing usgs data 
fig_usgs = px.scatter()
for index, row in usgs_large.iterrows():
    start_date = pd.to_datetime(row['date'])
    usgs_binned = bin_data(usgs, start_date, binning_time, date_range)
    relative_dates = (pd.to_datetime(usgs_binned.index) - start_date).days
    
    fig_usgs.add_scatter(x=relative_dates, y=usgs_binned['energy'], mode='lines+markers', name=f'Date: {start_date}')
    
fig_usgs.update_layout(title='Energy Before/After Large Earthquakes (USGS)',
                       xaxis_title='Days Before/After Large Earthquake',
                       yaxis_title='Energy',
                       legend_title='Date')

#graphing etas data 
fig_etas = px.scatter()
for index, row in etas_large.iterrows():
    start_date = pd.to_datetime(row['date'])
    etas_binned = bin_data(etas, start_date, binning_time, date_range)
    relative_dates = (pd.to_datetime(etas_binned.index) - start_date).days
    
    fig_etas.add_scatter(x=relative_dates, y=etas_binned['energy'], mode='lines+markers', name=f'Date: {start_date}')
    
fig_etas.update_layout(title='Energy Before/After Large Earthquakes (ETAS)',
                       xaxis_title='Days Before/After Large Earthquake',
                       yaxis_title='Energy',
                       legend_title='Date')

#finding the average energy spread before/after large events of usgs        
def average_spread(df:pd.DataFrame, df_large:pd.DataFrame) -> pd.DataFrame:
    dataframes = []
    for index, row in df_large.iterrows():
        start_date = pd.to_datetime(row['date'])
        df_binned = bin_data(df, start_date, binning_time, date_range)
        
        df_binned = df_binned.reset_index()
        index = df_binned.index
        df_binned['index'] = index
        df_binned = df_binned.set_index('index')
        df_binned = df_binned.drop(columns='date')

        dataframes.append(df_binned)
    
    avg_df = pd.concat(i for i in dataframes).groupby('index').mean()
    avg_df = avg_df.reset_index()
    return avg_df

usgs_avg = average_spread(usgs, usgs_large)
etas_avg = average_spread(etas, etas_large)

#coloring the markers by the difference in the data
color = ['red' if (i - j) < 0 else 'blue' for i, j in zip(etas_avg['energy'], usgs_avg['energy'])]

fig_difference = px.scatter()
fig_difference.add_scatter(x=etas_avg['index'], y=etas_avg['energy'] - usgs_avg['energy'], mode='lines+markers',
                           marker=dict(color=color), line=dict(color='black'))

fig_difference.update_layout(title='Average Difference In Trend Between ETAS & USGS',
                             xaxis_title='Index Value',
                             yaxis_title='Energy Difference')

#function to bin data before and after major event based on location
#* note that there is no timed (2 week) binning
def location_binning(data: pd.DataFrame, start_date: pd.Timestamp, date_range_: float) -> pd.DataFrame:
    date_range_start = start_date - pd.DateOffset(months=date_range_)
    date_range_end = start_date + pd.DateOffset(months=date_range_)
    
    return data[(data['date'] >= date_range_start) & (data['date'] <= date_range_end)]

#redefining magnitude cutoff value and range to prevent lag
magnitude_cutoff = 7
date_range = 6 

etas_large = etas_large[etas_large['mag'] >= magnitude_cutoff]
usgs_large = usgs_large[usgs_large['mag'] >= magnitude_cutoff]

#new dataframe to store binned data
binned_data_etas = pd.DataFrame()
for index, row in etas_large.iterrows():
    start_date = pd.to_datetime(row['date'])
    #binning data before/after major event
    etas_binned = location_binning(etas, start_date, date_range)
    relative_dates = (pd.to_datetime(etas_binned['date']) - start_date).dt.days
    
    binned_data_etas = pd.concat([
        binned_data_etas,
        pd.DataFrame({
            'Day 0': [start_date.strftime('%Y-%m-%d')] * len(etas_binned),
            'Latitude': etas_binned['latitude'],
            'Longitude': etas_binned['longitude'],
            'Relative_Dates': relative_dates,
            'Energy': etas_binned['energy']
        })
    ])

fig_etas_location = px.scatter(binned_data_etas, x='Longitude', y='Latitude', color='Relative_Dates', 
                               size='Energy', animation_frame='Day 0',
                               title='Energy Locations Before/After Large Earthquakes (ETAS)',
                               labels={'Relative_Dates': 'Days +/- Large Event'},
                               color_continuous_scale='Viridis', opacity=0.7, 
                               size_max=10, width=800, height=800
)

#new dataframe to store binned data
binned_data_usgs = pd.DataFrame()
for index, row in usgs_large.iterrows():
    start_date = pd.to_datetime(row['date'])
    #binning data before/after major event
    usgs_binned = location_binning(usgs, start_date, date_range)
    relative_dates = (pd.to_datetime(usgs_binned['date']) - start_date).dt.days
    
    binned_data_usgs = pd.concat([
        binned_data_usgs,
        pd.DataFrame({
            'Day 0': [start_date.strftime('%Y-%m-%d')] * len(usgs_binned),
            'Latitude': usgs_binned['latitude'],
            'Longitude': usgs_binned['longitude'],
            'Relative_Dates': relative_dates,
            'Energy': usgs_binned['energy']
        })
    ])

fig_usgs_location = px.scatter(binned_data_usgs, x='Longitude', y='Latitude', color='Relative_Dates', 
                size='Energy',animation_frame='Day 0',
                title='Energy Locations Before/After Large Earthquakes (USGS)',
                labels={'Relative_Dates': 'Days +/- Large Event'},
                color_continuous_scale='Viridis', opacity=0.7,
                size_max=10, width=800, height=800
)

def is_mainshock(x) -> bool:
    return x == 'b'

#creating mainshock column
etas_aftershock['is_mainshock'] = etas_aftershock['aftershock'].apply(is_mainshock)
etas_aftershock['next_aftershock'] = etas_aftershock['aftershock'].shift(-1)

#assign mainshock id and corresponding aftershock
def assign_mainshock_id(row):
    if row['is_mainshock']:
        #mainshock - take the integer part from the next row's 'aftershock' value
        if '.' in str(row['next_aftershock']):
            return int(str(row['next_aftershock']).split('.')[0])
        else:
            return -1  #invalid aftershock
    else:
        #aftershock - take the integer part from the current row's 'aftershock' value
        if '.' in str(row['aftershock']):
            return int(str(row['aftershock']).split('.')[0])
        else:
            return -1  #invalid aftershock

etas_aftershock['mainshock_id'] = etas_aftershock.apply(assign_mainshock_id, axis=1)
etas_aftershock.drop('next_aftershock', axis=1, inplace=True)
etas_aftershock['aftershock_num'] = etas_aftershock['aftershock'].apply(lambda x: int(x.split('.')[1]) if x != 'b' else -1)

max_aftershock_num = etas_aftershock.groupby('mainshock_id')['aftershock_num'].max().reset_index()
max_aftershock_num.rename(columns={'aftershock_num': 'max_aftershock_num'}, inplace=True)

etas_aftershock = pd.merge(etas_aftershock, max_aftershock_num, on='mainshock_id', how='left')

mainshocks = etas_aftershock.loc[etas_aftershock['aftershock'] == 'b']
top_50_earthquakes = mainshocks.sort_values(by='mag', ascending=False).head(50)

fig_mag_vs_aftershocks = go.Figure(data=go.Scatter(x=top_50_earthquakes['mag'], y=top_50_earthquakes['max_aftershock_num'], mode='markers', name='Data'))
fig_mag_vs_aftershocks.update_layout(title='Magnitude vs. Number of Aftershocks',
                  xaxis_title='Magnitude',
                  yaxis_title='Number of Aftershocks')

#quadratic curve of best fit
def quadratic_model(x, a, b, c):
    return a*(x-b)**2 + c

x_data = top_50_earthquakes['mag']

popt, pcov = curve_fit(quadratic_model, x_data, top_50_earthquakes['max_aftershock_num'], p0=[3,2,-16])
a_opt, b_opt, c_opt = popt
x_model = np.linspace(min(x_data), max(x_data), 100)
y_model = quadratic_model(x_model, a_opt, b_opt, c_opt)

fig_mag_vs_aftershocks.add_trace(go.Scatter(x=x_model, y=y_model, mode='lines', name='Quadratic Curve of Best Fit'))

#correlation between largest earthquakes and aftershock
correlation = top_50_earthquakes['mag'].corr(top_50_earthquakes['max_aftershock_num'])
print(f"The correlation between magnitude and no. of aftershocks for large earthquakes is: {correlation}")

correlation = top_50_earthquakes['mag'].corr(np.log(top_50_earthquakes['max_aftershock_num']))
print(f"The correlation between magnitude and log of no. of aftershocks for large earthquakes is: {correlation}")

#correlation between earthquakes and aftershock
correlation = mainshocks['mag'].corr(mainshocks['max_aftershock_num'])
print(f"The general correlation between magnitude and no. of aftershocks is: {correlation}")

fig_mag_vs_aftershocks_log = go.Figure(data=go.Scatter(x=top_50_earthquakes['mag'], y=np.log(top_50_earthquakes['max_aftershock_num']), mode='markers', name='Data'))
fig_mag_vs_aftershocks_log.update_layout(title='Magnitude vs. Number of Aftershocks (Log)',
                  xaxis_title='Magnitude',
                  yaxis_title='Number of Aftershocks')

def distance_calculation(lat_diff, long_diff):
    return np.sqrt(np.square(lat_diff) + np.square(long_diff))

#distance between main and aftershocks
etas_aftershock['distance_to_mainshock'] = - 1
etas_aftershock['lat_distance'] = np.nan
etas_aftershock['long_distance'] = np.nan

for index, mainshock in etas_aftershock[etas_aftershock['aftershock'] == 'b'].iterrows():
    long = mainshock['longitude']
    lat = mainshock['latitude']
    mainshock_id = mainshock['mainshock_id']
    
    # Calculate the distance for each aftershock of this mainshock
    aftershocks = etas_aftershock[(etas_aftershock['mainshock_id'] == mainshock_id) & (etas_aftershock['mainshock_id'] != -1)]
    for aftershock_index, aftershock in aftershocks.iterrows():
        after_long = aftershock['longitude']
        after_lat = aftershock['latitude']
        
        lat_diff = after_lat - lat
        long_diff = after_long - long
        etas_aftershock.at[aftershock_index, 'lat_distance'] = lat_diff
        etas_aftershock.at[aftershock_index, 'long_distance'] = long_diff
        
        etas_aftershock.at[aftershock_index, 'distance_to_mainshock'] = distance_calculation(lat_diff, long_diff)

#max distance between main and aftershocks
etas_aftershock['max_distance'] = -1

for mainshock_id in etas_aftershock['mainshock_id'].unique():
    if mainshock_id == -1:
        continue
    # maximum distance for this mainshock's aftershocks
    max_distance = etas_aftershock[etas_aftershock['mainshock_id'] == mainshock_id]['distance_to_mainshock'].max()
    etas_aftershock.loc[etas_aftershock['mainshock_id'] == mainshock_id, 'max_distance'] = max_distance
    
fig_aftershock_distances = px.scatter(etas_aftershock, x="lat_distance", y="long_distance", color="distance_to_mainshock",
                title="Spatial Diagram", width=800, height=800)
fig_aftershock_distances.update_layout(xaxis_title='X', yaxis_title='Y')

#aftershock distance for earthquakes with 10+ aftershocks
over_10_aftershocks = etas_aftershock[etas_aftershock['max_aftershock_num'] >= 10]
fig_aftershock_distances_over_10 = px.scatter(over_10_aftershocks, x="lat_distance", y="long_distance", color="distance_to_mainshock",
                title="Spatial Diagram (10+ Aftershocks)", width=800, height=800)
fig_aftershock_distances_over_10.update_layout(xaxis_title='X', yaxis_title='Y')


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Capstone Research Spr 2024", style={'textAlign': 'center'}),
    html.H2("Vishal Kamalakrishnan", style={'textAlign': 'center'}),
    html.P("This Capstone research, under Professor Geoffrey Fox, builds on the work of the Independent Study group of Fall 2023 with the following goals:"),
    html.Ul(children=[
        html.Li("Establish differences and similarities towards generated earthquake data from the ETAS model and recorded earthquake data from USGS"),
        html.Li("Establish how the data of ETAS is generated and suggest optimal paramaters to match USGS data"),
        html.Li("Establish similarities and differences on using ETAS data vs USGS data as training data for earthquake energy models"),
        html.Li("Suggest improvements towards existing earthquake models based on observations from USGS data")
    ], style={'textAlign': 'left', 'padding': '20px'}),
    html.P("The analysis on ETAS was doing using modified parameters and code that sought to more closely resemble USGS data and create a more accurate representation of "),
    dcc.Tabs(id="tabs", value='tab-large-event-analysis', children=[
        dcc.Tab(label='Large Event Analysis', value='tab-large-event-analysis', children=[
            dcc.Tabs(id='large-event-analysis-tabs', value='tab-earthquake-locations', children=[
                dcc.Tab(label='Earthquake Locations', value='tab-earthquake-locations', children=[
                    html.P("These plots look at earthquake locations before and after a large earthquake for both USGS and ETAS data"), 
                    html.Div([
                        dcc.Graph(id='usgs-location-plot', figure=fig_usgs_location),
                        dcc.Graph(id='etas-location-plot', figure=fig_etas_location)
                    ], style={'display': 'flex', 'flex-direction': 'row'}),
                    html.H4("Observations"), 
                    html.Ul(children=[
                        html.Li("Both ETAS and USGS etablish clustering on earthquake close in time to a large earthquake"),
                        html.Li("ETAS shows a more even spread of events while USGS shows more 'radomness' across the spread of earthquakes"),
                        html.Li("There are a significantly higher amount of earthquakes clustered after large events in USGS"),
                    ], style={'textAlign': 'left', 'padding': '20px'}),
                ]),
                dcc.Tab(label='Energy Analysis', value='tab-energy-analysis', children=[
                    html.P("These plots look at energy before and after a large earthquake for USGS and ETAS"), 
                    html.Div([
                        dcc.Graph(id='usgs-energy-plot', figure=fig_usgs),
                        dcc.Graph(id='etas-energy-plot', figure=fig_etas),
                        dcc.Graph(id='average-difference-plot', figure=fig_difference)
                    ]),
                    html.H4("Observations"), 
                    html.Ul(children=[
                        html.Li("USGS shows a much higher spike of energy close to the day of large earthquakes compared to ETAS"),
                        html.Li("ETAS (modified) shows a higher average amount of energy and is more consistent across the board"),
                        html.Li("ETAS shows a higher amount of large earthquakes compared to USGS"),
                    ], style={'textAlign': 'left', 'padding': '20px'}),
                ])
            ])
        ]),
        dcc.Tab(label='Aftershock Analysis', value='tab-aftershock-analysis', children=[
            dcc.Tabs(id='aftershocks-tabs', value='tab-aftershocks', children=[
                dcc.Tab(label='Aftershock Correlations', value='tab-aftershock-correlationss', children=[
                    html.P("These plots look at the corelation between magnitude and number of aftershocks for ETAS"), 
                    html.Div([
                        dcc.Graph(id='mag-vs-aftershocks', figure=fig_mag_vs_aftershocks),
                        dcc.Graph(id='mag-vs-aftershocks-log', figure=fig_mag_vs_aftershocks_log)
                    ]),
                    html.H4("Observations"), 
                    html.Ul(children=[
                        html.Li("Except for outliers, the magnitude of the earthquake is strongly correlated with Log of number of aftershocks"),
                    ], style={'textAlign': 'left', 'padding': '20px'}),
                ]),
                dcc.Tab(label='Aftershock Distances', value='tab-aftershock-distances', children=[
                html.P("These plots look at the distance of aftershocks from their associated mainshock in ETAS"), 
                    html.Div([
                        dcc.Graph(id='aftershock-distances-plot', figure=fig_aftershock_distances),
                        dcc.Graph(id='aftershock-distances-over-10-plot', figure=fig_aftershock_distances_over_10),
                    ],  style={'display': 'flex', 'flex-direction': 'row'}),
                    html.H4("Observations"), 
                    html.Ul(children=[
                        html.Li("Aftershocks show a wide range of distance from the corresponding mainshock"),
                        html.Li("Most of the aftershocks are clustered within distance 1 (which is approx 70 miles)"),
                        html.Li("More clustering is present when there are over 10+ aftershocks"),
                        html.Li("Spatial location of aftershocks does not seem to have a high degree of correlation with the location of the mainshock - the randomness of earthquakes is at play"),
                    ], style={'textAlign': 'left', 'padding': '20px'}),
                ])
            ])
        ]),
        dcc.Tab(label='Cell Grid Analysis', value='tab-cell-grids', children=[
            html.Div([
                html.Iframe(src="./assets/etas_forcast.html", style={"height": "800px", "width": "50%"}),
                html.Iframe(src="./assets/usgs_forcast.html", style={"height": "800px", "width": "50%"})
            ],  style={'display': 'flex', 'flex-direction': 'row'}),
            
        ])
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)
