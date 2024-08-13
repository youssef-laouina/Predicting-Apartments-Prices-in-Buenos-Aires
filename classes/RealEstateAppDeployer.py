import logging
from geopy.geocoders import Nominatim  # type: ignore
from dash import Dash, html, dcc, Input, Output, State, dash_table 
from dash_extensions.javascript import assign
import dash_leaflet as dl
import scipy.stats as sps
from scipy.special import inv_boxcox
import pandas as pd
from flask_caching import Cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure caching
cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 60})


class CoordinateConverter:
    def __init__(self, user_agent="coordinateconverter"):
        self.geolocator = Nominatim(user_agent=user_agent)

    def get_address(self, latitude, longitude):
        try:
            location = self.geolocator.reverse(
                (latitude, longitude), exactly_one=True, timeout=10, addressdetails=True, language='en', zoom=16  # type: ignore	
            )  # type: ignore
            return location.address  # type: ignore
        except Exception as e:
            logger.error(f"Error retrieving address: {e}")
            return "Not Found!"

class RealEstateApp:
    def __init__(self, df_full, model, scaler):
        self.app = Dash(__name__)
        self.converter = CoordinateConverter()
        self.df_full = df_full
        self.model = model
        self.scaler = scaler

        self.app.layout = self.build_layout()
        self.configure_callbacks()
        
        cache.init_app(self.app.server)  # type: ignore  # Initialize cache with the Flask server

    def build_layout(self):
        # Define event handlers
        eventHandlers = dict( 
            click=assign("function(e, ctx){ctx.setProps({data: {lat: e.latlng.lat, lng: e.latlng.lng}})}")
        )

        # Build layout
        layout = html.Div([
            dcc.Store(id='address_cache'),  # Store for caching addresses
            dcc.Store(id='prediction_cache'),  # Store for caching predictions
            self.build_header(),
            self.build_content(eventHandlers)
        ], style={'background-color': 'white', 'max-width': '1200px', 'margin': '0 auto', 'backgroundColor': '#FFEBD0', 'padding': '20px', 'border': '2px solid #17616E', 'border-radius': '10px'})
        
        return layout

    def build_header(self):
        return html.Div([
            html.H1("Real Estate Price Prediction", style={'text-align': 'center'}),
        ], style={'font-weight': 'bold', 'font-family':'Calibri','color':'#FD8916', 'background-color': '#17616E', 'width': '60%', 'padding':'1px','text-align': 'center', 'margin': '0 auto', 'border': '0px solid white', 'border-radius': '10px', 'margin-bottom': '10px'})

    def build_content(self, eventHandlers):
        return html.Div([
            self.build_form_section(),
            self.build_map_section(eventHandlers)
        ], style={'display': 'flex', 'justify-content': 'space-between'})

    def build_form_section(self):
        return html.Div([
            self.build_form_inputs(),
            self.build_prediction_output()
        ], style={'width': '40%', 'padding': '20px', 'border': '2px solid #17616E', 'border-radius': '10px', 'margin-right': '10px', 'background-color': 'white'})

    def build_form_inputs(self):
        return html.Div([
            html.Div([
                dcc.Input(id='covered_surface', type='number', placeholder='Surface Covered in m\u00b2',
                          style={'margin': '10px', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #FD8916'}),

                dcc.Input(id='latitude', type='number', placeholder='Latitude',
                          style={'margin': '10px', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #FD8916'}),

                dcc.Input(id='longitude', type='number', placeholder='Longitude',
                          style={'margin': '10px', 'padding': '10px', 'border-radius': '8px', 'border': '1px solid #FD8916'}),
            ], style={'width': '100%', 'border': '0px solid #007BFF', 'border-radius': '10px'}),

            html.Div([
                html.Button('Predict Price', id='predict_button', n_clicks=0,
                            style={'font-weight': 'bold', 'margin': '10px', 'padding': '10px', 'background-color': '#FD8916', 'color': '#173b61', 'border-radius': '8px', 'border': '1px solid #173b61', 'float': 'right'}),
            ], style={'width': '100%', 'border': '0px solid #007BFF', 'border-radius': '10px', 'display': 'flex', 'justify-content': 'right'}),
        ], style={'border': '2px solid #17616E', 'border-radius': '10px'})

    def build_prediction_output(self):
        return html.Div(id='prediction_output', style={'width': '100%', 'height': 'auto', 'margin-top': '10px','border': '2px solid #17616E', 'border-radius': '10px', 'padding': '0px', 'display':'none'},
                        children=[
                            self.build_prediction_detail('Predicted Price:', 'predicted_price', '#FD8916'),
                            self.build_prediction_detail('Apartment Address:', 'address', '#17616E'),
                            self.build_municipality_info()
                        ])

    def build_prediction_detail(self, label, id_, color):
        return html.Div([
            html.Div(label, style={'display': 'inline-block', 'font-weight': 'bold', 'font-family':'Calibri'}),
            html.Div(id=id_, style={'font-weight': 'bold', 'font-family':'Calibri', 'display': 'inline-block', 'margin-left': '20px', 'color': color}),
        ], style={'padding': '10px', 'margin-bottom': '10px'})

    def build_municipality_info(self):
        return html.Div([
            html.Div("Municipality Statistics:", style={'display': 'inline-block', 'font-weight': 'bold', 'font-family':'Calibri'}),
            html.Div(id='municipality_name', style={'font-weight': 'bold', 'font-family':'Calibri','color':'#FD8916', 'margin': '5px', 'display': 'flex', 'justify-content': 'center'}),
            html.Div(id='municipality_info', style={'font-weight': 'bold', 'font-family':'Calibri', 'color': '#17616E', 'margin': '10px', 'display': 'flex', 'justify-content': 'center'}),
        ], style={'padding': '10px', 'margin-bottom': '10px'})

    def build_map_section(self, eventHandlers):
        return html.Div([
            dl.Map(
                children=[
                    dl.TileLayer(),
                    dl.Marker(id='marker', position=[33.6086, -7.6327])],
                eventHandlers=eventHandlers,
                style={'height': '60vh', 'width': '100%'},  # Set height to 80% of the viewport height
                center=[-34.6037, -58.3816],  # Center around Buenos Aires
                zoom=12,  # Set zoom level to view the city
                id='map'
            ),
            html.Div(id='coordinates', style={'marginTop': '20px', 'font-weight': 'bold', 'font-family':'Calibri', 'fontSize': '20px', 'color': 'black', 'backgroundColor': 'white', 'border-radius': '10px', 'padding': '10px', 'text-align': 'center'})
        ], style={'width': '60%', 'padding': '20px', 'border': '0px solid #FAF3DD', 'border-radius': '10px', 'background-color': '#173b61'})

    def configure_callbacks(self):
        self.app.callback(
            Output('prediction_output', 'style'),
            Input('predict_button', 'n_clicks'),
            prevent_initial_call=False
        )(self.show_prediction_output)

        self.app.callback(
            [Output('latitude', 'value'),
             Output('longitude', 'value'),
             Output('coordinates', 'children'),
             Output('marker', 'position')],
            Input('map', 'data')
        )(self.update_coordinates)

        self.app.callback(
            Output('address', 'children'),
            [Input('predict_button', 'n_clicks')],
            [State('latitude', 'value'),
             State('longitude', 'value')]
        )(self.update_address)

        self.app.callback(
            Output('predicted_price', 'children'),
            [Input('predict_button', 'n_clicks')],
            [State('covered_surface', 'value'),
             State('latitude', 'value'),
             State('longitude', 'value')]
        )(self.update_predicted_price)

        self.app.callback(
            [Output('municipality_name', 'children'),
             Output('municipality_info', 'children')],
            [Input('predict_button', 'n_clicks')],
            [State('latitude', 'value'),
             State('longitude', 'value')]
        )(self.update_municipality_info)

    def show_prediction_output(self, n_clicks):
        if n_clicks > 0:
            return {'display': 'block', 'width': '100%', 'height': 'auto', 'margin-top': '10px', 'border': '2px solid #17616E', 'border-radius': '10px', 'padding': '0px'}
        return {'display': 'none'}

    def update_coordinates(self, data):
        if data:
            return data['lat'], data['lng'], f"Coordinates: Latitude {data['lat']:.6f}, Longitude {data['lng']:.6f}", [data['lat'], data['lng']]
        return None, None, "Click on the map to select coordinates", [33.6086, -7.6327]

    def update_address(self, n_clicks, lat, lon):
        if n_clicks > 0 and all(v is not None for v in [lat, lon]):
            address = self.converter.get_address(lat, lon)
            logger.info(f"Retrieved address: {address}")
            return address
        return "No coordinates provided"

    def update_predicted_price(self, n_clicks, surface, lat, lon):
        if n_clicks > 0 and all(v is not None for v in [surface, lat, lon]):

            # check if city is in Argentina by using the get_address function
            try:
                city = [element.strip() for element in self.converter.get_address(lat, lon).split(",")][-3]
            except:
                return 'So you want to buy an apartment in the sea... Very funny 😄'

            if city != 'Autonomous City of Buenos Aires':
                return '❌ The city is not in the Autonomous City of Buenos Aires! Please select another location.'

            # Prepare the feature vector
            features = pd.DataFrame([[lat, lon, surface]], columns=['latitude', 'longitude', 'surface_covered_in_m2'])

            # Transform the feature `surface_covered_in_m2` using the 'box-cox' transformation
            best_lambda = -0.3
            features.surface_covered_in_m2 = sps.boxcox(features.surface_covered_in_m2, best_lambda)

            # Scale the feature vector
            features_scaled = self.scaler.transform(features)      
            features_df = pd.DataFrame(features_scaled, columns=features.columns)  # type: ignore

            # Predict Apartment Price 
            prediction = self.model.predict(features_df)

            # Inverse Box-Cox transformation
            prediction_orginal_scale = inv_boxcox(prediction, best_lambda)[0]

            # Return the predicted price
            return f'${prediction_orginal_scale:,.0f}'
        return ''

    def update_municipality_info(self, n_clicks, lat, lon):
        if n_clicks > 0 and all(v is not None for v in [lat, lon]):
            try:
                # Get address
                address = self.converter.get_address(lat, lon)
                # Filter data by municipality
                municipality = [element.strip().lower() for element in address.split(",")]
                info_municipality = self.df_full[(self.df_full.municipality.str.lower().str.contains(municipality[1]))
                                            | (self.df_full.municipality.str.lower().str.contains(municipality[2]))
                                            | (self.df_full.municipality.str.lower().str.contains(municipality[3]))
                ]
            except Exception:
                return '', f"🔱 Poseidon has put up a ‘No Humans Allowed’ sign for the sea apartments. Apparently, he's not into ocean-side neighbors who don't appreciate underwater real estate!"
            
            if info_municipality.shape[0] == 0:
                return '', f"🤷🏻‍♂️ It seems that we do not have any information about this location"
            else:
                # Extract municipality name
                municipality_name = info_municipality.municipality.unique()[0]

                # Create table
                municipality_df = pd.DataFrame(info_municipality.price_aprox_usd.describe())[1:]  # Exclude 'count'

                # Create 'Measure' column
                municipality_df['Measure'] = 0

                # Order columns & rename `price_usd` to `Value`
                municipality_df = municipality_df[['Measure', 'price_aprox_usd']].rename(columns={'price_aprox_usd': 'Value'})  # type: ignore 

                # Format table
                municipality_df.Value = municipality_df.Value.apply(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)  # type: ignore

                # Rename rows to proper headings
                municipality_df['Measure'] = ['Average Price (USD)', 'Price Standard Deviation (USD)', 'Minimum Price (USD)', '25th Percentile Price (USD)', 
                                        'Median Price (USD)', '75th Percentile Price (USD)','Maximum Price (USD)']

                # Return the table            
                return municipality_name, dash_table.DataTable(
                    municipality_df.to_dict('records'),[{'name': 'Measure', 'id': 'Measure'}, {'name': 'Value', 'id': 'Value'}],
                    style_cell={'textAlign': 'left', 'font-family': 'Calibri', 'padding': '10px', 'border': '1px solid #17616E'},
                    style_header={'backgroundColor': '#17616E', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                    style_table={'width': '100%', 'margin': '0 auto'}
                    )
        
        return '', html.Div('')

    def run(self):
        self.app.run_server(debug=True)
