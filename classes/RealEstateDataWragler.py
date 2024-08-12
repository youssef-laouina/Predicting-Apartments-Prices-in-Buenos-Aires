import pandas as pd
import logging


class DataWrangler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.DataFrame()
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def wrangle(self):

        logging.debug('Reading CSV file from %s', self.filepath)
        self.df = pd.read_csv(self.filepath, low_memory=False)
        
        logging.debug('Subsetting data: Selecting only Apartments in Capital Federal, less than $400,000')
        self._subset_data()
        
        logging.debug('Removing outliers: Strategy --> Quantiles based on get_quantile_category() analysis')
        self._remove_outliers()
        
        logging.debug('Extracting relevant information from columns')
        self._split_columns()
        
        logging.debug('Dropping null columns: Strategy --> get_nan_columns() analysis (50%+ null columns will be dropped.)')
        self._drop_null_columns()
        
        logging.debug('Dropping irrelevant columns.')
        self._drop_unused_columns()

        logging.debug('Removing empty rows from municipality column.')
        self._remove_empty_rows()
        
        logging.debug('Dropping null values.')
        self._drop_null_values()

        proper_headings = ['municipality', 'latitude', 'longitude', 'surface_covered_in_m2', 'price_aprox_usd']
        logging.debug('Wrangling complete, returning dataframe with proper headings')
        return self.df[proper_headings].reset_index(drop=True)

    def _subset_data(self):
        mask_cf = self.df["place_with_parent_names"].str.contains("capital federal", case=False)
        mask_apt = self.df["property_type"] == "apartment"
        mask_price = self.df["price_aprox_usd"] <= 400_000
        self.df = self.df[mask_cf & mask_apt & mask_price]

    def _remove_outliers(self):
        mask_sc = self.df['surface_covered_in_m2'].between(34, 500)
        self.df = self.df[mask_sc]

    def _split_columns(self):
        self.df[["latitude", "longitude"]] = self.df["lat-lon"].str.split(",", expand=True).astype(float)
        self.df['municipality'] = self.df.place_with_parent_names.str.split('|', expand=True)[3]

    def _drop_null_columns(self):
        nan_cols = self._nan_columns(self.df)
        self.df.drop(columns=nan_cols, inplace=True)

    def _drop_unused_columns(self):
        # drop `lat-lon`
        self.df.drop(columns="lat-lon", inplace=True)

        # drop `place_with_parent_names` because we already extracted `department_municipality`
        self.df.drop(columns="place_with_parent_names", inplace=True)

        # drop `property_type`because of high cardinality
        self.df.drop(columns=['property_type'], inplace=True)

        # drop `operation`, `currency` and `properati_url` because these are irrelevant  
        self.df.drop(columns=['operation', 'currency', 'properati_url'], inplace=True)

        # drop `price` and `price_aprox_local_currency` because we already have `price_aprox_usd` 
        self.df.drop(columns=['price', 'price_aprox_local_currency'], inplace=True)

        # drop `rooms` and `surface_total_in_m2` because we already have `surface_covered_in_m2``
        self.df.drop(columns=['rooms', 'surface_total_in_m2'], inplace=True)
 
    def _drop_null_values(self):
        self.df.dropna(inplace=True)

    def _remove_empty_rows(self):
        self.df = self.df[self.df['municipality'].str.len() > 0]

    @staticmethod
    def _nan_columns(df):
        return df.columns[df.isna().mean() > 0.5]
    