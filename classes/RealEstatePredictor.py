import pandas as pd
import numpy as np
import scipy as sp
import logging
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


class RealEstateRegressor:
    def __init__(self, df: pd.DataFrame):
        self.df_full = df
        self.scaler = None
        self.model = None
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.debug('RealEstateRegressor initialized.')

    def process_data(self):
        logging.debug('Processing data...')
        self._subset_data()
        self._remove_influential_points()
        self._apply_boxcox()
        self._scale_data()
        logging.debug('Data processing complete.')

    def fit_and_store_model(self, model_path='results/model.pkl', scaler_path='results/scaler.pkl'):
        logging.debug('Fitting model...')
        self.model = self._fit_gb_model()
        logging.debug('Model fitting complete.')
        self._store_model(model_path, scaler_path)
    
    def _subset_data(self):
        """Prepare features and target variable, excluding 'municipality'."""
        df = self.df_full.drop(columns=['municipality'], axis=1)
        self.X = df.drop(columns=['price_aprox_usd'], axis=1)
        self.y = df.price_aprox_usd
        logging.debug('Features and target variable prepared.')

    def _split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        logging.debug('Data split into training and testing sets.')

    def _get_influential_points(self):
        """Identify influential points using OLS regression."""
        self._split_data()
        X_train_const = sm.add_constant(self.X_train)
        fitted_model = sm.OLS(self.y_train, X_train_const).fit()
        logging.debug('OLS regression model fitted to identify influential points.')
        return fitted_model.get_influence().summary_frame()

    def _remove_influential_points(self):
        """Remove points with high leverage or high residuals."""
        influence_df = self._get_influential_points()
        high_leverage = influence_df.cooks_d > 0.0003
        high_residual = np.abs(influence_df.standard_resid) > 3
        influential_points = high_leverage & high_residual
        self.X_train = self.X_train[~influential_points]
        self.y_train = self.y_train[~influential_points]
        logging.debug('Influential points removed from training data.')

    def _apply_boxcox(self, best_lambda=-0.3):
        """Apply Box-Cox transformation to specified features."""
        self.y_train = sp.stats.boxcox(self.y_train, best_lambda)  # type: ignore
        self.X_train['surface_covered_in_m2'] = sp.stats.boxcox(self.X_train['surface_covered_in_m2'], best_lambda)  # type: ignore
        logging.debug('Box-Cox transformation applied to features.')

    def _scale_data(self):
        """Scale the features and target variable."""
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        logging.debug('Data scaled using StandardScaler.')

    def _fit_gb_model(self, n_estimators=1000, random_state=42):
        """Fit the Gradient Boosting Regressor model."""
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(self.X_train, self.y_train)  # type: ignore
        logging.debug('Gradient Boosting Regressor model fitted.')
        return model

    def _store_model(self, model_path='results/model.pkl', scaler_path='results/scaler.pkl'):
        """Store the trained model and scaler using pickle."""
        if self.scaler is None or self.model is None:
            logging.error('Model and scaler must be fitted before storing.')
            return

        with open(scaler_path, 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)
        logging.debug('Scaler stored at %s.', scaler_path)

        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        logging.debug('Model stored at %s.', model_path)

    def run_pipeline(self):
        """Run the entire pipeline to process data, fit the model, and store the results."""
        logging.debug('Pipeline started.')
        self.process_data()
        self.fit_and_store_model()
        logging.debug('Pipeline completed successfully.')
