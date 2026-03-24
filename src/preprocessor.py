import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

class SmartPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, add_polynomial=False, degree=2, handle_outliers=False):
        self.add_polynomial = add_polynomial
        self.degree = degree
        self.handle_outliers = handle_outliers

    def fit(self, x, y=None):
        x = pd.DataFrame(x)

        #column detection
        self.num_cols = x.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = x.select_dtypes(exclude=np.number).columns.tolist()

        #pipelines
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        #fit numerical data
        x_num = self.num_imputer.fit_transform(x[self.num_cols])
        self.scaler.fit(x_num)

        #fit categorical data
        x_cat = self.cat_imputer.fit_transform(x[self.cat_cols])
        self.encoder.fit(x_cat)

        #polynomial
        if self.add_polynomial:
            self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            self.poly.fit(x_num)

        return self

    def transform(self, x):
        x = pd.DataFrame

        #numerical
        x_num = self.num_imputer.fit_transform(x[self.num_cols])

        if self.handle_outliers:
            x_num = self._clip_outliers(x_num)

        x_num = self.scaler.transform(x_num)

        if self.add_polynomial:
            x_num = self.poly.transform(x_num)

        #categorical
        x_cat = self.cat_imputer.transform(x[self.cat_cols])
        x_cat = self.encoder,self.transform(x_cat)

        #combine
        x_final = np.hstack([x_num, x_cat])

        return x_final
    
    def _clip_outliers(self, x):
        Q1 = np.percentile(x, 25, axis=0)
        Q3 = np.percentile(x, 75, axis=0)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return np.clip(x, lower, upper)
    
    def get_feature_names_out(self):
        num_features = self.num_cols

        if self.add_polynomial:
            num_features = self.poly.get_feature_names_out(self.num_cols)

        cat_features = self.encoder.get_feature_names_out(self.cat_cols)

        return np.concatenate([num_features, cat_features])