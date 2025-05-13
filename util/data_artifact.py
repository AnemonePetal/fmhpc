import pandas as pd
import numpy as np

class Artifact:
    def __init__(self, df):
        self.df = df

    def constant(self, col, value):
        self.df[col] = value
        return self.df
    
    def gaussian(self, col, mean, std):
        self.df[col] = np.random.normal(mean, std, len(self.df))
        return self.df

    def uniform(self, col, low, high):
        self.df[col] = np.random.uniform(low, high, len(self.df))
        return self.df
    
    def poisson(self, col, lam):
        self.df[col] = np.random.poisson(lam, len(self.df))
        return self.df

    def add_gaussian_noise(self, col, mean, std,time_ranges,farmnodes=None):
        for start_time, end_time in time_ranges:
            if start_time > end_time:
                raise ValueError('Start time must be less than end time')
            if start_time == '':
                mask = (self.df['timestamp'] <= end_time)
            elif end_time == '':
                mask = (self.df['timestamp'] >= start_time)
            else:
                mask = (self.df['timestamp'] >= start_time) & (self.df['timestamp'] <= end_time)
            if farmnodes!=None:
                mask = mask & (self.df['instance'].isin(farmnodes))

            self.df.loc[mask,col] = self.df.loc[mask,col] + np.random.normal(mean, std, len(self.df.loc[mask,col]))
        return self.df