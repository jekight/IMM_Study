import pandas as pd
import numpy as np

class GenerateMeasurements():
    '''
    GenerateMeasurements is a class that contains various methods that adds noise to flight profiles:

    Author: Jeremy Kight
    '''
    @staticmethod
    def create_measurements(df=pd.DataFrame(),x_std=0.1,y_std=0.1,z_std=0.1,dt=10):
        if df.empty:
            return
        if 'X' not in df.columns:
            raise ValueError("Please ensure there are x coordinates in the dataframe.")
        if 'Y' not in df.columns:
            raise ValueError("Please ensure there are y coordinates in the dataframe.")
        if 'Z' not in df.columns:
            raise ValueError("Please ensure there are z coordinates in the dataframe.")
        # Create zero mean white Gaussian noise
        x_noise = np.random.normal(0,x_std,np.shape(df.X))
        y_noise = np.random.normal(0,y_std,np.shape(df.Y))
        z_noise = np.random.normal(0,z_std,np.shape(df.Z))
        # Add noise to X,Y,Z 
        df.loc[:,'X'] = df.loc[:,'X'] + x_noise
        df.loc[:,'Y'] = df.loc[:,'Y'] + y_noise
        df.loc[:,'Z'] = df.loc[:,'Z'] + z_noise
        # Filter on Time every 10 seconds
        df = df.iloc[::dt].reset_index(drop=True)
        return df
    
    @staticmethod
    def save_measurements(df=pd.DataFrame(),filename=None):
        if df.empty:
            print('Dataframe is empty.')
            return 
        # Save dataframe as csv to working directory if filename is given
        if filename:
            df.to_csv(path_or_buf=filename+'.csv',index=False)
        return
    
    @staticmethod
    def create_measurements_dict(df=pd.DataFrame(),x_std=0.1,y_std=0.1,z_std=0.1,dt=10):
        if df.empty:
            return None
        if 'X' not in df.columns:
            raise ValueError("Please ensure there are x coordinates in the dataframe.")
        if 'Y' not in df.columns:
            raise ValueError("Please ensure there are y coordinates in the dataframe.")
        if 'Z' not in df.columns:
            raise ValueError("Please ensure there are z coordinates in the dataframe.")
        # Create zero mean white Gaussian noise
        x_noise = np.random.normal(0,x_std,np.shape(df.X))
        y_noise = np.random.normal(0,y_std,np.shape(df.Y))
        z_noise = np.random.normal(0,z_std,np.shape(df.Z))
        # Add noise to X,Y,Z 
        df.loc[:,'X'] = df.loc[:,'X'] + x_noise
        df.loc[:,'Y'] = df.loc[:,'Y'] + y_noise
        df.loc[:,'Z'] = df.loc[:,'Z'] + z_noise
        # Filter on Time every X seconds
        df = df.iloc[::dt].reset_index(drop=True)
        return df.to_dict('records')