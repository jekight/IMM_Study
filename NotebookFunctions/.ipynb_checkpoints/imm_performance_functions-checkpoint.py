import os 
import pandas as pd 
import numpy as np
import configparser
from InitialState.initial_state import InitialState
from KalmanFilters.kalman_filters import KalmanFilters
from IMM.imm import IMM
from Calcs.calcs import *

def filter_truth(truth_df=pd.DataFrame(),kf_df=pd.DataFrame()):
    if truth_df.empty or kf_df.empty:
        return pd.DataFrame()
    # Filter Truth for times in kf_df
    sub_truth_df = pd.DataFrame()
    for time in kf_df.Time.unique().tolist():
        row = truth_df.loc[truth_df.Time==time,:].reset_index(drop=True)
        sub_truth_df = pd.concat([sub_truth_df,row]).reset_index(drop=True)
        continue 
    return sub_truth_df 

def calculate_errors(truth_df=pd.DataFrame(),kf_df=pd.DataFrame()):
    if truth_df.empty or kf_df.empty:
        return pd.DataFrame()
    # Filter Truth
    sub_truth_df = filter_truth(truth_df,kf_df)
    # Calculate RMSE
    data = []
    for i in range(1,len(sub_truth_df)+1):
        rmse_x = rmse(sub_truth_df.X[:i],kf_df.X[:i])
        rmse_y = rmse(sub_truth_df.Y[:i],kf_df.Y[:i])
        rmse_z = rmse(sub_truth_df.Z[:i],kf_df.Z[:i])
        rmse_combined = rmse(
            np.array([sub_truth_df.X[:i],sub_truth_df.Y[:i],sub_truth_df.Z[:i]]).T,
            np.array([kf_df.X[:i],kf_df.Y[:i],kf_df.Z[:i]]).T
        )
        distance = ve3d(sub_truth_df.X[i-1],sub_truth_df.Y[i-1],sub_truth_df.Z[i-1],kf_df.X[i-1],kf_df.Y[i-1],kf_df.Z[i-1])
        row = {
            'Time':sub_truth_df.Time[i-1],
            'RMSE_X':rmse_x,
            'RMSE_Y':rmse_y,
            'RMSE_Z':rmse_z,
            'RMSE_COMBINED':rmse_combined,
            'VE3D':distance
        }
        data.append(row)
        continue 
    error_df = pd.DataFrame.from_dict(data)
    return error_df

def execute_kf_cv():
    # Filter configurations
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    # Data
    meas_data = pd.read_csv(os.path.join('Data','quadcopter_profile2_measurements.csv')).to_dict('records')
    # Kalman Filter
    data = []
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            # Constant Velocity
            s = init_state.get_state(model=1)
            kf_cv = KalmanFilters(
                model=1,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_cv']),
                process_noise=float(configs['PROCESS_NOISE']['kf_cv'])
            )
            pred_sig = kf_cv.get_uncertainties()
            meas_sig = kf_cv.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'X':kf_cv.X[0],
                'Y':kf_cv.X[1],
                'Z':kf_cv.X[2],
                'VX':kf_cv.X[3],
                'VY':kf_cv.X[4],
                'VZ':kf_cv.X[5],
                'pred_x_sig':pred_sig[0],
                'pred_y_sig':pred_sig[1],
                'pred_z_sig':pred_sig[2],
                'pred_vx_sig':pred_sig[3],
                'pred_vy_sig':pred_sig[4],
                'pred_vz_sig':pred_sig[5],
                'meas_x_sig':meas_sig[0],
                'meas_y_sig':meas_sig[1],
                'meas_z_sig':meas_sig[2],
                'meas_vx_sig':meas_sig[3],
                'meas_vy_sig':meas_sig[4],
                'meas_vz_sig':meas_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf_cv.predict()
            pred_sig = kf_cv.get_uncertainties()
            kf_cv.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            meas_sig = kf_cv.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'X':kf_cv.X[0],
                'Y':kf_cv.X[1],
                'Z':kf_cv.X[2],
                'VX':kf_cv.X[3],
                'VY':kf_cv.X[4],
                'VZ':kf_cv.X[5],
                'pred_x_sig':pred_sig[0],
                'pred_y_sig':pred_sig[1],
                'pred_z_sig':pred_sig[2],
                'pred_vx_sig':pred_sig[3],
                'pred_vy_sig':pred_sig[4],
                'pred_vz_sig':pred_sig[5],
                'meas_x_sig':meas_sig[0],
                'meas_y_sig':meas_sig[1],
                'meas_z_sig':meas_sig[2],
                'meas_vx_sig':meas_sig[3],
                'meas_vy_sig':meas_sig[4],
                'meas_vz_sig':meas_sig[5],
            }
            data.append(data_dict)
            continue
    data_df = pd.DataFrame(data)
    return data_df 

def execute_kf_ca():
    # Filter configurations
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    # Data
    meas_data = pd.read_csv(os.path.join('Data','quadcopter_profile2_measurements.csv')).to_dict('records')
    # Kalman Filter
    data = []
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            # Constant Velocity
            s = init_state.get_state(model=2)
            kf_ca = KalmanFilters(
                model=2,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_ca']),
                process_noise=float(configs['PROCESS_NOISE']['kf_ca'])
            )
            pred_sig = kf_ca.get_uncertainties()
            meas_sig = kf_ca.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'X':kf_ca.X[0],
                'Y':kf_ca.X[1],
                'Z':kf_ca.X[2],
                'VX':kf_ca.X[3],
                'VY':kf_ca.X[4],
                'VZ':kf_ca.X[5],
                'pred_x_sig':pred_sig[0],
                'pred_y_sig':pred_sig[1],
                'pred_z_sig':pred_sig[2],
                'pred_vx_sig':pred_sig[3],
                'pred_vy_sig':pred_sig[4],
                'pred_vz_sig':pred_sig[5],
                'meas_x_sig':meas_sig[0],
                'meas_y_sig':meas_sig[1],
                'meas_z_sig':meas_sig[2],
                'meas_vx_sig':meas_sig[3],
                'meas_vy_sig':meas_sig[4],
                'meas_vz_sig':meas_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf_ca.predict()
            pred_sig = kf_ca.get_uncertainties()
            kf_ca.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            meas_sig = kf_ca.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'X':kf_ca.X[0],
                'Y':kf_ca.X[1],
                'Z':kf_ca.X[2],
                'VX':kf_ca.X[3],
                'VY':kf_ca.X[4],
                'VZ':kf_ca.X[5],
                'pred_x_sig':pred_sig[0],
                'pred_y_sig':pred_sig[1],
                'pred_z_sig':pred_sig[2],
                'pred_vx_sig':pred_sig[3],
                'pred_vy_sig':pred_sig[4],
                'pred_vz_sig':pred_sig[5],
                'meas_x_sig':meas_sig[0],
                'meas_y_sig':meas_sig[1],
                'meas_z_sig':meas_sig[2],
                'meas_vx_sig':meas_sig[3],
                'meas_vy_sig':meas_sig[4],
                'meas_vz_sig':meas_sig[5],
            }
            data.append(data_dict)
            continue
    data_df = pd.DataFrame(data)
    return data_df 

def execute_kf_h():
    # Filter configurations
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    # Data
    meas_data = pd.read_csv(os.path.join('Data','quadcopter_profile2_measurements.csv')).to_dict('records')
    # Kalman Filter
    data = []
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            # Hovering
            s = init_state.get_state(model=1)
            kf_h = KalmanFilters(
                model=3,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_h']),
                process_noise=float(configs['PROCESS_NOISE']['kf_h'])
            )
            pred_sig = kf_h.get_uncertainties()
            meas_sig = kf_h.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'X':kf_h.X[0],
                'Y':kf_h.X[1],
                'Z':kf_h.X[2],
                'VX':kf_h.X[3],
                'VY':kf_h.X[4],
                'VZ':kf_h.X[5],
                'pred_x_sig':pred_sig[0],
                'pred_y_sig':pred_sig[1],
                'pred_z_sig':pred_sig[2],
                'pred_vx_sig':pred_sig[3],
                'pred_vy_sig':pred_sig[4],
                'pred_vz_sig':pred_sig[5],
                'meas_x_sig':meas_sig[0],
                'meas_y_sig':meas_sig[1],
                'meas_z_sig':meas_sig[2],
                'meas_vx_sig':meas_sig[3],
                'meas_vy_sig':meas_sig[4],
                'meas_vz_sig':meas_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf_h.predict()
            pred_sig = kf_h.get_uncertainties()
            kf_h.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            meas_sig = kf_h.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'X':kf_h.X[0],
                'Y':kf_h.X[1],
                'Z':kf_h.X[2],
                'VX':kf_h.X[3],
                'VY':kf_h.X[4],
                'VZ':kf_h.X[5],
                'pred_x_sig':pred_sig[0],
                'pred_y_sig':pred_sig[1],
                'pred_z_sig':pred_sig[2],
                'pred_vx_sig':pred_sig[3],
                'pred_vy_sig':pred_sig[4],
                'pred_vz_sig':pred_sig[5],
                'meas_x_sig':meas_sig[0],
                'meas_y_sig':meas_sig[1],
                'meas_z_sig':meas_sig[2],
                'meas_vx_sig':meas_sig[3],
                'meas_vy_sig':meas_sig[4],
                'meas_vz_sig':meas_sig[5],
            }
            data.append(data_dict)
            continue
    data_df = pd.DataFrame(data)
    return data_df 

def execute_imm_ncv_ca():
    # Filter configurations
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    # Data
    meas_data = pd.read_csv(os.path.join('Data','quadcopter_profile2_measurements.csv')).to_dict('records')        
    time = []
    x = []
    y = []
    z = []
    prob_cv = []
    prob_ca = []
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            # Constant Velocity
            s = init_state.get_state(model=1)
            kf_cv = KalmanFilters(
                model=1,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_cv']),
                process_noise=float(configs['PROCESS_NOISE']['kf_cv'])
            )
            # Constant Acceleration
            s = init_state.get_state(model=2)
            kf_ca = KalmanFilters(
                model=2,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_ca']),
                process_noise=float(configs['PROCESS_NOISE']['kf_ca'])
            )
            # Initialize IMM
            mu = np.array([0.7,0.3])
            M = np.array([[0.9,0.1],[0.1,0.9]])
            imm = IMM([kf_cv,kf_ca],mu,M)
            # Store Values
            time.append(meas_data[i]['Time'])
            x.append(imm.X[0])
            y.append(imm.X[1])
            z.append(imm.X[2])
            prob_cv.append(imm.mu[0])
            prob_ca.append(imm.mu[1])
            continue
        else:
            # IMM procedure
            imm.predict()
            imm.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            # Store values
            time.append(meas_data[i]['Time'])
            x.append(imm.X[0])
            y.append(imm.X[1])
            z.append(imm.X[2])
            prob_cv.append(imm.mu[0])
            prob_ca.append(imm.mu[1])
            continue
    kf_df = pd.DataFrame(
        {
            'Time':time,
            'X':x,
            'Y':y,
            'Z':z,
            'PROB_NCV':prob_cv,
            'PROB_CA':prob_ca,
        }
    )
    return kf_df

def execute_imm():
    # Filter configurations
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    # Data
    meas_data = pd.read_csv(os.path.join('Data','quadcopter_profile2_measurements.csv')).to_dict('records')        
    time = []
    x = []
    y = []
    z = []
    prob_cv = []
    prob_ca = []
    prob_h = []
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            # Constant Velocity
            s = init_state.get_state(model=1)
            kf_cv = KalmanFilters(
                model=1,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_cv']),
                process_noise=float(configs['PROCESS_NOISE']['kf_cv'])
            )
            # Constant Acceleration
            s = init_state.get_state(model=2)
            kf_ca = KalmanFilters(
                model=2,
                dt=10,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_ca']),
                process_noise=float(configs['PROCESS_NOISE']['kf_ca'])
            )
            # Hovering
            s = init_state.get_state(model=1)
            kf_h = KalmanFilters(
                model=3,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_h']),
                process_noise=float(configs['PROCESS_NOISE']['kf_h'])
            )
            # Initialize IMM
            mu = np.array([0.6,0.25,0.15])
            M = np.array([[0.6, 0.3, 0.1],   # Probabilities of transitioning from model 1 to 1, 2, and 3
                [0.15, 0.7, 0.15],     # Probabilities of transitioning from model 2 to 1, 2, and 3
                [0.1, 0.3, 0.6] # Probabilities of transitioning from model 3 to 1, 2, and 3
            ])  
            imm = IMM([kf_cv,kf_ca,kf_h],mu,M)
            # Store Values
            time.append(meas_data[i]['Time'])
            x.append(imm.X[0])
            y.append(imm.X[1])
            z.append(imm.X[2])
            prob_cv.append(imm.mu[0])
            prob_ca.append(imm.mu[1])
            prob_h.append(imm.mu[2])
        else:
            # IMM procedure
            imm.predict()
            imm.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            # Store values
            time.append(meas_data[i]['Time'])
            x.append(imm.X[0])
            y.append(imm.X[1])
            z.append(imm.X[2])
            prob_cv.append(imm.mu[0])
            prob_ca.append(imm.mu[1])
            prob_h.append(imm.mu[2])
    kf_df = pd.DataFrame(
        {
            'Time':time,
            'X':x,
            'Y':y,
            'Z':z,
            'PROB_NCV':prob_cv,
            'PROB_CA':prob_ca,
            'PROB_H':prob_h,
        }
    )
    return kf_df

def execute_imm2():
    # Filter configurations
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    # Data
    meas_data = pd.read_csv(os.path.join('Data','quadcopter_profile2_measurements.csv')).to_dict('records')        
    time = []
    x = []
    y = []
    z = []
    prob_cv = []
    prob_ca = []
    prob_h = []
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            # Constant Velocity
            s = init_state.get_state(model=1)
            kf_cv = KalmanFilters(
                model=1,
                dt=5,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_cv']),
                process_noise=float(configs['PROCESS_NOISE']['kf_cv'])
            )
            # Constant Acceleration
            s = init_state.get_state(model=2)
            kf_ca = KalmanFilters(
                model=2,
                dt=10,
                s=s,
                meas_noise=float(configs['MEAS_NOISE']['kf_ca']),
                process_noise=float(configs['PROCESS_NOISE']['kf_ca'])
            )
            # Hovering
            s = init_state.get_state(model=1)
            kf_h = KalmanFilters(
                model=4,
                dt=5,
                s=s[:3],
                meas_noise=float(configs['MEAS_NOISE']['kf_h']),
                process_noise=float(configs['PROCESS_NOISE']['kf_h'])
            )
            # Initialize IMM
            mu = np.array([0.6,0.25,0.15])
            M = np.array([[0.6, 0.3, 0.1],   # Probabilities of transitioning from model 1 to 1, 2, and 3
                [0.15, 0.7, 0.15],     # Probabilities of transitioning from model 2 to 1, 2, and 3
                [0.1, 0.3, 0.6] # Probabilities of transitioning from model 3 to 1, 2, and 3
            ])  
            imm = IMM([kf_cv,kf_ca,kf_h],mu,M)
            # Store Values
            time.append(meas_data[i]['Time'])
            x.append(imm.X[0])
            y.append(imm.X[1])
            z.append(imm.X[2])
            prob_cv.append(imm.mu[0])
            prob_ca.append(imm.mu[1])
            prob_h.append(imm.mu[2])
        else:
            # IMM procedure
            imm.predict()
            imm.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            # Store values
            time.append(meas_data[i]['Time'])
            x.append(imm.X[0])
            y.append(imm.X[1])
            z.append(imm.X[2])
            prob_cv.append(imm.mu[0])
            prob_ca.append(imm.mu[1])
            prob_h.append(imm.mu[2])
    kf_df = pd.DataFrame(
        {
            'Time':time,
            'X':x,
            'Y':y,
            'Z':z,
            'PROB_NCV':prob_cv,
            'PROB_CA':prob_ca,
            'PROB_H':prob_h,
        }
    )
    return kf_df

