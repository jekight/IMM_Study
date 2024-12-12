import os 
import pandas as pd 
import numpy as np
import configparser
from InitialState.initial_state import InitialState
from KalmanFilters.kalman_filters import KalmanFilters
from IMM.imm import IMM
from Calcs.calcs import *
from GenerateTruth.generate_truth import GenerateTruth as gt
from GenerateMeasurements.generate_measurements import GenerateMeasurements as gm

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

def execute_imm_ncv_ca(meas_data=None,configs=None):
    # Data
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

def execute_imm(meas_data=None,configs=None):      
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

def run_simulations_old():
    # Load Truth Data
    truth_df = gt.quadcopter_profile2()
    # Load Kalman Filter Configs
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    simulations = []
    # 1000 Simulations
    for i in range(1000):
        # Create Measurements
        meas_data = gm.create_measurements_dict(truth_df.copy(),0.4,0.5,0.15,5)
        # Execute IMM Filters
        try:
            imm_df1 = execute_imm_ncv_ca(meas_data,configs)
            imm_df2 = execute_imm(meas_data,configs)
        except:
            sim = {
                'Test':i+1,
                'RMSE_PC_OVERALL':np.nan,
                'VE_PC_OVERALL':np.nan,
                'RMSE_PC_QM':np.nan,
                'VE_PC_QM':np.nan,
            }
            simulations.append(sim)
            continue 
        # Calculate errors
        error_df1 = calculate_errors(truth_df,imm_df1)
        error_df2 = calculate_errors(truth_df,imm_df2)
        # Calculate Overall Percent Change
        rmse_pc_overall = np.mean(((error_df2.RMSE_COMBINED - error_df1.RMSE_COMBINED)/error_df1.RMSE_COMBINED) * 100)
        ve_pc_overall = np.mean(((error_df2.VE3D - error_df1.VE3D)/error_df1.VE3D) * 100)
        # Calculate Percent Change For Quadcopter Motion
        sub_error1 = error_df1.loc[error_df1.Time >= 301,:].reset_index(drop=True)
        sub_error2 = error_df2.loc[error_df2.Time >= 301,:].reset_index(drop=True)
        rmse_pc_quadcopter = np.mean(((sub_error1.RMSE_COMBINED - sub_error1.RMSE_COMBINED)/sub_error1.RMSE_COMBINED) * 100)
        ve_pc_quadcopter = np.mean(((sub_error2.VE3D - sub_error2.VE3D)/sub_error2.VE3D) * 100)
        # Store Results
        sim = {
            'Test':i+1,
            'RMSE_PC_OVERALL':rmse_pc_overall,
            'VE_PC_OVERALL':ve_pc_overall,
            'RMSE_PC_QM':rmse_pc_quadcopter,
            'VE_PC_QM':ve_pc_quadcopter,
        }
        simulations.append(sim)
        del meas_data
        del imm_df1
        del imm_df2 
        del error_df1
        del error_df2
        continue 
    sim_df = pd.DataFrame(simulations)
    return sim_df

def run_simulations():
    # Load Truth Data
    truth_df = gt.quadcopter_profile2()
    # Load Kalman Filter Configs
    configs = configparser.ConfigParser()
    configs.read(os.path.join(os.getcwd(),'Configs','configs.ini'))
    simulations = []
    # 1000 Simulations
    n = 0
    while n != 1000:
        # Create Measurements
        meas_data = gm.create_measurements_dict(truth_df.copy(),0.4,0.5,0.15,5)
        # Execute IMM Filters
        try:
            imm_df1 = execute_imm_ncv_ca(meas_data,configs)
            imm_df2 = execute_imm(meas_data,configs)
        except:
            del meas_data
            continue 
        # Calculate errors
        error_df1 = calculate_errors(truth_df,imm_df1)
        error_df2 = calculate_errors(truth_df,imm_df2)
        # Calculate Overall Percent Change
        rmse_pc_overall = np.mean(((error_df2.RMSE_COMBINED - error_df1.RMSE_COMBINED)/error_df1.RMSE_COMBINED) * 100)
        ve_pc_overall = np.mean(((error_df2.VE3D - error_df1.VE3D)/error_df1.VE3D) * 100)
        # Calculate Percent Change For Quadcopter Motion
        sub_error1 = error_df1.loc[error_df1.Time >= 301,:].reset_index(drop=True)
        sub_error2 = error_df2.loc[error_df2.Time >= 301,:].reset_index(drop=True)
        rmse_pc_quadcopter = np.mean(((sub_error2.RMSE_COMBINED - sub_error1.RMSE_COMBINED)/sub_error1.RMSE_COMBINED) * 100)
        ve_pc_quadcopter = np.mean(((sub_error2.VE3D - sub_error1.VE3D)/sub_error1.VE3D) * 100)
        # Store Results
        sim = {
            'Test':n+1,
            'RMSE_PC_OVERALL':rmse_pc_overall,
            'VE_PC_OVERALL':ve_pc_overall,
            'RMSE_PC_QM':rmse_pc_quadcopter,
            'VE_PC_QM':ve_pc_quadcopter,
        }
        simulations.append(sim)
        n += 1
        del meas_data, imm_df1,imm_df2,error_df1,error_df2,sim,sub_error1,sub_error2
        continue 
    sim_df = pd.DataFrame(simulations)
    return sim_df

