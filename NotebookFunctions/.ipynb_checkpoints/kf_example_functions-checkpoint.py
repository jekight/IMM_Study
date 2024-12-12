import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from InitialState.initial_state import InitialState
from KalmanFilters.kalman_filters import KalmanFilters
from Charts.charts import Charts

def straight_level_const_vel(meas_noise=1,process_noise=0.0001,leg_pos=-0.2,height=500,width=800):
    truth1_df = pd.read_csv(os.path.join('Data','profile1_truth.csv'))
    meas1 = pd.read_csv(os.path.join('Data','profile1_measurements.csv')).to_dict('records')
    data = []
    meas_data = meas1
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            s = init_state.get_state(model=1)
            kf = KalmanFilters(model=1,dt=10,s=s,meas_noise=meas_noise,process_noise=process_noise)
            init_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
                'pred_x_sig':init_sig[0],
                'pred_y_sig':init_sig[1],
                'pred_z_sig':init_sig[2],
                'pred_vx_sig':init_sig[3],
                'pred_vy_sig':init_sig[4],
                'pred_vz_sig':init_sig[5],
                'meas_x_sig':init_sig[0],
                'meas_y_sig':init_sig[1],
                'meas_z_sig':init_sig[2],
                'meas_vx_sig':init_sig[3],
                'meas_vy_sig':init_sig[4],
                'meas_vz_sig':init_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf.predict()
            pred_sig = kf.get_uncertainties()
            kf.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            meas_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
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
    meas_df = pd.DataFrame(meas_data)
    ch = Charts()
    fig = ch.performance_subplot(
        truth_df = truth1_df,
        meas_df = meas_df,
        data_df = data_df,
        show_pred = True,
        show_meas = True,
        xy_title = 'Filter Performance',
        pos_sig_title = 'Position Uncertainty',
        vel_sig_title = 'Velocity Uncertainty',
        leg_pos = leg_pos,
        height = height,
        width = width,
    )
    return fig

def straight_level_const_vel_missed_meas(meas_noise=1,process_noise=0.0001,leg_pos=-0.2,height=500,width=800):
    truth1_df = pd.read_csv(os.path.join('Data','profile1_truth.csv'))
    meas1 = pd.read_csv(os.path.join('Data','profile1_measurements.csv')).to_dict('records')
    data = []
    meas_data = meas1
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            s = init_state.get_state(model=1)
            kf = KalmanFilters(model=1,dt=10,s=s,meas_noise=meas_noise,process_noise=process_noise)
            init_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
                'pred_x_sig':init_sig[0],
                'pred_y_sig':init_sig[1],
                'pred_z_sig':init_sig[2],
                'pred_vx_sig':init_sig[3],
                'pred_vy_sig':init_sig[4],
                'pred_vz_sig':init_sig[5],
                'meas_x_sig':init_sig[0],
                'meas_y_sig':init_sig[1],
                'meas_z_sig':init_sig[2],
                'meas_vx_sig':init_sig[3],
                'meas_vy_sig':init_sig[4],
                'meas_vz_sig':init_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf.predict()
            pred_sig = kf.get_uncertainties()
            if i >= 7 and i <= 12:
                meas_sig = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
            else:
                kf.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
                meas_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
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
    meas_df = pd.DataFrame(meas_data)
    idx = (meas_df.index >= 7) & (meas_df.index <=12)
    meas_df = meas_df.loc[~idx,:].reset_index(drop=True)
    ch = Charts()
    fig = ch.performance_subplot(
        truth_df = truth1_df,
        meas_df = meas_df,
        data_df = data_df,
        show_pred = True,
        show_meas = True,
        xy_title = 'Filter Performance',
        pos_sig_title = 'Position Uncertainty',
        vel_sig_title = 'Velocity Uncertainty',
        leg_pos = leg_pos,
        height = height,
        width = width,
    )
    return fig

def straight_level_const_acc(meas_noise=1,process_noise=0.0001,leg_pos=-0.2,height=500,width=800):
    truth_df = pd.read_csv(os.path.join('Data','profile1_truth.csv'))
    meas = pd.read_csv(os.path.join('Data','profile1_measurements.csv')).to_dict('records')
    data = []
    meas_data = meas
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            s = init_state.get_state(model=2)
            kf = KalmanFilters(model=2,dt=10,s=s,meas_noise=meas_noise,process_noise=process_noise)
            init_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
                'pred_x_sig':init_sig[0],
                'pred_y_sig':init_sig[1],
                'pred_z_sig':init_sig[2],
                'pred_vx_sig':init_sig[3],
                'pred_vy_sig':init_sig[4],
                'pred_vz_sig':init_sig[5],
                'meas_x_sig':init_sig[0],
                'meas_y_sig':init_sig[1],
                'meas_z_sig':init_sig[2],
                'meas_vx_sig':init_sig[3],
                'meas_vy_sig':init_sig[4],
                'meas_vz_sig':init_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf.predict()
            pred_sig = kf.get_uncertainties()
            kf.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            meas_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
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
    meas_df = pd.DataFrame(meas_data)
    ch = Charts()
    fig = ch.performance_subplot(
        truth_df = truth_df,
        meas_df = meas_df,
        data_df = data_df,
        show_pred = True,
        show_meas = True,
        xy_title = 'Filter Performance',
        pos_sig_title = 'Position Uncertainty',
        vel_sig_title = 'Velocity Uncertainty',
        leg_pos = leg_pos,
        height = height,
        width = width,
    )
    return fig

def maneuver_const_acc(meas_noise=1,process_noise=0.0001,leg_pos=-0.2,height=500,width=800):
    truth_df = pd.read_csv(os.path.join('Data','profile2_truth.csv'))
    meas = pd.read_csv(os.path.join('Data','profile2_measurements.csv')).to_dict('records')
    data = []
    meas_data = meas
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
        elif i == 2:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            s = init_state.get_state(model=2)
            kf = KalmanFilters(model=2,dt=10,s=s,meas_noise=meas_noise,process_noise=process_noise)
            init_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
                'pred_x_sig':init_sig[0],
                'pred_y_sig':init_sig[1],
                'pred_z_sig':init_sig[2],
                'pred_vx_sig':init_sig[3],
                'pred_vy_sig':init_sig[4],
                'pred_vz_sig':init_sig[5],
                'meas_x_sig':init_sig[0],
                'meas_y_sig':init_sig[1],
                'meas_z_sig':init_sig[2],
                'meas_vx_sig':init_sig[3],
                'meas_vy_sig':init_sig[4],
                'meas_vz_sig':init_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf.predict()
            pred_sig = kf.get_uncertainties()
            kf.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            meas_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
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
    meas_df = pd.DataFrame(meas_data)
    ch = Charts()
    fig = ch.performance_subplot(
        truth_df = truth_df,
        meas_df = meas_df,
        data_df = data_df,
        show_pred = True,
        show_meas = True,
        xy_title = 'Filter Performance',
        pos_sig_title = 'Position Uncertainty',
        vel_sig_title = 'Velocity Uncertainty',
        leg_pos = leg_pos,
        height = height,
        width = width,
    )
    return fig

def maneuver_const_vel(meas_noise=1,process_noise=0.0001,leg_pos=-0.2,height=500,width=800):
    truth_df = pd.read_csv(os.path.join('Data','profile2_truth.csv'))
    meas = pd.read_csv(os.path.join('Data','profile2_measurements.csv')).to_dict('records')
    data = []
    meas_data = meas
    for i in range(len(meas_data)):
        if i == 0:
            init_state = InitialState(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            continue
        elif i == 1:
            init_state.update_state(meas_data[i]['Time'],meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z'])
            s = init_state.get_state(model=1)
            kf = KalmanFilters(model=1,dt=10,s=s,meas_noise=meas_noise,process_noise=process_noise)
            init_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
                'pred_x_sig':init_sig[0],
                'pred_y_sig':init_sig[1],
                'pred_z_sig':init_sig[2],
                'pred_vx_sig':init_sig[3],
                'pred_vy_sig':init_sig[4],
                'pred_vz_sig':init_sig[5],
                'meas_x_sig':init_sig[0],
                'meas_y_sig':init_sig[1],
                'meas_z_sig':init_sig[2],
                'meas_vx_sig':init_sig[3],
                'meas_vy_sig':init_sig[4],
                'meas_vz_sig':init_sig[5],
            }
            data.append(data_dict)
            continue
        else:
            kf.predict()
            pred_sig = kf.get_uncertainties()
            kf.update(np.array([meas_data[i]['X'],meas_data[i]['Y'],meas_data[i]['Z']]))
            meas_sig = kf.get_uncertainties()
            data_dict = {
                'Time':meas_data[i]['Time'],
                'x_kf':kf.X[0],
                'y_kf':kf.X[1],
                'z_kf':kf.X[2],
                'vx_kf':kf.X[3],
                'vy_kf':kf.X[4],
                'vz_kf':kf.X[5],
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
    meas_df = pd.DataFrame(meas_data)
    ch = Charts()
    fig = ch.performance_subplot(
        truth_df = truth_df,
        meas_df = meas_df,
        data_df = data_df,
        show_pred = True,
        show_meas = True,
        xy_title = 'Filter Performance',
        pos_sig_title = 'Position Uncertainty',
        vel_sig_title = 'Velocity Uncertainty',
        leg_pos = leg_pos,
        height = height,
        width = width,
    )
    return fig

