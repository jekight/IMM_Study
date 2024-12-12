import pandas as pd
import numpy as np
from MotionModels.motion_models import MotionModels

class GenerateTruth():
    '''
    GenerateTruth is a class that contains various methods that calculates flight profiles in x,y,z coordinates:

    Author: Jeremy Kight
    '''
    @staticmethod
    # Simple demo straight and level profile
    def demo_profile1():
        # Time step
        dt = 1
        # Time duration
        t_f = 210
        # Initial position
        s = np.array([0,0,10])
        # Velocities
        v = np.array([0.2,0.2,0])
        # Motion Model
        mm = MotionModels()
        x,y,z = mm.constant_velocity(dt,t_f,s,v)
        # Create time vector
        t = np.arange(1,t_f+2*dt,dt)
        # Create dataframe
        df = pd.DataFrame(
            {
                'Time':t,
                'X':x,
                'Y':y,
                'Z':z
            }
        )
        return df
    
    # Demo 
    @staticmethod
    def demo_profile2():
        mm = MotionModels()
        dt = 1
        # Straight and Level
        t_f = 120
        s = np.array([0,0,10])
        v = np.array([0.2,0.2,0])
        x,y,z = mm.constant_velocity(dt,t_f,s,v)
        t1 = np.arange(1,t_f+2*dt,dt)
        # Coordinated right turn
        t2 = np.arange(t_f+dt,t_f+180 + dt,dt)
        total_yaw = np.radians(-45)
        total_pitch = np.radians(0)
        total_roll = np.radians(0)
        yaw_angle = total_yaw/len(t2)
        pitch_angle = total_pitch/len(t2)
        roll_angle = total_roll/len(t2)
        for i in range(len(t2)):
            roll = np.array([
                [1, 0, 0],
                [0, np.cos(roll_angle), -np.sin(roll_angle)],
                [0, np.sin(roll_angle), np.cos(roll_angle)]
            ])
            pitch = np.array([
                [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
                [0, 1, 0],
                [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
            ])
            yaw = np.array([
                [np.cos(yaw_angle),-np.sin(yaw_angle),0],
                [np.sin(yaw_angle),np.cos(yaw_angle),0],
                [0,0,1]
            ])
            R = np.dot(yaw,np.dot(pitch,roll))
            new_pos = R.dot(np.array([x[-1],y[-1],z[-1]]))
            x = np.append(x,[new_pos[0]])
            y = np.append(y,[new_pos[1]])
            z = np.append(z,[new_pos[2]])
            continue
        # Create dataframe
        df = pd.DataFrame(
            {
                'Time':np.append(t1,t2),
                'X':x,
                'Y':y,
                'Z':z
            }
        )
        return df
    
    # Quadcopter Motions
    @staticmethod
    def quadcopter_profile():
        # Initialize Motion Model Class
        mm = MotionModels()
        # Time between updates
        dt = 1
        # Straight and Level for 3 minutes
        t_f = 60*3
        s = np.array([0,0,10])
        v = np.array([0.2,0.2,0])
        x,y,z = mm.constant_velocity(dt,t_f,s,v)
        t = np.arange(0,t_f+dt,dt)
        model = ['Constant Velocity'] * len(t)
        # Decelerate for 2 seconds to stop
        t_f = 2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.array([0.2,0.2,0])
        a = np.array([-0.1,-0.1,0])
        xi,yi,zi = mm.constant_acceleration(dt,t_f,s,v,a)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Constant Acceleration'] * len(t_new)
        # Hover for 2 Minutes
        t_f = 60*2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.zeros(3)
        xi,yi,zi = mm.constant_velocity(dt,t_f,s,v)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Hovering'] * len(t_new)
        # Accelerate Straight Up for 30 Seconds
        t_f = 30
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.zeros(3)
        a = np.array([0,0,0.05])
        xi,yi,zi = mm.constant_acceleration(dt,t_f,s,v,a)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Constant Acceleration'] * len(t_new)
        # Decelerate for 2 seconds to stop
        t_f = 2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.array([0,0,z[-1]-z[-2]])
        a = np.array([0,0,-1*(v[2]/t_f)])
        xi,yi,zi = mm.constant_acceleration(dt,t_f,s,v,a)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Constant Acceleration'] * len(t_new)
        # Hover for 2 Minutes
        t_f = 60*2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.zeros(3)
        xi,yi,zi = mm.constant_velocity(dt,t_f,s,v)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Hovering'] * len(t_new)
        # Create dataframe
        df = pd.DataFrame(
            {
                'Time':t,
                'X':x,
                'Y':y,
                'Z':z,
                'Model':model
            }
        )
        return df

    # Quadcopter Profile 2 
    @staticmethod
    def quadcopter_profile2():
        mm = MotionModels()
        dt = 1
        # Straight and Level
        t_f = 120
        s = np.array([0,0,10])
        v = np.array([0.2,0.2,0])
        x,y,z = mm.constant_velocity(dt,t_f,s,v)
        t = np.arange(0,t_f+dt,dt)
        model = ['Constant Velocity'] * len(t)
        # Coordinated right turn
        t_f = 180
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        total_yaw = np.radians(-45)
        total_pitch = np.radians(0)
        total_roll = np.radians(0)
        yaw_angle = total_yaw/len(t_new)
        pitch_angle = total_pitch/len(t_new)
        roll_angle = total_roll/len(t_new)
        roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_angle), -np.sin(roll_angle)],
            [0, np.sin(roll_angle), np.cos(roll_angle)]
        ])
        pitch = np.array([
            [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
            [0, 1, 0],
            [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
        ])
        yaw = np.array([
            [np.cos(yaw_angle),-np.sin(yaw_angle),0],
            [np.sin(yaw_angle),np.cos(yaw_angle),0],
            [0,0,1]
        ])
        R = np.dot(yaw,np.dot(pitch,roll))
        for i in range(len(t_new)):
            new_pos = R.dot(np.array([x[-1],y[-1],z[-1]]))
            x = np.append(x,[new_pos[0]])
            y = np.append(y,[new_pos[1]])
            z = np.append(z,[new_pos[2]])
            continue
        t = np.append(t,t_new)
        model += ['Constant Acceleration'] * len(t_new)
        # Straight and Level
        t_f = 60
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.array([-0.2,-0.2,0])
        xi,yi,zi = mm.constant_velocity(dt,t_f,s,v)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Constant Velocity'] * len(t_new)
        # Decelerate for 2 seconds to stop
        t_f = 2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.array([0.2,0.2,0])
        a = np.array([-0.1,-0.1,0])
        xi,yi,zi = mm.constant_acceleration(dt,t_f,s,v,a)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Constant Acceleration'] * len(t_new)
        # Hover for 2 Minutes
        t_f = 60*2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.zeros(3)
        xi,yi,zi = mm.constant_velocity(dt,t_f,s,v)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Hovering'] * len(t_new)
        # Accelerate Straight Up for 30 Seconds
        t_f = 30
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.zeros(3)
        a = np.array([0,0,0.05])
        xi,yi,zi = mm.constant_acceleration(dt,t_f,s,v,a)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Constant Acceleration'] * len(t_new)
        # Decelerate for 2 seconds to stop
        t_f = 2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.array([0,0,z[-1]-z[-2]])
        a = np.array([0,0,-1*(v[2]/t_f)])
        xi,yi,zi = mm.constant_acceleration(dt,t_f,s,v,a)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Constant Acceleration'] * len(t_new)
        # Hover for 2 Minutes
        t_f = 60*2
        t_new = np.arange(t[-1]+dt,t[-1]+t_f+dt,dt)
        s = np.array([x[-1],y[-1],z[-1]])
        v = np.zeros(3)
        xi,yi,zi = mm.constant_velocity(dt,t_f,s,v)
        x = np.append(x,xi[1:])
        y = np.append(y,yi[1:])
        z = np.append(z,zi[1:])
        t = np.append(t,t_new)
        model += ['Hovering'] * len(t_new)
        # Create dataframe
        df = pd.DataFrame(
            {
                'Time':t,
                'X':x,
                'Y':y,
                'Z':z,
                'Model':model
            }
        )
        return df
        # return t,x,y,z,model


    # Save profile
    @staticmethod
    def save_profile(df=pd.DataFrame(),filename=None):
        if df.empty:
            print('Dataframe is empty.')
            return 
        # Save dataframe as csv to working directory if filename is given
        if filename:
            df.to_csv(path_or_buf=filename+'.csv',index=False)
        return
