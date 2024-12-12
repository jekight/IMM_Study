import numpy as np

class MotionModels():
    '''
    MotionModels is a class that contains various methods that describe different motion models:
        - Constant velocity
        - Constant acceleration
        - Constant jerk

    Author: Jeremy Kight
    '''

    # Constant x,y,z velocity
    def constant_velocity(self,dt=10,t_f=600,s=np.array([0,0,0]),v=np.array([0,0,0])):
        '''
        Inputs:
            - dt: time step
            - t_f: time duration (seconds)
            - s: Initial position in meters [x,y,z]
            - v: Velocity vector in m/s [vx,vy,vz]
        Outputs:
            - x,y,z of profile
        '''
        if type(s) == list:
            s = np.asarray(s)
        if type(v) == list:
            v = np.asarray(v)
        if len(s) != 3:
            raise ValueError("Please ensure the initial state, s, contains x,y, and z coordinates.")
        if len(v) != 3:
            raise ValueError("Please ensure the velocity vector, v, contains the velocities in the x,y, and z directions.")
        # Initialize time vector
        t = np.arange(0,t_f+dt,dt)
        # Calculate displacements
        x = s[0] + v[0]*t
        y = s[1] + v[1]*t
        z = s[2] + v[2]*t 
        return x,y,z
    
    # Constant x,y,z acceleration
    def constant_acceleration(self,dt=10,t_f=600,s=np.array([0,0,0]),v=np.array([0,0,0]),a=np.array([0,0,0])):
        '''
        Inputs:
            - dt: time step
            - t_f: time duration (seconds)
            - s: Initial position in meters [x,y,z]
            - v: Initial Velocity in m/s [vx,vy,vz]
            - a: Acceleration in m/s^2 [ax,ay,az]
        Outputs:
            - x,y,z of profile
        '''
        if type(s) == list:
            s = np.asarray(s)
        if type(v) == list:
            v = np.asarray(v)
        if type(a) == list:
            a = np.asarray(a)
        if len(s) != 3:
            raise ValueError("Please ensure the initial state, s, contains x,y, and z coordinates.")
        if len(v) != 3:
            raise ValueError("Please ensure the velocity vector, v, contains the velocities in the x,y, and z directions.")
        if len(a) != 3:
            raise ValueError("Please ensure the acceleration vector, a, contains the accelerations in the x,y, and z directions.")
        # Initialize time vector
        t = np.arange(0,t_f+dt,dt)
        # Calculate displacements
        x = s[0] + v[0]*t + 0.5*a[0]*(t**2)
        y = s[1] + v[1]*t + 0.5*a[1]*(t**2)
        z = s[2] + v[2]*t + 0.5*a[2]*(t**2)
        return x,y,z
    
    # Constant x,y,z jerk
    def constant_jerk(self,dt=10,t_f=600,s=np.array([0,0,0]),v=np.array([0,0,0]),a=np.array([0,0,0]),j=np.array([0,0,0])):
        '''
        Inputs:
            - dt: time step
            - t_f: time duration (seconds)
            - s: Initial position in meters [x,y,z]
            - v: Initial velocity in m/s [vx,vy,vz]
            - a: Initial acceleration in m/s^2 [ax,ay,az]
            - j: Jerk in m/s^3 [jx,jy,jz]
        Outputs:
            - x,y,z of profile
        '''
        if type(s) == list:
            s = np.asarray(s)
        if type(v) == list:
            v = np.asarray(v)
        if type(a) == list:
            a = np.asarray(a)
        if type(j) == list:
            j = np.asarray(j)
        if len(s) != 3:
            raise ValueError("Please ensure the initial state, s, contains x,y, and z coordinates.")
        if len(v) != 3:
            raise ValueError("Please ensure the velocity vector, v, contains the velocities in the x,y, and z directions.")
        if len(a) != 3:
            raise ValueError("Please ensure the acceleration vector, a, contains the accelerations in the x,y, and z directions.")
        if len(j) != 3:
            raise ValueError("Please ensure the jerk vector, j, contains the jerks in the x,y, and z directions.")
        # Initialize time vector
        t = np.arange(0,t_f+dt,dt)
        # Calculate displacements
        x = s[0] + v[0]*t + 0.5*a[0]*(t**2) + (1/6)*j[0]*(t**3)
        y = s[1] + v[1]*t + 0.5*a[1]*(t**2) + (1/6)*j[1]*(t**3)
        z = s[2] + v[2]*t + 0.5*a[2]*(t**2) + (1/6)*j[2]*(t**3)
        return x,y,z