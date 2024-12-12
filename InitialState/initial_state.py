import numpy as np

class InitialState():
    '''
    InitialState is a class that is used to determine the initial state based on the first three measurements

    Author: Jeremy Kight
    '''
    def __init__(self,time=None,x=None,y=None,z=None):
        self.time = np.array([time])
        self.x = np.array([x])
        self.y = np.array([y])
        self.z = np.array([z])
        self.vx = np.array([]) 
        self.vy = np.array([]) 
        self.vz = np.array([]) 
        self.ax = np.array([]) 
        self.ay = np.array([]) 
        self.az = np.array([]) 
        self.num_meas = 1

    def update_state(self,time=None,x=None,y=None,z=None):
        # Add time update
        self.time = np.append(self.time,time)
        # Add new position measurements
        self.x = np.append(self.x,x)
        self.y = np.append(self.y,y)
        self.z = np.append(self.z,z)
        # Update measurement tracker
        self.num_meas += 1
        # Calculate current velocities
        self.vx = np.append(self.vx,(self.x[-1]-self.x[-2])/(self.time[-1]-self.time[-2]))
        self.vy = np.append(self.vy,(self.y[-1]-self.y[-2])/(self.time[-1]-self.time[-2]))
        self.vz = np.append(self.vz,(self.z[-1]-self.z[-2])/(self.time[-1]-self.time[-2]))
        # Calculate acceleration
        if self.num_meas >= 3:
            self.ax = np.append(self.ax,(self.vx[-1]-self.x[-2])/(self.time[-1]-self.time[-2]))
            self.ay = np.append(self.ay,(self.vy[-1]-self.y[-2])/(self.time[-1]-self.time[-2]))
            self.az = np.append(self.az,(self.vz[-1]-self.z[-2])/(self.time[-1]-self.time[-2]))
        return
    
    def get_state(self,model=1):
        if model == 1:
            s = np.array(
                [
                    self.x[-1],
                    self.y[-1],
                    self.z[-1],
                    self.vx[-1],
                    self.vy[-1],
                    self.vz[-1],
                ]
            )
        else:
            s = np.array(
                [
                    self.x[-1],
                    self.y[-1],
                    self.z[-1],
                    self.vx[-1],
                    self.vy[-1],
                    self.vz[-1],
                    self.ax[-1],
                    self.ay[-1],
                    self.az[-1],
                ]
            )
        return s 