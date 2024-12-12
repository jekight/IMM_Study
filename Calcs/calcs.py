import numpy as np

def rmse(actual=np.array([]),predicted=np.array([])):
    if len(actual)==0 or len(predicted)==0:
        return np.array([])
    return np.sqrt(((actual-predicted)**2).mean())

def ve3d(x1,y1,z1,x2,y2,z2):
    # Calculate distance
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
