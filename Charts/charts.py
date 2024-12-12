import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Charts():
    '''
    Charts is a class that contains various methods that creates charts of interest:

    Author: Jeremy Kight
    '''

    @staticmethod
    def xy_2d_chart(truth_df=pd.DataFrame(),meas_df=pd.DataFrame(),data_df=pd.DataFrame(),title=None,leg_pos=None,height=250,width=500):
        # Initialize
        fig = go.Figure()
        # Chart Layout
        fig.update_layout({
            'margin':{'t':0,'b':0,'l':0,'r':0},
            'width':width,
            'height':height,
        })
        if title:
            fig.update_layout({
                'margin':{'t':30,'b':0,'l':0,'r':0},
                'title':{
                    'text':title,
                    'xanchor':'center',
                    'yanchor':'top',
                    'x':0.5,
                },
            })
        if leg_pos:
            fig.update_layout({
                'legend':{
                    'orientation':"h",
                    'yanchor':'bottom',
                    'y':leg_pos,
                    'xanchor':'center',
                    'x':0.5,
                    'bgcolor':'white',
                    'bordercolor':'black',
                    'borderwidth':2
                },
            })
        # Update Axes
        fig.update_yaxes({'title':'y position'})
        fig.update_xaxes({'title':'x position'})
        # Truth
        if not truth_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=truth_df.X,
                    y=truth_df.Y,
                    mode='lines',
                    line={'width':1,'color':'black'},
                    showlegend=True,
                    name='Truth'
                )
            )
        # Measurements
        if not meas_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=meas_df.X,
                    y=meas_df.Y,
                    mode='markers',
                    marker={'size':8,'symbol':'square','color':'red'},
                    opacity=0.6,
                    showlegend=True,
                    name='Measurements'
                )
            )
        # Kalman Filter
        if not data_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=data_df.x_kf,
                    y=data_df.y_kf,
                    mode='markers',
                    marker={'size':8,'symbol':'circle','color':'blue'},
                    opacity=0.6,
                    showlegend=True,
                    name='Kalman Filter'
                )
            )
        return fig 
    
    @staticmethod
    def pos_uncertainty_chart(data_df=pd.DataFrame(),show_pred=True,show_meas=True,title=None,leg_pos=None,height=250,width=500):
        # Initialize
        fig = go.Figure()
        # Chart Layout
        fig.update_layout({
            'margin':{'t':0,'b':0,'l':0,'r':0},
            'width':width,
            'height':height,
        })
        if title:
            fig.update_layout({
                'margin':{'t':30,'b':0,'l':0,'r':0},
                'title':{
                    'text':title,
                    'xanchor':'center',
                    'yanchor':'top',
                    'x':0.5,
                },
            })
        if leg_pos:
            fig.update_layout({
                'legend':{
                    'orientation':"h",
                    'yanchor':'bottom',
                    'y':leg_pos,
                    'xanchor':'center',
                    'x':0.5,
                    'bgcolor':'white',
                    'bordercolor':'black',
                    'borderwidth':2
                },
            })
        # Update Axes
        fig.update_yaxes({'title':'Uncertainty'})
        fig.update_xaxes({'title':'Time (seconds)'})
        # Uncertainty after prediction
        if not data_df.empty and show_pred:
            fig.add_trace(
                go.Scatter(
                    x=data_df.Time,
                    y=data_df.pred_x_sig,
                    mode='lines',
                    line={'width':1,'color':'blue'},
                    showlegend=True,
                    name='Uncertainty After Prediction'
                )
            )
        # Uncertainty after measurement
        if not data_df.empty and show_meas:
            fig.add_trace(
                go.Scatter(
                    x=data_df.Time,
                    y=data_df.meas_x_sig,
                    mode='lines',
                    line={'width':1,'color':'red'},
                    showlegend=True,
                    name='Uncertainty After Prediction'
                )
            )
        return fig

    @staticmethod
    def performance_subplot(truth_df=pd.DataFrame(),meas_df=pd.DataFrame(),data_df=pd.DataFrame(),show_pred=True,show_meas=True,xy_title=None,pos_sig_title=None,vel_sig_title=None,leg_pos=None,height=250,width=500):
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{'rowspan':2},{}],[None,{}]],
            subplot_titles=(xy_title,pos_sig_title,vel_sig_title)
        )
        # Chart Layout
        fig.update_layout({
            'margin':{'t':30,'b':0,'l':0,'r':0},
            'width':width,
            'height':height,
            'showlegend':True,
            'legend':{
                    'orientation':"h",
                    'yanchor':'bottom',
                    'y':leg_pos,
                    'xanchor':'center',
                    'x':0.5,
                    'bgcolor':'white',
                    'bordercolor':'black',
                    'borderwidth':2
                },
        })
        # XY Chart
        if not truth_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=truth_df.X,
                    y=truth_df.Y,
                    mode='lines',
                    line={'width':1,'color':'black'},
                    showlegend=True,
                    name='Truth'
                ),
                row=1,col=1,
            )
        if not meas_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=meas_df.X,
                    y=meas_df.Y,
                    mode='markers',
                    marker={'size':8,'symbol':'square','color':'red'},
                    opacity=0.6,
                    showlegend=True,
                    name='Measurements'
                ),
                row=1,col=1,
            )
        if not data_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=data_df.x_kf,
                    y=data_df.y_kf,
                    mode='markers',
                    marker={'size':8,'symbol':'circle','color':'blue'},
                    opacity=0.6,
                    showlegend=True,
                    name='Kalman Filter'
                ),
                row=1,col=1,
            )
        fig.update_yaxes({'title':'y position'},row=1,col=1)
        fig.update_xaxes({'title':'x position'},row=1,col=1)
        # Position Uncertainty
        if not data_df.empty and show_pred:
            fig.add_trace(
                go.Scatter(
                    x=data_df.Time,
                    y=data_df.pred_x_sig,
                    mode='markers',
                    marker={'size':8,'symbol':'diamond','color':'rgb(27,158,119)'},
                    showlegend=True,
                    opacity=0.6,
                    name='Uncertainty After Prediction'
                ),
                row=1,col=2,
            )
        if not data_df.empty and show_meas:
            fig.add_trace(
                go.Scatter(
                    x=data_df.Time,
                    y=data_df.meas_x_sig,
                    mode='markers',
                    marker={'size':8,'symbol':'diamond','color':'rgb(217,95,2)'},
                    showlegend=True,
                    opacity=0.6,
                    name='Uncertainty After Measurement'
                ),
                row=1,col=2,
            )
        fig.update_yaxes({'title':'Uncertainty'},row=1,col=2,)
        fig.update_xaxes({'title':'Time (seconds)'},row=1,col=2,)
        # Velocity Uncertainty
        if not data_df.empty and show_pred:
            fig.add_trace(
                go.Scatter(
                    x=data_df.Time,
                    y=data_df.pred_vx_sig,
                    mode='markers',
                    marker={'size':8,'symbol':'diamond','color':'rgb(27,158,119)'},
                    showlegend=False,
                    opacity=0.6,
                    name='Uncertainty After Prediction'
                ),
                row=2,col=2,
            )
        if not data_df.empty and show_meas:
            fig.add_trace(
                go.Scatter(
                    x=data_df.Time,
                    y=data_df.meas_vx_sig,
                    mode='markers',
                    marker={'size':8,'symbol':'diamond','color':'rgb(217,95,2)'},
                    showlegend=False,
                    opacity=0.6,
                    name='Uncertainty After Prediction'
                ),
                row=2,col=2,
            )
        fig.update_yaxes({'title':'Uncertainty'},row=2,col=2,)
        fig.update_xaxes({'title':'Time (seconds)'},row=2,col=2,)
        return fig

    @staticmethod
    def quadcopter_xyz(truth_df=pd.DataFrame(),meas_df=pd.DataFrame(),kf_df=pd.DataFrame(),leg_pos=None,height=None,width=None):
        fig = make_subplots(
            rows=3,
            cols=1,
            # specs=[[{'rowspan':2},{}],[None,{}]],
            subplot_titles=(
                'X Position vs. Time','Y Position vs. Time','Z Position vs. Time'
            )
        )
        # Chart Layout
        fig.update_layout({
            'margin':{'t':30,'b':0,'l':0,'r':0},
            'width':width,
            'height':height,
            'showlegend':True,
            'legend':{
                    'orientation':"h",
                    'yanchor':'bottom',
                    'y':leg_pos,
                    'xanchor':'center',
                    'x':0.5,
                    'bgcolor':'white',
                    'bordercolor':'black',
                    'borderwidth':2
                },
        })
        # X Position vs. Time Chart
        if not truth_df.empty and 'X' in truth_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=truth_df.Time,
                    y=truth_df.X,
                    mode='lines',
                    line={'width':1,'color':'black'},
                    showlegend=True,
                    name='Truth'
                ),
                row=1,col=1,
            )
        if not meas_df.empty and 'X' in meas_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=meas_df.Time,
                    y=meas_df.X,
                    mode='markers',
                    marker={'size':8,'symbol':'square','color':'red'},
                    opacity=0.6,
                    showlegend=True,
                    name='Measurements'
                ),
                row=1,col=1,
            )
        if not kf_df.empty and 'X' in kf_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=kf_df.Time,
                    y=kf_df.X,
                    mode='markers',
                    marker={'size':8,'symbol':'circle','color':'blue'},
                    opacity=0.6,
                    showlegend=True,
                    name='IMM Filter'
                ),
                row=1,col=1,
            )
        fig.update_yaxes({'title':'X Position'},row=1,col=1)
        fig.update_xaxes({'title':'Time (Seconds)'},row=1,col=1)
        # Y Position vs. Time Chart
        if not truth_df.empty and 'Y' in truth_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=truth_df.Time,
                    y=truth_df.Y,
                    mode='lines',
                    line={'width':1,'color':'black'},
                    showlegend=False,
                    name='Truth'
                ),
                row=2,col=1,
            )
        if not meas_df.empty and 'Y' in meas_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=meas_df.Time,
                    y=meas_df.Y,
                    mode='markers',
                    marker={'size':8,'symbol':'square','color':'red'},
                    opacity=0.6,
                    showlegend=False,
                    name='Measurements'
                ),
                row=2,col=1,
            )
        if not kf_df.empty and 'Y' in kf_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=kf_df.Time,
                    y=kf_df.Y,
                    mode='markers',
                    marker={'size':8,'symbol':'circle','color':'blue'},
                    opacity=0.6,
                    showlegend=False,
                    name='IMM Filter'
                ),
                row=2,col=1,
            )
        fig.update_yaxes({'title':'Y Position'},row=2,col=1)
        fig.update_xaxes({'title':'Time (Seconds)'},row=2,col=1)
        # Z Position vs. Time Chart
        if not truth_df.empty and 'Z' in truth_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=truth_df.Time,
                    y=truth_df.Z,
                    mode='lines',
                    line={'width':1,'color':'black'},
                    showlegend=False,
                    name='Truth'
                ),
                row=3,col=1,
            )
        if not meas_df.empty and 'Z' in meas_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=meas_df.Time,
                    y=meas_df.Z,
                    mode='markers',
                    marker={'size':8,'symbol':'square','color':'red'},
                    opacity=0.6,
                    showlegend=False,
                    name='Measurements'
                ),
                row=3,col=1,
            )
        if not kf_df.empty and 'Z' in kf_df.columns and 'Time' in truth_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=kf_df.Time,
                    y=kf_df.Z,
                    mode='markers',
                    marker={'size':8,'symbol':'circle','color':'blue'},
                    opacity=0.6,
                    showlegend=False,
                    name='IMM Filter'
                ),
                row=3,col=1,
            )
        fig.update_yaxes({'title':'Z Position'},row=3,col=1)
        fig.update_xaxes({'title':'Time (Seconds)'},row=3,col=1)
        return fig

    @staticmethod
    def truth_strip_chart(truth_df=pd.DataFrame(),height=None,width=None):
        # Initialize
        fig = go.Figure()
        # Chart Layout
        fig.update_layout({
            'margin':{'t':30,'b':0,'l':0,'r':0},
            'title':{
                    'text':'Quadcopter Actual Motion Model',
                    'xanchor':'center',
                    'yanchor':'top',
                    'x':0.5,
                },
            'width':width,
            'height':height,
        })
        # Update Axes
        fig.update_yaxes({'title':'Motion Model'})
        fig.update_xaxes({'title':'Time (Seconds)'})
        # Truth
        if not truth_df.empty and 'Model' in truth_df.columns:
            truth_df = truth_df.sort_values(by=["Model",'Time'])
            fig.add_trace(
                go.Scatter(
                    x=truth_df.Time,
                    y=truth_df.Model,
                    mode='markers',
                    marker={'size':10,'symbol':'circle','color':'blue','line':{'width':0,'color':'black'}},
                    showlegend=False,
                )
            )
        return fig
    
    @staticmethod
    def rmse_chart(rmse=np.array([]),time=np.array([]),title=None,height=None,width=None):
        # Initialize
        fig = go.Figure()
        # Chart Layout
        fig.update_layout({
            'margin':{'t':30,'b':0,'l':0,'r':0},
            'title':{
                    'text':title,
                    'xanchor':'center',
                    'yanchor':'top',
                    'x':0.5,
                },
            'width':width,
            'height':height,
        })
        # Update Axes
        fig.update_yaxes({'title':'RMSE'})
        fig.update_xaxes({'title':'Time (Seconds)'})
        # Plot
        fig.add_trace(
            go.Scatter(
                x=time,
                y=rmse,
                mode='lines',
                line={'width':1,'color':'black'},
                showlegend=False,
            )
        )
        return fig 

    @staticmethod
    def ve3d_chart(ve3d=np.array([]),time=np.array([]),title=None,height=None,width=None):
        # Initialize
        fig = go.Figure()
        # Chart Layout
        fig.update_layout({
            'margin':{'t':30,'b':0,'l':0,'r':0},
            'title':{
                    'text':title,
                    'xanchor':'center',
                    'yanchor':'top',
                    'x':0.5,
                },
            'width':width,
            'height':height,
        })
        # Update Axes
        fig.update_yaxes({'title':'Separation Distance'})
        fig.update_xaxes({'title':'Time (Seconds)'})
        # Plot
        fig.add_trace(
            go.Scatter(
                x=time,
                y=ve3d,
                mode='lines',
                line={'width':1,'color':'black'},
                showlegend=False,
            )
        )
        return fig 

    @staticmethod
    def probability_chart(time=pd.Series([]),prob_cv=pd.Series([]),prob_ca=pd.Series([]),prob_h=pd.Series([]),title=None,leg_pos=None,height=None,width=None):
        # Initialize
        fig = go.Figure()
        # Chart Layout
        fig.update_layout({
            'margin':{'t':0,'b':0,'l':0,'r':0},
            'width':width,
            'height':height,
        })
        if title:
            fig.update_layout({
                'margin':{'t':30,'b':0,'l':0,'r':0},
                'title':{
                    'text':title,
                    'xanchor':'center',
                    'yanchor':'top',
                    'x':0.5,
                },
            })
        if leg_pos:
            fig.update_layout({
                'legend':{
                    'orientation':"h",
                    'yanchor':'bottom',
                    'y':leg_pos,
                    'xanchor':'center',
                    'x':0.5,
                    'bgcolor':'white',
                    'bordercolor':'black',
                    'borderwidth':2
                },
            })
        # Update Axes
        fig.update_yaxes({'title':'Probability'})
        fig.update_xaxes({'title':'Time (seconds)'})
        if time.empty:
            return fig 
        if not prob_cv.empty:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=prob_cv,
                    mode='markers',
                    name = 'NCV'
                )
            )
        if not prob_ca.empty:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=prob_ca,
                    mode='markers',
                    name = 'CA'
                )
            )
        if not prob_h.empty:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=prob_h,
                    mode='markers',
                    name = 'Hovering'
                )
            )
        return fig 
    

    