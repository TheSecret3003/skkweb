import datetime
import platform
import csv
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot

# model_scaler = joblib.load("scalerprofileproduction.save")
# model_gmfileprod = joblib.load("gmprofileproduction.save")


def get_database1():
    df = pd.read_excel("processdata/database1.xlsx", engine="openpyxl")
    return df

def get_database2():
    df = pd.read_excel("processdata/database2.xlsx", engine="openpyxl", sheet_name=None)
    return df

def get_database3():
    df = pd.read_excel("processdata/database3.xlsx", engine="openpyxl")
    return df

def func1(rfpred, qd2, field_ioip):
    fname=[]
    rf=[]
    rfdif=[]
    rfqpeak=[]
    qpeak=[]
    rfmax=[]
    b=[]
    di=[]
    for k in qd2.index:
        fname.append(qd2['Field Name'][k])
        rf.append(qd2['EUR/IOIP'][k])
        rfqpeak.append(qd2['RF@qpeak'][k])
        qpeak.append(qd2['qpeak, stb/m'][k]/30)
        rfmax.append(qd2['RF Max'][k])
        b.append(qd2['b'][k])
        di.append(qd2['Di'][k])
        rfdif.append(abs(rfpred-qd2['EUR/IOIP'][k]))
    rf_dif=pd.DataFrame()
    rf_dif['FIELD_NAME']=fname
    rf_dif['RF']=rf
    rf_dif['RF@qpeak']=rfqpeak
    rf_dif['qpeak']=qpeak
    rf_dif['RF Max']=rfmax
    rf_dif['b']=b
    rf_dif['Di']=di
    rf_dif['RF_DIFFERENCE']=rfdif
    rf_dif.sort_values(by=['RF_DIFFERENCE'], inplace=True)
    rf_dif.reset_index(drop=True, inplace=True)
    qcummax=rfpred*field_ioip*(10**6)
    return rf_dif, qcummax

def func2(rfpred, rf_dif, database3, field_ioip, model_cluster, model_scaling):
    rf_qpeak=[]
    qcum_peak=[]
    rf_cluster_pred=[]
    b=[]
    di=[]
    rfqpeak=-1
    k=0
    while rfqpeak<0:
        low=rf_dif.iloc[k,:]
        up=rf_dif.iloc[k+1,:]
        rfqpeak=low['RF@qpeak']+((rfpred-low['RF'])/(up['RF']-low['RF']))*(up['RF@qpeak']-low['RF@qpeak'])        
        k+=1
    qcum_peak=rfqpeak*field_ioip*(10**6)
    rf_qpeak=rfqpeak
    clust=model_cluster.predict(model_scaling.transform([[rfqpeak,rfpred]]))[0]
    b=database3['b'][clust]
    di=database3['Di'][clust]/100
    rf_cluster_pred=clust

    return rf_qpeak, qcum_peak, rf_cluster_pred, b, di

def func3(rf_dif, df_ita, rf_qpeak):
    for k in rf_dif.index:
        if rf_qpeak<=3/6*rf_dif['RF Max'][k] and rf_qpeak>=1/6*rf_dif['RF Max'][k]:
        #if k>=0:
            rfqp=rf_qpeak
            fn=rf_dif['FIELD_NAME'][k]
            tbasis=df_ita[fn][df_ita[fn]['RF']<=rfqp]
            ttpeak=tbasis['t/tpeak'][len(tbasis)-1]
            tcum=tbasis['tcum'][len(tbasis)-1]
            tpeak=tcum/ttpeak
            qpeak=rf_dif['qpeak'][k]
            break

    return tpeak

# Defining Function
def f(x, t, N):
    return (t**x)/x - N

# Defining derivative of function
def g(x, t):
    return ((np.log(t)*x-1)*(t**x))/(x**2)

# Implementing Newton Raphson Method

def newtonRaphson(x0,e,niter,t,N):
    #print('\n\n*** NEWTON RAPHSON METHOD IMPLEMENTATION ***')
    step = 1
    flag = 1
    condition = True
    while condition:
        if g(x0,t) == 0.0:
            print('Divide by zero error!')
            break
        
        x1 = x0 - f(x0, t,N)/g(x0,t)
        #print('Iteration-%d, x1 = %0.6f and f(x1) = %0.6f' % (step, x1, f(x1,t,N)))
        x0 = x1
        step = step + 1
        
        if step > niter:
            flag = 0
            break
        
        condition = abs(f(x1,t,N)) > e
    
    #if flag==1:
        #print('\nRequired root is: %0.8f' % x1)
    #else:
        #print('\nNot Convergent.')
    
    return x1

def func4 (tpeak, qcum_peak):
    x0=3
    e=0.0001
    t=tpeak
    N=qcum_peak
    niter=1000
    q_exp=abs(newtonRaphson(x0,e,niter,t,N)-1)
    return q_exp

def func(x, qexp):

    y=x**(qexp)
    return y

def cum_prod(x,y):
    cumprod=0
    for i in range (len(x)):
        cumprod=cumprod+y[i]
    return cumprod
      
def forecast(qi, b, Di, t0, qcumpeak, qcummax):
    def q(qi, b, Di, t, t0):
        if b==0:
            q=qi*np.exp((-Di) * (t-t0))
        else:
            q=qi/((1 + (b * Di*(t-t0)))**(1/b))
        return q
    condition=True
    i=1
    qcum=qcumpeak
    x=[]
    x.append(t0)
    y=[]
    y.append(qi)
    
    while condition:
        x.append(t0+i)
        y.append(q(y[0],b,Di,x[i],x[0]))
        qcum=qcum+y[i]
        i=i+1
        if qcum>=qcummax or (t0+i)>=365*30:
            condition=False
    return x, y

def plot_prod(tpeak, qexp, fieldname,qcumpeak,qcummax,b,di):
    xi=[i for i in range (1,int(tpeak+1))]
    yi=[]
    for i in xi:
        yi.append(func(i, qexp))
    x=[]
    y=[]
    nmon=np.ceil(tpeak/30)
    for i in range(1,int(nmon+1)):
        x.append(i)
        qmon=0
        j=1
        if i!=nmon:
            while j<=30:
                qmon+=yi[j+(i-1)*30-1]
                j+=1
            y.append(qmon)
        else:
            while (j+(i-1)*30-1)<tpeak-1:
                qmon+=yi[j+(i-1)*30-1]
                j+=1
            y.append(qmon)

            
    x_forecast, y_forecast=forecast(max(y),b,di,nmon,qcumpeak,qcummax)
    # fig = go.Figure(layout_title_text=fieldname)
    # fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=go.scatter.Line(color="dimgrey"),showlegend=False))
    # fig.add_trace(go.Scatter(x=x_forecast,y=y_forecast,mode="lines",line=go.scatter.Line(color="dimgrey"),showlegend=False))
    # plt.figure(figsize=(16,8))
    # plt.plot(x,y)
    # plt.fill_between(np.array(x),np.array(y), color='r')
    # plt.plot(x_forecast,y_forecast)
    # plt.xlabel('Time, Month', fontstyle='italic',fontsize=12,color='dimgrey')
    # plt.ylabel('Oil Production, STB/M', fontstyle='italic',fontsize=12,color='dimgrey')
    # plt.title(fieldname, fontweight='bold',fontsize = 15)
    # plt.xticks(fontsize=15)
    # plt.xlim(0,500)
    # plt.yticks(fontsize=15)
    # plt.show()
    # plot_div = plot(fig, include_plotlyjs=False, output_type='div', config={'displayModeBar': False})
    df1 = pd.DataFrame( columns =['X_up', 'Y_up'])
    df2 = pd.DataFrame( columns =['X_down', 'Y_down'])
    # df['X'] = (list(x)+list(x_forecast[1:]))
    # df['Y'] = (list(y)+list(y_forecast[1:]))
    df1['X_up'] = list(x)
    df1['Y_up'] = list(y)
    df2['X_down'] = list(x_forecast)
    df2['Y_down'] = list(y_forecast)
    return df1,df2 # return x, y


# qd2 = get_database1()
# rf_dif, qcummax = func1(0.345,qd2,25.45)

# database3 = get_database3()
# rf_qpeak, qcum_peak, rf_cluster_pred, b, Di = func2(0.345, rf_dif, database3,25.45,model_gmfileprod, model_scaler)
# df_ita = get_database2()

# tpeak = func3(rf_dif, df_ita, rf_qpeak)

# q_exp = func4(tpeak, qcum_peak)

# df = plot_prod(tpeak, q_exp, "BANGKO" ,qcum_peak ,qcummax,b, Di)
# print(df)