from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template import loader
import csv

import json

from . import getdata
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

scale = joblib.load("processdata/scaler.save")
model_pca = joblib.load("processdata/pca.save")
model_gm = joblib.load("processdata/gm.save")
model_lr1 = joblib.load("processdata/lr1.save")
model_lr2 = joblib.load("processdata/lr2.save")
model_lr3 = joblib.load("processdata/lr3.save")
model_scaler = joblib.load("processdata/scalerprofileproduction.save")
model_gmfileprod = joblib.load("processdata/gmprofileproduction.save")



def index(request): 
    return render(request, template_name='index.html')


def results(request):

    FIELD_PORO_AVG = request.GET.get('FIELD_PORO_AVG','')
    FIELD_PERM_AVG = request.GET.get('FIELD_PERM_AVG','')
    FIELD_SWC_AVG = request.GET.get('FIELD_SWC_AVG','')
    FIELD_PRESS_INIT_AVG = request.GET.get('FIELD_PRESS_INIT_AVG','')
    FIELD_FVFOIL_AVG = request.GET.get('FIELD_FVFOIL_AVG','')
    FIELD_OILVISC_AVG = request.GET.get('FIELD_OILVISC_AVG','')
    FIELD_WATVISC_AVG =  request.GET.get('FIELD_WATVISC_AVG','')
    
    data = []
    if FIELD_PORO_AVG =='':
        return render(request, 'pages/recovery_factor.html',{'result':'','cluster':'','F1':'','F2':'','F3':'','F4':'','F5':'','F6':'','F7':''})
    data.append(float(FIELD_PORO_AVG))
    data.append(float(FIELD_PERM_AVG))
    data.append(float(FIELD_SWC_AVG))
    data.append(float(FIELD_PRESS_INIT_AVG))
    data.append(float(FIELD_FVFOIL_AVG))
    data.append(float(FIELD_OILVISC_AVG))
    data.append(float(FIELD_WATVISC_AVG))

    x_array = np.array(data)
    x_array = x_array.reshape(1,-1)
    x_scaled = scale.transform(x_array)
    scaled_data = []
    for x in x_scaled:
        for y in x:
            scaled_data.append(y)

    # print(scaled_data)
    x_pca_array = np.array(scaled_data)
    x_pca_array = x_pca_array.reshape(1,-1)
    # print(x_pca_array)
    x_pca = model_pca.transform(x_pca_array)

    # pca_data = pd.DataFrame(x_pca,columns=['PC1','PC2','PC3'])
    cluster = model_gm.predict(x_pca)
    # print(cluster)
    cluster = cluster+1
    if cluster == 1:
        rf = model_lr1.predict(x_pca)
    elif cluster == 2:
        rf = model_lr2.predict(x_pca)
    else:
        rf = model_lr3.predict(x_pca)

    output = round(rf[0], 3)
    F1 = round(float(FIELD_PORO_AVG),2)
    F2 = round(float(FIELD_PERM_AVG),2)
    F3 = round(float(FIELD_SWC_AVG),2)
    F4 = round(float(FIELD_PRESS_INIT_AVG),2)
    F5 = round(float(FIELD_FVFOIL_AVG),2)
    F6 = round(float(FIELD_OILVISC_AVG),2)
    F7 = round(float(FIELD_WATVISC_AVG),2)

    return render(request, 'pages/recovery_factor.html', {'result':output,'cluster':cluster[0],'F1':F1,'F2':F2,'F3':F3,'F4':F4,'F5':F5,'F6':F6,'F7':F7})

def profile(request):
    fieldname = request.GET.get('FIELD_NAME',False)
    rfpred = float(request.GET.get('RECOVERY_FACTOR',False))
    field_ioip = float(request.GET.get('FIELD_IOIP',False))
    qd2 = getdata.get_database1()
    rf_dif, qcummax = getdata.func1(rfpred,qd2,field_ioip)

    database3 = getdata.get_database3()
    rf_qpeak, qcum_peak, rf_cluster_pred, b, Di = getdata.func2(rfpred, rf_dif, database3,field_ioip,model_gmfileprod, model_scaler)
    df_ita = getdata.get_database2()

    tpeak = getdata.func3(rf_dif, df_ita, rf_qpeak)

    q_exp = getdata.func4(tpeak, qcum_peak)
    df1,df2 = getdata.plot_prod(tpeak, q_exp, fieldname,qcum_peak ,qcummax,b, Di)

    response = {
        "fieldname": fieldname,
        "data1": json.loads(df1.to_json(orient='columns')),
        "data2": json.loads(df2.to_json(orient='columns'))
    }

    return HttpResponse(json.dumps(response), content_type='application/json')

    



