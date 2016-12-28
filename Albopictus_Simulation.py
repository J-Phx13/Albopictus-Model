# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:33:58 2016

@author: Pengfei Jia
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

"""
Function-guass,invQuadratic,quadratic
"""
def guass(x,coff):
    return coff[0]*np.exp(-((x-coff[1])/coff[2])**2)
def invQuadratic(x,coff):
    return  min(1,1/abs(coff[0]*x**2+coff[1]*x+coff[2]))
def quadratic(x,coff):
    return  coff[0]*x**2+coff[1]*x+coff[2]
"""
daily temperature-driven function
"""
# egg incubation function (temperature-driven)
def f_E(tem):
    coff = [0.5070, 30.85, 12.82]; return guass(tem,coff)
# larva pupation function (temperature-driven)
def f_L(tem):
    coff = [0.1727, 28.40, 10.20]; return guass(tem,coff)
# pupa eclosion function (temperature-driven)
def f_P(tem):
    coff = [0.6020, 34.29, 15.07]; return guass(tem,coff)
# adult gestation function (temperature-driven)
def f_Ag(tem):
    return max(0,(tem-10)/77)
# larva mortality function (temperature-driven)
def m_L(tem):
    coff = [-0.1305, 3.868, 30.83]; return invQuadratic(tem,coff)
# pupa mortality function (temperature-driven)
def m_P(tem):
    coff = [-0.1502, 5.057, 3.517]; return invQuadratic(tem,coff)
# adult mortality function (temperature-driven)
def m_A(tem):
    coff = [-0.1921, 8.147, -22.98]; return invQuadratic(tem,coff)
# egg oviposition function (temperature-driven)
def beta(tem):
    coff = [-0.0163, 1.2897, -15.837]; return max(0,quadratic(tem,coff))
"""
---------------------daily precipitation-driven function---------------------
"""
# larva environmental capacity function (temperature-driven)
def k_L(m,pre):
    return m*(1+pre)
# pupa environmental capacity function (temperature-driven)
def k_P(m,pre):
    return m*(1+pre)
"""
---------------------model preparation---------------------
"""
#validating regions
region={'guangzhou':[2006,2007,2008,2009,2010,2011],\
'shanghai':[2008,2009,2010,2011,2012,2013]}
# cliamte data and validation data filepath
pathName='C:\\Users\\Phoenix\\Desktop\\'
"""
model parameters independent of climate factors
"""
noClimate = {'kappaL': 2.5e5, 'kappaP': 2.5e5, 'sexRatio': 0.5, \
'm_E': 0.05, 'muAe': 0.1, 'muR': 0.08, 'rAe': 0.4, 'rAb': 0.2, 'rAo': 0.2}
"""
diapause-related parameters
"""    
diapause = {'diaBegin': [21.0, 13.5], 'diaEnd': [10.5, 11.25],\
'diaMortality':10e-2,'diaHatch':10e-2,'diaAdult':9.5,'rate':0.99}
"""
Date description
"""
# month node    
totalMonth = np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
# year
yearLast = 6
intraYear = totalMonth[12]
totalData = yearLast*intraYear
"""
diapause information
"""
DiaPeriod = {'shanghai':\
{'diaBegin': np.zeros(yearLast-1), 'diaEnd': np.zeros(yearLast-1),\
'eggBegin': np.zeros(yearLast-1), 'eggEnd': np.zeros(yearLast-1)},\
'guangzhou':\
{'diaBegin': np.zeros(yearLast-1), 'diaEnd': np.zeros(yearLast-1),\
'eggBegin': np.zeros(yearLast-1), 'eggEnd': np.zeros(yearLast-1)}}
plt.close('all')
for city in region.keys():#city: guangzhou and shanghai
    print ''.join(['---------------',city,'---------------'])
    fileName=''.join([pathName,city,'.mat']) #city: guangzhou and shanghai
    """
    ---------------------climate data assignment---------------------
    T_: daily mean temperature; Tave_: 7-day average daily mean temperature
    Pnorm_: 14-day normalized sum synthesis of daily accumulated precipitation
    PPave_: 7-day average daily sunshine hour
    field_mci: monthly container indexes
    """    
    climateData = sio.loadmat(fileName)
    T_ = climateData['T']; Tave_ = climateData['Tave']
    Pave_ = climateData['Pave']; Pnorm_ = climateData['Pnorm']
    PPave_ = climateData['PPave'];  field_mci = climateData['MCI'] 
    """
    albopictus population initialization
    """
    albopictus = {\
    'E':np.zeros((totalData,1)),'Edia':np.zeros((totalData,1)),\
    'L':np.zeros((totalData,1)),'P':np.zeros((totalData,1)),\
    'Ae':np.zeros((totalData,1)),'Ab':np.zeros((totalData,1)),\
    'Ag':np.zeros((totalData,1)),'Ao':np.zeros((totalData,1)),\
    'TL':np.zeros((totalData,1)),'TA':np.zeros((totalData,1))}
    Initial = 1e6; albopictus['E'][0] = Initial; albopictus['Edia'][0] = Initial 
    """
    ---------------------albopictus population iteration---------------------
    """
    for yi in range(yearLast): # year iteration
        diaTime = {'diaBegin': totalMonth[6], 'diaEnd': totalMonth[6],\
        'eggBegin': totalMonth[12], 'eggEnd': totalMonth[12]}
        diaFlag = {'diaBegin': False, 'diaEnd': False, 'eggEnd': False,\
        'eggBegin': False, 'hatch': False, 'lay': False, 'adult': True}
        for dj in range(intraYear): # day iteration
            num = yi * intraYear + dj
            """
            diapause eggs incubation condtion
            """
            if dj>totalMonth[0] and diaFlag['eggEnd'] is False:
                diaFlag['hatch'] = (Tave_[yi][dj]>diapause['diaEnd'][0] \
            and PPave_[yi][dj]>diapause['diaEnd'][1])
            """
            diapause period ends
            """
            if diaFlag['diaEnd'] is False and \
            albopictus['Edia'][num] < diapause['rate']*albopictus['E'][num]:
                diaTime['diaEnd'] = dj
                diaFlag['diaEnd'] = True
            """
            diapause eggs finish hatching
            """
            if diaFlag['diaEnd'] is not False and diaFlag['eggEnd'] is False and \
            albopictus['Edia'][num] < (1-diapause['rate'])*albopictus['E'][num]:
                diaTime['eggEnd'] = dj
                diaFlag['eggEnd'] = True; diaFlag['hatch'] = False
            """
            diapause eggs laid condtion
            """
            if dj>totalMonth[8] and diaFlag['diaBegin'] is False:
                diaFlag['lay'] = (Tave_[yi][dj]<diapause['diaBegin'][0]\
                and PPave_[yi][dj]<diapause['diaBegin'][1])
            """
            diapause eggs begin being laid
            """
            if diaFlag['eggBegin'] is False and diaFlag['eggEnd'] is not False \
            and albopictus['Edia'][num] > (1-diapause['rate'])*albopictus['E'][num]:
                diaTime['eggBegin'] = dj 
                diaFlag['eggBegin'] = True
            """
            diapause period begins
            """
            if diaFlag['eggBegin'] is not False and \
            diaFlag['diaBegin'] is False and albopictus['Edia'][num] > \
            diapause['rate']*albopictus['E'][num]:
                diaTime['diaBegin'] = dj
                diaFlag['diaBegin'] = True; diaFlag['lay'] = False
            """
            precipitation-driven parameter assignment
            """
            KL = k_L(noClimate['kappaL'], Pnorm_[yi][dj])
            KP = k_P(noClimate['kappaP'], Pnorm_[yi][dj])            
            """
            temperature-driven parameter assignment
            """
            climate = {\
            'f_E': f_E(T_[yi][dj]), 'f_L': f_L(T_[yi][dj]), 'f_P': f_P(T_[yi][dj]), \
            'm_L': m_L(T_[yi][dj]), 'm_P': m_P(T_[yi][dj]), 'm_A': m_A(T_[yi][dj]), \
            'f_Ag': f_Ag(T_[yi][dj]), 'n_Ao': beta(T_[yi][dj])}         
            """
            differential equation iteration
            """            
            Delta = {'E':0,'Edia':0,'L':0,'P':0,'Ae':0,'Ab':0,'Ag':0,'Ao':0}
            
            Delta['E'] = \
            climate['n_Ao']*albopictus['Ao'][num]\
            -(noClimate['m_E']*diapause['diaMortality']\
            +climate['f_E']*diaFlag['hatch']*diapause['diaHatch'])\
            *albopictus['Edia'][num]\
            -(noClimate['m_E']+climate['f_E'])\
            *(albopictus['E'][num]-albopictus['Edia'][num])
        
            Delta['Edia'] = \
            climate['n_Ao']*diaFlag['lay']*albopictus['Ao'][num] \
            -(noClimate['m_E']*diapause['diaMortality']\
            +climate['f_E']*diapause['diaHatch']*diaFlag['hatch'])\
            *albopictus['Edia'][num]
            
            Delta['L'] = climate['f_E']*diapause['diaHatch']*diaFlag['hatch']\
            *albopictus['Edia'][num]+climate['f_E']*(albopictus['E'][num]\
            -albopictus['Edia'][num])-min(1,climate['m_L']*(1+albopictus['L'][num]/KL)\
            +climate['f_L'])*albopictus['L'][num]
            
            Delta['P'] = climate['f_L']*albopictus['L'][num]-min(1,(climate['m_P']+\
            climate['f_P']))*albopictus['P'][num]
            
            Delta['Ae'] = climate['f_P']*albopictus['P'][num]*noClimate['sexRatio']\
            *np.exp(-noClimate['muAe']*(1+albopictus['P'][num]/KP))-(climate['m_A']+\
            diaFlag['adult']*noClimate['rAe'])*albopictus['Ae'][num]
            
            Delta['Ab'] = diaFlag['adult']*(noClimate['rAe']*albopictus['Ae'][num]+\
            noClimate['rAo']*albopictus['Ao'][num])-(climate['m_A']+noClimate['muR']\
            +diaFlag['adult']*noClimate['rAb'])*albopictus['Ab'][num]
            
            Delta['Ag'] = diaFlag['adult']*noClimate['rAb']*albopictus['Ab'][num]-\
            (climate['m_A']+climate['f_Ag'])*albopictus['Ag'][num]
            
            Delta['Ao'] = climate['f_Ag']*albopictus['Ag'][num]-(climate['m_A']+\
            noClimate['muR']+diaFlag['adult']*noClimate['rAo'])*albopictus['Ao'][num]
            
            if num< totalData-1:
                albopictus['E'][num+1]=albopictus['E'][num]+Delta['E']
                albopictus['Edia'][num+1]=albopictus['Edia'][num]+Delta['Edia']
                albopictus['L'][num+1]=albopictus['L'][num]+Delta['L']
                albopictus['P'][num+1]=albopictus['P'][num]+Delta['P']
                albopictus['Ae'][num+1]=albopictus['Ae'][num]+Delta['Ae']
                albopictus['Ab'][num+1]=albopictus['Ab'][num]+Delta['Ab']
                albopictus['Ag'][num+1]=albopictus['Ag'][num]+Delta['Ag']
                albopictus['Ao'][num+1]=albopictus['Ao'][num]+Delta['Ao']
                albopictus['TL'][num+1]=albopictus['L'][num+1]+albopictus['P'][num+1]
                albopictus['TA'][num+1]=albopictus['Ab'][num+1]\
                +albopictus['Ag'][num+1]+albopictus['Ao'][num+1]
        if yi>0:
            print(region[city][yi])# print year
            """
            diapause information: 
            diapause ends, egg ends, egg begins, diapause begins
            """
            print(sorted(diaTime.items(), lambda x,y: cmp(x[1],y[1])))
            DiaPeriod[city]['diaEnd'][yi-1]=diaTime['diaEnd']
            DiaPeriod[city]['eggEnd'][yi-1]=diaTime['eggEnd']
            DiaPeriod[city]['eggBegin'][yi-1]=diaTime['eggBegin']
            DiaPeriod[city]['diaBegin'][yi-1]=diaTime['diaBegin']
    """
    figure axis preparation
    """
    x_tick=[0]
    [x_tick.extend(totalMonth[3::3]+intraYear*year) for year in range(yearLast-1)]
    x_tickLabel=['JAN']
    [x_tickLabel.extend(['APR','JUL','OCT','JAN']) for year in range(yearLast-1)]
    figureShow = ['E','Ab','L','Ag','P','Ao']
    figureTitle = \
    ['Egg','Blooding Adult','Larva','Gestating Adult','Pupa','Ovipositing Adult']
    x_label = {'shanghai': ''.join(['|<-------2009------->|<------2010------->|',\
            '<------2011------>|<-------2012------>|<-------2013------->|']),\
            'guangzhou': ''.join(['|<-------2007------->|<------2008------->|',\
            '<------2009------>|<-------2010------>|<-------2011------->|'])}     
    """
    ---------------------daily scale figure---------------------
    """
    nrow=3;ncolumn=2
    ncolor = ['r','r','b','b','g','g']
    fig,axes= plt.subplots(nrow,ncolumn,figsize=(17,9))
    for i in range(nrow):
        for j in range(ncolumn):
            axes[i,j].plot(albopictus[figureShow[ncolumn*i+j]][365:],color=ncolor[ncolumn*i+j])
            axes[i,j].set_xlim(1,(yearLast-1)*intraYear)
            axes[i,j].set_xticks(x_tick)
            axes[i,j].set_xticklabels(x_tickLabel,fontsize=10)
            axes[i,j].set_xlabel(x_label[city])
            
            axes[i,j].set_ylim([0,max(albopictus[figureShow[ncolumn*i+j]][365:])*1.1])
            axes[i,j].yaxis.get_major_formatter().set_powerlimits((0,1))
            axes[i,j].set_ylabel('Number of Population')
            axes[i,j].grid()
            
            axes[i,j].set_title(' '.join([figureTitle[ncolumn*i+j],'population']))
            for yi in range(yearLast-1):
                axes[i,j].axvspan(yi*intraYear,DiaPeriod[city]['diaEnd'][yi]+yi*intraYear,\
                facecolor='0.5',edgecolor='w',alpha=0.3)
                axes[i,j].axvspan(DiaPeriod[city]['diaBegin'][yi]+yi*intraYear,(yi+1)*intraYear,\
                facecolor='0.5',edgecolor='w',alpha=0.3)                
    plt.subplots_adjust(left=0.04,bottom=0.05,right=0.97,top=0.95,wspace=0.1,hspace=0.4)
    """
    ---------------------month scale figure---------------------
    """
    AeMonth = {'E':[],'L':[],'P':[],'Ab':[],'Ag':[],'Ao':[]}
    for key in AeMonth.keys():
        [AeMonth[key].append(albopictus[key][(yi+1)*intraYear+totalMonth[mj]\
    :(yi+1)*intraYear+totalMonth[mj+1]].mean()) for yi in range(yearLast-1)\
    for mj in range(len(totalMonth)-1)]
    """
    Monthly Simulated Larvae vs field container index
    """
    corr = np.corrcoef(AeMonth['L'],field_mci)
    print ('------------correlation coefficient------------')
    print corr


            
            
             
            
                
            
    