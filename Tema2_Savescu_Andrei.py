#!/usr/bin/env python
# coding: utf-8

# # 1. Testare RW pentru Cryptomoneda XRP si actiunile Tesla

# In[54]:


import warnings
warnings.simplefilter("ignore")

import pandas as pd
import scipy.stats as stat
import scipy
import datetime
import numpy as np
from arch.unitroot import VarianceRatio

from yahoo_fin.stock_info import *

start = datetime.datetime(2021,10,1)
end = datetime.datetime(2022,10,1)

stock_cryp = get_data('XRP-USD' , start_date = start, end_date = end)

stock_cryp.head()

stock_cryp=stock_cryp.dropna()
close = pd.DataFrame(stock["close"]).dropna()
simple_return = close.pct_change().dropna()
log_return = np.log(1+simple_return)


# In[55]:


stock_cryp["close"].plot(grid = False)


# In[28]:



x = np.asarray(close)
x = np.log(x)
N=len(x)
vr_1=[]
vrt = []
w=[]
varvrt = []
zvrt = []
q = []
lcl=[]
ucl=[]
vr_1=[]
p_value=[]
stderr=[]
alpha=0.05
k=5

def VRTest():
    for ii in range (1, k+1):
        a=2**ii
        q.append(a)
        vr = VarianceRatio(x, a)
        vrt.append(vr.vr)
        se=np.sqrt(vr._stat_variance)/np.sqrt(vr._nobs-1)
        stderr.append(se)
        lower=vr.vr-stat.norm.ppf(1-alpha/2)*se
        upper=vr.vr+stat.norm.ppf(1-alpha/2)*se
        one=1
        zvrt.append(vr.stat)
        lcl.append(lower)
        ucl.append(upper)
        vr_1.append(one)
        p_value.append(vr.pvalue)

    return vrt,p_value,zvrt,q,stderr 
VRTest()

confidence=1-alpha
get_ipython().run_line_magic('pylab', 'inline')
#pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots
#plt.ylim(min(lcl), max(ucl))
plt.xlim(0, 2**k+2)
plt.xlabel( 'q' )
plt.ylabel( 'VR(q)' )
plt.plot(q, vr_1,color='black', linestyle='dashed', label='VR(q)=1')
plt.plot(q, vrt, color='blue', marker='o',markerfacecolor='blue', markersize=8,label='VR(q)')
plt.plot(q, lcl,color='red', linestyle='dashed', label='LCL ' +str('{:.0%}'.format(confidence)))
plt.plot(q, ucl,color='red', linestyle='dashed', label='UCL ' +str('{:.0%}'.format(confidence)))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),fancybox=True, shadow=True, ncol=5)

plt.show()


# In[29]:


results=pd.DataFrame(columns=['q','VR test', 'Std. Error','z statistic', 'P-value'])
results['q']=q
results['VR test']=vrt
results['Std. Error']=stderr
results['z statistic']=zvrt
results['P-value']=p_value


# In[30]:


results


# In[31]:


z=abs(results["z statistic"]).max()
alpha_star=1-(1-alpha)**(1/k)
alpha_star
z_star=stat.norm.ppf(1-alpha_star/2)
z_star

mvr=pd.DataFrame(columns=['z Statistic','Critical z','Decision'])
mvr.at[0,'z Statistic']=z
mvr['Critical z']=z_star
if z<z_star:
    mvr['Decision']="Cannot reject the null hypothesis of random walk"
else:
    mvr['Decision']="Reject the null hypothesis of random walk"
mvr


# In[25]:


#Compute the covariance matrix
cov1=np.zeros((k,k))

q=np.asarray(q)

for i in range (0,k-1):
    for j in range (i+1,k):
        cov1[i][j]=2*(3*q[j]-q[i]-1)*(q[i]-1)/(3*q[j])
cov2=np.transpose(cov1)
cov=cov1+cov2

for i in range (0,k):
    #for j in range (i+1,k):
  
    cov[i][i]=2*(2*q[i]-1)*(q[i]-1)/(3*q[i])
    
Wald_Test=np.matmul(np.asarray(vrt)-1,np.linalg.inv(cov))
                    
Wald_Test=N*np.matmul(Wald_Test,np.transpose(np.asarray(vrt)-1))

p_val= 1-stat.chi2.cdf(Wald_Test,k)

mvrw=pd.DataFrame(columns=['Wald Test','Critical Chi2','P-value','Decision'])
mvrw.at[0,'Wald Test']=Wald_Test
mvrw['Critical Chi2']=stat.chi2.ppf(alpha/2,k)
mvrw['P-value']=p_val
if p_val>0.05:
    mvrw['Decision']="Cannot reject the null hypothesis of random walk"
else:
    mvrw['Decision']="Reject the null hypothesis of random walk"
mvrw


# In[43]:


stock_tsl = get_data('TSLA' , start_date = start, end_date = end)

stock_tsl.head()

stock_tsl=stock_tsl.dropna()
close_tsl = pd.DataFrame(stock_tsl["close"]).dropna()
simple_return2 = close_tsl.pct_change().dropna()
log_return2 = np.log(1+simple_return2)


# In[44]:


stock_tsl["close"].plot(grid = False)


# In[45]:


y = np.asarray(close_tsl)
y = np.log(y)
N=len(y)
vr_1=[]
vrt = []
w=[]
varvrt = []
zvrt = []
q = []
lcl=[]
ucl=[]
vr_1=[]
p_value=[]
stderr=[]
alpha=0.05
k=5

def VRTest():
    for ii in range (1, k+1):
        a=2**ii
        q.append(a)
        vr = VarianceRatio(y, a)
        vrt.append(vr.vr)
        se=np.sqrt(vr._stat_variance)/np.sqrt(vr._nobs-1)
        stderr.append(se)
        lower=vr.vr-stat.norm.ppf(1-alpha/2)*se
        upper=vr.vr+stat.norm.ppf(1-alpha/2)*se
        one=1
        zvrt.append(vr.stat)
        lcl.append(lower)
        ucl.append(upper)
        vr_1.append(one)
        p_value.append(vr.pvalue)

    return vrt,p_value,zvrt,q,stderr 
VRTest()

confidence=1-alpha
get_ipython().run_line_magic('pylab', 'inline')
plt.xlim(0, 2**k+2)
plt.xlabel( 'q' )
plt.ylabel( 'VR(q)' )
plt.plot(q, vr_1,color='black', linestyle='dashed', label='VR(q)=1')
plt.plot(q, vrt, color='blue', marker='o',markerfacecolor='blue', markersize=8,label='VR(q)')
plt.plot(q, lcl,color='red', linestyle='dashed', label='LCL ' +str('{:.0%}'.format(confidence)))
plt.plot(q, ucl,color='red', linestyle='dashed', label='UCL ' +str('{:.0%}'.format(confidence)))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),fancybox=True, shadow=True, ncol=5)

plt.show()


# In[47]:


results2=pd.DataFrame(columns=['q','VR test', 'Std. Error','z statistic', 'P-value'])
results2['q']=q
results2['VR test']=vrt
results2['Std. Error']=stderr
results2['z statistic']=zvrt
results2['P-value']=p_value


# In[48]:


results2


# In[49]:


z=abs(results2["z statistic"]).max()
alpha_star=1-(1-alpha)**(1/k)
alpha_star
z_star=stat.norm.ppf(1-alpha_star/2)
z_star

mvr=pd.DataFrame(columns=['z Statistic','Critical z','Decision'])
mvr.at[0,'z Statistic']=z
mvr['Critical z']=z_star
if z<z_star:
    mvr['Decision']="Cannot reject the null hypothesis of random walk"
else:
    mvr['Decision']="Reject the null hypothesis of random walk"
mvr


# In[51]:


cov1=np.zeros((k,k))

q=np.asarray(q)

for i in range (0,k-1):
    for j in range (i+1,k):
        cov1[i][j]=2*(3*q[j]-q[i]-1)*(q[i]-1)/(3*q[j])
cov2=np.transpose(cov1)
cov=cov1+cov2

for i in range (0,k):
  
    cov[i][i]=2*(2*q[i]-1)*(q[i]-1)/(3*q[i])
    
Wald_Test=np.matmul(np.asarray(vrt)-1,np.linalg.inv(cov))
                    
Wald_Test=N*np.matmul(Wald_Test,np.transpose(np.asarray(vrt)-1))

p_val= 1-stat.chi2.cdf(Wald_Test,k)

mvrw=pd.DataFrame(columns=['Wald Test','Critical Chi2','P-value','Decision'])
mvrw.at[0,'Wald Test']=Wald_Test
mvrw['Critical Chi2']=stat.chi2.ppf(alpha/2,k)
mvrw['P-value']=p_val
if p_val>0.05:
    mvrw['Decision']="Cannot reject the null hypothesis of random walk"
else:
    mvrw['Decision']="Reject the null hypothesis of random walk"
mvrw


# In ambele cazuri nu avem destule dovezi pentru a nu accepta ipoteza de Random Walk.

# # 2. Exemplu eficienta in forma semi-tare: Pretul actiunilor Adidas au scazut in utlima luna considerabil cand au anuntat investigarea si ulterior incetarea colaborarii cu Kanye West pentru productia brand-ului Yeezy.

# In[53]:


start = datetime.datetime(2022,10,1)
end = datetime.datetime(2022,10,27)

stock_ads = get_data('ADS.DE' , start_date = start, end_date = end)

stock_ads.head()

close = pd.DataFrame(stock["close"])

stock_ads["close"].plot(grid = False)


# In[ ]:




