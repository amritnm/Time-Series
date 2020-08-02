#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:17:25 2020

@author: amrit
"""

#Import libraries.
import numpy as np
import pandas as pd
from nsepy import get_history
from datetime import date
import matplotlib.pyplot as plt

import datetime
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

end1 = date(2020,7,31)
start1 = date(2008,1,1)

stock_df = yf.download("^NSEI", start=start1, end=end1)

from pmdarima.arima import auto_arima

#Prepare the data.

df = stock_df[['Close']]
df.head()
"""
                  Close
Date                   
2008-01-01  6144.350098
2008-01-02  6179.399902
2008-01-03  6178.549805
2008-01-04  6274.299805
2008-01-07  6279.100098 
"""

df.tail()
"""
                   Close
Date                    
2020-07-24  11194.150391
2020-07-27  11131.799805
2020-07-28  11300.549805
2020-07-29  11202.849609
2020-07-30  11102.150391
"""

# Understanding the pattern.

df.plot()

#Test for Stationarity.
from pmdarima.arima import ADFTest

adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(df)
#(0.017371366462767886, False)

#Train and Test Split.
train_period = (df.index >= "2016-01-01") & (df.index <= "2019-12-31")
test_period = (df.index >= "2020-01-01") & (df.index <= "2020-07-31")
df_train = df[train_period]
df_test = df[test_period]

plt.plot(df_train)
plt.plot(df_test)

# Building an AUTO ARIMA Model.
arima_model = auto_arima(df_train, start_p = 0, d = 1, start_q = 0, 
                         max_p = 3, max_d = 3, max_q = 3, start_P = 0,
                         D = 1, start_Q = 0, max_P = 3, max_D = 3, 
                         max_Q = 3, m = 12, seasonal = True, 
                         error_action = 'warn', trace = True, 
                         supress_warnings = True, stepwise = True, 
                         random_state = 20, n_fits = 50)

arima_model.summary()
#<class 'statsmodels.iolib.summary.Summary'>
"""
                                     SARIMAX Results                                      
==========================================================================================
Dep. Variable:                                  y   No. Observations:                  978
Model:             SARIMAX(0, 1, 1)x(0, 1, 1, 12)   Log Likelihood               -5624.825
Date:                            Sun, 02 Aug 2020   AIC                          11257.650
Time:                                    17:02:22   BIC                          11277.139
Sample:                                         0   HQIC                         11265.070
                                            - 978                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0002      0.118      0.002      0.998      -0.232       0.232
ma.L1          0.0679      0.022      3.093      0.002       0.025       0.111
ma.S.L12      -0.9825      0.015    -67.650      0.000      -1.011      -0.954
sigma2      6504.2440    218.364     29.786      0.000    6076.259    6932.229
===================================================================================
Ljung-Box (Q):                       45.50   Jarque-Bera (JB):               580.64
Prob(Q):                              0.25   Prob(JB):                         0.00
Heteroskedasticity (H):               2.29   Skew:                             0.40
Prob(H) (two-sided):                  0.00   Kurtosis:                         6.72
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""    

#Forecasting on the test data.
prediction = pd.DataFrame(arima_model.predict(n_periods = 144),index = df_test.index)
prediction.columns = ['prediction_close']
prediction

plt.figure(figsize = (8,5))
plt.plot(df_train, label = 'Training')
plt.plot(df_test, label = 'Testing')
plt.plot(prediction, label = 'Predicted')
plt.legend(loc = 'Left corner')
plt.show()

from sklearn.metrics import r2_score
df_test['predicted_close'] = prediction
r2_score(df_test['Close'], df_test['predicted_close'])
# -2.562668470276911
