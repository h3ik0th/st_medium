# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% 

# dependencies

# number crunching
import pandas as pd
import datetime as dt
from dateutil import relativedelta as rdt
import numpy as np

# system
import os
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# plotting
import matplotlib.pyplot as plt
import plotly.express as px

# time series analysis and forecasting
import pmdarima as pmd
from pmdarima.arima import arima
import statsmodels.api as sm
from scipy.stats import normaltest

# web app
import streamlit as st


# constants
ALPHA = 0.05     # significance level
MSEAS = 12       # seasonality: 12 months
TESTP = 24       # 24 months to reserve for test dataset


wpath = os.path.dirname(__file__)
# st.sidebar.write(wpath)

# %%
st.sidebar.title("**Forecasting:**")
st.sidebar.markdown("**Python Web App**")
st.title("**Time Series Forecasting with Python Web App**")


image = Image.open(wpath + "/uberbaggies.jpg")
st.sidebar.image(image, caption="Mike Von, unsplash.com")


st.write("**PREPARING:**")
# read the source data file
df = pd.read_csv(wpath + "/pants.csv")


# %%
# enable the end user to upload a csv file:
st.sidebar.write("_" * 30)
st.sidebar.write("**file uploader:**")

uploaded_file = st.sidebar.file_uploader(
    label="Upload the csv file containing the time series:",
    type="csv",
    accept_multiple_files=False,
    help='''Upload a csv file that contains your time series.     
        required structure:     
        first column = dates;        
        second column = observed numerical values of Pants popularity y;
        third column = exogenous variable X: 'home cave index';      
        first row = column headers;     
        length = max. 1,200 rows
        ''')   
if uploaded_file is not None:
    if uploaded_file.size <= 10000:    # size < 10 kB
        df = pd.read_csv(uploaded_file)
st.sidebar.write("_" * 30)


# convert objects/strings to datetime and numbers; set datetime index
# df = df.dropna()
df.columns = ["Date", "Pants", "Peak"]                   # rename columns
df.Date = df["Date"]    
df.Date = pd.to_datetime(df.Date)                        # convert imported object/string to datetime
df.set_index(df.Date, inplace=True)                      # set Date as index
df["year"] = df.index.year.astype("Int64")
df["month"] = df.index.month.astype("Int64")
df = df.sort_index(axis=0)
df.Date = df.Date.dt.date
df.index = df.index.date
n_obs = len(df["Pants"].dropna())                        # number of actual observations
n_x = len(df["Peak"].dropna())                           # number of X, exogenous values
n_x = min(n_x, n_obs + 12)                               # limit forecast to maximally 12 months after end of actual


df = df.loc[df["year"] > 2010]
y = pd.Series(df["Pants"].dropna())
st.write("==  scrollable dataframe after the end user has uploaded her time series file:")
st.dataframe(df.style.highlight_max(axis=0))



# %%
# plot the time series 
fig = px.line(df, x="Date", y=["Pants"], 
    title="chart: sweatpants popularity", width=1000)
st.plotly_chart(fig, use_container_width=False)


# %%
# create a pivot table and show it on the website
st.write("==  pivot table to aggregate and filter the dataframe:")
pivot = pd.pivot_table(
    df, values='Pants', index='month', columns='year',
    aggfunc='mean', margins=False, margins_name="Avg", fill_value=0)
pivot.transpose()
st.write(pivot)



# %%
st.write("_" * 30)
st.write("**DIAGNOSTICS and TRANSFORMATIONS:**")
# check the time series for normality - null hypothesis: normally distributed => rejected
st.write("==  'metric' widget, showing the test result and by how much it deviates from the threshold;")
st.write("==  here, the widget is filled with the results of a test for normality")
st.write("**original time series: normally distributed?**")

p = normaltest(y)[1]
printme = f'normality test: p-value > {ALPHA:.3f}?  p = {p:.3f}'
st.write(printme)
delta = p - ALPHA
# write test result on website, using the "metric" widget
st.metric(label="p-value", value=f'{p:.3f}', delta=f'{delta:.3f}', delta_color="normal")


# try a Box-Cox transformation
y_bc, _ = pmd.preprocessing.BoxCoxEndogTransformer(lmbda2=1.E-6).fit_transform(y)


# normality test => passing with flying colors
st.write("." * 30)
st.write("**Box-Cox-transformed time series: normally distributed?**")
p = normaltest(y_bc)[1]
printme = f'normality test: p-value > {ALPHA:.3f}?  p = {p:.3f}'
st.write(printme)
delta = p - ALPHA
# write test result on website, using the "metric" widget
st.metric(label="p-value", value=f'{p:.3f}', delta=f'{delta:.3f}', delta_color="normal")



# %%
# define tests for order of first differencing 
def diff_order(y):
    n_kpss = pmd.arima.ndiffs(y, alpha=ALPHA, test='kpss', max_d=6)
    n_adf = pmd.arima.ndiffs(y, alpha=ALPHA, test='adf', max_d=6)
    n_diff = max(n_adf, n_kpss)
    n_diffs = {"ADF":n_adf, "KPSS":n_kpss, "decision":n_diff} 
    return n_diffs

# call the stationarity tests to determine the order of differencing:
st.write("." * 30)
st.write("\n**Box-Cox-transformed data: need first-differencing?**")
n_diffs = diff_order(y_bc)
df_ndiffs = pd.DataFrame.from_dict(n_diffs, orient="index")
df_ndiffs.columns = ["order of first differencing"]
# write test result on website: order of differencing
st.dataframe(df_ndiffs)


# define tests for order of seasonal differencing
def diffseas_order(y, m):
    n_ocsb = pmd.arima.OCSBTest(m=m).estimate_seasonal_differencing_term(y)
    n_ch = pmd.arima.CHTest(m=m).estimate_seasonal_differencing_term(y)
    ns_diff = max(n_ocsb, n_ch)
    ns_diffs = {"OCSB":n_ocsb, "CH":n_ch, "decision":ns_diff}
    return ns_diffs

# call the tests for seasonal differencing
st.write("." * 30)
st.write("\n**Box-Cox-transformed data: need seasonal differencing?**")
ns_diffs = diffseas_order(y_bc, m=MSEAS)
df_nsdiffs = pd.DataFrame.from_dict(ns_diffs, orient="index")
df_nsdiffs.columns = ["order of seasonal differencing"]
# write test result on website: order of differencing
st.dataframe(df_nsdiffs)


# combine the recommendations for differencing
# and apply them to the Box-Cox transformed time series

# copy the original dataframe to df2
# insert the Box-Cox transformed series as a new column y_bc
df2 = df.copy()
s = pd.Series(y_bc)
s.index = df2.index[:len(s)]
df2["y_bc"] = s  


def diff_order_apply(y, m):
    n_diff = diff_order(y).get("decision")
    ns_diff = diffseas_order(y, m).get("decision")
    if n_diff * ns_diff != 0:
        df2["y_bc_diff"] = df2["y_bc"].diff(n_diff).diff(ns_diff)
    elif n_diff + ns_diff != 0:
        df2["y_bc_diff"] = df2["y_bc"].diff(max(n_diff, ns_diff))
    else:
        df2["y_bc_diff"] = df2["y_bc"]
    


# compute the differenced series and show it on the website
diff_order_apply(y_bc, MSEAS)
# df2.dropna(how="any", inplace=True)
y_bc_diff = df2["y_bc_diff"]
st.write("." * 30)
st.write("==  scrollable dataframe after transformations and differencing:")
st.dataframe(df2)  #.style.highlight_max(axis=0))


# check the differenced and transformed series for normality
st.write("." * 30)
st.write("\n**Box-Cox-transformed _and_ differenced time series: normally distributed?**")
p = normaltest(y_bc_diff.dropna())[1]
printme = f'normality test: p-value > {ALPHA:.3f}?  p = {p:.3f}'
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
st.write(printme)
delta = p - ALPHA
# write test result on website, using the "metric" widget
st.metric(label="p-value", value=f'{p:.3f}', delta=f'{delta:.3f}', delta_color="normal")



# %%
# ask user for start and end date of prediction period (test + forecast months)
st.write("_" * 30)
st.write("**SETTING UP the MODEL:**")
end_df2 = max(df2["Date"])                                # last date in time series file
endAct = df2["Pants"].last_valid_index()                  # date of last actual observation
startPred = st.sidebar.date_input(
    label="select the start of the prediction period (maximally 24 trailing months of the actual observations):",
    value=endAct - rdt.relativedelta(months=12),         # default: test/forecast begins 12 months before end of actual time series
    min_value=endAct - rdt.relativedelta(months=24),     # min: test/forecast period should not comprise more than 24 historical actual months
    max_value=endAct - rdt.relativedelta(months=1))      # max: latest forecast start = first month after end of actual observations
n_test = max(1, (endAct.month - startPred.month) + 12*(endAct.year - startPred.year))    # months in test period: between start of predictions and end of actual
n_pred = max(1, (end_df2.month - startPred.month) + 12*(end_df2.year - startPred.year))  # months in prediction period
n_act_df = end_df2.month - endAct.month + 12*(end_df2.year - endAct.year)
st.sidebar.write("selected: " + str(n_test) + " months in _test_ period")
st.sidebar.write(str(n_pred) + " months in _prediction_ period _(test + forecast)_; forecast will based on exogenous values in uploaded file")
st.sidebar.write("_" * 30)


# split dataset into training and test set
if pd.isna(n_test):
    n_test = TESTP

dtrain = df2[:(-n_act_df - n_test)].copy()
dtest = df2[(-n_act_df - n_test): -n_act_df].copy()
dpred = df2[(-n_act_df - n_test):].copy()

st.write("training period from - to:")
st.write(dtrain.iloc[[0,-1]])

st.write("test period from - to:")
st.write(dtest.iloc[[0,-1]])

y_train = dtrain["Pants"]
y_test = dtest["Pants"]

X_train = dtrain["Peak"].to_frame()
X_test = dtest["Peak"].to_frame()
X_pred = dpred["Peak"].to_frame()


# write longer explanations in an expander textbox on the website
st.sidebar.write("." * 30)
st.write("==  expandable textbox:")
expander = st.expander('''left sidebar: end user choices for SARIMAX model parameters (p, q, P, Q):''')
expander.write('''
    The end user has the option to enter SARIMAX parameters in the
    number input fields shown in the sidebar on the left.   
    The input fields display default values the user can overwrite, 
    within the reasonable lower and upper limits the model designer has defined.''')


# order of differencing the stationarity tests recommend:
n_diff = diff_order(y).get("decision")
ns_diff = diffseas_order(y, MSEAS).get("decision")


st.sidebar.write("==  radio buttons: **Which Model?**")
model = st.sidebar.radio("Which model to run:", ("SARIMAX", "Prophet", "LSTM"), index=0)
if model == "SARIMAX":
    st.sidebar.write(" => will run a _SARIMAX_ forecast")
elif "Prophet":
    st.sidebar.write(" => will run a _Prophet_ forecast")
else:
    st.sidebar.write(" => will run a _LSTM_ forecast")


st.sidebar.write("." * 30)
st.sidebar.write("== selectbox: **Which Model?**")
model = st.sidebar.selectbox(
        "Which model to run:", ("SARIMAX", "Prophet", "LSTM"), index=0)   
st.sidebar.write(" => will run a _", model, "_ forecast")


# user-selected SARIMA values:
# offer the user to overwrite the default values for the SARIMA parameters,
# in number input boxes on the didebar of the website:
st.sidebar.write("_" * 30)
st.sidebar.markdown('**Overwrite** the **SARIMAX** default parameters?')
p = st.sidebar.number_input('AR term p', min_value=0, max_value=5, value=0, step=1)
q = st.sidebar.number_input('MA term q', min_value=0, max_value=5, value=1, step=1)
P = st.sidebar.number_input('seasonal AR term P', min_value=0, max_value=5, value=3, step=1)
Q = st.sidebar.number_input('seasonal MA term Q', min_value=0, max_value=5, value=0, step=1)


# %%
# SARIMA model
# insert the user inputs for the parameters in the SARIMA model
from pmdarima.pipeline import Pipeline
pdq = (p, n_diff, q)
PDQm = (P, ns_diff, Q, 12)
intercept = False

pipe = Pipeline([
    ('boxcox', pmd.preprocessing.BoxCoxEndogTransformer(lmbda2=1e-6)),
    ('arima', pmd.arima.ARIMA(order=pdq, seasonal_order=PDQm, with_intercept=intercept, 
        suppress_warnings=True))
                ])

# %%
st.write("_" * 30)
# TRAINING: fit the SARIMA model
st.write("**TRAINING:**")
res = pipe.fit(y=y_train, X=X_train)


# %%
# TRAINING: get the SARIMA model summary and write the summary on the website
st.write("_**SARIMAX Model Summary:**_")
sum = pipe.summary()
# write dictionary to website:
st.write(sum)


# %%
# long textbox with explanations
st.write("  ")
st.write("==  scrollable textbox:")
txt = st.text_area('SARIMAX Summary - Review the Quality of the Model:', 
    '''(1) Ljung-Box: Does the p-value exceed 0.05? Then the residuals consist of white noise, as they should. The model has not failed to identify any valid signal in the observations that could be used to improve the forecast accuracy. 
    (2) Jarque-Bera: Does the p-value exceed 0.05? Then we can conclude that the residuals are normally distributed. Otherwise, with non-normal residuals, confidence intervals would be less reliable.
    (3) Heteroskedasticity: Does the p-value exceed 0.05? Then we can conclude that the variance of the residuals is time-invariant. This is an important criterion for forecast quality.
    (4) Is any SARIMA coefficient close to 0? Then the AR or MA term is negligible for the model, particularly if it also shows a large p-value that exceeds 0.05.''')


# %%
# define prediction accuracy metrics
def prediction_accuracy(y_hat, y_act):
    mae = np.mean(np.abs(y_hat - y_act))                             # MAE
    mape = np.mean(np.abs(y_hat - y_act)/np.abs(y_act))              # MAPE
    rmse = np.mean((y_hat - y_act)**2)**.5                           # RMSE
    corr = np.corrcoef(y_hat, y_act)[0,1]                            # correlation of prediction and actual
    return({'MAE':mae, 
            'MAPE':mape, 
            'RMSE':rmse, 
            'Corr':corr})


# %%
# TRAINING:
# get prediction values for training period
# then compare predictions with actual observation in training set
# in-sample prediction: training dataset
y_hat_train, conf_int_train = pipe.predict_in_sample(
    X=X_train,
    return_conf_int=True, 
    alpha=ALPHA, 
    inverse_transform=True)
resid_train = y_hat_train - y_train
train_accuracy = prediction_accuracy(y_hat_train, y_train)
df_train_accuracy = pd.DataFrame.from_dict(train_accuracy, orient="index")
df_train_accuracy.columns = ["training: accuracy"]


# %%
# TESTING:
st.write("_" * 30)
st.write("**TESTING:**")
# compute predictions for the months in the test dataset 
# and compare them with the actual observations
# prediction: test dataset
y_hat_test, conf_int_test = pipe.predict(
    X=X_test,
    n_periods=n_test,                                  
    return_conf_int=True, 
    alpha=ALPHA, 
    inverse_transform=True)
resid_test = y_hat_test - y_test
test_accuracy = prediction_accuracy(y_hat_test, y_test)
df_test_accuracy = pd.DataFrame.from_dict(test_accuracy, orient="index")
df_test_accuracy.columns = ["testing: accuracy"]


# %% 
# TESTING vs TRAINING: prediction accuracy
# show the prediction accuracy for training and test dataset, column by column, on the website
df_accuracy = pd.concat([df_train_accuracy, df_test_accuracy], axis=1)
df_accuracy["test vs training"] = df_accuracy["testing: accuracy"] - df_accuracy["training: accuracy"]
st.write("prediction accuracy: testing vs training dataset")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
st.write(df_accuracy)


# %%
# TESTING:
# combine numpy arrays: predicted test and actual values:
# st.write("." * 30)
# combine predictions of training and testing period in a single column vector:
yhat = np.hstack((y_hat_train, y_hat_test))
# combine: column of predictions and column of actual observations:
y_yhat = np.vstack((y, yhat))
# copy the original dataframe:
df_fc = df2.copy()
df_fc = df_fc[["Date", "year", "month"]]

# fill the copied dataframe with the column vectors for predictions, actual observations, and errors
s0 = pd.Series(y_yhat[0])
s0.index = df_fc.index[:len(s0)]
df_fc["actual"] = s0

s1 = pd.Series(y_yhat[1])
s1.index = df_fc.index[:len(s1)]
df_fc["predictions"] = s1

df_fc["variance"] = df_fc["predictions"] - df_fc["actual"]
df_fc["perc.var"] = df_fc["predictions"] / df_fc["actual"] -1



# %%
st.write("_" * 30)
# FORECASTING out-of-sample:
st.write("**FORECASTING:**")
# compute a forecast for the months beyond the test dataset
y_hat, conf_int = pipe.predict(
    X=X_pred,    
    n_periods=len(X_pred),                                  
    return_conf_int=True, 
    alpha=ALPHA, 
    inverse_transform=True)


# %%
# FORECASTING:
# combine numpy arrays: predicted test and actual values:
yhat = np.hstack((y_hat_train, y_hat))
# copy the original dataframe:
df_fc = df2.copy()
df_fc = df_fc.rename(columns={"Pants":"actual"})
# fill the copied dataframe with the column vectors for predictions, actual observations, and errors
s2 = pd.Series(yhat)
s2.index = df_fc.index[:len(s2)]
df_fc["forecast"] = s2
df_fc["variance"] = df_fc["forecast"] - df_fc["actual"]
df_fc["perc.var"] = df_fc["forecast"] / df_fc["actual"] -1
df_fc = df_fc[["Date", "actual", "forecast", "variance", "perc.var", "Peak"]]


# plot a chart with predictions vs actual observations on the website
fig = px.line(df_fc, x="Date", y=["actual", "forecast"], 
    title="sweatpants popularity: forecast vs. actual", width=1000)
st.plotly_chart(fig, use_container_width=False)

st.dataframe(df_fc)

# %%
# download button: offer the end user the option to download the dataframe 
# with the prediction results as a .csv file

@st.cache      # cache the conversion to csv to prevent computation on every rerun: @st.cache
def convert_df(df):   # convert dataframe to .csv
    return df_fc.to_csv().encode('utf-8')

file_csv = convert_df(df)   # conversion function: dataframe to csv


# download button in the sidebar
st.sidebar.write("_" * 30)
st.sidebar.download_button(
    label="download dataframe with predictions: file = df.csv", 
    data=file_csv,             
    file_name='df.csv',
    mime='text/csv',
    key="btDL_side")

# second download button
st.download_button(
    label="download dataframe with predictions: file = df.csv", 
    data=file_csv,             
    file_name='df.csv',
    mime='text/csv',
    key="btDL")


# %%
# almost done, we can soon relax
# and also demonstrate how to display a relaxed picture
st.write("_" * 30)
st.write("done for today:")
from PIL import Image
image = Image.open(wpath + "/mini.jpg")
st.image(image, caption="author's property")


# %%
st.write("_" * 30)
# long textbox with explanations
txt = st.text_area('==  scrollable textbox with explanations:', 
    '''We have set up a website for our forecast app with just a few lines of Python code taken from the Streamlit package.
    Streamlit methods, similar to the print method that would display output in a Jupyter cell or in a terminal window,
    write the output to the website. We also inserted a couple of Streamlit controls such as radio buttons.

    Now the non-Pythonistas among our end users can run a forecast model from their standard browsers. They don't need 
    to install, maintain and update Python on their local computers. They can even access the model via mobile devices.
    Nor do they need learn how to run a Python script or a Jupyter notebook.

    (1) The end user can upload a .csv file with updated actual observations.
    (2) Before kicking off the forecasting process, the user has the option to overwrite the 
    default parameters in the sidebar.  
    (3) After the model has completed the forecast, the end user can download a csv file with the results.''')
