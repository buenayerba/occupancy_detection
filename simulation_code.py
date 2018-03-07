# Load Required Libraries

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# load k-NN estimator
from sklearn.neighbors import KNeighborsClassifier
# load ANN
from sklearn.neural_network import MLPClassifier
# load TPOT
from tpot import TPOTClassifier
# load Random Forrest
from sklearn.ensemble import RandomForestClassifier
# load naive-bayes estimator
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
#load decision tree estimator
from sklearn import tree
# Load support vector machines
from sklearn.svm import SVC


plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13

# Define Helper Functions
def read_data(file_path):
    data = pd.read_csv(file_path,header = 0)
    data["date"] = pd.to_datetime(data["date"])
    #data["year"] = data["date"].dt.year
    #data["month"] = data["date"].dt.month
    #data["day"] = data["date"].dt.day
    #data["hour"] = data["date"].dt.hour
    #data["minute"] = data["date"].dt.minute
    #data["second"] = data["date"].dt.second    
    data['SecMid'] = data["date"].dt.hour * 60 * 60 + data["date"].dt.minute * 60 + data["date"].dt.second
    data['Weekday'] = data[['date']].apply(lambda x: dt.datetime.strftime(x['date'], '%A'), axis=1)
    data['WeekStatus'] = np.where((data['Weekday'] == 'Sunday') | (data['Weekday'] == 'Saturday'), 0, 1)
        
    del data['Weekday']
    #del data['date']
    
    return data

tr_data = read_data("datatraining.txt")
ts_one_data = read_data("datatest.txt")
ts_two_data = read_data("datatest2.txt")

frames = [tr_data, ts_one_data, ts_two_data]
result = pd.concat(frames)
result = result.sample(frac = 1)


tr_data = result.iloc[: int(0.7*len(result)), :]
ts_one_data = result.iloc[int(0.7*len(result)) : int(0.85*len(result)), :]
ts_two_data = result.iloc[int(0.85*len(result)) :, :]

def plot_signal(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%A, %b %d'))
    ax.set_xlim([min(x) , max(x)])
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.grid(True)
    plt.subplots_adjust(hspace=0.5)
    
def analyze(clf):
	clf.fit(tr_features, tr_labels)
	trn_pred = clf.predict(tr_features)
	print "Training Data Accuracy: ", ("%.3f" % (accuracy_score(tr_labels, trn_pred) * 100))
	test_pred = clf.predict(ts_one_features)
	print "Testing Data Set One Accuracy: ", ("%.3f" % (accuracy_score(ts_one_labels, test_pred) * 100))
	test_pred2 = clf.predict(ts_two_features)
	print "Testing Data Set Two Accuracy: ", ("%.3f" % (accuracy_score(ts_two_labels, test_pred2) * 100))
   
#--------------------------------------------------------------------
#------ Plotting Individual Signals ---------------------------------
#--------------------------------------------------------------------
# plot_signal(ts_two_data["date"], ts_two_data["CO2"], 'CO2')
# plot_signal(ts_two_data["date"], ts_two_data["Occupancy"], 'Occupancy')
# plot_signal(ts_two_data["date"], ts_two_data["Light"], 'Light')
# plot_signal(ts_two_data["date"], ts_two_data["Temperature"], 'Temperature')
# plot_signal(ts_two_data["date"], ts_two_data["Humidity"], 'Humidity')
# plot_signal(ts_two_data["date"], ts_two_data["HumidityRatio"], 'HumidityRatio')
# plt.show()


#--------------------------------------------------------------------
#------------- Plotting Bar Diagram ---------------------------------
#--------------------------------------------------------------------

# Scale and average features by occupancy for plotting
# scaled_data =  tr_data[["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"]]
# scaled_data.loc[:,"Temperature"] = scale(scaled_data["Temperature"])
# scaled_data.loc[:,"Humidity"] = scale(scaled_data["Humidity"])
# scaled_data.loc[:,"Light"] = scale(scaled_data["Light"])
# scaled_data.loc[:,"CO2"] = scale(scaled_data["CO2"])

# min_max_scaler = MinMaxScaler()

# scaled_data.loc[:,"CO2"] = min_max_scaler.fit_transform(scaled_data["CO2"])
# scaled_data.loc[:,"Temperature"] = min_max_scaler.fit_transform(scaled_data["Temperature"])
# scaled_data.loc[:,"Humidity"] = min_max_scaler.fit_transform(scaled_data["Humidity"])
# scaled_data.loc[:,"Light"] = min_max_scaler.fit_transform(scaled_data["Light"])
# 
# summary_occupancy = scaled_data[["Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"]].\
#         groupby(["Occupancy"],as_index = False).mean()
#     
# n_groups = 4
# fig, ax = plt.subplots(figsize = (8,5))
# index = np.arange(n_groups)
# bar_width = 0.35
# 
# rects_no = plt.bar(index,np.asarray(summary_occupancy[["Temperature","Humidity","Light","CO2"]])[0], 
#                  bar_width,
#                  color="grey",
#                  label="Not Occupied")
#  
#     
# rects_o = plt.bar(index + bar_width,np.asarray(summary_occupancy[["Temperature","Humidity","Light","CO2"]])[1], 
#                  bar_width,
#                  label="Occupied")
# 
# plt.xlabel("Features")
# plt.ylabel("Average")
# plt.xticks(index + bar_width, ("Temperature", "Humidity","Light", "CO2")) 
# plt.legend(loc = "upper right")
# plt.title("MinMaxScaler")
#  
# plt.tight_layout()
# # plt.show()
# 


	
#-----------------------------------------------------------------------
#--------------------Scaled Data----------------------------------------
#-----------------------------------------------------------------------

# scaled_data_tr =  tr_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid", "WeekStatus" ,"Occupancy"]]
# scaled_data_tr.loc[:,"Temperature"] = scale(scaled_data_tr["Temperature"])
# scaled_data_tr.loc[:,"Humidity"] = scale(scaled_data_tr["Humidity"])
# scaled_data_tr.loc[:,"Light"] = scale(scaled_data_tr["Light"])
# scaled_data_tr.loc[:,"CO2"] = scale(scaled_data_tr["CO2"])
# 
# scaled_data_ts1 =  ts_one_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid", "WeekStatus", "Occupancy"]]
# scaled_data_ts1.loc[:,"Temperature"] = scale(scaled_data_ts1["Temperature"])
# scaled_data_ts1.loc[:,"Humidity"] = scale(scaled_data_ts1["Humidity"])
# scaled_data_ts1.loc[:,"Light"] = scale(scaled_data_ts1["Light"])
# scaled_data_ts1.loc[:,"CO2"] = scale(scaled_data_ts1["CO2"])
# 
# scaled_data_ts2 =  ts_two_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid", "WeekStatus", "Occupancy"]]
# scaled_data_ts2.loc[:,"Temperature"] = scale(scaled_data_ts2["Temperature"])
# scaled_data_ts2.loc[:,"Humidity"] = scale(scaled_data_ts2["Humidity"])
# scaled_data_ts2.loc[:,"Light"] = scale(scaled_data_ts2["Light"])
# scaled_data_ts2.loc[:,"CO2"] = scale(scaled_data_ts2["CO2"])

#--------------------------------------------------------------------
#--------------------MinMax Normalized Data--------------------------
#--------------------------------------------------------------------


# scaled_data_tr =  tr_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid", "WeekStatus" ,"Occupancy"]]
# scaled_data_tr.loc[:,"CO2"] = min_max_scaler.fit_transform(scaled_data_tr["CO2"])
# scaled_data_tr.loc[:,"Temperature"] = min_max_scaler.fit_transform(scaled_data_tr["Temperature"])
# scaled_data_tr.loc[:,"Humidity"] = min_max_scaler.fit_transform(scaled_data_tr["Humidity"])
# scaled_data_tr.loc[:,"Light"] = min_max_scaler.fit_transform(scaled_data_tr["Light"])
# 
# scaled_data_ts1 =  ts_one_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid", "WeekStatus" ,"Occupancy"]]
# scaled_data_ts1.loc[:,"CO2"] = min_max_scaler.fit_transform(scaled_data_ts1["CO2"])
# scaled_data_ts1.loc[:,"Temperature"] = min_max_scaler.fit_transform(scaled_data_ts1["Temperature"])
# scaled_data_ts1.loc[:,"Humidity"] = min_max_scaler.fit_transform(scaled_data_ts1["Humidity"])
# scaled_data_ts1.loc[:,"Light"] = min_max_scaler.fit_transform(scaled_data_ts1["Light"])
# 
# scaled_data_ts2 =  ts_two_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid", "WeekStatus" ,"Occupancy"]]
# scaled_data_ts2.loc[:,"CO2"] = min_max_scaler.fit_transform(scaled_data_ts2["CO2"])
# scaled_data_ts2.loc[:,"Temperature"] = min_max_scaler.fit_transform(scaled_data_ts2["Temperature"])
# scaled_data_ts2.loc[:,"Humidity"] = min_max_scaler.fit_transform(scaled_data_ts2["Humidity"])
# scaled_data_ts2.loc[:,"Light"] = min_max_scaler.fit_transform(scaled_data_ts2["Light"])



#--------------------------------------------------------------------
#-------------------- Predictors and Labels--------------------------
#--------------------------------------------------------------------
# This part of the code contains different combinations of predictors for testing

# Labels
tr_labels = tr_data["Occupancy"]
ts_one_labels = ts_one_data["Occupancy"]
ts_two_labels = ts_two_data["Occupancy"]


#-------------------- SCALED FEATURES -------------------------------

# tr_features = scaled_data_tr[["Humidity", "CO2", "WeekStatus", "SecMid"]]
# ts_one_features = scaled_data_ts1[["Humidity", "CO2", "WeekStatus", "SecMid"]]
# ts_two_features = scaled_data_ts2[["Humidity","CO2", "WeekStatus", "SecMid"]]

# tr_features = scaled_data_tr[["CO2", "WeekStatus", "SecMid"]]
# ts_one_features = scaled_data_ts1[["CO2", "WeekStatus", "SecMid"]]
# ts_two_features = scaled_data_ts2[["CO2", "WeekStatus", "SecMid"]]

# tr_features = scaled_data_tr[["CO2"]]
# ts_one_features = scaled_data_ts1[["CO2"]]
# ts_two_features = scaled_data_ts2[["CO2"]]

# tr_features = scaled_data_tr[["Humidity", "WeekStatus", "SecMid"]]
# ts_one_features = scaled_data_ts1[["Humidity", "WeekStatus", "SecMid"]]
# ts_two_features = scaled_data_ts2[["Humidity", "WeekStatus", "SecMid"]]

# tr_features = scaled_data_tr[["Temperature","Humidity","Light","CO2","HumidityRatio", "WeekStatus", "SecMid"]]
# ts_one_features = scaled_data_ts1[["Temperature","Humidity","Light","CO2","HumidityRatio", "WeekStatus", "SecMid"]]
# ts_two_features = scaled_data_ts2[["Temperature","Humidity","Light","CO2","HumidityRatio", "WeekStatus", "SecMid"]]


#-------------------- UNMODIFIED FEATURES ----------------------------
# 
# tr_features = tr_data[["Humidity", "CO2", "WeekStatus", "SecMid", "HumidityRatio"]]
# ts_one_features = ts_one_data[["Humidity", "CO2", "WeekStatus", "SecMid", "HumidityRatio"]]
# ts_two_features = ts_two_data[["Humidity","CO2", "WeekStatus", "SecMid", "HumidityRatio"]]

# tr_features = tr_data[["CO2", "Humidity"]]
# ts_one_features = ts_one_data[["CO2", "Humidity"]]
# ts_two_features = ts_two_data[["CO2", "Humidity"]]

# tr_features = tr_data[["WeekStatus", "SecMid", "CO2", "Humidity"]]
# ts_one_features = ts_one_data[["WeekStatus", "SecMid", "CO2", "Humidity"]]
# ts_two_features = ts_two_data[["WeekStatus", "SecMid", "CO2", "Humidity"]]

# tr_features = tr_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "WeekStatus", "SecMid"]]
# ts_one_features = ts_one_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "WeekStatus", "SecMid"]]
# ts_two_features = ts_two_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "WeekStatus", "SecMid"]]

# tr_features = tr_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid"]]
# ts_one_features = ts_one_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid"]]
# ts_two_features = ts_two_data[["Temperature","Humidity","Light","CO2","HumidityRatio", "SecMid"]]


# tr_features = tr_data[["Temperature","Humidity","Light","CO2","HumidityRatio"]]
# ts_one_features = ts_one_data[["Temperature","Humidity","Light","CO2","HumidityRatio"]]
# ts_two_features = ts_two_data[["Temperature","Humidity","Light","CO2","HumidityRatio"]]

# tr_features = tr_data[["WeekStatus"]]
# ts_one_features = ts_one_data[["WeekStatus"]]
# ts_two_features = ts_two_data[["WeekStatus"]]

# tr_features = tr_data[["SecMid"]]
# ts_one_features = ts_one_data[["SecMid"]]
# ts_two_features = ts_two_data[["SecMid"]]

tr_features = tr_data[["CO2", "Temperature"]]
ts_one_features = ts_one_data[["CO2", "Temperature"]]
ts_two_features = ts_two_data[["CO2", "Temperature"]]


# tr_features = tr_data[["HumidityRatio"]]
# ts_one_features = ts_one_data[["HumidityRatio"]]
# ts_two_features = ts_two_data[["HumidityRatio"]]

# tr_features = tr_data[["WeekStatus", "Light", "SecMid"]]
# ts_one_features = ts_one_data[["WeekStatus", "Light", "SecMid"]]
# ts_two_features = ts_two_data[["WeekStatus", "Light", "SecMid"]]

# tr_features = tr_data[["WeekStatus", "Light"]]
# ts_one_features = ts_one_data[["WeekStatus", "Light"]]
# ts_two_features = ts_two_data[["WeekStatus", "Light"]]

# tr_features = tr_data[["Temperature", "WeekStatus", "SecMid"]]
# ts_one_features = ts_one_data[["Temperature", "WeekStatus", "SecMid"]]
# ts_two_features = ts_two_data[["Temperature", "WeekStatus", "SecMid"]]

# tr_features = tr_data[["CO2", "WeekStatus", "SecMid"]]
# ts_one_features = ts_one_data[["CO2", "WeekStatus", "SecMid"]]
# ts_two_features = ts_two_data[["CO2", "WeekStatus", "SecMid"]]

# tr_features = tr_data[["Humidity"]]
# ts_one_features = ts_one_data[["Humidity"]]
# ts_two_features = ts_two_data[["Humidity"]]

# tr_features = tr_data[["WeekStatus", "SecMid"]]
# ts_one_features = ts_one_data[["WeekStatus", "SecMid"]]
# ts_two_features = ts_two_data[["WeekStatus", "SecMid"]]

# tr_features = tr_data[["Temperature","Humidity","CO2"]]
# ts_one_features = ts_one_data[["Temperature","Humidity","CO2"]]
# ts_two_features = ts_two_data[["Temperature","Humidity","CO2"]]
#  
# tr_features = tr_data[["Humidity","CO2","HumidityRatio", "Temperature"]]
# ts_one_features = ts_one_data[["Humidity","CO2","HumidityRatio", "Temperature"]]
# ts_two_features = ts_two_data[["Humidity","CO2","HumidityRatio", "Temperature"]]

# tr_features = tr_data[["Humidity","CO2"]]
# ts_one_features = ts_one_data[["Humidity","CO2"]]
# ts_two_features = ts_two_data[["Humidity","CO2"]]



#--------------------------------------------------------------------
#--------------------------------------------------------------------
#-------------------- Training --------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

print tr_features.head(1)

#--------------------------------------------------------------------
print "-"*30, "\nPrediction with scikit-learn"
# --------------------------------------------------------------------
# print "\nUsing ___ features for prediction"
# print "Naive Bayes"
# clf = BernoulliNB()
# analyze(clf)

#--------------------------------------------------------------------
print "\nUsing ___ features for prediction"
print "Gaussian Bayes"
clf = GaussianNB()
#clf = BernoulliNB()

analyze(clf)

#--------------------------------------------------------------------
print "\nUsing ___ features for prediction"
print "Decision Trees:"
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)
analyze(clf)

#--------------------------------------------------------------------
print "\nUsing ___ features for prediction"
print "Random Forrest"
clf = RandomForestClassifier()
analyze(clf)

#--------------------------------------------------------------------
print "\nUsing ___ features for prediction"
print "Support Vector Machines:"
clf = SVC()
analyze(clf)
#--------------------------------------------------------------------
print "\nUsing ___ features for prediction"
print "K-NN:"
clf = KNeighborsClassifier(n_neighbors=3)
analyze(clf)

#--------------------------------------------------------------------
print "\nUsing ___ features for prediction"
print "ANN:"
clf = MLPClassifier(solver='adam', alpha=1e-3,
                     hidden_layer_sizes=(30, 30), random_state=1)

analyze(clf)