from datetime import datetime
import logging
import numpy as np
from os import path
import time


#some constants

MIN_LINE_SIZE = 30
X_NUM_COLS = 8 #'temp', 'humidity', 'light', 'CO2', 'dust', 'status' and time ('hour' and 'day')
NUM_CSV_COLS = 7
SEP_CHAR = ","
POWER_CUT = 0.01 #cut value for positive power consumption
ON_MIN_DURATION = 5 # more than 5 minute on is considered as ON status
OFF_MIN_DURATION = 5 # more than 5 minutes off is considered as OFF status

STATUS_ON = 1
STATUS_OFF = 0

ACTION_NOTHING = 0
ACTION_TURN_ON = 1
ACTION_TURN_OFF = 2


class EngineError(Exception):
    def __init__(self, msg, details=True):
        global log_to_file

        if not log_to_file:
            details = False

        if details:
            self.__message = msg + ", see details in log file"
        else:
            self.__message = msg
    def __str__(self):
        return repr(self.__message)



def process_data(filename):

    file_size = path.getsize(filename)

    max_num_lines = int(file_size/MIN_LINE_SIZE)

    raw_data = np.empty([max_num_lines, X_NUM_COLS], float) #plus 1 col for power
    power = np.empty(max_num_lines, float)
    saving_time =  np.empty(max_num_lines, float)

    nrows=0
    temp_values = np.zeros(X_NUM_COLS+1, float)
    with open(filename, 'rt') as f:
        headers = f.readline()[:-1].split(SEP_CHAR)
        for line in f:
            values = line[:-1].split(SEP_CHAR)
            if len(values) == NUM_CSV_COLS:
                try:
                    #get time
                    t = datetime.strptime(values[0], "\"%Y-%m-%d %H:%M:%S\"")
                    temp_values[X_NUM_COLS-2] = t.weekday()
                    temp_values[X_NUM_COLS-1] = t.hour + t.minute/60.0 + t.second/3600.0

                    #get remaining columns
                    for i in range(1, len(values)):
                        temp_values[i-1] = float(values[i])
                    #TODO: remove bad values (outliers)
                except Exception as e: #pass the ill-format line
                    print e
                    pass
                power[nrows] = temp_values[0]
                saving_time[nrows] = time.mktime(t.timetuple())
                for i in range(1, X_NUM_COLS):
                    raw_data[nrows, i-1] = temp_values[i]

                nrows += 1


    raw_data = np.resize(raw_data,[nrows, X_NUM_COLS])
    power = np.resize(power, nrows)
    saving_time = np.resize(saving_time, nrows)
    power[np.where(power<0)] = 0


    action = np.empty(nrows, int)
    status = np.empty(nrows, int)
    status.fill(STATUS_OFF)
    action.fill(ACTION_NOTHING)


    #detect power ON status
    status[np.where(power>POWER_CUT)] = STATUS_ON

    #detect TURN ON, TURN OFF action
    action_id = []
    for i in range(1, nrows):
        status_diff = status[i] - status[i-1]
        if status_diff == 1:
            action[i] = ACTION_TURN_ON
            action_id.append(i)
        elif status_diff == -1:
            action[i] = ACTION_TURN_OFF
            action_id.append(i)



    min_diff = OFF_MIN_DURATION*60 # 5 minutes time 60 secs
    num_actions = len(action_id)
    action_id = np.array(action_id)

    #remove any (TURN_OFF, TURN_ON) tuple whose time interval is too short (less than 5, for example)
    i = num_actions-1
    while i>0:
        if action[action_id[i]] == ACTION_TURN_ON:
            time_diff = saving_time[action_id[i]] - saving_time[action_id[i-1]]

            if time_diff < min_diff:
                action[action_id[i]] = ACTION_NOTHING # remove TURN ON
                action[action_id[i-1]] = ACTION_NOTHING # remove TURN OFF
            i -= 2
        else:
            i -= 1

    #remove any (TURN_ON, TURN_OFF) tuple whose time interval is too short (less than 5, for example)
    min_diff = ON_MIN_DURATION*60 # 5 minutes time 60 secs
    i=0

    while i<num_actions-1:
        if action[action_id[i]] == ACTION_TURN_ON:
            #find the next TURN OFF action
            j=i+1
            while (j<num_actions and action[action_id[j]] != ACTION_TURN_OFF):
                j += 1
            if j<num_actions: # found TURN OFF action
                time_diff = saving_time[action_id[j]] - saving_time[action_id[i]]
                if time_diff < min_diff:
                    action[action_id[i]] = ACTION_NOTHING # remove TURN ON
                    action[action_id[j]] = ACTION_NOTHING # remove TURN OFF
            i=j+1
        else:
            i += 1

    #now fix the power status column

    prev_status = status[0]
    for i in range(1, nrows):
        if action[i] == ACTION_NOTHING:
            status[i] = prev_status
        elif action[i] == ACTION_TURN_ON:
            status[i] = STATUS_ON
        else:
            status[i] = STATUS_ON
        prev_status = status[i]


    # reduce the data size
    n_obs = 10 # sampling every 10 timepoints (i.e. 5 minutes)
    nfrows = int(nrows/n_obs) #number of data rows after sampling

    x = np.empty([nfrows, X_NUM_COLS], float)
    y = np.empty(nfrows, int)
    y.fill(ACTION_NOTHING)

    sampled_id = n_obs
    i = 0

    feat_period = 5*60 #extract feature from previous 5 minutes
    while sampled_id <nrows:

        #find any near by actions
        for j in range(n_obs):
            if action[sampled_id+j] != ACTION_NOTHING :
                sampled_id += j
                break

        #collect samples which are obs_period seconds from sampled_id sample
        feature_ids=[]
        cur_id = sampled_id-1
        while (cur_id>=0) and ((saving_time[sampled_id]-saving_time[cur_id]) < feat_period):
            feature_ids.append(cur_id)
            cur_id -= 1
        if len(feature_ids)>0:
            #extract feature from these samples
            for j in range(5):
                x[i, j] = np.mean(raw_data[feature_ids, j])
            x[i, 5] = raw_data[sampled_id, 5] #week day
            x[i, 6] = raw_data[sampled_id, 6] #hour
            x[i, 7] = status[sampled_id-1]
            y[i] = action[sampled_id]
            i += 1

        #move on to the next id for sampling
        sampled_id += n_obs

    return [x, y]


def train_model(data, method):
    """Train model with give method (forest/tree)
    Note: Currently decision tree pruning is not support by
    sklearn. Unless we decide to use decision tree, pruning is not
    neccesary.

    """
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier

    try:
        if (method=="tree"):
            model = tree.DecisionTreeClassifier()
            method_name="Decision Tree"
        elif (method=="forest"):
            model = RandomForestClassifier(n_estimators=100)
            method_name="Random Forest"
        else:
            raise EngineError("Unexpected error")
        logging.info("train a " + method_name + " model")
        model = model.fit(data[0], data[1])
    except Exception as e:
        logging.error(e)
        raise EngineError("Failed to train model")
    return model

def evaluate_model(data, model):
    from sklearn.metrics import confusion_matrix
    logging.info("evaluate model")
    #first partition data into train and test data

    try:
        x = data[0]
        y = data[1]
        pred_y = model.predict(x)
        con_mat = confusion_matrix(y, pred_y)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to evaluate model")
    return con_mat