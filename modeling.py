from datetime import datetime
import logging
import numpy as np
from os import path
import time

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#some constants

# data processing and feature extraction parameters
MIN_LINE_SIZE = 30      #minimum length of a data line. Used for guessing the number of data rows
NUM_FEATURES = 7        #number of features for prediction
NUM_CSV_COLS = 7        #number of columns in CVS file, plus 1 ('temp', 'humidity', 'light'
                        # , 'CO2', 'dust', and time ('hour' and 'day'))

SEP_CHAR = ","          #seprating character for csv file
POWER_CUT = 0.01        #cut value for positive power consumption
ON_MIN_DURATION = 5     # more than 5 minute on is considered as ON status
OFF_MIN_DURATION = 5    # more than 5 minutes off is considered as OFF status
PAST_DURATION = 5       # 5 minutes of past data is used for feature extraction
SAMPLING_SIZE = 60  #

SENSOR_RANGES=((0, 50),(0,100),(0, 100), (0,2000),(0,100)) #valid ranges of temp, humidity, light, CO2, dust sensors

#air conditioner status labels
STATUS_ON = 1           #aircon is ON
STATUS_OFF = 0          #aircon is OFF

#class labels
ACTION_NOTHING = 0      #user does nothing
ACTION_TURN_ON = 1      #user turns on aircon
ACTION_TURN_OFF = 2     #user turn off aircon


#random forest/decision tree parameter
MIN_SPLIT = 5           #minimum leaf size for spliting
NUM_TREES = 100         #number of trees to make forest

NUM_FOLDS = 3           #number of folds for cross-validation
NUM_CLASS = 2           #binary classification

log_to_file = False     #log message to file?

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

    raw_data = np.empty([max_num_lines, NUM_CSV_COLS], float) #plus 1 col for power
    power = np.empty(max_num_lines, float)
    saving_time =  np.empty(max_num_lines, float)

    nrows=0
    temp_values = np.zeros(NUM_CSV_COLS, float)
    with open(filename, 'rt') as f:
        headers = f.readline()[:-1].split(SEP_CHAR)
        for line in f:
            values = line[:-1].split(SEP_CHAR)
            if len(values) == NUM_CSV_COLS:
                try:
                    #get time
                    t = datetime.strptime(values[0], "\"%Y-%m-%d %H:%M:%S\"")
                    temp_values[NUM_CSV_COLS-2] = t.weekday()
                    temp_values[NUM_CSV_COLS-1] = t.hour + t.minute/60.0 + t.second/3600.0

                    #get power column
                    temp_values[0] = float(values[1])
                    #get reamaining columns (`temp`, `humidity`, `CO2` and `dust`)
                    for i in range(2, len(values)):
                        v = float(values[i])
                        if (v<SENSOR_RANGES[i-2][0]) or (v>SENSOR_RANGES[i-2][1]):
                            raise ValueError("sensor value is out of range")
                        temp_values[i-1] = v

                    power[nrows] = temp_values[0]
                    saving_time[nrows] = time.mktime(t.timetuple())
                    for i in range(1, NUM_CSV_COLS-1):
                        raw_data[nrows, i-1] = temp_values[i]
                    nrows += 1

                except Exception as e: #pass the ill-format line
                    #logging.warning(e)
                    pass



    raw_data = np.resize(raw_data,[nrows, NUM_CSV_COLS])
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
            status[i] = STATUS_OFF
        prev_status = status[i]


    # reduce the data size
    n_obs = SAMPLING_SIZE # sampling every 10 timepoints (i.e. 5 minutes)
    nfrows = int(nrows/n_obs) #number of data rows after sampling

    x = np.empty([nfrows, NUM_FEATURES], float)
    y = np.empty(nfrows, int)
    y.fill(ACTION_NOTHING)
    sampled_status = np.empty(nfrows, int)

    sampled_id = n_obs
    i = 0

    feat_period = PAST_DURATION*60 #extract feature from previous 5 minutes
    while sampled_id <nrows:

        #find any near by actions
        for j in range(n_obs):
            new_id = sampled_id+j
            if new_id < nrows and action[new_id] != ACTION_NOTHING :
                sampled_id = new_id
                break

        #collect samples which are obs_period seconds from sampled_id sample
        feature_ids=[]
        cur_id = sampled_id-1
        while (cur_id>=0) and ((saving_time[sampled_id]-saving_time[cur_id]) < feat_period):
            feature_ids.append(cur_id)
            cur_id -= 1

        if len(feature_ids)>0:
            #extract feature from these samples
            for j in range(NUM_CSV_COLS-2):
                x[i, j] = np.mean(raw_data[feature_ids, j])
                #x[i, 2*j] = np.mean(raw_data[feature_ids, j])
                #x[i, 2*j+1] = np.std(raw_data[feature_ids, j])
            x[i, NUM_FEATURES-2] = raw_data[sampled_id, NUM_CSV_COLS-2] #week day
            x[i, NUM_FEATURES-1] = raw_data[sampled_id, NUM_CSV_COLS-1] #hour
            sampled_status[i] = status[sampled_id-1]
            y[i] = action[sampled_id]
            i += 1

        #move on to the next id for sampling
        sampled_id += n_obs

    nrows = i
    x = np.resize(x, [nrows, NUM_FEATURES])
    y = np.resize(y, nrows)
    #sampled_status = np.resize(sampled_status, nrows)
    on_ids = []
    off_ids = []
    for i in range(nrows):
        if sampled_status[i] == STATUS_ON:
            on_ids.append(i)
        else:
            off_ids.append(i)
    return [x[on_ids, :], y[on_ids], x[off_ids, :], y[off_ids]]

def train_model(x, y, method):
    if method == "tree":
        return train_tree(x, y)
    elif method == "forest":
        return train_forest(x, y)
    else:
        raise EngineError("Unexpected error")

def train_forest(x, y):
    """Train a random forest with given data
    """
    try:
        model = RandomForestClassifier(n_estimators=NUM_TREES, min_samples_split=MIN_SPLIT)
        logging.info("train a random forest model")
        model = model.fit(x, y)
    except Exception as e:
        logging.error(e)
        raise EngineError("Failed to train model")
    return model


def train_tree(x, y):
    """Train a decision tree
    Note: Currently decision tree pruning is not support by
    sklearn. Unless we decide to use decision tree, pruning is not
    neccesary.
    """
    try:
        model = tree.DecisionTreeClassifier()
        logging.info("train a decision tree model")
        model = model.fit(x, y)
    except Exception as e:
        logging.error(e)
        raise EngineError("Failed to train model")
    return model

def evaluate_model(x, y, method):
    from sklearn.metrics import confusion_matrix
    logging.info("evaluate model")


    #first partition data into train and test data
    from sklearn.cross_validation import StratifiedKFold
    try:
        skf = StratifiedKFold(y, NUM_FOLDS)
        precisions = np.zeros([NUM_FOLDS, NUM_CLASS], float)
        recalls = np.zeros([NUM_FOLDS, NUM_CLASS], float)
        con_mats = []
        i=0
        for train, test in skf:
            x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]
            if method=="tree":
                model = train_forest(x_train, y_train)
            elif method=="forest":
                model = train_forest(x_train, y_train)
            else:
                raise EngineError("unexpected error")
            pred_y1 = model.predict(x_train)
            pred_y2 = model.predict(x_test)
            con_mats.append([confusion_matrix(y_train, pred_y1), confusion_matrix(y_test, pred_y2)])
            i += 1
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to evaluate model")
    return con_mats
