from datetime import datetime
import logging
import numpy as np
from os import path
import time

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# data processing and feature extraction parameters
MIN_LINE_SIZE = 30      #minimum length of a data line. Used for guessing the number of data rows
NUM_FEATURES = 7        #number of features for prediction
NUM_CSV_COLS = 7        #number of columns in CVS file ('time', `power`, 'temp', 'humidity', 'light'
                        # , 'CO2', 'dust', `Temp.ext`, `Humidity.ext`)

SEP_CHAR = ","          #seprating character for csv file
POWER_CUT = 0.01        #cut value for positive power consumption
POWER_MIN_NA_CHUNK_LEN = 10   #a chunk of more than 10 continous NA values (-1) for `power` is set to be removed
POWER_MIN_GOOD_CHUNK_LEN = 20   #a good chunk must have at least 20 observations (10 minutes)
ON_MIN_DURATION = 10     #more than 5 minute on is considered as ON status
OFF_MIN_DURATION = 10    #more than 5 minutes off is considered as OFF status
PAST_DURATION = 5       #5 minutes of past data is used for feature extraction
SAMPLING_SIZE = 30      #1 for 30 secs, 2 for 1 minutes, 10 for 5 minutes, 60 for 30 minutes and so on...

SENSOR_RANGES=((0, 50),(0,100),(0, 100), (0,2000),(0,100), (0, 50), (0,100)) #valid ranges of temp, humidity, light, CO2, dust sensors

#air conditioner status labels
STATUS_ON = 1           #aircon is ON
STATUS_OFF = 0          #aircon is OFF

#user actions
ACTION_NOTHING = 0      #user does nothing
ACTION_TURN_ON = 1      #user turns on aircon
ACTION_TURN_OFF = 2     #user turn off aircon


#random forest/decision tree parameter
MIN_SPLIT = 4           #minimum leaf size for spliting
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

def read_raw_data(filename):
    """This function reads csv file to a numpy array
    """

    #guess the number of rows for data array allocation
    file_size = path.getsize(filename)
    max_num_lines = int(file_size/MIN_LINE_SIZE)

    #prepare arrays
    raw_data = np.empty([max_num_lines, NUM_CSV_COLS+2], float) #plus 2 cols for `weekday` and `hour`

    #read row by row
    nrows=0
    temp_values = np.zeros(NUM_CSV_COLS+2, float)
    with open(filename, 'rt') as f:
        headers = f.readline()[:-1].split(SEP_CHAR)
        if len(headers) < NUM_CSV_COLS:
            raise EngineError("Expect more columns (" + str(NUM_CSV_COLS) + ") in the CSV file")

        for line in f:
            values = line[:-1].split(SEP_CHAR)
            if len(values) >= NUM_CSV_COLS:
                try:
                    #get time, convert to timestamp (float)
                    t = datetime.strptime(values[0], "\"%Y-%m-%d %H:%M:%S\"")
                    temp_values[0] = time.mktime(t.timetuple())
                    temp_values[NUM_CSV_COLS] = t.weekday()
                    temp_values[NUM_CSV_COLS+1] = t.hour + t.minute/60.0 + t.second/3600.0
                    #get reamaining columns (`power`, `temp`, `humidity`, `CO2` and `dust`, `Temp.ext`, `Humidity.ext`)
                    for i in range(1, NUM_CSV_COLS):
                        temp_values[i] = float(values[i])

                    #now it is safe to save sensor values into data array
                    raw_data[nrows, :] = temp_values
                    nrows += 1

                except Exception as e: #pass the ill-format line
                    #logging.warning(e)
                    pass


    if nrows==0:
        raise EngineError("File has no data")
    #resize the arrays to their correct sizes
    return np.resize(raw_data,[nrows, NUM_CSV_COLS+2])

def get_good_power_chunks(power):
    """This functions removes all bad chunks of consecutive NAs power values (-1 values)
    @:param power the power column of raw data array
    @:return a list of good chunks, each consist of a starting index and its length
    """
    i=0
    power_na_chunks = []
    nrows = power.shape[0]

    #get all chunks of NAs (-1) from `power`
    while i<nrows:
        if power[i] == -1:
            j = i+1
            while  (j < nrows) and (power[j] == -1):
                j += 1
            power_na_chunks.append((i, j-i))
            i = j
        else:
            i += 1

    #retain only good chunks
    power_good_chunks = []
    start_id = 0
    for id, len in power_na_chunks:
        if (len >= POWER_MIN_NA_CHUNK_LEN):
            if (id - start_id >= POWER_MIN_GOOD_CHUNK_LEN):
                power_good_chunks.append((start_id, id))
            start_id = id + len
    if nrows-start_id >=POWER_MIN_GOOD_CHUNK_LEN:
        power_good_chunks.append((start_id, nrows))
    return power_good_chunks


def process_chunk(start_id, end_id, raw_data):
    """This function extracts features and assigns labels from a good chunk of raw data
    Following step is taken:
        + determine aircon status from `power`
        + identify user actions (classification labels) from the aircon status
        + remove spurious actions (TURN ON, TURN OFF that are too close to each other)
        + fix aircon status to match with the user actions (after removing spurious ones)
        + sub-sample and extract data features (outlier samples will be removed)

    @:param raw_data the whole raw data array
    @:param start_id where the chunk starts
    @:param end_id where the chunk ends
    @:return a list of feature array, label vector and aircon status vector
    """

    nrows = end_id - start_id
    saving_time = raw_data[start_id:end_id, 0]
    power = raw_data[start_id:end_id, 1]

    #prepare vectors to hold `user action` and `aircon status`
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

    num_actions = len(action_id)            #total number of actions

    action_id = np.array(action_id)         #row index of actions

    #remove any (TURN_OFF, TURN_ON) tuple whose time interval is too short (less than 5, for example)
    min_diff = OFF_MIN_DURATION*60 # define the minimum time interval, for example 5 minutes time 60 secs
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
    min_diff = ON_MIN_DURATION*60 # define the minimum time interval, for example 5 minutes time 60 secs
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

    #once actions are correctly recognized, now it's time to fix the aircon status

    prev_status = status[0]
    for i in range(1, nrows):
        if action[i] == ACTION_NOTHING:
            status[i] = prev_status
        elif action[i] == ACTION_TURN_ON:
            status[i] = STATUS_ON
        else:
            status[i] = STATUS_OFF
        prev_status = status[i]



    # sub-sampling (reduce the data size), but retain all data points having an action
    n_obs = SAMPLING_SIZE       #sampling every 10 timepoints (i.e. 5 minutes)
    nfrows = int(nrows/n_obs)   #number of data rows after sampling

    if nfrows <=0:
        return None

    #allocate data arrays
    x = np.empty([nfrows, NUM_FEATURES], float)
    y = np.empty(nfrows, int)
    y.fill(ACTION_NOTHING)
    sampled_status = np.empty(nfrows, int)

    sampled_id = n_obs
    i = 0

    feat_period = PAST_DURATION*60 #extract feature from previous 5 minutes

    #sub-sampling is happening in this loop
    while sampled_id <nrows:

        #find any near by actions. actions is precious, we must retain all actions
        for j in range(n_obs):
            new_id = sampled_id+j
            if new_id < nrows and action[new_id] != ACTION_NOTHING :
                sampled_id = new_id
                break

        #collect samples which are obs_period seconds from sampled_id sample
        feature_ids=[]
        cur_id = start_id + sampled_id-1
        while (cur_id>=0) and ((raw_data[start_id+sampled_id, 0] - raw_data[cur_id, 0]) < feat_period):
            #check if the sample at cur_id is an outlier
            bOutlier = False
            for j in range(2, NUM_CSV_COLS):
                if (raw_data[cur_id, j]<SENSOR_RANGES[j-2][0]) or (raw_data[cur_id, j]>SENSOR_RANGES[j-2][1]):
                    bOutlier = True
                    break
            if not bOutlier:
                feature_ids.append(cur_id)

            cur_id -= 1

        if len(feature_ids)>0:
            #extract feature from these samples
            for j in range(2, NUM_CSV_COLS):
                x[i, j-2] = np.mean(raw_data[feature_ids, j])
                #x[i, 2*j] = np.mean(raw_data[feature_ids, j])
                #x[i, 2*j+1] = np.std(raw_data[feature_ids, j])
            x[i, NUM_FEATURES-2] = raw_data[sampled_id, NUM_CSV_COLS] #week day
            x[i, NUM_FEATURES-1] = raw_data[sampled_id, NUM_CSV_COLS+1] #hour
            sampled_status[i] = status[sampled_id-1]
            y[i] = action[sampled_id]
            i += 1

        #move on to the next id for sampling
        sampled_id += n_obs


    nrows = i #correct number of data rows
    if nrows <=0:
        return None

    #resize arrays to their correct sizes
    x = np.resize(x, [nrows, NUM_FEATURES])
    y = np.resize(y, nrows)
    sampled_status = np.resize(sampled_status, nrows)
    return [x, y, sampled_status]

def process_data(filename):
    """this function is to read raw sensor data from csv file then do preprocessing to prepare data for model learning.
    It takes these following processing steps:
        - read raw data
        - convert POSIX time column to `hour` (hour of day) and `weekday` (day of week) then add to the data
        - remove any bad data chunk (consecutive observations of power=-1)
        - identify data chunks of good quality
        - assign labels and extract features from these good chunks
        - merge the output chunks (of previous step)
        - return the processed data
    @:param the raw data file
    @:return a list of 4 arrays including: 1 feature matrix and 1 label vector for the aircon in ON state,
             1 feature matrix and 1 label vector for the aircon in OFF state
    """

    raw_data = read_raw_data(filename)
    #remove observations with continous power value of -1
    power_good_chunks = get_good_power_chunks(raw_data[:,1])

    #extract features for each chunk
    data_chunks = []
    for start_id, end_id in power_good_chunks:
        data = process_chunk(start_id, end_id, raw_data)
        if data:
            data_chunks.append(data)



    #now merge all the chunks
    nrows = 0
    for i in range(len(data_chunks)):
        nrows += data_chunks[i][0].shape[0]

    x = np.empty([nrows, NUM_FEATURES], float)
    y = np.empty(nrows, int)
    status = np.empty(nrows, int)

    i = 0
    for chunk_x, chunk_y, chunk_status in data_chunks:
        chunk_nrows = chunk_x.shape[0]
        x[i:i+chunk_nrows,:] = chunk_x
        y[i:i+chunk_nrows] = chunk_y
        status[i:i+chunk_nrows] = chunk_status
        i += chunk_nrows

    #separate data into two, regarding to aircon status (ON or OFF)
    on_ids = []         #list to hold row index for ON status
    off_ids = []        #list to hold row index for OFF status
    for i in range(nrows):
        if status[i] == STATUS_ON:
            on_ids.append(i)
        else:
            off_ids.append(i)

    #"""get the number of actions
    cnt1 = 0
    cnt2 = 0

    for i in range(nrows):
        if y[i] == ACTION_TURN_ON:
            cnt1 += 1
        elif y[i] == ACTION_TURN_OFF:
            cnt2 += 1
    print "There are " + str(cnt1) + " TURN ON in " + str(len(off_ids)) + " observations of aircon OFF status"
    print "There are " + str(cnt2) + " TURN OFF in " + str(len(on_ids)) + " observations of aircon ON status"

    #"""

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
    """this function evaluate prediction performance using cross validation
    @:param: x is a numpy array that holds predictor values
    @:param y is a numpy vector that holds output labels
    @:param method is `tree` (for decision tree) or `forest` (for random forest)

    @:return a list of confusion matrices (length equals the number of cross-validation folds)
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.cross_validation import StratifiedKFold

    logging.info("evaluate model")

    try:
        #first partition data into train and test data
        skf = StratifiedKFold(y, NUM_FOLDS)
        precisions = np.zeros([NUM_FOLDS, NUM_CLASS], float)
        recalls = np.zeros([NUM_FOLDS, NUM_CLASS], float)
        con_mats = []
        i=0
        for train, test in skf:     #for each fold
            #make train/test datasets
            x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]

            #learn model
            if method=="tree":
                model = train_forest(x_train, y_train)
            elif method=="forest":
                model = train_forest(x_train, y_train)
            else:
                raise EngineError("unexpected error")

            #make predictions on train data
            pred_y1 = model.predict(x_train)

            #make prediction on test data
            pred_y2 = model.predict(x_test)

            #collect the confusion matrice
            con_mats.append([confusion_matrix(y_train, pred_y1), confusion_matrix(y_test, pred_y2)])
            i += 1
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to evaluate model")
    return con_mats
