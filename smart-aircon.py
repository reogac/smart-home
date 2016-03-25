#!/usr/bin/python


import argparse
import logging
import pickle as pk
import sys


MIN_LINE_SIZE = 30
X_NUM_COLS = 8 #'temp', 'humidity', 'light', 'CO2', 'dust', 'status' and time ('hour' and 'day')
NUM_CSV_COLS = 7
SEP_CHAR = ","
POWER_CUT = 0.01 #cut value for positive power consumption

STATUS_ON = 1
STATUS_OFF = 0

ACTION_NOTHING = 0
ACTION_TURN_ON = 1
ACTION_TURN_OFF = -1


SENSOR_CSV = "sensor.csv"
SENSOR_PICKLE = "sensor.pk"
MODEL_PICKLE = "model.pk"
log_to_file = False

predictors = ('ac_status', 'temp', 'humidity', 'light', 'CO2', 'dust', 'day', 'hour')
label = 'action'

power_cut = 0.01 #cut value for positive power consumption

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

def load_data(filename):
    import pandas as pd
    logging.info("load processed data from \'"+ filename + "\'")
    try:
        df = pd.read_pickle(filename)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to load data")
    return df

def save_data(data, filename):
    """
    TODO: convert data frame to numpy matrix then save to file
    """
    logging.info("saving the data into \'"+ filename + "\'")
    try:
        data.to_pickle(filename)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to save data")

def process_data(filename):
    from datetime import datetime
    import numpy as np
    from os import path
    import time

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



    min_diff = 5*60 # 5 minutes time 60 secs
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

"""def process_data(filename):
    from datetime import datetime
    import numpy as np
    import pandas as pd

    logging.info("load sensor data from \'" + filename + "\'")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to read csv file")
    df = df.loc[0:5000,]

    logging.info("pre-processing the data")
    try:
        nrows = len(df)
        # convert string to time object
        df['time'] = df.time.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        # add two columns to the data frame to represent hour of the the day, and day of the week
        df['hour'] = df.time.apply(lambda x: x.hour + x.minute/60.0)
        df['day'] = df.time.apply(lambda x: x.day)

        df['ac_status'] = np.zeros(nrows, int)
        df.loc[df.power > power_cut, 'ac_status'] = 1
        df['action'] = np.zeros(nrows, int)
        for i in range(1, nrows):
            if (df.ac_status[i] == 1) and (df.ac_status[i-1]) == 0:
                df.loc[i, 'action'] = 1 #TURN ON
            elif (df.ac_status[i] == 0) and (df.ac_status[i-1]) == 1:
                df.loc[i, 'action'] = -1 #TURN OFF
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to process data")
    return df
"""

def save_model(model, filename):
    logging.info("save model to \'" + filename + "\'")
    try:
        pk.dump(model, open(filename, 'wb'), 2)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to save model")

def load_model(filename):

    logging.info("load model from \'" + filename + "\'")
    try:
        model = pk.load(open(filename, 'rb'))
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to load model")
    return model

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
        x = data.as_matrix(predictors)
        y = data.loc[:,label]
        model = model.fit(x, y)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to train model")
    return model

def evaluate_model(data, method):
    from sklearn.metrics import confusion_matrix
    logging.info("evaluate model")
    #first partition data into train and test data

    try:
        x = data.as_matrix(predictors)
        y = data.loc[:, label]
        pred_y = model.predict(x)
        con_mat = confusion_matrix(y, pred_y)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to evaluate model")
    return con_mat

def predict(model, inputs):
    import numpy as np
    logging.info("make a prediction")
    try:
        input_vect = np.array(inputs)
        pred = model.predict(input_vect)
    except Exception as e:
        logging.error(str(e))
        raise EngineError("Failed to make a prediction")
    return pred


def reinforce():
    logging.info("reinformance learning has not been implemented yet!")
    raise EngineError("reinformance learning has not been implemented yet!", False)

def parse_sensors(sensors):
    import numpy as np
    if not sensors:
        raise EngineError("Sensor data is not provided", False)
    sensors = sensors.replace(' ', '') #remove all whitespaces
    assignments = sensors.split(",") #retrieve sensor value assignments
    pairs = {}
    for a in assignments:
        pair = a.split("=")
        if len(pair) != 2:
            raise EngineError("Failed to parse the sensor data: " + a, False)
        p = pair[0]
        v = float(pair[1])
        if p not in predictors:
            raise EngineError(p + " is not a predictor", False)
        pairs[p] = v

    sensor_values = np.zeros(len(predictors), float)


    for i in range(len(predictors)):
        p = predictors[i]

        if not pairs.has_key(p):
            raise EngineError("the \'" + p + "\' variable is missing", False)
        sensor_values[i] = pairs[p]

    return sensor_values.reshape(1,-1)

def process(args):
    global log_to_file
    global SENSOR_CSV

    if args.log:
        log_to_file = True
        logging.basicConfig(filename=args.log, level=logging.INFO)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    if (args.command == 'process'):

        if not args.csv_file:
            args.csv_file = SENSOR_CSV

        df = process_data(args.csv_file)
        save_data(df, args.data_file)

    elif (args.command == 'train'):
        if (args.csv_file):
            df = process_data(args.csv_file)
            save_data(df, args.data_file)
        else:
            df = load_data(args.data_file)
        model = train_model(df, args.classifier)
        save_model(model, args.model_file)
    elif (args.command == 'predict'):

        from sklearn import tree
        from sklearn.ensemble import RandomForestClassifier

        inputs = parse_sensors(args.sensors)
        model = load_model(args.model_file)
        action = predict(model, inputs)
        if action==1:
            print "TURN_ON"
        elif action==-1:
            print "TURN_ON"
        else:
            print "DO_NOTHING"

    elif (args.command == 'evaluate'):
        if (args.csv_file):
            df = process_data(args.csv_file)
        else:
            df = load_data(args.data_file)
        print evaluate_model(df, args.classifier)
    elif (args.command == 'reinforce'):
        reinforce()
    else:
        raise EngineError("unknown command")

def main():
    """"
      this program can do following tasks:
    - read raw data in csv file, process then save to file for training model
    - read raw data in csv file or processed data then train model; save the trained model to file
    - read trained model from file then make a prediction of user's action given input sensor data
    - reinforce model with feedbacks from user (not implement yet)
    - prediction evaluation with cross validation (not implement yet)
    """

    global SENSOR_PICKLE
    global MODEL_PICKLE
    try:
        parser = argparse.ArgumentParser(description="Home air conditioner controller smart engine",
                                         usage='%(prog)s command [options]')
        parser.add_argument("command", choices=['process', 'train', 'predict', 'reinforce','evaluate'],
                            help="tell the program what to do")
        parser.add_argument("-c", "--classifier", choices=['tree', 'forest'], dest="classifier", default="tree",
                            help="Select classification model (Decision tree or Random forest)")
        parser.add_argument("-t", "--csv_file", dest="csv_file",
                            help="file containing the original sensor data, default name = \'" + SENSOR_CSV + "\'")
        parser.add_argument("-d", "--data_file", dest="data_file", default=SENSOR_PICKLE,
                            help="file to save/load the processed sensor data, default name = \'" + SENSOR_PICKLE + "\'")
        parser.add_argument("-m", "--model", dest="model_file", default=MODEL_PICKLE,
                            help="file to save/load the prediction model, default name = \'" + MODEL_PICKLE + "\'")
        parser.add_argument("-s", "--sensors", dest="sensors",
                            help="sensor data for which prediction should be made, here is a sample format: "
                                 "\"ac_status=1, temp=37, humidity=50, dust=100, CO2=1000, light=30, day=2, hour=19.5\"")
        parser.add_argument("-l", "--log", dest="log", help="where to save log messages")
        args = parser.parse_args()

        process(args)
    except EngineError as e:
        sys.stderr.write(str(e) + "\n")
    except Exception as e:
        sys.stderr.write(str(e) + "\n")

if __name__ == "__main__" : main()
