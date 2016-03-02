import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
#from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
import pickle as pk
#from sklearn.externals import joblib

args = None

#model_file = "forest/forest.pkl"
model_file = "model.dat"

database = "sensor-data.csv"
power_cut = 0.01 #cut value for positive power consumption

def predict():
    pass

def process_raw_data():
    df = pd.read_csv(database)
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


    return df

def train(is_load = False):
    df = process_raw_data()
    if is_load:
        forest = pk.load(open(model_file, 'rb'))
    else:
        #forest = RandomForestClassifier(n_estimators=100)
        forest = tree.DecisionTreeClassifier()
        forest = forest.fit(df.as_matrix(('temp', 'humidity', 'light', 'CO2', 'dust', 'hour', 'day')),
                            df.action)
        pk.dump(forest, open(model_file, 'wb'))

    forest = pk.load(open(model_file, 'rb'))
    predicts = forest.predict(df.as_matrix(('temp', 'humidity', 'light', 'CO2', 'dust', 'hour', 'day')))
    con_matrix = confusion_matrix(df.action, predicts)

    print(con_matrix)

def collect_sensor():
    pass

def colect_action():
    pass

def process():
    #process_raw_data()
    train(True)

def main():
  """"
  this program is to do the following job:
    - train model from data
    - collect sensors data and store to database
    - predict user action taken on air conditioner
    - collect user feed back
  :return:


  global args
  parser = argparse.ArgumentParser(description="A simple photo organizer")
  parser.add_argument("src_dir", help="photo directory")
  parser.add_argument("dest_dir", help="target directory")
  parser.add_argument("-r", "--recursive", action="store_true",
                      dest="recursive", help="process sub-directories")
  parser.add_argument("-d", "--dry", dest="dry", action="store_true",
                      help="dry run")
  parser.add_argument("-c", "--copy", dest="copy",
                      help="copy photos instead of moving",
                      action="store_true")
  parser.add_argument("-v", "--video", dest="video",
                      help="handle videos", action="store_true")
  parser.add_argument("-s", "--separate", dest="separate",
                      help="handle videos and photos separately",
                      action="store_true")
  parser.add_argument("-t", "--use-ctime", dest="ctime",
                      help="use file created time if neither exif information"
                           " nor file name can reveal the photo taken time",
                      action="store_true")
  parser.add_argument("-o", "--overwrite", dest="overwrite",
                      help="overwriting existing files", action="store_true")
  parser.add_argument("-m", "--months", dest="months", help="a string of names"
                                                            "separated by commas that be used as month directories,"
                                                            "or \'system\' (default) for using system month names,"
                                                            "or \'number\' for using month number as names, ",
                      type=get_months, default="system")
  parser.add_argument("-y", "--year-prefix",
                      help="prefix for year directory", dest="year")

  args = parser.parse_args()

  try:
      process()
      except PhorgError as e:
      print e
  """
  df = process()

if __name__ == "__main__" : main()
