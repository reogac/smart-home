from threading import Thread
import time
import serial
import Queue
from datetime import datetime
import numpy as np

PROBING_INTERVAL = 1 #reading at 1sec interval. It should be not to small
SAVING_INTERVAL = 5
FW_KEY = "(02)"
NUM_SENSORS = 5
PREDICTION_INTERVAL = 6
SENSOR_DATA_BUFFER_SIZE = 10
SENSOR_DATA_FILE_NAME = "sensor-data.csv"

class MyThread(Thread):
    def __init__(self, my_name):
        self.alive = True
        self.my_name = my_name
        Thread.__init__(self)

    def set_kill(self):
        print "kill " + self.my_name + " thread!"
        self.alive = False

class ReaderWriter(MyThread):
    def __init__(self, data):
        self.port = serial.Serial(
                    port='/dev/ttyO2',
                    baudrate=9600, timeout=0,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
        self.queue = data
        MyThread.__init__(self, "Port reader/writer")

    def run(self):
        self.port.flush() #clearing input buffer
        bf = ""
        while self.alive:

            #Tell the port that I want more data
            self.port.write(FW_KEY)

            while self.port.inWaiting():
                bf += self.port.read()
            l = len(bf)
            if (l >= 32) :
                print bf
                self.queue.put(bf[0:32])
                bf = bf[32:]
            else:
                print "got this: " + bf
            time.sleep(PROBING_INTERVAL)
        self.port.close()
        print self.my_name + " says BYE"


class Predictor:
    def __init__(self):
        self.last_prediction_time = datetime.now()

    def handle_data(self, data):
        current_time = datetime.now()
        time_diff = current_time - self.last_prediction_time
        if time_diff.total_seconds() >= PREDICTION_INTERVAL:
            #make a prediciton here
            print "Make a prediction"
            self.last_prediction_time = current_time


class SensorDataManager:
    def __init__(self):
        self.last_saving_time = datetime.now()
        self.buffer=[]

    def handle_data(self, data):
        current_time = datetime.now()
        time_diff = current_time - self.last_saving_time
        if time_diff.total_seconds() >= SAVING_INTERVAL:
            print "collecting data"
            self.buffer.append(data)
            self.last_saving_time = current_time

    def save(self):
        if len(self.buffer) >= SENSOR_DATA_BUFFER_SIZE:
            self.flush()

    def flush(self): #save data to disk
        print "write data to disk"
        with open(SENSOR_DATA_FILE_NAME, "wt") as f:
            for data in self.buffer:
                f.write(str(data))
        self.buffer = []


class Framework:
    def __init__(self):
        self.sensor_data = Queue.Queue(100) #queue to get sensor data

        self.port_reader_writer = ReaderWriter(self.sensor_data) #port read/write thread
        self.predictor = Predictor() #predictor
        self.data_manager = SensorDataManager()

    def parse_sensor_data(self, data):
        sensors={}
        try:
            sensors["time"] = str(datetime.now())
            sensors["dust"] = 0.01*int(data[10:15])
            sensors["temp"] = 0.1*int(data[17:21])
            sensors["humidity"] = int(data[23:26])
            sensors["light"] = int(data[28:31])
            print sensors
        except ValueError as e:
            print e

        return sensors

    def run(self):
        self.port_reader_writer.start()
        while True:
            try:
                data = self.sensor_data.get(timeout=PROBING_INTERVAL)
                data = self.parse_sensor_data(data)
                self.data_manager.handle_data(data) #save data to disk
                self.predictor.handle_data(data) #make prediction

            except (KeyboardInterrupt, SystemExit):
                print "user interupt"
                self.data_manager.flush()
                self.port_reader_writer.set_kill() #must be killed before the command sender thread
                break
            except Queue.Empty as e:
                pass

        print "BYE"


framework = Framework()
framework.run()
