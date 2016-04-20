from threading import Thread
import time
import serial
import Queue
from datetime import datetime
import numpy as np

PROBING_INTERVAL = 1 #reading at 1sec interval. It should be not to small
SAVING_INTERVAL_NUM = 5
FW_KEY = "(02)"
NUM_SENSORS = 5
AMBIENT_SENSORS = 1

def parse_data(data):
    sensors={}
    try:
        sensors["dust"] = 0.01*int(data[10:15])
        sensors["temp"] = 0.1*int(data[17:21])
        sensors["humidity"] = int(data[23:26])
        sensors["light"] = int(data[28:31])
        sensors["time"] = str(datetime.now())
        print sensors
    except ValueError as e:
        print e


class MyThread(Thread):
    def __init__(self, my_name):
        self.alive = True
        self.my_name = my_name
        Thread.__init__(self)

    def set_kill(self):
        print "kill " + self.my_name + " thread!"
        self.alive = False

class CommandSender(MyThread):
    def __init__(self, command):
        self.command = command
        MyThread.__init__(self, "Command sender thread")

    def run(self):
        while self.alive:
            self.command.put(AMBIENT_SENSORS)
            time.sleep(PROBING_INTERVAL)

class ReaderWriter(MyThread):
    def __init__(self, command, data):
        self.port = serial.Serial(
                    port='/dev/ttyO2',
                    baudrate=9600, timeout=0,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
        print self.port.isOpen()
        self.queue = data
        self.command = command
        MyThread.__init__(self, "Port reader writer thread")

    def run(self):
        self.port.flush() #clearing input buffer
        bf = ""
        while (self.alive):
            c = self.command.get()
            if c==AMBIENT_SENSORS:
                #Tell the port that I want more data
                self.port.write(FW_KEY)
                print "write"
            while self.port.inWaiting():
                bf += self.port.read()
            l = len(bf)
            if (l >= 32) :
                print bf
                self.queue.put(bf[0:32])
                bf = bf[32:]
            else:
                print "got this: " + bf

        self.port.close()


class Predictor(MyThread):
    def __init__(self, input_queue):
        self.input_queue = input_queue
        MyThread.__init__(self, "Predictor thread")

    def run(self):
        while self.alive:
            try:
                sensors = self.input_queue.get(timeout=1)
                #now make a prediction
                print sensors
            except Queue.Empty as e:
                print "pass the empty queue exception"
                pass



sensor_data = Queue.Queue(100) #queue to get sensor data
port_command = Queue.Queue(100) #queue of commands to be sent to port
command_sender = CommandSender(port_command) #port command sender thread
port_reader_writer = ReaderWriter(port_command, sensor_data) #port read/write thread
pred_input = Queue(100)
pred = Predictor(pred_input) #prediction thread


command_sender.start()
port_reader_writer.start()
pred.start()
while True:
    try:
        data = sensor_data.get(timeout=1)
        pred_input.put(parse_data(data))
    except (KeyboardInterrupt, SystemExit):
        pred.set_kill()
        port_reader_writer.set_kill()
        time.sleep(1)
        command_sender.set_kill()
        break
