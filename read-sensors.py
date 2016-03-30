#!/usr/bin/python

import threading
import time
import serial
import Queue
from datetime import datetime
PROBING_INTERVAL = 1 #reading at 1sec interval. It should be not to small
SAVING_INTERVAL_NUM = 5
FW_KEY = "(02)"
class ReaderWriter(threading.Thread):
    def __init__(self, data, key):
        self.port = serial.Serial(
                    port='/dev/ttyO2',
                    baudrate=9600, timeout=0,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )
        print self.port.isOpen()
        self.alive = True
        self.queue = data
        self.key = key
        threading.Thread.__init__(self)

    def run(self):
        self.port.flush() #clearing input buffer
        bf = ""
        while (self.alive):
            #Tell the port that I want more data
            self.port.write(self.key)
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
            time.sleep(PROBING_INTERVAL)

        self.port.close()

    def set_stop(self):
        print "kill it"
        self.alive = False
 
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


data = Queue.Queue()
rw = ReaderWriter(data, FW_KEY)
rw.start()

while True:
    try:
        msg = data.get()
        sensors = parse_data(msg)
    except (KeyboardInterrupt, SystemExit):
        rw.set_stop()
        break

