import serial
import datetime

class TriggerArduino(object):
    def __init__(self):
        self.connected = False

    def StartArduino(self):
        ser = serial.Serial("COM5", 9600, writeTimeout=0)
        while not self.connected:
            print 'Initialising Arduino...'
            ser_in = ser.read()
            self.connected = True
            return ser

    @staticmethod
    def TurnLEDOFF(ser):
        print "LED OFF : Valve closed"
        ser.write('L')

    @staticmethod
    def TurnLEDON(ser):
        print "LED ON : Valve open"
        ser.write('H')

    @staticmethod
    def close(ser):
        print "Ending Arduino Session"
        ser.close()


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    serial = TriggerArduino().StartArduino()
    TriggerArduino.TurnLEDON(serial)
    while (datetime.datetime.now() - starttime).total_seconds() < 2:
        A = 1
    TriggerArduino.TurnLEDOFF(serial)