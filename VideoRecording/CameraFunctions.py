import cv2
import sys
import os
from ArduinoFunctions import TriggerArduino
import datetime
import csv

filesep = os.path.sep
sys.path.append("C:\\opencv\\build\\python\\2.7")


## Main file
class RunCamera(object):
    # Define camera parameters
    def __init__(self, fps, imgwidth, imgheight, stimulus_on_time, stimulus_off_time, total_experiment_time,
                 resultdirectory, savefilename, display):
        self.framerate = fps
        self.imgwidth = imgwidth
        self.imgheight = imgheight
        self.display = display

        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.currentstimulus = 0
        self.total_experiment_time = total_experiment_time
        self.totalframes = round(self.total_experiment_time * self.framerate)

        self.ResultDirectory = resultdirectory
        self.savefilename = savefilename

        # Start Camera
        print ('Starting Camera...')
        self.cap = cv2.VideoCapture(0)  # Initialise camera

        self.outputavi = None
        self.initialise_camera()

        # Save print statements
        self.string_list = []

        # Start Arduino
        self.serialconnection = TriggerArduino().StartArduino()

        # To verify FPS
        self._numFrames = 0

    def initialise_camera(self):
        print "Initialising Camera Parameters and avi file"

        self.cap.set(3, self.imgwidth)  # change Width  800x600 for small monitor;1280x1024 for HD monitor
        self.cap.set(4, self.imgheight)  # change Height
        self.cap.set(cv2.cv.CV_CAP_PROP_FPS, self.framerate)

        # For the fourcc to work, add C:\opencv\build\x86\vc10\bin to System PATH
        self.cap.set(6, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'))
        fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
        if self.outputavi is None:
            self.outputavi = cv2.VideoWriter(os.path.join(self.ResultDirectory, self.savefilename + '.avi'), fourcc,
                                             self.framerate, (self.imgwidth, self.imgheight), False)

        print self.outputavi

    def CaptureAndSaveFrames(self):
        print 'Capturing a total of %3.2f frames at %3.2f frames per sec' % (self.totalframes, self.framerate)
        print 'Begin experiment...'

        starttime = datetime.datetime.now()
        triggered_flag = False  # LED ON or not?

        while self._numFrames < self.totalframes:  # Read until frames specified by user

            # Read frame and save first
            (grabbed, frame) = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if grabbed == True:
                if self.display:
                    cv2.imshow('Frame', gray)
                    key = cv2.waitKey(1) & 0xFF
                self.outputavi.write(gray)  # Write frame

            elapsed_time = (datetime.datetime.now() - starttime).total_seconds()
            self.print_framenumber_and_time(self._numFrames, elapsed_time)

            # If time has elapsed switch on arduino to turn on valve
            if self.currentstimulus <= len(self.stimulus_on_time) - 1:
                if self.stimulus_on_time[self.currentstimulus] <= elapsed_time <= self.stimulus_off_time[
                    self.currentstimulus]:
                    if triggered_flag == False:
                        TriggerArduino.TurnLEDON(self.serialconnection)
                        self.string_list.append('Valve ON')
                        triggered_flag = True

                elif elapsed_time >= self.stimulus_off_time[self.currentstimulus]:
                    TriggerArduino.TurnLEDOFF(self.serialconnection)
                    self.string_list.append('Valve OFF')
                    self.currentstimulus += 1
                    triggered_flag = False

            self.update()  # Increase frame number

        self.print_fps((datetime.datetime.now() - starttime).total_seconds())
        self.end_camera_session()
        self.saveascsv()

        TriggerArduino.close(self.serialconnection)

    def update(self):
        self._numFrames += 1

    def print_fps(self, end_time):

        self.string_list.append("[INFO] elasped time: {:.2f}".format(end_time))
        self.string_list.append("[INFO] approx. FPS: {:.2f}".format(self._numFrames / end_time))

        print("[INFO] elasped time: {:.2f}".format(end_time))
        print("[INFO] approx. FPS: {:.2f}".format(self._numFrames / end_time))

    def print_framenumber_and_time(self, framenumber, time):
        self.string_list.append("Capturing Frame %d Elapsed Time %1.3f " % (framenumber, time))
        print "Capturing Frame %d Elapsed Time %1.3f " % (framenumber, time)

    def saveascsv(self):
        with open(os.path.join(self.ResultDirectory, 'FrameNumber_ElapsedTime.csv'),
                  'wb') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f)
            for ii in self.string_list:
                w.writerow([ii])


    def end_camera_session(self):
        # Release everything if job is finished
        print "Ending Camera Session"
        self.cap.release()
        self.outputavi.release()
        cv2.destroyAllWindows()
