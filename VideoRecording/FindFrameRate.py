"""
Steps for capturing image and tracking fish
1. Find Camera FPS.
"""

import cv2
import sys
import os
import datetime
import platform

filesep = os.path.sep

if platform.system().find('Windows') == 0:
    sys.path.append("C:\\opencv\\build\\python\\2.7")


class FPS(object):
    def __init__(self, imgwidth, imgheight, numtestframes, display=True):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
        self.imgwidth = imgwidth
        self.imgheight = imgheight
        self.numtestframes = numtestframes
        self.display = display

        print ('Starting Camera...')
        self.cap = cv2.VideoCapture(0)  # Initialise camera
        print ('Done Starting Camera...')

        self.cap.set(3, self.imgwidth)  # change Width
        self.cap.set(4, self.imgheight)  # change Height

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

    def captureimages(self):
        # Capture some images to calculate framerate
        print("[INFO] sampling frames from webcam...")
        start = self.start()

        while self._numFrames < self.numtestframes:  # Read until frames specified by user
            elapsed_time = (datetime.datetime.now() - start._start).total_seconds()
            print "Capturing Frame %d Elapsed Time %1.3f " % (self._numFrames, elapsed_time)

            # Read frame
            (grabbed, frame) = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.display:
                cv2.imshow('frame', gray)
                key = cv2.waitKey(1) & 0xFF

            self.update()

        self.stop()
        self.print_framerate()

        self.cap.release()
        cv2.destroyAllWindows()

        return self.fps()

    def print_framerate(self):
        print("[INFO] elasped time: {:.2f}".format(self.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps()))
