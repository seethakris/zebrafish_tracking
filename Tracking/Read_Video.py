import cv2

print cv2.__version__
import os
import numpy as np
from time import time
import sys
import matplotlib.pyplot as plt
from Trackfish import TrackFish, DrawContour, PlotIfError, FirstFewFrames, PlotifDarkorStimulus, FindStimulus
from DrawROI import DrawROI
import csv
import imutils


class ProcessVideo(object):
    def __init__(self, DirectoryName, FishLength, framesforbg, resize, minimumarea, updateROIoffirstframe):
        self.DirectoryName = DirectoryName
        self.FishLength = FishLength
        self.framesforbg = framesforbg
        self.resize = resize
        self.minimumarea = minimumarea
        self.updateROIoffirstframe = updateROIoffirstframe
        self.avifiles = [f for f in os.listdir(DirectoryName) if f.endswith('.mov')]

        self.ResultFolder = os.path.join(self.DirectoryName, 'TrackingResults')
        self.ResultFolderImages = os.path.join(self.DirectoryName, 'TrackedAvi')
        if not os.path.exists(self.ResultFolder):
            os.mkdir(self.ResultFolder)
        if not os.path.exists(self.ResultFolderImages):
            os.mkdir(self.ResultFolderImages)

        self.cap = None
        self.firstframe = None
        self.outputavi = None
        self.outputavi_bgimage = None
        self.imgsize = None
        self.selectroiwindowname = 'Press c to crop r to reset q to quit'

        # For saving
        self.contourA = []
        self.contourB = []
        self.centroidA = []
        self.centroidB = []
        self.errormessages = []

    def read_video(self):
        for ii in self.avifiles[0:1]:
            Experiments_with_Stimulus = ['1800', '2400', '2700', '2800', '3000']
            stimuluspresentfiles = [ff for ff in Experiments_with_Stimulus if ii.find(ff) > 0]
            if stimuluspresentfiles:
                print 'This File contains a odor stimulus!!!!'

            print 'Analysing...', ii
            framenumber = 1  # Counter for frame number
            stimulus_flag = False

            # Open capture and background subtraction parameter
            self.cap = cv2.VideoCapture(os.path.join(self.DirectoryName, ii))
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 57500)
            fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            start_time = time()

            while self.cap.isOpened():
                print 'Frame %s sizeof append %s' % (framenumber, np.shape(self.errormessages))

                # Open video
                ret, frame = self.cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.resize == True:
                        gray = imutils.resize(gray, width=800)
                        frame = imutils.resize(frame, width=800)

                    self.imgsize = np.shape(gray)  # Get image size

                    # Preprocess the frame
                    data = self.preprocess_frame(grayscaleimage=gray)
                    if framenumber == 1:
                        print 'Calculating Frame 1'
                        # Initialise avi for writing results, Define ROI or load existing ROI
                        self.firstframe = data  # Frame operations
                        self.initialise_avi_stream(savefilename=ii[:-4] + ' Tracking')
                        FirstFrameROIs = self.firstframeoperations()
                    else:
                        # if the whole frame is dark, it was during light off period, so skip it.
                        if self.find_dark_frames(data=data):
                            self.appendmessagetocontours(message='Dark')
                            PlotifDarkorStimulus(rgbimage=frame, textonplot='Darkness ' + str(framenumber),
                                                 avifile=self.outputavi)
                            framenumber += 1
                            continue
                        if stimuluspresentfiles:
                            if self.find_stimulus_frames(data=data, ledROI=FirstFrameROIs[-1],
                                                         currentframenumber=framenumber):
                                stimulus_flag = True
                                num_frame_afterstimulus = 0
                                self.appendmessagetocontours(message='StimulusON')
                                PlotifDarkorStimulus(rgbimage=frame, textonplot='StimulusON ' + str(framenumber),
                                                     avifile=self.outputavi)
                                framenumber += 1
                                continue

                        if stimulus_flag:
                            num_frame_afterstimulus += 1
                            print 'Waiting After stimulus frames to end...'
                            if num_frame_afterstimulus == 100:
                                stimulus_flag = False
                                meniscusA -= 15
                                meniscusB -= 15
                            else:
                                self.appendmessagetocontours(message='AfterStimuluspause')
                                framenumber += 1
                                continue

                        # Use selected fish roi for the second frame as reference
                        if framenumber > self.framesforbg:
                            print 'Calculating Frames for tracking Tank A'
                            fgmaskA = self.trackframes(data=data, tankROIs=FirstFrameROIs, framenumber=framenumber,
                                                       fgbgarray=fgbg, tanktoanalyse='TankA', meniscus=meniscusA)

                            print 'Calculating Frames for tracking Tank B'
                            fgmaskB = self.trackframes(data=data, tankROIs=FirstFrameROIs, framenumber=framenumber,
                                                       fgbgarray=fgbg, tanktoanalyse='TankB', meniscus=meniscusB)
                            self.errormessages.append('OK')
                        else:
                            print 'Calculating Frames for background'
                            fgmaskA, meniscusA = self.trackframesforbgcalculation(data=data, tankROIs=FirstFrameROIs,
                                                                                  fgbgarray=fgbg,
                                                                                  tanktoanalyse='TankA')
                            fgmaskB, meniscusB = self.trackframesforbgcalculation(data=data, tankROIs=FirstFrameROIs,
                                                                                  fgbgarray=fgbg,
                                                                                  tanktoanalyse='TankB')
                            self.errormessages.append('Background')

                        # Plot as avi bgimage and rgbimage

                        PlotIfError(fullimagesize=self.imgsize, bgimageA=fgmaskA, bgimageB=fgmaskB,
                                    tankROIA=FirstFrameROIs[1],
                                    tankROIB=FirstFrameROIs[0], ResultFolder=self.ResultFolderImages,
                                    framenumber=framenumber, avifile=self.outputavi_bgimage,
                                    textonplot='FrameNumber ' + str(framenumber))

                        self.createavioftrackeddata(rgbimage=frame, tankROIA=FirstFrameROIs[1],
                                                    tankROIB=FirstFrameROIs[0], framenumber=framenumber)

                else:
                    break

                framenumber += 1
                if framenumber == 3000:
                    print 'Filename %s Elapsed Time, %s' % (ii, time() - start_time)
                    print 'Filename %s Total Time Per Frame %s' % (ii, (time() - start_time) / (framenumber - 1))
                    # self.saveascsv(centroid=self.centroidA, contour=self.contourA, savefilename=ii[:-4] + ' TankA')
                    # self.saveascsv(centroid=self.centroidB, contour=self.contourB, savefilename=ii[:-4] + ' TankB')
                    self.release_windows()
                    # break

    def appendmessagetocontours(self, message):
        self.contourA.append(self.contourA[-1])
        self.centroidA.append(self.centroidA[-1])
        self.contourB.append(self.contourB[-1])
        self.centroidB.append(self.centroidB[-1])
        self.errormessages.append(message)

    def find_dark_frames(self, data):
        # print 'dark frame..intensity, ', np.mean(data[:])
        if np.all(np.mean(data[:]) < 50):  # Check for dark
            print 'DARK'
            skip_flag = True
        else:
            skip_flag = False
        return skip_flag

    def find_stimulus_frames(self, data, ledROI, currentframenumber):
        # Check for intensity of led
        print 'LED intensity, ', FindStimulus(frame=data, ledROI=ledROI).get_average_ofroi()
        # if FindStimulus(frame=data, ledROI=ledROI).get_average_ofroi() > 10:
        #     print 'Stimulus ON!'
        #     skip_flag = True
        if currentframenumber in range(57694, 58077):
            print 'Stimulus ON!'
            skip_flag = True
        else:
            skip_flag = False
        return skip_flag

    @staticmethod
    def get_centroid(roi):
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
        x_centre = x + (w / 2)
        y_center = y + (h / 2)

        return x_centre, y_center

    def release_windows(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.outputavi.release()

    def createavioftrackeddata(self, rgbimage, tankROIA, tankROIB, framenumber):
        DrawContour(rgbimage=rgbimage, fishtankroiA=tankROIA[1], fishtankroiB=tankROIB[0],
                    contourA=[self.contourA[-2], self.contourA[-1]],
                    contourB=[self.contourB[-2], self.contourB[-1]], centroidA=self.centroidA[-1],
                    centroidB=self.centroidB[-1], avifile=self.outputavi, framenumber=framenumber)

    def initialise_avi_stream(self, savefilename):
        # Create avi file for saving
        csvfile = os.path.join(self.DirectoryName, 'EstimatedFrameRate.csv')
        framerate = np.loadtxt(csvfile)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if self.outputavi is None:
            print 'Initialising .avi file', self.imgsize
            self.outputavi = cv2.VideoWriter(os.path.join(self.ResultFolderImages, savefilename + '.avi'), fourcc,
                                             framerate, (self.imgsize[1], self.imgsize[0]), True)
            self.outputavi_bgimage = cv2.VideoWriter(
                os.path.join(self.ResultFolderImages, savefilename + '_bgimage.avi'),
                fourcc, framerate, (self.imgsize[1], self.imgsize[0]), True)

    def preprocess_frame(self, grayscaleimage):
        # Preprocess data
        denoised = cv2.fastNlMeansDenoising(grayscaleimage, None, 10, 10, 5)
        blur = cv2.GaussianBlur(denoised, (5, 5), 0)
        # blur = cv2.medianBlur(grayscaleimage, 11)
        return blur

    def firstframeoperations(self):
        ## Clear everythin first
        # For saving
        self.contourA = []
        self.contourB = []
        self.centroidA = []
        self.centroidB = []
        self.errormessages = []

        if self.updateROIoffirstframe:
            filepath = os.path.join(self.ResultFolder, 'ROIs2.npy')
        else:
            filepath = os.path.join(self.ResultFolder, 'ROIs.npy')

        if os.path.exists(filepath):  # If ROI has already been made
            FirstFrameROIs = np.load(filepath)
        else:
            # Get ROI of tank and initial poition of fish from the first frame
            FirstFrameROIs = DrawROI(self.ResultFolder, self.selectroiwindowname, self.firstframe,
                                     self.updateROIoffirstframe).drawROIfortank()

        self.centroidA.append(self.get_centroid(FirstFrameROIs[3]))  # Update centroids
        self.centroidB.append(self.get_centroid(FirstFrameROIs[2]))
        self.contourA.append(list(FirstFrameROIs[3]))  # Update rectangle surrounding fish's contours
        self.contourB.append(list(FirstFrameROIs[2]))
        self.errormessages.append('FirstFrame')

        return FirstFrameROIs

    def trackframesforbgcalculation(self, data, tankROIs, fgbgarray, tanktoanalyse='TankA'):
        if tanktoanalyse == 'TankA':
            fgmask, meniscus = FirstFewFrames(grayimage=data, fishtank_roi=tankROIs[1],
                                              fgbgarray=fgbgarray).update_bg_alone_getminiscus()
            meniscus = 20
            self.contourA.append(list(tankROIs[3]))
            self.centroidA.append(self.get_centroid(tankROIs[3]))
        else:
            fgmask, meniscus = FirstFewFrames(grayimage=data, fishtank_roi=tankROIs[0],
                                              fgbgarray=fgbgarray).update_bg_alone_getminiscus()
            self.contourB.append(list(tankROIs[2]))
            self.centroidB.append(self.get_centroid(tankROIs[2]))

        return fgmask, meniscus

    def trackframes(self, data, tankROIs, fgbgarray, framenumber, meniscus, tanktoanalyse='TankA'):
        # If there is mistake in tracking just plot current fgbg and save the previous centroids.

        if tanktoanalyse == 'TankA':
            fgmask, tempcontour, tempcentroid = TrackFish(grayimage=data, fishtank_roi=tankROIs[1],
                                                          previouscontour=self.contourA[-1],
                                                          fgbgarray=fgbgarray, minarea=self.minimumarea,
                                                          fishlength=self.FishLength, meniscus=meniscus).get_contours()
            self.contourA.append(tempcontour)
            self.centroidA.append(tempcentroid)

        else:
            fgmask, tempcontour, tempcentroid = TrackFish(grayimage=data, fishtank_roi=tankROIs[0],
                                                          previouscontour=self.contourB[-1],
                                                          fgbgarray=fgbgarray, minarea=self.minimumarea,
                                                          fishlength=self.FishLength, meniscus=meniscus).get_contours()
            self.contourB.append(tempcontour)
            self.centroidB.append(tempcentroid)
        return fgmask

    def saveascsv(self, centroid, contour, savefilename):
        # Save as CSV, A and B seperately
        cx = []
        cy = []
        rx = []
        ry = []
        rw = []
        rh = []
        framenumber = []
        errormessage = []

        for ii in xrange(0, len(centroid)):
            framenumber.append(ii)
            errormessage.append(self.errormessages[ii])
            cx.append(centroid[ii][0])
            cy.append(centroid[ii][1])
            rx.append(contour[ii][0])
            ry.append(contour[ii][1])
            rw.append(contour[ii][2])
            rh.append(contour[ii][3])
        with open(os.path.join(self.ResultFolder, savefilename + '.csv'), 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['FrameNumber', 'Status', 'CentroidX', 'CentroidY', 'RectX', 'RectY', 'RectW', 'RectH'])
            rows = zip(framenumber, errormessage, cx, cy, rx, ry, rw, rh)
            for row in rows:
                writer.writerow(row)


if __name__ == '__main__':
    DirectoryName = '/Users/seetha/Desktop/Alarm_Response/TrackingFishExample/'
    lengthoffish = 10
    framesforbgcalculation = 10
    minimumarea = 50
    resize = True
    updateROI = False
    ProcessVideo(DirectoryName=DirectoryName, FishLength=lengthoffish, framesforbg=framesforbgcalculation,
                 resize=resize, minimumarea=minimumarea, updateROIoffirstframe=updateROI).read_video()
