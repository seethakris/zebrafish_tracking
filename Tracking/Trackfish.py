import cv2
import numpy as np
import os
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


class TrackFish(object):
    def __init__(self, grayimage, fishtank_roi, previouscontour, fgbgarray, minarea, fishlength, meniscus):
        self.grayimage = grayimage

        self.fishtank_roi = fishtank_roi
        self.previouscontour = previouscontour

        self.fgbgarray = fgbgarray
        self.minarea = minarea
        self.fishlength = fishlength
        self.meniscus = meniscus
        self.fishtank = []

    def get_contours(self):
        self.fishtank = self.getimage_fromroi(self.grayimage, self.fishtank_roi)
        fgmask = self.fgbgarray.apply(self.fishtank)

        # Dilate image for better result
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)

        # Find Contour
        (_, cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 0:
            finalcontour = list(self.previouscontour)
            finalcentroid = self.get_centroid_ofrect(finalcontour)
        else:
            larger_cnts = self.remove_small_contours(bgimage=fgmask, contours=cnts, min_area=self.minarea)
            if len(larger_cnts) == 0:
                finalcontour = list(self.previouscontour)
                finalcentroid = self.get_centroid_ofrect(finalcontour)
            else:
                finalcontour, finalcentroid = self.check_contours(larger_cnts, fgmask)

        return fgmask, finalcontour, finalcentroid

    def remove_small_contours(self, bgimage, contours, min_area):
        selected_contours = []
        print 'Number of contours in this frame %d' % len(contours)
        for c in contours:
            print 'Area of contour..', cv2.contourArea(c)
            if min_area > cv2.contourArea(c):
                continue

            if self.findintensity_of_contour(bgimage=bgimage, currentcontour=c) < 20:
                continue

            cx, cy = self.get_centroid_of_contour(c)
            # IF centroid is too close to the borders, then skip (prevent reflections from being found as fish
            if cy < self.meniscus:
                print 'Contour too close', cy, self.meniscus
                continue

            selected_contours.append(c)

        print 'Number of contours exceeding thresholds %d' % len(selected_contours)
        return selected_contours

    def check_contours(self, contours, fgmask):
        # What to do with the contours found
        if len(contours) == 1:  # If there is just one contour select it
            selected_contour = contours[0]
        elif len(contours) > 1:  # if there are more than one contours, get the most overlapping or the nearest contour
            length_of_intersect = np.zeros(len(contours))
            for jj, ii in enumerate(contours):
                print 'Calculating intersect'
                contour_rect = cv2.boundingRect(ii)
                length_of_intersect[jj] = self.calculate_overlap_withpreviouscontour(currentcontour=contour_rect)

            if np.all(length_of_intersect) == 0:
                distance = np.zeros(len(contours))
                for jj, ii in enumerate(contours):
                    print 'Calculating nearest distance'
                    contour_rect = cv2.boundingRect(ii)
                    distance[jj] = self.calculate_distance(currentcontour=contour_rect)

                selected_contour = contours[np.argmin(distance)]

            else:  # Find the maximum intersected ROI
                selected_contour = contours[np.argmax(length_of_intersect)]

        centroid_contour = self.get_centroid_of_contour(selected_contour)
        selected_contour = cv2.boundingRect(selected_contour)

        selected_contour = list(selected_contour)

        # Fix contour according to length of fish
        selected_contour[0] -= self.fishlength
        selected_contour[1] -= self.fishlength
        selected_contour[2] += self.fishlength
        selected_contour[3] += self.fishlength

        return selected_contour, centroid_contour

    def findintensity_of_contour(self, bgimage, currentcontour):
        mask = np.zeros(self.fishtank.shape, np.uint8)
        cv2.drawContours(mask, [currentcontour], 0, 255, -1)
        intensity = cv2.mean(self.fishtank, mask=mask)[0]
        print 'Intensity of contour', intensity
        # cv2.imshow('Bg image', mask)
        # cv2.waitKey()

        return intensity

    def calculate_overlap_withpreviouscontour(self, currentcontour):
        image1 = self.fishtank
        roi1 = self.previouscontour
        roi2 = currentcontour

        pixels_previousroi = self.getimage_fromroi(image1, roi1)
        pixels_currentroi = self.getimage_fromroi(image1, roi2)

        thresh_previousroi = np.where(cv2.threshold(pixels_previousroi, 1, 255, cv2.THRESH_BINARY)[1])[0]
        thresh_currentroi = np.where(cv2.threshold(pixels_currentroi, 1, 255, cv2.THRESH_BINARY)[1])[0]
        intersect = len(set(thresh_previousroi).intersection(thresh_currentroi))
        # print 'Intersections with previous contour', intersect
        return intersect

    def calculate_distance(self, currentcontour):
        image = self.fishtank
        roi1 = self.previouscontour
        roi2 = currentcontour

        # Get centroid and calculate euclidean distaance
        px, py = self.get_centroid_ofrect(roi1)
        cx, cy = self.get_centroid_ofrect(roi2)

        # distance
        distance = math.sqrt(((cx - px) ** 2) + ((cy - py) ** 2))

        return distance

    def getimage_fromroi(self, image, roi):
        # Get image within roi from bigger image
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]

        newimage = np.zeros(np.shape(image), dtype=np.uint8)
        newimage[y:y + h, x:x + w] = image[y:y + h, x:x + w]
        return newimage

    @staticmethod
    def get_centroid_of_contour(contours):
        moment = cv2.moments(contours)
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])

        return cx, cy

    @staticmethod
    def get_centroid_ofrect(roi):
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
        x_centre = x + (w / 2)
        y_center = y + (h / 2)

        return x_centre, y_center


class FirstFewFrames(object):
    def __init__(self, grayimage, fishtank_roi, fgbgarray):
        self.grayimage = grayimage
        self.fishtank_roi = fishtank_roi
        self.fgbgarray = fgbgarray

    def update_bg_alone_getminiscus(self):
        self.fishtank = self.getimage_fromroi(self.grayimage, self.fishtank_roi)
        fgmask = self.fgbgarray.apply(self.fishtank)

        meniscus = self.find_meniscus()

        return fgmask, meniscus

    def getimage_fromroi(self, image, roi):
        # Get image within roi from bigger image
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
        if image.ndim == 3:
            newimage = np.zeros(np.shape(image), dtype=np.uint8)
            newimage[y:y + h, x:x + w, :] = image[y:y + h, x:x + w, :]
        else:
            newimage = np.zeros(np.shape(image), dtype=np.uint8)
            newimage[y:y + h, x:x + w] = image[y:y + h, x:x + w]
        return newimage

    def find_meniscus(self):
        centroid_tank = self.get_centroid_ofrect(self.fishtank_roi)
        y_fishtank = list(self.fishtank[:, centroid_tank[0]])
        min1 = min(i for i in y_fishtank if i > 10)
        print 'Found Meniscus is at...', y_fishtank.index(min1)
        # Check if meniscus found is actually on top, else assign something
        if y_fishtank.index(min1) < 80:
            meniscus = y_fishtank.index(min1)
            meniscus -= 5
        else:
            print 'Cant find meniscus.. using default value'
            meniscus = 50  # User defined

        return meniscus

    @staticmethod
    def get_centroid_ofrect(roi):
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
        x_centre = x + (w / 2)
        y_center = y + (h / 2)

        return x_centre, y_center


class PlotIfError(object):
    def __init__(self, fullimagesize, bgimageA, bgimageB, tankROIA, tankROIB, ResultFolder, avifile, framenumber,
                 textonplot):
        self.bgimageA = bgimageA
        self.bgimageB = bgimageB
        self.bgimage = np.add(self.bgimageA, self.bgimageB)
        self.avifile = avifile
        self.ResultFolder = ResultFolder
        self.framenumber = framenumber
        self.textonplot = textonplot
        self.tankROIA = tankROIA
        self.tankROIB = tankROIB

        self.rgbimage = np.zeros((fullimagesize[0], fullimagesize[1], 3), dtype="uint8")
        self.create_rgtankimage(self.tankROIA)
        self.create_rgtankimage(self.tankROIB)
        # self.plot_and_saveimage()
        self.plottoavi()

    # def plot_and_saveimage(self):
    #     cv2.imshow('Error in Frame', self.bgimage)
    #     cv2.imwrite(os.path.join(self.ResultFolder, str(self.framenumber) + '.tif'), self.bgimage)

    def plottoavi(self):
        cv2.putText(self.rgbimage, self.textonplot, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2, cv2.LINE_AA)
        self.avifile.write(self.rgbimage)

    def create_rgtankimage(self, roi):
        # Make bg image rgb
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
        self.rgbimage[y:y + h, x:x + w, :] = np.repeat(self.bgimage[y:y + h, x:x + w, np.newaxis], 3, axis=2)


class PlotifDarkorStimulus(object):
    def __init__(self, rgbimage, textonplot, avifile):
        cv2.putText(rgbimage, textonplot, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2, cv2.LINE_AA)
        avifile.write(rgbimage)


class DrawContour(object):
    def __init__(self, rgbimage, fishtankroiA, fishtankroiB, contourA, contourB, centroidA, centroidB, avifile,
                 framenumber):
        self.rgbimage = rgbimage
        self.fishtankroiA = fishtankroiA
        self.fishtankroiB = fishtankroiB
        self.contourA = contourA  # Contour is made up of previous contour and next contour
        self.contourB = contourB
        self.centroidA = centroidA  # Contour is made up of previous contour and next contour
        self.centroidB = centroidB
        self.framenumber = framenumber
        self.avifile = avifile
        self.color = [(0, 255, 0), (0, 0, 255)]
        self.drawcontour()

    def drawcontour(self):
        # Draw previous and current contours
        for index, roi in enumerate(self.contourA):
            # print index, roi, self.color[index]
            cv2.rectangle(self.rgbimage, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), color=self.color[index],
                          thickness=1)  # Green is previous, red is current
            cv2.circle(self.rgbimage, (self.centroidA[0], self.centroidA[1]), radius=1, color=self.color[index],
                       thickness=1)

        for index, roi in enumerate(self.contourB):
            # print index, roi
            cv2.rectangle(self.rgbimage, (roi[0], roi[1]),
                          (roi[0] + roi[2], roi[1] + roi[3]), color=self.color[index],
                          thickness=1)  # Green is previous, red is current
            cv2.circle(self.rgbimage, (self.centroidB[0], self.centroidB[1]), radius=1, color=self.color[index],
                       thickness=1)
        cv2.putText(self.rgbimage, 'FrameNumber ' + str(self.framenumber), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # self.figure_frame(self.rgbimage)

        print 'Saving as avi'
        self.avifile.write(self.rgbimage)


class FindStimulus(object):
    def __init__(self, frame, ledROI):
        self.ledROI = ledROI
        self.frame = frame

        # cv2.rectangle(self.frame, (ledROI[0], ledROI[1]),
        #               (ledROI[0] + ledROI[2], ledROI[1] + ledROI[3]), color=0,
        #               thickness=1)
        # cv2.imshow('Frame', frame)
        # cv2.waitKey(1)

    def get_average_ofroi(self):
        x = self.ledROI[0]
        y = self.ledROI[1]
        w = self.ledROI[2]
        h = self.ledROI[3]

        meanled = np.median(self.frame[y:y + h, x:x + w])
        return meanled
