import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
import cv2
import imutils
import itertools
import imageio


class CollectDataandCorrectDistance(object):
    def __init__(self, DirectoryName, background_frames, distance_threshold):
        self.DirectoryName = DirectoryName
        self.background_frames = background_frames
        self.FiguresFolder = os.path.join(self.DirectoryName, 'Figures')
        self.CorrectedDistancesFolder = os.path.join(self.DirectoryName, 'CorrectedCentroid')
        self.distance_threshold = distance_threshold
        self.OriginalVideoFolder = os.path.join(os.path.normpath(self.DirectoryName + os.sep + os.pardir))

        if not os.path.exists(self.FiguresFolder):
            os.mkdir(self.FiguresFolder)
        if not os.path.exists(self.CorrectedDistancesFolder):
            os.mkdir(self.CorrectedDistancesFolder)

    def get_csvfiles(self):
        csvfilenames = [f for f in os.listdir(self.DirectoryName) if f.endswith('.csv')]
        # Collect TankA and TankB Filename
        TankAcsv = [ii for ii in csvfilenames if ii.find('TankA') > 0]
        TankBcsv = [ii for ii in csvfilenames if ii.find('TankB') > 0]

        return TankAcsv, TankBcsv

    def getdatfromfile(self, Tankname):
        # Get Data for plotting
        TankData = pd.read_csv(os.path.join(self.DirectoryName, Tankname))

        return TankData

    def find_darkframe_stimulusframe(self, Tank, filename, pad):
        Frames = Tank['Status']
        DarkFrames = Frames[Frames == 'Dark'].index
        StimulusOnFrames = Frames[Frames == 'StimulusON'].index
        StimulusOnFrames = np.asarray(range(57692, 58300))  # np.asarray(range(19300, 20000))
        if not StimulusOnFrames.size:
            StimulusOnFrames = []

        if DarkFrames.size:
            DarkFrames1 = DarkFrames[DarkFrames < 60000]  # First Dark cycle
            print DarkFrames1[-1]
            DarkFrames1 = list(itertools.chain(range(DarkFrames1[0] - pad, DarkFrames1[0]), DarkFrames1,
                                               range(DarkFrames1[-1], DarkFrames1[-1] + pad)))
            print DarkFrames1[-1]
            DarkFrames2 = DarkFrames[DarkFrames > 80000]  # Second Dark Cycle

            if not DarkFrames2.size:
                DarkFrames2 = []
            else:
                DarkFrames2 = list(itertools.chain(range(DarkFrames2[0] - pad, DarkFrames2[0]), DarkFrames2,
                                                   range(DarkFrames2[-1], DarkFrames2[-1] + pad)))
        else:
            DarkFrames1 = []
            DarkFrames2 = []

        BlackoutFrames = [i for i in itertools.chain(DarkFrames1, DarkFrames2, StimulusOnFrames)]

        return BlackoutFrames

    def correct_centroids(self, Tank, distance, filename):
        # Check for large jumps. Check if the meniscus was tracked instead of the fish.
        # Recalculate distance. Resave centroids
        ii = 0
        centroid = np.vstack((Tank['CentroidX'], Tank['CentroidY']))
        rectangle = np.vstack((Tank['RectX'], Tank['RectY'], Tank['RectW'], Tank['RectH']))
        BlackoutFrames = self.find_darkframe_stimulusframe(Tank, filename=filename, pad=10)

        # Remove artifcats due to background calculation
        centroid[:, :self.background_frames] = np.tile(centroid[:, self.background_frames + 1],
                                                       (self.background_frames, 1)).T
        rectangle[:, :self.background_frames] = np.tile(rectangle[:, self.background_frames + 1],
                                                        (self.background_frames, 1)).T

        read_video = self.read_videofile(filename)

        while ii < np.size(Tank['CentroidX']) - 1:
            if ii in BlackoutFrames:
                centroid[:, ii] = centroid[:, ii - 1]
                centroid[:, ii + 1] = centroid[:, ii - 1]
                ii += 1
                continue

            # if ii > 60000:
            #     self.distance_threshold = 30

            current_centroid = centroid[:, ii]
            current_rect = rectangle[:, ii]
            next_centroid = centroid[:, ii + 1]
            # If the jump in y is not large, dont consider for videos with no moving meniscus
            #  (Maybe here the fish wasn't properly tracked)

            if distance[ii] > self.distance_threshold: #(current_centroid[1] - next_centroid[1]) > 10:
                dist_current_next = self.find_distance_between_twocentroids(current_centroid, next_centroid)
                print ii, current_centroid, next_centroid, dist_current_next
                # Maybe meniscus was found
                correctflag = True  # Check subsequent frames to see if it needs correction
                self.read_and_display_particularframe(videofile=read_video, framenumber=ii + 1,
                                                      centroid1=current_centroid, centroid2=next_centroid)
                while correctflag:

                    # Find distance between current and next centroid and second next centroid
                    dist_current_next = self.find_distance_between_twocentroids(current_centroid, next_centroid)

                    # print ii, dist_current_next, current_centroid, next_centroid
                    # If next centroid is tracked wrong.
                    ## Top half
                    if ii == np.size(distance) - 1 and dist_current_next > self.distance_threshold:  # If reach end
                        # print ii, np.size(distance)
                        print current_centroid
                        centroid[:, ii + 1] = current_centroid
                        rectangle[:, ii + 1] = current_rect
                        break

                    if dist_current_next > self.distance_threshold:
                        # Check if y is on top or bottom.
                        centroid[:, ii + 1] = current_centroid
                        rectangle[:, ii + 1] = current_rect
                        ii += 1
                        next_centroid = centroid[:, ii + 1]  # Get subsequent centroid

                    else:
                        correctflag = False
                        ii += 1
            else:
                ii += 1

        Tank['CentroidX'], Tank['CentroidY'] = centroid
        Tank['RectX'], Tank['RectY'], Tank['RectW'], Tank['RectH'] = rectangle
        # Tank = Tank[:-1]
        corrected_euclideandistance = self.get_speed(Tank=Tank)
        self.plot_andsave_correctedspeed(Tank=Tank, correcteddistance=corrected_euclideandistance,
                                         uncorrecteddistance=distance, filename=filename)

    def find_distance_between_twocentroids(self, centroidA, centroidB):
        distance = np.sqrt(
            ((centroidA[0] - centroidB[0]) ** 2) + ((centroidA[1] - centroidB[1]) ** 2))

        return distance

    def read_videofile(self, filename):
        filename = os.path.join(self.OriginalVideoFolder, filename[:-10] + '.mp4')
        vid = imageio.get_reader(filename, 'ffmpeg')

        return vid

    def read_and_display_particularframe(self, videofile, framenumber, centroid1, centroid2):
        image = videofile.get_data(framenumber)
        image = imutils.resize(image, width=800)
        fig = plt.figure()
        fig.suptitle('image #{}'.format(framenumber), fontsize=12)
        plt.imshow(image)
        plt.plot(centroid1[0], centroid1[1], 'r*', markersize=10)
        plt.plot(centroid2[0], centroid2[1], 'g*', markersize=10)
        plt.xlim((0, np.size(image, 1)))
        plt.ylim((np.size(image, 0), 0))
        # print 'Intensity of area, ', image[centroid1[0], centroid1[1]], centroid1, centroid2
        plt.grid('off')
        plt.show()
        plt.close()

    def create_avi(self, TanknameA, TanknameB, Filename):
        TankA = pd.read_csv(os.path.join(TanknameA))
        TankB = pd.read_csv(os.path.join(TanknameB))
        centroidA = np.vstack((TankA['CentroidX'], TankA['CentroidY']))
        centroidB = np.vstack((TankB['CentroidX'], TankB['CentroidY']))
        print np.shape(centroidA), np.shape(centroidB)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        avifile = cv2.VideoWriter(os.path.join(self.CorrectedDistancesFolder, 'Corrected.avi'), fourcc,
                                  60, (800, 333), True)
        cap = cv2.VideoCapture(Filename)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 36000)
        count = 0
        while cap.isOpened():
            # print count
            ret, frame = cap.read()
            if ret:
                frame = imutils.resize(frame, width=800)
                # print np.shape(frame)
                cv2.circle(frame, (centroidA[0, count], centroidA[1, count]), radius=3, color=(0, 0, 255),
                           thickness=2)
                cv2.circle(frame, (centroidB[0, count], centroidB[1, count]), radius=3, color=(0, 0, 255),
                           thickness=2)
            avifile.write(frame)
            count += 1
            if count == 10000:
                break
        cap.release()
        avifile.release()

    def get_speed(self, Tank):
        # Calculate speed between frames and then bin per second
        Frames = np.vstack((Tank['CentroidX'], Tank['CentroidY'])).T
        bx = Frames[:-1, 0]
        by = Frames[:-1, 1]
        ax = Frames[1:, 0]
        ay = Frames[1:, 1]

        distance = ((bx - ax) ** 2) + ((by - ay) ** 2)
        euclidean_distance = np.sqrt(distance)

        return euclidean_distance

    def plot_andsave_correctedspeed(self, Tank, uncorrecteddistance, correcteddistance, filename):
        # Plot and save corrected speed
        fs = plt.figure(figsize=(10, 3))
        ax1 = plt.subplot(121)
        ax1.plot(uncorrecteddistance)
        plt.xlabel('Frames')
        plt.ylabel('Distance in pixels')
        ax1.locator_params(axis='y', nbins=4)
        plt.xlim((0, np.size(uncorrecteddistance)))
        plt.ylim((0, np.max(uncorrecteddistance)))
        plt.xticks(rotation=90)
        plt.title('Uncorrected')

        ax2 = plt.subplot(122)
        ax2.plot(correcteddistance)
        plt.xlabel('Frames')
        plt.ylabel('Distance in pixels')
        plt.xlim((0, np.size(correcteddistance)))
        plt.ylim((0, np.max(uncorrecteddistance)))
        ax2.locator_params(axis='y', nbins=4)
        plt.xticks(rotation=90)
        plt.title('Corrected')

        plt.tight_layout()
        plt.savefig(os.path.join(self.FiguresFolder, filename[:-4] + '_Corrected_Distances.pdf'))
        plt.close()
        Tank.to_csv(os.path.join(self.CorrectedDistancesFolder, filename))


class PlotSpatialMaps(object):
    def __init__(self, DirectoryName, pixel_to_mm):
        self.DirectoryName = DirectoryName
        self.FiguresFolder = os.path.join(self.DirectoryName, 'Figures')
        self.CorrectedDistancesFolder = os.path.join(self.DirectoryName, 'CorrectedCentroid')
        self.pixel_to_mm = pixel_to_mm
        self.Experiments_with_Stimulus = ['1800', '2400', '2700', '2800', '3000']
        self.stimuluspresentfiles = []

        # Get experiment parameters
        self.stimulusA, self.stimulusB, self.fishA, self.fishB, self.framerate = [], [], [], [], []
        self.ylim_fishtank, self.xlim_fishtankA, self.xlim_fishtankB = [], [], []
        self.getexperimentparameters()

    def get_csvfiles(self):
        csvfilenames = [f for f in os.listdir(self.CorrectedDistancesFolder) if f.endswith('.csv')]
        # Collect TankA and TankB Filename
        TankAcsv = [ii for ii in csvfilenames if ii.find('TankA') > 0]
        TankBcsv = [ii for ii in csvfilenames if ii.find('TankB') > 0]

        return TankAcsv, TankBcsv

    def getdatfromfile(self, Tankname):
        # Get Data for plotting
        TankData = pd.read_csv(os.path.join(self.CorrectedDistancesFolder, Tankname))

        return TankData

    def getexperimentparameters(self):
        file = os.path.join(os.path.normpath(self.DirectoryName + os.sep + os.pardir), 'ExperimentParameters.csv')
        with open(file) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'StimulusType':
                    [self.stimulusA, self.stimulusB] = row[1].split('_')
                if row[0] == 'Fishname':
                    [self.fishA, self.fishB] = row[1].split('_')
                if row[0] == 'FrameRate':
                    self.framerate = float(row[1])

        file = np.load(os.path.join(self.DirectoryName, 'ROIs.npy'))
        self.ylim_fishtank = (file[0][1], file[0][3])
        self.xlim_fishtankB = (file[0][0], file[0][0] + file[0][2])
        self.xlim_fishtankA = (file[1][0], file[1][0] + file[1][2])

    def find_dark_stimulus_frames(self, Tank, filename):
        # Find frames with darkness and stimulus ON
        Frames = Tank['Status']
        DarkFrames = Frames[Frames == 'Dark'].index
        if DarkFrames.size:
            DarkFrames1 = DarkFrames[DarkFrames < 30000]  # First Dark cycle
            DarkFrames2 = DarkFrames[DarkFrames > 70000]  # Second Dark Cycle
            if not DarkFrames2.size:
                DarkFrames2 = None
        else:
            DarkFrames1 = None
            DarkFrames2 = None

        self.stimuluspresentfiles = [ff for ff in self.Experiments_with_Stimulus if filename.find(ff) > 0]
        ### Only get stimulus frames for those files where stimulus was given
        if self.stimuluspresentfiles:
            StimulusOnFrames = Frames[Frames == 'StimulusON'].index
            StimulusOnFrames = StimulusOnFrames[(StimulusOnFrames > 30000) & (StimulusOnFrames < 50000)]

            if np.size(StimulusOnFrames) == 0:
                ## Stimulus at 600
                # start = 38600
                # finish = 38850
                ## Stimulus at 900
                start = 57692
                finish = 58075
                print 'No stimulus frames found in stimulus files. Using %s to %s' % (start, finish)
                StimulusOnFrames = range(start, finish)

            StimulusAfterFrames = range(StimulusOnFrames[-1], StimulusOnFrames[-1] + 100)
        else:
            StimulusOnFrames = None
            StimulusAfterFrames = None
        print np.shape(DarkFrames1), np.shape(DarkFrames2), np.shape(StimulusOnFrames), np.shape(StimulusAfterFrames)
        return DarkFrames1, DarkFrames2, StimulusOnFrames, StimulusAfterFrames

    def get_speed(self, Tank):
        # Calculate speed between frames and then bin per second
        Frames = np.vstack((Tank['CentroidX'], Tank['CentroidY'])).T
        bx = Frames[:-1, 0]
        by = Frames[:-1, 1]
        ax = Frames[1:, 0]
        ay = Frames[1:, 1]

        distance = ((bx - ax) ** 2) + ((by - ay) ** 2)
        euclidean_distance = np.sqrt(distance)

        return euclidean_distance

    def timepoints_for_plotting_beforeandafterdark(self, Tank, distance, DarkFrames1, DarkFrames2, StimulusONFrames,
                                                   filename, tankflag='TankA'):
        ## Plot depending on what exists in the file
        framerate = int(np.round(self.framerate))
        label_in_seconds = self.convert_frames_to_minutes(Tank['CentroidX'])
        fs = plt.figure(figsize=(15, 6))
        gs = plt.GridSpec(2, 6, height_ratios=[2, 1])
        ax2 = fs.add_subplot(gs[1, 0:3])
        ax2.set_xlim((0, label_in_seconds[-1]))
        dict_file = {}

        # If there are darkframes, plot before
        if DarkFrames1 is not None:
            spatialx = Tank['CentroidX'][:DarkFrames1[0]]  # Spatial
            spatialy = Tank['CentroidY'][:DarkFrames1[0]]
            distancey = distance[:DarkFrames1[0]]  # Distance Measure
            distancex = label_in_seconds[range(0, DarkFrames1[0])]

            colors = sns.color_palette('winter', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 0])
            ax1.set_title('Before Dark')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            if self.stimuluspresentfiles:
                dict_file['AB_BeforeDarkBeforeStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))
            else:
                dict_file['AF_BeforeDarkAfterStimulius'] = np.vstack((spatialx, spatialy, distancex, distancey))

        if DarkFrames1 is None and StimulusONFrames is not None and self.stimuluspresentfiles:  # If there were no darkframes presented before stiulus
            spatialx = Tank['CentroidX'][:StimulusONFrames[0]]  # Spatial
            spatialy = Tank['CentroidY'][:StimulusONFrames[0]]
            distancey = distance[:StimulusONFrames[0]]  # Distance Measure
            distancex = label_in_seconds[range(0, StimulusONFrames[0])]

            colors = sns.color_palette('winter', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 0])
            ax1.set_title('Before Stimulus')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            dict_file['AB_BeforeStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        # If there are darkframes and no stimulus frames, plot what happens after darkness till the end
        if DarkFrames1 is not None and StimulusONFrames is None:
            spatialx = Tank['CentroidX'][DarkFrames1[-1]:-1]  # Spatial
            spatialy = Tank['CentroidY'][DarkFrames1[-1]:-1]
            distancey = distance[DarkFrames1[-1]:]  # Distance Measure
            distancex = label_in_seconds[range(DarkFrames1[-1], np.size(distance))]

            colors = sns.color_palette('cool', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 1])
            ax1.set_title('After Second Dark')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            dict_file['AG_AfterDarkAfterStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        # If there are darkframes and stimulus frames then plot after dark and before stimulus
        elif DarkFrames1 is not None and StimulusONFrames is not None:
            spatialx = Tank['CentroidX'][DarkFrames1[-1]:StimulusONFrames[0]]  # Spatial
            spatialy = Tank['CentroidY'][DarkFrames1[-1]:StimulusONFrames[0]]
            distancey = distance[DarkFrames1[-1]:StimulusONFrames[0]]  # Distance Measure
            distancex = label_in_seconds[range(DarkFrames1[-1], StimulusONFrames[0])]

            colors = sns.color_palette('summer', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 1])
            ax1.set_title('Before Stimulus')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            dict_file['AC_AfterDarkBeforeStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        # If there are two darkframes (before and after stimulus) , plot the second
        if DarkFrames2 is not None:
            print 'Plotting Baseline'
            spatialx = Tank['CentroidX'][DarkFrames2[-1]:-1]  # Spatial
            spatialy = Tank['CentroidY'][DarkFrames2[-1]:-1]
            distancey = distance[DarkFrames2[-1]:]  # Distance Measure
            distancex = label_in_seconds[range(DarkFrames2[-1], np.size(distance))]
            print  np.shape(spatialx), np.shape(spatialy), np.shape(distancey), np.shape(distancex)
            colors = sns.color_palette('cool', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 5])
            ax1.set_title('After Second Dark')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            dict_file['AG_AfterDarkAfterStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        # If none of these are present, just plot everything
        if DarkFrames1 is None and DarkFrames2 is None and StimulusONFrames is None:
            spatialx = Tank['CentroidX'][:-1]  # Spatial
            spatialy = Tank['CentroidY'][:-1]
            distancey = distance  # Distance Measure
            distancex = label_in_seconds[range(0, np.size(distance))]
            # print np.shape(distancex), np.shape(spatialx), np.shape(spatialy), np.shape(distancey)
            colors = sns.color_palette('autumn', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 0])
            print np.shape(distancex), np.shape(spatialx), np.shape(spatialy), np.shape(distancey)
            ax1.set_title('Free Swimming')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            if filename.find('Before') > 0:
                dict_file['AA_FreeSwimmingBeforeStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))
            else:
                dict_file['AH_FreeSwimmingAfterStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        return fs, gs, ax2, dict_file  # Return figure handles to continue plotting

    def timepoints_for_plotting_stimulus(self, fs, gs, ax2, dict_file, Tank, distance, DarkFrames1, DarkFrames2,
                                         StimulusONFrames, StimulusAfterPause, tankflag='TankA'):

        framerate = int(np.round(self.framerate))
        label_in_seconds = self.convert_frames_to_minutes(Tank['CentroidX'])

        if StimulusONFrames is not None:
            spatialx = Tank['CentroidX'][StimulusONFrames[0]:StimulusAfterPause[-1]]  # Spatial
            spatialy = Tank['CentroidY'][StimulusONFrames[0]:StimulusAfterPause[-1]]
            distancey = distance[StimulusONFrames[0]:StimulusAfterPause[-1]]  # Distance Measure
            distancex = label_in_seconds[range(StimulusONFrames[0], StimulusAfterPause[-1])]

            colors = sns.color_palette('Blues', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 2])
            ax1.set_title('During Stimulus')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            dict_file['AD_DuringStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        if StimulusONFrames is not None and DarkFrames2 is None:
            spatialx = Tank['CentroidX'][StimulusAfterPause[-1]:-1]  # Spatial
            spatialy = Tank['CentroidY'][StimulusAfterPause[-1]:-1]
            distancey = distance[StimulusAfterPause[-1]:]  # Distance Measure
            distancex = label_in_seconds[range(StimulusAfterPause[-1], np.size(distance))]

            colors = sns.color_palette('YlOrRd', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 3])
            ax1.set_title('After Stimulus')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            print np.shape(distancex), np.shape(spatialx), np.shape(spatialy), np.shape(distancey)
            dict_file['AE_AfterStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        if StimulusONFrames is not None and DarkFrames2 is not None:
            spatialx = Tank['CentroidX'][StimulusAfterPause[-1]:DarkFrames2[0]]  # Spatial
            spatialy = Tank['CentroidY'][StimulusAfterPause[-1]:DarkFrames2[0]]
            distancey = distance[StimulusAfterPause[-1]:DarkFrames2[0]]  # Distance Measure
            distancex = label_in_seconds[range(StimulusAfterPause[-1], DarkFrames2[0])]

            colors = sns.color_palette('YlOrRd', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 3])
            ax1.set_title('After Stimulus')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy, xd=distancex, yd=distancey, ax1=ax1,
                                  ax2=ax2, colors=colors, tankflag=tankflag)
            dict_file['AE_AfterStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

        return fs, dict_file  # Return figure handle for saving

    # plt.tight_layout()

    def plot_before_andafter_alone(self, Tank, distance, DarkFrames1, DarkFrames2, StimulusONFrames, StimulusAfterPause,
                                   tankflag='TankA'):

        framerate = int(np.round(self.framerate))
        label_in_seconds = self.convert_frames_to_minutes(Tank['CentroidX'])
        fs = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 1])
        gs.update(hspace=0.5)

        dict_file = {}

        if DarkFrames1 is not None:
            # Plot Everything before stimulus except a few frames before and after Darkness
            spatialx = np.hstack((Tank['CentroidX'][:DarkFrames1[0] - framerate * 2],
                                  Tank['CentroidX'][DarkFrames1[-1] + framerate * 2:StimulusONFrames[0]]))  # Spatial
            spatialy = np.hstack((Tank['CentroidY'][:DarkFrames1[0] - framerate * 2],
                                  Tank['CentroidY'][DarkFrames1[-1] + framerate * 2:StimulusONFrames[0]]))  # Spatial
            distancey = np.hstack(
                (distance[:DarkFrames1[0] - framerate * 2],
                 distance[DarkFrames1[-1] + framerate * 2:StimulusONFrames[0]]))
            distancex = label_in_seconds[range(0, np.size(distancey))]

            colors = sns.color_palette('cool', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 0])
            ax1.set_title('Before Stimulus')
            ax2 = fs.add_subplot(gs[1, 0:3])
            ax2.set_xlim((0, label_in_seconds[-1]))
            ax2.set_title('Before Stimulus')
            print np.shape(distancex), np.shape(spatialx), np.shape(spatialy), np.shape(distancey)
            dict_file['AllBeforeStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy,
                                  xd=distancex, yd=distancey,
                                  ax1=ax1, ax2=ax2, colors=colors, tankflag=tankflag)
        else:
            # Plot Everything before stimulus except a few frames before and after Darkness
            spatialx = Tank['CentroidX'][:StimulusAfterPause[0] - framerate * 2]  # Spatial
            spatialy = Tank['CentroidY'][:StimulusAfterPause[0] - framerate * 2]
            distancey = distance[:StimulusAfterPause[0] - framerate * 2]  # Distance Measure
            distancex = label_in_seconds[range(0, np.size(distancey))]
            print 'DarkFrames are None', np.shape(distancex), np.shape(spatialx), np.shape(spatialy), np.shape(
                distancey)
            colors = sns.color_palette('cool', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 0])
            ax1.set_title('Before Stimulus')
            ax2 = fs.add_subplot(gs[1, 0:3])
            ax2.set_xlim((0, label_in_seconds[-1]))
            ax2.set_title('Before Stimulus')
            print np.shape(distancex), np.shape(spatialx), np.shape(spatialy), np.shape(distancey)
            dict_file['AllBeforeStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy,
                                  xd=distancex, yd=distancey,
                                  ax1=ax1, ax2=ax2, colors=colors, tankflag=tankflag)

        if DarkFrames2 is not None:
            spatialx = np.hstack((Tank['CentroidX'][StimulusAfterPause[-1]:DarkFrames2[0] - framerate * 2],
                                  Tank['CentroidX'][DarkFrames2[-1] + framerate * 2:-1]))  # Spatial
            spatialy = np.hstack((Tank['CentroidY'][StimulusAfterPause[-1]:DarkFrames2[0] - framerate * 2],
                                  Tank['CentroidY'][DarkFrames2[-1] + framerate * 2:-1]))  # Spatial
            distancey = np.hstack(
                (distance[StimulusAfterPause[-1]:DarkFrames2[0] - framerate * 2],
                 distance[DarkFrames2[-1] + framerate * 2:]))
            distancex = label_in_seconds[range(StimulusAfterPause[-1], StimulusAfterPause[-1] + np.size(distancey))]
            print 'DarkFrames2 are not None', np.shape(distancex), np.shape(spatialx), np.shape(spatialy), np.shape(
                distancey)

            dict_file['AllAfterStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))

            colors = sns.color_palette('summer', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 1])
            ax1.set_title('After Stimulus')
            ax2 = fs.add_subplot(gs[2, 0:3])
            ax2.set_xlim((0, label_in_seconds[-1]))
            ax2.set_title('After Stimulus')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy,
                                  xd=distancex, yd=distancey,
                                  ax1=ax1, ax2=ax2, colors=colors, tankflag=tankflag)
        else:
            spatialx = Tank['CentroidX'][StimulusAfterPause[-1]:-1]  # Spatial
            spatialy = Tank['CentroidY'][StimulusAfterPause[-1]:-1]
            distancey = distance[StimulusAfterPause[-1]:]  # Distance Measure
            distancex = label_in_seconds[range(StimulusAfterPause[-1], np.size(distance))]
            dict_file['AllAfterStimulus'] = np.vstack((spatialx, spatialy, distancex, distancey))
            colors = sns.color_palette('summer', (np.size(spatialx) / framerate) + 5)
            ax1 = fs.add_subplot(gs[0, 1])
            ax1.set_title('After Stimulus')
            ax2 = fs.add_subplot(gs[2, 0:3])
            ax2.set_xlim((0, label_in_seconds[-1]))
            ax2.set_title('After Stimulus')
            self.plot_spatialmaps(framerate=framerate, xs=spatialx, ys=spatialy,
                                  xd=distancex, yd=distancey,
                                  ax1=ax1, ax2=ax2, colors=colors, tankflag=tankflag)

        return fs, dict_file

    def plot_spatialmaps(self, framerate, xs, ys, xd, yd, colors, ax1, ax2, tankflag='TankA'):
        count = 0  # Plot spatial maps
        yd = self.convert_distance_to_mm(distance=yd)
        for ii in xrange(0, np.size(ys) + 1, framerate):
            ax1.plot(xs[ii:ii + framerate], ys[ii:ii + framerate], linewidth=3, color=colors[count], alpha=0.5)
            ax2.plot(xd[ii:ii + framerate], yd[ii:ii + framerate], linewidth=3, color=colors[count], alpha=0.5)
            count += 1

        ax1.set_ylim(self.ylim_fishtank)
        if tankflag == 'TankA':
            ax1.set_xlim(self.xlim_fishtankA)
        else:
            ax1.set_xlim(self.xlim_fishtankB)

        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Distance (mm)')
        ax2.locator_params(axis='y', nbins=4)
        ax2.set_ylim((0, 8))
        ax1.axis('off')
        ax2.grid('off')
        ax1.invert_yaxis()
        # self.convert_frames_to_seconds(ax2)

    def save_combined_figure(self, figurehandle, filename):
        figurehandle.savefig(os.path.join(self.FiguresFolder, filename[:-4] + '.pdf'))
        plt.close(figurehandle)

    def save_dict(self, dict_file, filename):
        DictFolder = os.path.join(self.DirectoryName, 'DictFileforconditions')
        if not os.path.exists(DictFolder):
            os.mkdir(DictFolder)

        np.save(os.path.join(DictFolder, filename[:-4] + '.npy'), dict_file)

    def convert_frames_to_minutes(self, array):
        ## Convert frames to seconds
        time = 1.0 / self.framerate
        n = np.size(array)
        label_in_seconds = np.linspace(0, n * time, n + 1) / 60

        return label_in_seconds

    def convert_distance_to_mm(self, distance):
        distance = distance * self.pixel_to_mm
        return distance
