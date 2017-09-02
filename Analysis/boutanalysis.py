import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.signal import argrelextrema
import csv
import collections
from matplotlib.backends.backend_pdf import PdfPages
import math
import scipy.stats as stats
import statsmodels.nonparametric.api as smnp
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import itertools
import pandas as pd
import cv2
from mpl_toolkits.mplot3d import Axes3D
from numpy import convolve, r_, ones
from sklearn.decomposition import PCA as sklearnPCA

sns.set_context('paper', font_scale=1)


class CollectParameter(object):
    def __init__(self, DirectoryName, StimulusNames, pixel_to_mm, analysistimebin, framerate, slow_speed):

        self.DirectoryName = DirectoryName
        self.StimulusNames = StimulusNames

        self.FiguresFolder = os.path.join(self.DirectoryName, 'Figures')
        if not os.path.exists(self.FiguresFolder):
            os.mkdir(self.FiguresFolder)

        self.pixel_to_mm = pixel_to_mm
        self.slow_speed = slow_speed
        self.framerate = framerate

        if analysistimebin == 10:
            self.sizeofmatrix = 37800
        elif analysistimebin == 5:
            self.sizeofmatrix = 18900

        figure = False
        if figure:
            self.pp = PdfPages(os.path.join(self.FiguresFolder, self.StimulusNames + '_BoutAnalysis.pdf'))
        self.beforeafterfiles = self.get_npyfiles(StimulusName=self.StimulusNames)
        self.TimeBinDataFrame, self.TimeBins = self.divide_by_time_bins(StimulusName=self.StimulusNames,
                                                                        timebins=analysistimebin, figureflag=figure)
        if figure:
            self.pp.close()

        self.distance = self.concatenate_timebins(self.TimeBinDataFrame, rownumber=3)  # distance
        self.X = self.concatenate_timebins(self.TimeBinDataFrame, rownumber=0)  # X
        self.Y = self.concatenate_timebins(self.TimeBinDataFrame, rownumber=1)  # Y
        self.dict_boutscount = {key: [] for key in
                                ['Fishnumber', 'Size', 'DistanceTravelled', 'Starttime', 'PauseorBout', 'Timebin',
                                 'Interval']}

        # self.pp = PdfPages(os.path.join(self.FiguresFolder, self.StimulusNames + '_Findminimumspeed.pdf'))
        self.calculate_bouts(self.distance, figureflag=False)
        self.boutscount = pd.DataFrame.from_dict(self.dict_boutscount)
        # self.pp.close()

    def get_npyfiles(self, StimulusName):
        npyfilenames = [f for f in os.listdir(os.path.join(self.DirectoryName, StimulusName)) if
                        f.endswith('.npy')]

        return npyfilenames

    def divide_by_time_bins(self, StimulusName, timebins, figureflag):
        new_dict = collections.OrderedDict()
        time_dict = []
        for ii in self.beforeafterfiles:
            print ii
            if figureflag:
                fs = plt.figure(figsize=(8, 8))
            subplotcount = 2
            count = timebins
            dictfile_before_after = np.load(os.path.join(self.DirectoryName, StimulusName, ii))[()]

            # Check size of data before stimulus to correct
            this_data = dictfile_before_after['AllBeforeStimulus'][:, :self.framerate * (timebins * 60)]
            checkdata = np.size(this_data, 1)
            #
            # newshpe = np.lib.pad(this_data,
            #                      ((0, np.size(this_data, 0)), (0, self.sizeofmatrix - checkdata)),
            #                      'constant', constant_values=(0))
            # dictfile_before_after['AllBeforeStimulus'] = newshpe
            # checkdata = np.size(dictfile_before_after['AllBeforeStimulus'], 1)

            print self.sizeofmatrix - checkdata, np.shape(dictfile_before_after['AllBeforeStimulus'])

            if checkdata == self.sizeofmatrix:
                for keys, data in dictfile_before_after.iteritems():
                    # print keys
                    if keys.find('AllBeforeStimulus') >= 0:
                        new_dict[ii[:-4] + '_' + keys] = data[:, :self.framerate * (timebins * 60)]
                        if figureflag:
                            self.plot_xy_distance_bybins(figurehandle=fs,
                                                         data=new_dict[ii[:-4] + '_' + keys],
                                                         plottitle='Before ' + ii,
                                                         subplot=1)
                    else:
                        for kk in xrange(0, np.size(data, 1),
                                         self.framerate * (timebins * 60)):  # 10 minutes = 600 seconds
                            newkey = '%s_%2dminutes' % (ii[:-4], count)
                            time_dict.append(count)
                            new_dict[newkey] = data[:, kk:kk + self.framerate * (timebins * 60)]
                            count += timebins
                            if figureflag:
                                self.plot_xy_distance_bybins(figurehandle=fs,
                                                             data=new_dict[newkey],
                                                             plottitle=newkey,
                                                             subplot=subplotcount)
                            subplotcount += 1
                            if count == 30:
                                break

            if figureflag:
                self.pp.savefig(bbox_inches='tight')
                plt.close()

        return new_dict, np.unique(time_dict)

    def plot_xy_distance_bybins(self, figurehandle, data, plottitle, subplot, bin=2000):
        # Plot X and Y differences
        ax1 = figurehandle.add_subplot(3, 1, subplot)
        # plt.plot(data[0, 600:bin] - data[0, 601:bin + 1], label='X', alpha=0.5, linewidth=2)
        # plt.plot(data[1, 600:bin] - data[1, 601:bin + 1], label='Y', alpha=0.5, linewidth=2)
        plt.plot(self.convert_distance_to_mm(data[3, 600:bin + 1]), color='g', label='distance', alpha=0.5, linewidth=1)
        smooth_data = self.smooth_func(data[3, 600:bin + 1], 5)
        plt.plot(self.convert_distance_to_mm(smooth_data), color='r', label='distance', alpha=0.5, linewidth=1)

        minima = np.where(smooth_data == 0)[0]
        plt.plot(minima, smooth_data[minima], 'm*', markersize=5)
        plt.axhline(y=0, linewidth=1, color='k')

        ax1.set_ylim((-0.1, 1))
        plt.title(plottitle)

        # ax2 = ax1.twinx()
        # ax2.plot(data[0, 600:bin], label='X', alpha=0.5, linewidth=2)
        # ax2.plot(data[1, 600:bin], label='Y', alpha=0.5, linewidth=2)
        # ax2.set_ylim((0, 600))

    def concatenate_timebins(self, BefAftDataFrame, rownumber):
        CompiledDataFrame = collections.OrderedDict()
        for keys, data in BefAftDataFrame.iteritems():
            compileddata = data[rownumber, 600:]
            if np.size(compileddata) == self.sizeofmatrix - 600:
                if keys.find('AllBeforeStimulus') > 0:
                    if 'Before' in CompiledDataFrame:
                        CompiledDataFrame['Before'] = np.vstack((CompiledDataFrame['Before'], compileddata))
                    else:
                        CompiledDataFrame['Before'] = compileddata

                elif keys.find('minutes') > 0:
                    string = keys[keys.find('minutes') - 2:keys.find('minutes')]
                    if string + 'minutes' in CompiledDataFrame:
                        CompiledDataFrame[string + 'minutes'] = np.vstack(
                            (CompiledDataFrame[string + 'minutes'], compileddata))
                    else:
                        CompiledDataFrame[string + 'minutes'] = compileddata

        for keys, data in CompiledDataFrame.iteritems():
            print keys, np.shape(data)

        return CompiledDataFrame

    def calculate_bouts(self, combineddata, figureflag=False):
        current_palette = sns.color_palette()
        plotcolors = {}
        count = 0
        for keys in combineddata.iterkeys():
            plotcolors[keys] = current_palette[count]
            count += 1
        if figureflag:
            fs = plt.figure(figsize=(8, 6))
            gs = plt.GridSpec(4, 2)
            subplotcount = 0
            boutpausex = 0
        for keys, data in combineddata.iteritems():
            for ii in xrange(0, np.size(data, 0)):
                print 'Finding minima', keys, np.shape(data)
                smooth_data = self.smooth_func(data[ii, :], window_len=5)
                minima = np.where(smooth_data == 0)[0]

                ## Find length of pauses
                # Group by pauses
                pause_data = 1 * (smooth_data == 0)
                size1, time1 = self.group_by_zeros(pausedata=pause_data)

                # Check distance inbetween pauses. If distance travelled is too small, convert it to 0 and get time and size again
                # fs = plt.figure()
                # ax1 = fs.add_subplot(111)
                # plt.plot(smooth_data[1:500])
                # plt.plot(1 - pause_data[1:500], '.', markersize=10, color='r')


                for zz in xrange(0, len(time1) - 1):
                    thisbout_length = np.sum(data[ii, time1[zz]:time1[zz + 1]] * self.pixel_to_mm)
                    if thisbout_length < 0.15:
                        pause_data[time1[zz]:time1[zz + 1]] = 1

                pausecount_size, pausecount_time = self.group_by_zeros(pausedata=pause_data)

                pause_interval = [a - b for a, b in zip(pausecount_time[1:], pausecount_time[:-1])]
                pause_interval = [0] + pause_interval
                print np.shape(pausecount_size), np.shape(pause_interval)

                # pausecount_size = self.convert_frames_to_ms(frames=np.asarray(pausecount_size), millisecondflag=False)
                # pausecount_time = self.convert_frames_to_ms(frames=np.asarray(pausecount_time), millisecondflag=False)
                # plt.plot(1-pause_data[1:500], '.', markersize=10, color='k')
                # plt.show()
                # Plot pauses
                if figureflag:
                    axpause = plt.subplot(gs[3, 0])
                    axpause.plot(np.ones(np.size(pausecount_size)) * boutpausex, pausecount_size, '.',
                                 color=plotcolors[keys], alpha=0.5, markersize=10)

                self.dict_boutscount['Fishnumber'].extend(['Fish' + str(ii)] * np.size(pausecount_size))
                self.dict_boutscount['Size'].extend(pausecount_size)
                self.dict_boutscount['Starttime'].extend(pausecount_time)
                self.dict_boutscount['DistanceTravelled'].extend([0] * np.size(pausecount_size))
                self.dict_boutscount['Timebin'].extend([keys] * np.size(pausecount_size))
                self.dict_boutscount['PauseorBout'].extend(['Pause'] * np.size(pausecount_size))
                self.dict_boutscount['Interval'].extend(pause_interval)

                # Find bouts. Plot bouts from start to end
                for numzero in xrange(0, np.size(minima) - 1):
                    if (minima[numzero + 1] - minima[numzero]) > 1:
                        boutcount = smooth_data[minima[numzero]:minima[numzero + 1]]  # Get bouts as time between zeros
                        boutdistance = np.sum(boutcount)
                        boutcount_size = np.size(boutcount)

                        if numzero == 0:
                            bout_interval = 0
                        else:
                            bout_interval = minima[numzero] - minima[numzero - 1]

                        framestoms = self.convert_frames_to_ms(
                            frames=np.linspace(0, np.size(boutcount) - 1, np.size(boutcount)), millisecondflag=False)

                        # Savedatadframe
                        self.dict_boutscount['Fishnumber'].append('Fish' + str(ii))
                        self.dict_boutscount['Size'].append(boutcount_size)
                        self.dict_boutscount['Starttime'].append(minima[numzero])
                        self.dict_boutscount['DistanceTravelled'].append(boutdistance)
                        self.dict_boutscount['Timebin'].append(keys)
                        self.dict_boutscount['PauseorBout'].append('Bouts')
                        self.dict_boutscount['Interval'].append(bout_interval)

                        # Plot bouts
                        if figureflag:
                            ax1 = plt.subplot(gs[subplotcount, :])
                            ax1.plot(framestoms, self.convert_distance_to_mm(boutcount),
                                     color=plotcolors[keys],
                                     alpha=0.5)
                            axbout = plt.subplot(gs[3, 1])
                            axbout.plot(boutpausex, boutcount_size, '.', color=plotcolors[keys], alpha=0.5,
                                        markersize=10)
            if figureflag:
                ax1.set_title(keys)
                ax1.set_ylabel('Distance \n Moved (mm)')
                ax1.set_xlabel('Time (Seconds)')
                ax1.locator_params(axis='y', nbins=3)
                boutpausex += 1
                subplotcount += 1

        if figureflag:
            self.set_axis_paraemter(axis=axbout, datalength=len(combineddata.keys()),
                                    xticklabel=['Swims'] * len(combineddata.keys()))
            self.set_axis_paraemter(axis=axpause, datalength=len(combineddata.keys()),
                                    xticklabel=['Pauses'] * len(combineddata.keys()))

            plt.tight_layout()
            plt.savefig(os.path.join(self.FiguresFolder, self.StimulusNames + '_Findminimumspeed.tif'),
                        bbox_inches='tight')
            # self.pp.savefig(bbox_inches='tight')
            plt.close()

    def set_axis_paraemter(self, axis, datalength, xticklabel):
        axis.set_xlim((-1, datalength + 1))
        axis.set_xticks(range(0, datalength))
        axis.set_xticklabels(xticklabel)
        axis.set_ylabel('Number of frames')
        for tick in axis.get_xticklabels():
            tick.set_rotation(45)

    @staticmethod
    def smooth_func(x, window_len=10):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        #         w = np.hanning(window_len)
        w = ones(window_len, 'd')
        y = convolve(w / w.sum(), s, mode='valid')
        # print 'Size of y...', shape(y)
        return y[window_len / 2:-window_len / 2 + 1]

    def convert_distance_to_mm(self, distance):
        distance = distance * self.pixel_to_mm
        return distance

    def convert_frames_to_ms(self, frames, millisecondflag=True):
        if millisecondflag:
            frames1 = (frames / self.framerate) * 1000
        else:
            frames1 = (frames / self.framerate)
        return frames1

    @staticmethod
    def group_by_zeros(pausedata):
        b = range(len(pausedata))
        pause_size = []
        pause_time = []
        for group in itertools.groupby(iter(b), lambda x: pausedata[x]):
            if group[0] == 1:
                lis = list(group[1])
                pause_size.append(len(lis))
                pause_time.append(min(lis))
        return pause_size, pause_time
