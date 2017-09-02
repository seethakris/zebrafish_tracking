import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns
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

sns.set_context('paper', font_scale=1.5)


class CollectParameter(object):
    def __init__(self, DirectoryName, StimulusName, pixel_to_mm, analysistimebin, histtimebin, framerate, zoom_axis,
                 slow_speed):
        self.DirectoryName = DirectoryName
        self.StimulusName = StimulusName
        self.FiguresFolder = os.path.join(self.DirectoryName, 'Figures')

        if not os.path.exists(self.FiguresFolder):
            os.mkdir(self.FiguresFolder)

        self.pixel_to_mm = pixel_to_mm
        self.slow_speed = slow_speed
        self.histtimebin = histtimebin
        self.framerate = framerate
        self.zoom_axis = zoom_axis
        self.xlimittank = 350
        self.ylimittank = 200
        self.axiscount = ['Before', '10minutes',
                          '20minutes']  # ['Before', ' 5minutes', '10minutes', '15minutes', '20minutes', '25minutes']
        self.edgecolors = sns.color_palette('colorblind', len(self.axiscount))
        self.width_ratios = [2, 2, 2, 0.25]

        self.beforeafterfiles = self.get_npyfiles()
        self.pp = PdfPages(os.path.join(self.FiguresFolder, self.StimulusName + str(analysistimebin) + 'TimeBin3.pdf'))
        TimeBinDataFrame, self.TimeBins = self.divide_by_time_bins(timebins=analysistimebin)
        TimeBinDataFrame = self.adjustTankAcentroid(TimeBinDataFrame)

        CentroidDataFramePerSecond, SpeedDataFrameTimeBin, SlowSpeedTimeBin = self.find_speed(TimeBinDataFrame)
        CombinedDataFrame_Speed = self.concatenate_centroids_speeds(SpeedDataFrameTimeBin, use_columnflag=10)

        CombinedDataFrame_Y = self.concatenate_centroids_speeds(TimeBinDataFrame, use_columnflag=1)
        CombinedDataFrame_X = self.concatenate_centroids_speeds(TimeBinDataFrame, use_columnflag=0)
        Combined_Centroid = self.concatenate_centroids_speeds(CentroidDataFramePerSecond, use_columnflag=11)

        # # Plot stuff
        self.plotcolors = self.get_colors()

        self.SlowSpeedCount = self.plot_pdf_withtimebins(CombinedDataFrame_Speed, SlowSpeedTimeBin)
        self.SlowSpeedCount['StimulusName'] = self.StimulusName
        temp = self.getpositionduringodor(TimeBinDataFrame, self.SlowSpeedCount)

        self.SlowSpeedCount = pd.concat([self.SlowSpeedCount, temp], axis=1)
        # print self.SlowSpeedCount
        self.plot_spatialmaps_as_heatmap(CombinedDataFrame_X, CombinedDataFrame_Y)
        # self.plot_centroidduringslowspeedbouts(Speed=CombinedDataFrame_Speed, Centroid=Combined_Centroid)
        self.plot_centroids_duringslowspeed_perfish(Speed=SpeedDataFrameTimeBin, Centroid=CentroidDataFramePerSecond)
        self.pp.close()

    def getpositionduringodor(self, DataFrame, SlowSpeedCount):
        CompiledDataFrame = {key: [] for key in ['Xdata', 'Ydata', 'type']}

        for keys, data in DataFrame.iteritems():
            if keys.find('10minutes') > 0:
                CompiledDataFrame['Xdata'].append(data[0, 0])
                CompiledDataFrame['Ydata'].append(data[1, 0])
                CompiledDataFrame['type'].append(keys)
        CompiledDataFrame = pd.DataFrame.from_dict(CompiledDataFrame)

        return CompiledDataFrame

    def get_npyfiles(self):
        npyfilenames = [f for f in os.listdir(os.path.join(self.DirectoryName, self.StimulusName)) if
                        f.endswith('.npy')]

        return npyfilenames

    def adjustTankAcentroid(self, DataFrame):
        for keys, data in DataFrame.iteritems():
            # Adjust tank data to start from 0,0
            # if keys.find('TankB') > 0:
            x = data[0, :]
            # x = x - np.min(x) #To make left side of tank at 0
            x = np.abs(x - np.max(x)) #To make odor valve at 0
            DataFrame[keys][0, :] = x

        return DataFrame

    def find_mean_acrossanimal(self, DataFrame, columnflag):
        CombineDataFrame = collections.OrderedDict()
        MeanDataFrame = {}
        for keys, data in DataFrame.iteritems():
            data = self.convert_distance_to_mm(data)
            if keys.find('AllBeforeStimulus') > 0:
                if 'Before' in CombineDataFrame:
                    CombineDataFrame['Before'] = np.vstack((CombineDataFrame['Before'], data[columnflag, :30001]))
                else:
                    CombineDataFrame['Before'] = data[columnflag, :30001]
            elif keys.find('minutes') > 0:
                string = keys[keys.find('minutes') - 2:keys.find('minutes')]
                if string + 'minutes' in CombineDataFrame:
                    CombineDataFrame[string + 'minutes'] = np.vstack(
                        (CombineDataFrame[string + 'minutes'], data[columnflag, :30001]))
                else:
                    CombineDataFrame[string + 'minutes'] = data[columnflag, :30001]

        for keys, data in CombineDataFrame.iteritems():
            MeanDataFrame['mean' + keys] = np.squeeze(np.mean(data, axis=0))
            MeanDataFrame['error' + keys] = np.squeeze(stats.sem(data, axis=0))

        return MeanDataFrame

    def find_speed(self, BefAftDataFrame):
        DataFrame = {}
        DataFrame_Centroid = {}
        SlowSpeed_count = {}
        time = []
        for keys, data in BefAftDataFrame.iteritems():
            count = 1
            time = []
            distance_inseconds = []
            meanCentroid_inseconds = []
            data1 = self.convert_distance_to_mm(data)
            for kk in xrange(0, np.size(data, 1), self.framerate):
                distance_inseconds.append(np.sum(data1[3, kk:kk + self.framerate]))
                seconddata = data[:, kk:kk + self.framerate]
                meanCentroid_inseconds.append([np.mean(seconddata[0, :]), np.mean(seconddata[1, :])])
                time.append(count)
                count += 1

            DataFrame[keys] = distance_inseconds
            DataFrame_Centroid[keys] = meanCentroid_inseconds
            # Find count of slow speed bins per fish
            SlowSpeed_count[keys] = np.size(np.where(np.asarray(distance_inseconds) <= self.slow_speed)[0])
        DataFrame['Timeinseconds'] = time

        return DataFrame_Centroid, DataFrame, SlowSpeed_count

    def convert_distance_to_mm(self, distance):
        distance = distance * self.pixel_to_mm
        return distance

    def get_colors(self):
        colors = sns.color_palette()  # sns.color_palette('colorblind')
        plotcolor = {}
        plotcolor['Before'] = colors[0]
        plotcolor['After'] = colors[1]
        count = 1
        if self.TimeBins is not None:
            for ii in self.TimeBins:
                plotcolor['%2dminutes' % ii] = colors[count]
                count += 1
        # print plotcolor.keys()
        return plotcolor

    def divide_by_time_bins(self, timebins):
        new_dict = collections.OrderedDict()
        time_dict = []
        for ii in self.beforeafterfiles:
            count = timebins
            dictfile_before_after = np.load(os.path.join(self.DirectoryName, self.StimulusName, ii))[()]
            for keys, data in dictfile_before_after.iteritems():
                if keys.find('AllBeforeStimulus') >= 0:
                    new_dict[ii[:-4] + '_' + keys] = data[:, 100:100 + self.framerate * (timebins * 60)]
                else:
                    for kk in xrange(0, np.size(data, 1),
                                     self.framerate * (timebins * 60)):  # 10 minutes = 600 seconds
                        newkey = '%s_%2dminutes' % (ii[:-4], count)
                        time_dict.append(count)
                        new_dict[newkey] = data[:, kk:kk + self.framerate * (timebins * 60)]
                        count += timebins
                        if count == 30:
                            break

        return new_dict, np.unique(time_dict)

    def concatenate_centroids_speeds(self, Dataframe, use_columnflag=10):
        CompiledDataFrame = collections.OrderedDict()

        for keys, data in Dataframe.iteritems():
            if use_columnflag not in [10, 11]:  # Random number
                compiledata = data[use_columnflag, :]
            else:
                compiledata = data

            if keys.find('AllBeforeStimulus') > 0:
                if 'Before' in CompiledDataFrame:
                    CompiledDataFrame['Before'] = np.concatenate((CompiledDataFrame['Before'], compiledata), axis=0)
                else:
                    CompiledDataFrame['Before'] = compiledata

            elif keys.find('minutes') > 0:
                string = keys[keys.find('minutes') - 2:keys.find('minutes')]
                if string + 'minutes' in CompiledDataFrame:
                    CompiledDataFrame[string + 'minutes'] = np.concatenate(
                        (CompiledDataFrame[string + 'minutes'], compiledata), axis=0)
                else:
                    CompiledDataFrame[string + 'minutes'] = compiledata

        # Find max value. Display size of data

        max_data = []
        for keys, data in CompiledDataFrame.iteritems():
            # print keys, np.shape(data)
            if use_columnflag == 10:
                max_data.append(np.max(data))
        if use_columnflag == 10:
            CompiledDataFrame['clip_speed'] = np.round(np.max(np.asarray(max_data))) + 1

        return CompiledDataFrame

    def plot_pdf_withtimebins(self, DataFrame, SlowSpeedDataFrame):
        with sns.axes_style('dark'):
            # Figure 1 with all pdfs
            fs1 = plt.figure(figsize=(15, 12))
            gs = plt.GridSpec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 2, 1])
            gs.update(hspace=0.25)

            # Plot histogram of data
            ax1 = fs1.add_subplot(gs[0, :-1])
            ax11 = ax1.twinx()
            Before, bin_edges1 = np.histogram(DataFrame['Before'], bins=self.histtimebin,
                                              normed=True)  # Get time bins from before scenario for histogram
            for keys in ['10minutes', '20minutes', 'Before']:
                if keys.find('Time') < 0 and keys.find('clip_speed') < 0:
                    ax1.hist(DataFrame[keys], bins=bin_edges1, histtype="bar", normed=False,
                             color=self.plotcolors[keys],
                             stacked=True, alpha=0.9, label=keys)
                    sns.distplot(DataFrame[keys], bins=bin_edges1, hist=False, kde=True, ax=ax11,
                                 kde_kws={"color": self.plotcolors[keys], "alpha": 0.8, "lw": 4, "label": keys})

            # Axis parameters
            ax1.set_xlim((-0.01, self.zoom_axis[1]))
            ax1.set_title('%s (n=%s)' % (self.StimulusName, len(self.beforeafterfiles)))

            ax11.set_ylim((0, 1))
            ax1.set_ylim((0, 3000))
            ax11.locator_params(axis='y', nbins=6)

            ax1.locator_params(axis='y', nbins=4)
            ax11.locator_params(axis='x', nbins=6)

            ax11.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
            ax11.set_ylabel('Probability of occurences')
            ax1.set_ylabel('Histogram')

            # Plot difference
            ax2 = fs1.add_subplot(gs[1, :-1], sharex=ax1)
            ax2.set_ylim((-0.3, 1))
            a21 = plt.axes([.4, .45, .20, .10], axisbg='w')
            self.create_inset(insetaxis=a21, bigaxis=ax2, mark_inset_loc1=3, mark_inset_loc2=4,
                              insetaxis_xlim=self.zoom_axis, lineflag=1)
            # a21.set_ylim(0, 0.2)

            for keys, data in DataFrame.iteritems():
                if keys.find('Before') < 0 and keys.find('Time') < 0 and keys.find('clip_speed') < 0:
                    Current, bin_edges = np.histogram(data, bins=bin_edges1, normed=True)
                    diff = Current - Before
                    # smootheddiff = self.smooth_hanning(diff, 3)
                    ax2.plot(bin_edges1[:-1], diff, alpha=0.6, linewidth=3, color=self.plotcolors[keys])
                    plot_higher_speed = [index for index, data1 in enumerate(bin_edges1) if
                                         self.zoom_axis[1] + 1 >= data1 >= self.zoom_axis[0] - 1]
                    a21.plot(bin_edges1[plot_higher_speed], diff[plot_higher_speed],
                             color=self.plotcolors[keys], alpha=0.6, linewidth=3)
            ax1.set_xlim((-0.01, self.zoom_axis[1]))
            ax2.axhline(0, color='gray')
            ax2.locator_params(axis='y', nbins=6)
            ax2.set_xlabel('Speed (mm/s)')
            ax2.set_ylabel('Difference with Before')

            # Plot cdf
            ax3 = fs1.add_subplot(gs[2, 0])
            for keys, data in DataFrame.iteritems():
                if keys.find('Time') < 0 and keys.find('clip_speed') < 0:
                    n, bins, patches = plt.hist(data, 1000, normed=True, histtype='step', cumulative=True,
                                                lw=2, color=self.plotcolors[keys], label=keys)

            ## Display KStest results for cumulative histogram
            cdf_A1 = []
            cdfkeys = []
            # fig2 = plt.figure()
            for keys, data in DataFrame.iteritems():
                if keys.find('Time') < 0 and keys.find('clip_speed') < 0:
                    countsA1, bin_edges = np.histogram(data, bins=500, range=[0, 10])
                    print np.shape(countsA1)
                    cdf_A1.append(np.cumsum(countsA1))
                    cdfkeys.append(keys)
            # plt.plot(bin_edges[1:], np.cumsum(countsA1), label =keys)
            # plt.legend()
            # fig2.show()

            for k1, cdf1 in zip(cdfkeys, cdf_A1):
                for k2, cdf2 in zip(cdfkeys, cdf_A1):
                    if k1 != k2:
                        t, p = stats.ks_2samp(cdf1, cdf2)
                        print "%s : KS test of %s and %s is p=%0.9f, t=%0.6f" % (self.StimulusName, k1, k2, p, t)

            ax3.grid(True)
            ax3.set_ylim(0, 1.05)
            ax3.set_xlim((0.1, self.zoom_axis[1]))
            ax3.locator_params(axis='y', nbins=6)
            ax3.set_xlabel('Speed (mm/s)')
            ax3.set_ylabel('Cumulative histogram')

            # Plot boxplot - count of slow speed per animal
            # Find count per timebin
            ax4 = fs1.add_subplot(gs[2, 1])
            CountSlowSpeed, BoxplotPalette = self.find_slowspeedcount(DataFrame, SlowSpeedDataFrame)
            # print CountSlowSpeed
            sns.boxplot(data=(CountSlowSpeed / 600) * 100,
                        order=['Before', '10minutes', '20minutes'])

            sns.swarmplot(data=(CountSlowSpeed / 600) * 100, linewidth=1, alpha=0.5,
                          order=['Before', '10minutes', '20minutes'])
            x = [0, 1, 2]
            dfcount = CountSlowSpeed.as_matrix()
            dfcount = (dfcount[:, [2, 0, 1]] / 600.0) * 100
            for ii in xrange(0, np.size(dfcount, 0)):
                ax4.plot(x, dfcount[ii, :], '-', linewidth=0.5, color='gray')

            plt.xticks(rotation=90)
            ax4.set_ylabel('Percentage time \n with speed < %0.1f mm/s' % self.slow_speed, fontsize=12)
            ax4.set_ylim(0, 100)
            ax4.locator_params(axis='y', nbins=4)

            ax5 = fs1.add_subplot(gs[2, 2])
            for keys in CountSlowSpeed.keys():
                if keys.find('Before') < 0:
                    sns.regplot(x="Before", y=keys, data=CountSlowSpeed,
                                color=self.plotcolors[keys], scatter=True, ci=None, marker='+', scatter_kws={"s": 40})
            ax5.locator_params(axis='x', nbins=6)
            ax5.locator_params(axis='y', nbins=6)
            # ax5.set_ylim(0, 800)
            # ax5.set_xlim((0, 800))
            ax5.set_xlabel("Before odor \n with speed < %0.1f mm/s" % self.slow_speed)
            ax5.set_ylabel('')

            self.pp.savefig(bbox_inches='tight')
            plt.close(fs1)

            return CountSlowSpeed

    def plot_ypos_for_timebin(self, CentroidDataFrame, MeanDataFrame, columnflag=1):
        # Second figure
        with sns.axes_style('dark'):
            fs2 = plt.figure(figsize=(10, 8))
            gs = plt.GridSpec(3, 1)
            xlabelinmin = self.convert_frames_to_minutes(np.zeros(30000))
            plotmeanbefore = True
            plotmean10 = True
            plotmean20 = True

            for keys, data in CentroidDataFrame.iteritems():
                data = self.convert_distance_to_mm(data)
                if keys.find('AllBeforeStimulus') > 0:
                    ax1 = fs2.add_subplot(gs[0, :])
                    ax1.plot(xlabelinmin, data[columnflag, :30001], color=self.plotcolors['Before'], lw=3, alpha=0.5)
                    ax1.plot(xlabelinmin, MeanDataFrame['meanBefore'], color='k', lw=3)

                    if plotmeanbefore:
                        ax1.fill_between(xlabelinmin, MeanDataFrame['meanBefore'] - MeanDataFrame['errorBefore'],
                                         MeanDataFrame['meanBefore'] + MeanDataFrame['errorBefore'], color='gray')
                        ax1.set_title('Before')
                        plotmeanbefore = False

                elif keys.find('10minutes') > 0:
                    ax1 = fs2.add_subplot(gs[1, :])
                    ax1.plot(xlabelinmin, data[columnflag, :30001], color=self.plotcolors['10minutes'], lw=3, alpha=0.5)
                    ax1.plot(xlabelinmin, MeanDataFrame['mean10minutes'], color='k', lw=3)

                    if plotmean10:
                        ax1.fill_between(xlabelinmin, MeanDataFrame['mean10minutes'] - MeanDataFrame['error10minutes'],
                                         MeanDataFrame['mean10minutes'] + MeanDataFrame['error10minutes'], color='gray')
                        ax1.set_title('10 minutes')
                        plotmean10 = False

                elif keys.find('20minutes') > 0:
                    ax1 = fs2.add_subplot(gs[2, :])
                    ax1.plot(xlabelinmin, data[columnflag, :30001], color=self.plotcolors['20minutes'], lw=3, alpha=0.5)
                    ax1.plot(xlabelinmin, MeanDataFrame['mean20minutes'], color='k', lw=3)
                    if plotmean20:
                        ax1.fill_between(xlabelinmin, MeanDataFrame['mean20minutes'] - MeanDataFrame['error20minutes'],
                                         MeanDataFrame['mean20minutes'] + MeanDataFrame['error20minutes'], color='gray')
                        ax1.set_title('20 minutes')
                        plotmean20 = False

                ax1.set_xlabel('Time (minutes)')
                ax1.set_ylabel('Y (mm)')
                ax1.set_xlim((0, 8))
                ax1.set_ylim((0, 400 * self.pixel_to_mm))
                ax1.locator_params(axis='y', nbins=4)

            plt.tight_layout()
            self.pp.savefig(bbox_inches='tight')
            plt.close(fs2)

    def plot_spatialmaps_as_heatmap(self, X, Y):
        ## Spatial maps as heatmaps
        fs3 = plt.figure(figsize=(10, 5))
        gs = plt.GridSpec(1, len(self.axiscount)+1, width_ratios=self.width_ratios)

        cmap = plt.cm.plasma
        cmap.set_under(color='black')

        for keys, data in Y.iteritems():
            create_image_fromcentroid = np.zeros((self.xlimittank, self.ylimittank))
            a = np.vstack((data.astype(int), X[keys].astype(int)))

            for ii in xrange(0, np.size(a, 1)):
                create_image_fromcentroid[a[0, ii], a[1, ii]] += 1
            create_image_fromcentroid = (create_image_fromcentroid / self.framerate) * 100

            ## Plot as heatmap
            blur = cv2.blur(create_image_fromcentroid, (3, 3))
            ax1 = fs3.add_subplot(gs[0, self.axiscount.index(keys)])
            plt.imshow(blur, cmap=cmap, interpolation='sinc', aspect='auto', vmax=500, vmin=0.1)
            if self.axiscount.index(keys) == len(self.axiscount):
                ax2 = fs3.add_subplot(gs[0, 3])
                cbar = plt.colorbar(cax=ax2, ax=ax1, ticks=range(0, 500, 1000))
                cbar.ax.set_ylabel('Time (ms)')

            ax1.set_title(keys)
            ax1.set_ylim((self.xlimittank, 0))
            ax1.set_xlim((0, self.ylimittank))
            ax1.axis('off')
            ax1.grid('off')

        plt.tight_layout()
        self.pp.savefig(bbox_inches='tight')
        plt.close(fs3)

    def plot_centroidduringslowspeedbouts(self, Speed, Centroid):
        fs4 = plt.figure(figsize=(10, 5))
        gs = plt.GridSpec(1, len(self.axiscount))

        for keys, data in Speed.iteritems():
            if keys.find('Time') < 0 and keys.find('clip_speed') < 0:
                low_speed = np.where(data < self.slow_speed)[0]
                create_image_fromcentroid = np.zeros((self.xlimittank, self.ylimittank))
                for ii in low_speed:
                    a = Centroid[keys][ii, :].astype(int)
                    create_image_fromcentroid[a[1], a[0]] += 1
                create_image_fromcentroid = (create_image_fromcentroid / self.framerate) * 10

                markersizearray = []
                for ii in low_speed:
                    a = Centroid[keys][ii, :].astype(int)
                    markersizearray.append(create_image_fromcentroid[a[1], a[0]])
                markersizearray = np.asarray(markersizearray) ** 2

                ax3 = fs4.add_subplot(gs[0, self.axiscount.index(keys)])
                # plt.imshow(create_image_fromcentroid, cmap='plasma', interpolation='sinc', aspect='auto')
                ax3.scatter(Centroid[keys][low_speed, 0], Centroid[keys][low_speed, 1],
                            s=markersizearray, facecolors='none',
                            edgecolors=self.edgecolors[self.axiscount.index(keys)], lw=2)
                ax3.set_aspect('equal')
                ax3.set_ylim((self.xlimittank, 0))
                ax3.set_xlim((0, self.ylimittank))
                # ax3.axis('off')
                plt.xticks([])
                plt.yticks([])
                ax3.grid('off')
        # plt.tight_layout()
        self.pp.savefig(bbox_inches='tight')
        plt.close(fs4)

    def plot_centroids_duringslowspeed_perfish(self, Speed, Centroid):
        # Get fish name and assign unique color to each
        fs5 = plt.figure(figsize=(10, 5))
        gs = plt.GridSpec(2, len(self.axiscount), height_ratios=[2, 0.05])

        name = []

        for keys in Speed.iterkeys():
            if keys.find('Timeinseconds') < 0:
                fishname = keys[keys.find('Fish'):keys.find('Fish') + keys[keys.find('Fish'):].find('_')]
                tankname = keys[keys.find('Tank'):keys.find('Tank') + 5]
                name.append(fishname + '_' + tankname)
        fishnames = list(set(name))
        color_palette = sns.color_palette("gist_ncar", np.size(fishnames))
        color_per_fish = dict(zip(fishnames, color_palette))

        ax1 = fs5.add_subplot(gs[1, 1:])
        gradient = np.linspace(0, np.size(fishnames), np.size(fishnames))
        gradient = np.vstack((gradient, gradient))
        color_palette = matplotlib.colors.ListedColormap(color_palette)
        ax1.imshow(gradient, aspect='auto', cmap=color_palette)
        ax1.get_yaxis().set_ticks([])
        ax1.set_xlabel('Fish Number')
        ax1.grid('off')

        for keys, data in Speed.iteritems():
            if keys.find('Timeinseconds') < 0:
                fishname = keys[keys.find('Fish'):keys.find('Fish') + keys[keys.find('Fish'):].find('_')]
                tankname = keys[keys.find('Tank'):keys.find('Tank') + 5]
                name = fishname + '_' + tankname
                time_bin = keys[keys.find('Tank') + 6:]
                centroid_array = np.asarray(Centroid[keys])
                data_array = np.asarray(data)

                # print name, time_bin, np.shape(data_array), np.shape(centroid_array)

                low_speed = np.where(data_array < self.slow_speed)[0]
                create_image_fromcentroid = np.zeros((self.xlimittank, self.ylimittank))
                for ii in low_speed:
                    a = centroid_array[ii, :].astype(int)
                    create_image_fromcentroid[a[1], a[0]] += 1
                create_image_fromcentroid = (create_image_fromcentroid / self.framerate) * 10

                markersizearray = []
                for ii in low_speed:
                    a = centroid_array[ii, :].astype(int)
                    markersizearray.append(create_image_fromcentroid[a[1], a[0]])
                markersizearray = np.asarray(markersizearray) ** 2

                # if markersizearray != []:
                # print name, time_bin, np.max(markersizearray)
                axisindex = [index for index, ii in enumerate(self.axiscount) if time_bin.find(ii) >= 0]
                ax3 = fs5.add_subplot(gs[0, axisindex[0]])

                ax3.scatter(centroid_array[low_speed, 0], centroid_array[low_speed, 1],
                            s=markersizearray, facecolors='none',
                            edgecolors=color_per_fish[name], lw=2)
                ax3.set_aspect('equal')
                ax3.set_ylim((self.xlimittank, 0))
                ax3.set_xlim((0, self.ylimittank))
                # ax3.axis('off')
                # plt.xticks([])
                # plt.yticks([])
                ax3.grid('off')
        self.pp.savefig(bbox_inches='tight')
        plt.close(fs5)

    def find_slowspeedcount(self, DataFrame, SlowSpeedDataFrame):
        # Find number of events with slow speed
        CountDataFrame = collections.OrderedDict()
        BoxplotPalette = {}
        SlowSpeedDataFrame = collections.OrderedDict(sorted(SlowSpeedDataFrame.items()))

        for keys in DataFrame.iterkeys():
            # print 'BLAH', keys
            for slowkeys, count in SlowSpeedDataFrame.iteritems():
                # print slowkeys
                if slowkeys[20:].find(keys) > 20:  # Topwards the end
                    if keys in CountDataFrame:
                        CountDataFrame[keys].append(count)
                        # CountDataFrame[keys+'type'].append(slowkeys)
                    else:
                        CountDataFrame[keys] = [count]
                        # CountDataFrame[keys+'type'] = [slowkeys]
                        BoxplotPalette[keys] = self.plotcolors[keys]
        # print CountDataFrame
        CountDataFrame = collections.OrderedDict(sorted(CountDataFrame.items()))
        # print CountDataFrame
        BoxplotPalette = collections.OrderedDict(sorted(BoxplotPalette.items()))
        PandasDataFrame = pd.DataFrame()

        for keys, data in CountDataFrame.iteritems():
            if np.size(data) != len(self.beforeafterfiles):
                Nanlist = [np.NaN] * (len(self.beforeafterfiles) - np.size(data))
                data = data + Nanlist

            PandasDataFrame[keys] = np.asarray(data)
        return PandasDataFrame, BoxplotPalette

    def create_inset(self, insetaxis, bigaxis, mark_inset_loc1, mark_inset_loc2, insetaxis_xlim, lineflag=0):
        insetaxis.locator_params(axis='y', nbins=1)
        insetaxis.set_xlim((insetaxis_xlim[0], insetaxis_xlim[1]))
        plt.xticks(([]))
        if lineflag:
            insetaxis.axhline(0, color='gray')
            insetaxis.locator_params(axis='y', nbins=2)
        mark_inset(bigaxis, insetaxis, loc1=mark_inset_loc1, loc2=mark_inset_loc2, fc="none", ec="0.5")

    @staticmethod
    def statsmodels_univariate_kde(data, kernel, bw, gridsize, cut, clip):
        """Compute a univariate kernel density estimate using statsmodels."""
        if clip is None:
            clip = (-np.inf, np.inf)
        fft = kernel == "gau"
        kde = smnp.KDEUnivariate(data)
        kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
        grid, y = kde.support, kde.density
        return grid, y

    @staticmethod
    def smooth_hanning(x, window_len, window='hanning'):
        s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = np.hanning(window_len)
        y = np.convolve(w / w.sum(), s, mode='valid')
        return y[:-window_len + 1]

    def convert_frames_to_minutes(self, array):
        ## Convert frames to seconds
        time = 1.0 / self.framerate
        n = np.size(array)
        label_in_seconds = np.linspace(0, n * time, n + 1) / 60
        return label_in_seconds
