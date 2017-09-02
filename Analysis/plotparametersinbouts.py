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
from mpl_toolkits.mplot3d import Axes3D
from numpy import convolve, r_, ones
from sklearn.decomposition import PCA as sklearnPCA


class CollectParameter(object):
    def __init__(self, DirectoryName, pixel_to_mm, analysistimebin, framerate, slow_speed):

        self.DirectoryName = DirectoryName
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

        self.StimulusNames = [ii for ii in os.listdir(DirectoryName) if
                              ii.find('.DS') < 0 and ii.find('Figures') < 0 and ii.find('Stats') < 0]
        self.CombinedTimeDataFrame = []
        for ii in self.StimulusNames[0:1]:
            self.ThisStimulus = ii
            self.beforeafterfiles = self.get_npyfiles(StimulusName=ii)

            self.TimeBinDataFrame, self.TimeBins = self.divide_by_time_bins(StimulusName=ii, timebins=analysistimebin)
            self.CombinedTimeDataFrame.append(
                self.concatenate_timebins_perstimulus(self.TimeBinDataFrame, timebins=analysistimebin))

        self.All, self.Celltype, self.numrowsperfish, self.numrowspertimebin = self.combineddata(
            self.CombinedTimeDataFrame)

        self.plotcolors = self.get_colors()
        # self.get_pca_fish_combined()

    def get_npyfiles(self, StimulusName):
        npyfilenames = [f for f in os.listdir(os.path.join(self.DirectoryName, StimulusName)) if
                        f.endswith('.npy')]

        return npyfilenames

    def divide_by_time_bins(self, StimulusName, timebins):
        new_dict = {}
        time_dict = []
        for ii in self.beforeafterfiles:
            count = timebins
            dictfile_before_after = np.load(os.path.join(self.DirectoryName, StimulusName, ii))[()]
            checkdata = np.size(dictfile_before_after['AllBeforeStimulus'][3, :self.framerate * (timebins * 60)])
            if checkdata == self.sizeofmatrix:
                for keys, data in dictfile_before_after.iteritems():
                    if keys.find('AllBeforeStimulus') >= 0:
                        new_dict[ii[:-4] + '_' + keys] = self.convert_distance_to_mm(
                            data[:, :self.framerate * (timebins * 60)])
                    else:
                        for kk in xrange(0, np.size(data, 1),
                                         self.framerate * (timebins * 60)):  # 10 minutes = 600 seconds
                            newkey = '%s_%2dminutes' % (ii[:-4], count)
                            time_dict.append(count)
                            new_dict[newkey] = self.convert_distance_to_mm(
                                data[:, kk:kk + self.framerate * (timebins * 60)])
                            count += timebins
                            if count == 30:
                                break

        return new_dict, np.unique(time_dict)

    def convert_distance_to_mm(self, distance):
        distance = distance * self.pixel_to_mm
        return distance

    def concatenate_timebins_perstimulus(self, BefAftDataFrame, timebins):
        CompiledDataFrame = {}
        for keys, data in BefAftDataFrame.iteritems():
            compileddata = data[3, 600:]
            if np.size(compileddata) == self.sizeofmatrix - 600:
                if keys.find('AllBeforeStimulus') > 0:
                    print keys
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

    def combineddata(self, CombinedDataFrame):
        CellType = {key: [] for key in ['StimulusType', 'TimeBin']}
        numberofcells_perfish = {key: 0 for key in self.StimulusNames}
        numberofcells_perfish_perkey = {}
        combineddata = np.array([])
        for ii in xrange(0, np.size(CombinedDataFrame)):
            for keys, data in CombinedDataFrame[ii].iteritems():
                if keys == 'Before':
                    CellType['StimulusType'].extend([self.StimulusNames[ii]] * np.size(data, 0))
                    CellType['TimeBin'].extend([keys] * np.size(data, 0))
                    numberofcells_perfish[self.StimulusNames[ii]] += np.size(data, 0)
                    numberofcells_perfish_perkey[self.StimulusNames[ii] + '_' + keys] = np.size(data, 0)
                    combineddata = np.vstack((combineddata, data)) if combineddata.size else data

        print np.shape(CellType['StimulusType'])
        CellType = pd.DataFrame.from_dict(CellType)
        print CellType.head()
        return combineddata, CellType, numberofcells_perfish, numberofcells_perfish_perkey

    def get_pca_fish_combined(self):
        fs = plt.figure(figsize=((8, 5)))
        sklearn_pca = sklearnPCA(n_components=3)
        Y_sklearn = sklearn_pca.fit_transform(self.All)

        Dataframe = pd.DataFrame({'PCA1': Y_sklearn[:, 0], 'PCA2': Y_sklearn[:, 1], 'PCA3': Y_sklearn[:, 2]})
        Dataframe = pd.concat([Dataframe, self.Celltype], axis=1)
        print Dataframe['PCA1'][0]

        ax1 = fs.add_subplot(121)
        for ii in xrange(0, np.size(Dataframe, 0)):
            ax1.scatter(Dataframe['PCA2'][ii], Dataframe['PCA3'][ii],
                        c=self.plotcolors[Dataframe['TimeBin'][ii]],
                        marker=self.plotcolors[Dataframe['StimulusType'][ii]], s=50, alpha=0.5)
        #
        # ax2 = fs.add_subplot(122)
        # for ii in xrange(0, np.size(Dataframe, 0)):
        #     ax2.scatter(Dataframe['PCA2'][ii], Dataframe['PCA3'][ii],
        #                 c=self.plotcolors[Dataframe['TimeBin'][ii]],
        #                 marker=self.plotcolors[Dataframe['StimulusType'][ii]], s=50, alpha=0.5)

        # pLot in 2d Space
        # count1 = 0
        # for ii in xrange(0, np.size(CombinedDataFrame)):
        #     stimulusname = self.StimulusNames[ii]
        #     for timekeys in CombinedDataFrame[ii].iterkeys():
        #         count2 = self.numrowspertimebin[stimulusname + '_' + timekeys]
        #         print stimulusname, timekeys, count1, count2


        plt.xlim((-50, 100))
        plt.ylim((-100, 100))
        plt.tight_layout()
        plt.show()

    def get_colors(self):
        colors = sns.color_palette('colorblind')
        plotcolor = {}
        plotcolor['Before'] = colors[0]
        plotcolor['After'] = colors[1]
        count = 1
        if self.TimeBins is not None:
            for ii in self.TimeBins:
                plotcolor['%2dminutes' % ii] = colors[count]
                count += 1
        marker = itertools.cycle(('.', 'o', '+', 's', '*'))
        for ii in self.StimulusNames:
            plotcolor[ii] = marker.next()
        print plotcolor

        return plotcolor

    @staticmethod
    def smooth_func(x, window_len=10):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        #         w = np.hanning(window_len)
        w = ones(window_len, 'd')
        y = convolve(w / w.sum(), s, mode='valid')
        # print 'Size of y...', shape(y)
        return y[window_len / 2:-window_len / 2 + 1]
