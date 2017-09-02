import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
import collections
from matplotlib.backends.backend_pdf import PdfPages
import math


class CollectParameter(object):
    def __init__(self, DirectoryName, pixel_to_mm, histtimebin):

        self.DirectoryName = DirectoryName
        self.FiguresFolder = os.path.join(self.DirectoryName, 'Figures')
        self.DictFolder = os.path.join(self.DirectoryName, 'DictFileforconditions')
        self.pp = PdfPages(os.path.join(self.FiguresFolder, 'SwimParameterFigures.pdf'))
        self.pixel_to_mm = pixel_to_mm
        self.histtimebin = histtimebin

        ## Combine dict from different files if they exist according to tank
        TankAnpy, TankBnpy = self.get_npyfiles()
        self.TankAdicts, self.TankABefAft = self.load_dict_and_combine(Tanknpy=TankAnpy)
        self.TankBdicts, self.TankBBefAft = self.load_dict_and_combine(Tanknpy=TankBnpy)

        self.stimulusA, self.stimulusB, self.fishA, self.fishB, self.framerate = [], [], [], [], []
        self.getexperimentparameters()

        self.iterate_and_find_speed(TankA=self.TankAdicts, TankB=self.TankBdicts)
        self.iterate_and_find_speed(TankA=self.TankABefAft, TankB=self.TankBBefAft)

        for ii in [5, 10, 15]:
            # Plot different timebins for speed after stimulus
            newdictA, newdictB = self.divide_after_stimulus_intobins(self.TankABefAft, self.TankBBefAft, timebins=ii)
            self.iterate_and_find_speed(TankA=newdictA, TankB=newdictB)

        self.pp.close()

    def get_npyfiles(self):
        npyfilenames = [f for f in os.listdir(self.DictFolder) if f.endswith('.npy')]
        TankAnpy = [ii for ii in npyfilenames if ii.find('TankA') > 0]
        TankBnpy = [ii for ii in npyfilenames if ii.find('TankB') > 0]
        return TankAnpy, TankBnpy

    def load_dict_and_combine(self, Tanknpy):
        # Combine dict, raise error if the same key exists. Fix it in that case
        new_dictfile = {}
        dictfile_before_after = []

        for ii in Tanknpy:
            if ii.find('Before') == 0:
                dictfile_before_after = np.load(os.path.join(self.DictFolder, ii))[()]
            else:
                dictfile = np.load(os.path.join(self.DictFolder, ii))[()]
                print dictfile.keys()
                if any(k in dictfile for k in new_dictfile) and any(k in new_dictfile for k in dictfile):
                    raise Exception("Check Dict Labels")
                else:
                    new_dictfile.update(dictfile)
        # print new_dictfile.keys()
        new_dictfile = collections.OrderedDict(sorted(new_dictfile.items()))
        # rename keys
        if dictfile_before_after == []:
            raise Exception("Check Before and After Dict Labels")
        else:
            return new_dictfile, dictfile_before_after

    def divide_after_stimulus_intobins(self, TankA, TankB, timebins):
        framerate = int(np.round(self.framerate))
        new_dictA = {}
        new_dictB = {}
        count = timebins
        for keys, dataA in TankA.iteritems():
            print 'Tank distance dataframe size', np.shape(dataA)
            if keys.find('Before') > 0:
                new_dictA[keys] = dataA
                new_dictB[keys] = TankB[keys]
            else:
                for kk in xrange(0, np.size(dataA, 1) + 1, framerate * (timebins * 60)):  # 10 minutes = 600 seconds
                    newkey = str(count) + 'minutes_' + keys
                    count += timebins
                    new_dictA[newkey] = dataA[:, kk:kk + framerate * (timebins * 60)]
                    new_dictB[newkey] = TankB[keys][:, kk:kk + framerate * (timebins * 60)]
        new_dictA = collections.OrderedDict(sorted(new_dictA.items()))
        new_dictB = collections.OrderedDict(sorted(new_dictB.items()))
        # print 'New Dictionary Keys, ', new_dictA.keys()
        return new_dictA, new_dictB

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

    def iterate_and_find_speed(self, TankA, TankB):
        framerate = int(np.round(self.framerate))
        subplot_number = math.ceil(len(TankA.keys()) / 2.)

        DataFrameA = []  # Combine all speed to one dataframe
        DataFrameB = []

        for keys, dataA in TankA.iteritems():
            count = 1
            time = []
            distance_insecondsA = []
            distance_insecondsB = []
            dataB = self.convert_distance_to_mm(TankB[keys])
            dataA = self.convert_distance_to_mm(dataA)

            for kk in xrange(0, np.size(dataA, 1) + 1, framerate):
                distance_insecondsA.append(np.sum(dataA[3, kk:kk + framerate]))
                distance_insecondsB.append(np.sum(dataB[3, kk:kk + framerate]))
                time.append(count)
                count += 1

            DataFrameA.append(distance_insecondsA)
            DataFrameB.append(distance_insecondsB)

        self.boxplot_of_the_stimuli(TankA.keys(), DataFrameA, DataFrameB)
        self.plot_cumsum_ofspeed(TankA.keys(), DataFrameA, DataFrameB)
        self.plot_pdf_ofspeed(TankA.keys(), DataFrameA, DataFrameB)
        self.plot_cumulative_histogram(TankA.keys(), DataFrameA, DataFrameB)

    def boxplot_of_the_stimuli(self, TimeBinType, DataFrameA, DataFrameB):
        # Plot All together
        fs2 = plt.figure(figsize=(15, 8))
        ax1 = fs2.add_subplot(1, 2, 1)
        ax1 = sns.boxplot(data=DataFrameA, linewidth=2)
        ax1 = sns.stripplot(data=DataFrameA, jitter=True, linewidth=1, alpha=0.5)
        ax1.set_xticklabels(TimeBinType)
        ax1.set_ylabel('Speed (mm/s)')
        ax1.set_ylim((-3, 20))
        plt.xticks(rotation=90)
        plt.title(self.stimulusA)

        ax2 = fs2.add_subplot(1, 2, 2)
        sns.boxplot(data=DataFrameB)
        ax1 = sns.stripplot(data=DataFrameB, jitter=True, linewidth=1, alpha=0.5)
        ax2.set_ylim((-3, 20))
        ax2.set_ylabel('Speed (mm/s)')
        ax2.set_xticklabels(TimeBinType)
        plt.xticks(rotation=90)
        plt.title(self.stimulusB)

        plt.tight_layout()
        self.pp.savefig(bbox_inches='tight')
        plt.close(fs2)

    def plot_cumsum_ofspeed(self, TimeBinType, DataFrameA, DataFrameB):
        fs1 = plt.figure(figsize=(10, 5))
        gs = plt.GridSpec(1, 3)
        ax1 = fs1.add_subplot(gs[0, 0])
        ax2 = fs1.add_subplot(gs[0, 1])
        ax1.set_title(self.stimulusA)
        ax2.set_title(self.stimulusB)
        for idx, ii in enumerate(DataFrameA):
            values, base = np.histogram(ii, bins=self.histtimebin)
            cumulative = np.cumsum(values)
            ax1.plot(base[:-1], cumulative, label=TimeBinType[idx])

        for idx, ii in enumerate(DataFrameB):
            values, base = np.histogram(ii, bins=self.histtimebin)
            cumulative = np.cumsum(values)
            ax2.plot(base[:-1], cumulative, label=TimeBinType[idx])

        # Put a legend to the right of the current axis
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        self.pp.savefig(bbox_inches='tight')
        plt.close(fs1)

    def plot_pdf_ofspeed(self, TimeBinType, DataFrameA, DataFrameB):
        fs1 = plt.figure(figsize=(12, 5))
        gs = plt.GridSpec(1, 3)

        if len(TimeBinType) > 2:
            colors = sns.color_palette("husl", len(TimeBinType))
        else:
            colors = sns.color_palette()

        ax1 = fs1.add_subplot(gs[0, 0])
        for idx, ii in enumerate(DataFrameA):
            # kde = scipy.stats.gaussian_kde(ii)
            # t_range = np.linspace(-2, 30, 200)
            # ax1.plot(t_range, kde(t_range), lw=2, alpha=0.6, label=TimeBinType[idx], color=colors[idx])
            sns.distplot(ii, hist=False, color=colors[idx])
            # plt.xlim(-1, 20)

        ax2 = fs1.add_subplot(gs[0, 1])
        for idx, ii in enumerate(DataFrameB):
            sns.distplot(ii, hist=False, color=colors[idx], label=TimeBinType[idx])
            # kde = scipy.stats.gaussian_kde(ii)
            # t_range = np.linspace(-2, 30, 200)
            # ax2.plot(t_range, kde(t_range), lw=2, alpha=0.6, label=TimeBinType[idx], color=colors[idx])
            # plt.xlim(-1, 20)
        ax1.set_title(self.stimulusA)
        ax2.set_title(self.stimulusB)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        self.pp.savefig(bbox_inches='tight')
        plt.close(fs1)

    def plot_cumulative_histogram(self, TimeBinType, DataFrameA, DataFrameB):

        fs1 = plt.figure(figsize=(10, 10))
        gs = plt.GridSpec(1, 3)

        if len(TimeBinType) > 2:
            colors = sns.color_palette("husl", len(TimeBinType))
        else:
            colors = sns.color_palette()

        ax1 = fs1.add_subplot(gs[0, 0])
        for idx, ii in enumerate(DataFrameA):
            mu = np.mean(ii)
            sigma = np.std(ii)
            n, bins, patches = plt.hist(ii, self.histtimebin, normed=1,
                                        histtype='step', cumulative=True, lw=2, color=colors[idx],
                                        label=TimeBinType[idx])
            plt.grid(True)
            plt.ylim(0, 1.05)

        ax2 = fs1.add_subplot(gs[0, 1])
        for idx, ii in enumerate(DataFrameB):
            mu = np.mean(ii)
            sigma = np.std(ii)
            n, bins, patches = plt.hist(ii, self.histtimebin, normed=1,
                                        histtype='step', cumulative=True, lw=2, color=colors[idx],
                                        label=TimeBinType[idx])

            plt.grid(True)
            plt.ylim(0, 1.05)

        ax1.set_title(self.stimulusA)
        ax2.set_title(self.stimulusB)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        self.pp.savefig(bbox_inches='tight')
        plt.close(fs1)

    def convert_distance_to_mm(self, distance):
        distance = distance * self.pixel_to_mm
        return distance
