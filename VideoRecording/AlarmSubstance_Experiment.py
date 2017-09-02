import numpy as np
import os
from FindFrameRate import FPS
from CameraFunctions import RunCamera
import csv
import time

filesep = os.path.sep

if __name__ == '__main__':
    ## Pause for some minutes before starting
    time.sleep(300)

    imgwidth = 1200
    imgheight = 500

    numtestframes = 300

    ##Experiment times
    total_experiment_time = 3000 # in seconds
    stimulus_on = [900]  # In seconds
    stimulus_off = [906]  # in seconds

    ## Details for saving file
    # Write as valveA_valveB if different
    FishName = 'Fish199_Fish200'
    StimulusType = 'SS500ul_E3'
    FishType = 'ABWT'
    FishAge = '7dpf'
    SS_made_date = '08022017'
    Experiment_date = '17022017_1120'

    find_framerate = False
    display_duringconfig = True
    display_duringexp = True

    ####################################################################################################################
    ##MakeDir if not available
    ResultDirectory = os.path.join('D:\\Seetha_AlarmSubstance\\', FishName, 'Results')
    if not os.path.exists(ResultDirectory):
        os.makedirs(ResultDirectory)

    if find_framerate:
        ## Step1. Find FPS of Camera
        print 'Finding Frame Rate...'
        framerate = FPS(imgwidth, imgheight, numtestframes,
                        display=display_duringconfig).captureimages()  # Display to configure and then shut off for accurate frame rate estimation

        ## Save frame rate and other experiment parameters
        np.savetxt(os.path.join(ResultDirectory, 'EstimatedFrameRate.csv'), [framerate], fmt='%3.6f')

    else:
        List = [FishName, FishType, FishAge, StimulusType, 'RecordingTime' + str(total_experiment_time) + 's']
        filename_for_saving = '_'.join(List)
        print 'Saving file as... ', filename_for_saving

        # Step2: Start Camera and setup for recording

        if os.path.exists(os.path.join(ResultDirectory, 'EstimatedFrameRate.csv')):
            framerate = np.loadtxt(os.path.join(ResultDirectory,
                                                'EstimatedFrameRate.csv'))  # load estimated frame rate. If not present give error
        else:
            print ('FrameRate has not been calibrated !!!!')
            exit()

        RunCamera(fps=framerate, imgwidth=imgwidth, imgheight=imgheight, stimulus_on_time=stimulus_on,
                  stimulus_off_time=stimulus_off, total_experiment_time=total_experiment_time,
                  resultdirectory=ResultDirectory, savefilename=filename_for_saving,
                  display=display_duringexp).CaptureAndSaveFrames()

        # Save experiment parameters
        # Create Dictionary for saving
        itemDict = {}
        itemDict['Fishname'] = FishName
        itemDict['FishAge'] = FishAge
        itemDict['FishType'] = FishType
        itemDict['StimulusType'] = StimulusType
        itemDict['ImageWidth'] = imgwidth
        itemDict['ImageHeight'] = imgheight
        itemDict['FrameRate'] = framerate
        itemDict['StimulusONTime'] = stimulus_on
        itemDict['StimulusOFFTime'] = stimulus_off
        itemDict['TotalExperimentTime'] = total_experiment_time
        itemDict['SSMadeDate'] = SS_made_date
        itemDict['ExperimentDate'] = Experiment_date

        with open(os.path.join(ResultDirectory, 'ExperimentParameters.csv'), 'wb') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f)
            for key, value in itemDict.items():
                w.writerow([key, value])


