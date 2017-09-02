import cv2
import os
import numpy as np

refPt = []
cropping = False


class DrawROI(object):
    def __init__(self, ResultFolder, windowname, frame, updateROIoffirstframe=False):
        self.selectroiwindowname = windowname
        self.frame = frame
        self.refPt = []
        self.cropping = False
        self.ResultFolder = ResultFolder
        self.updateROIoffirstframe = updateROIoffirstframe

        if self.updateROIoffirstframe:
            self.num_regions = 4
            self.ROIfilename = 'ROIs2'
        else:
            self.num_regions = 10
            self.ROIfilename = 'ROIs'

    def drawROIfortank(self):
        self.displayROIorder()  # Display what each ROI means

        # Clone frame and wait for mouse clicks
        clone = self.frame.copy()
        cv2.namedWindow(self.selectroiwindowname)
        cv2.setMouseCallback(self.selectroiwindowname, self.click_and_crop)

        while True:
            cv2.imshow(self.selectroiwindowname, self.frame)  # Display frame
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region, and all points
            if key == ord("r"):
                self.frame = clone.copy()
                self.refPt = []

            # if the 'c' key is pressed, crop the region
            elif key == ord("c"):
                if len(self.refPt) == self.num_regions:  # There needs to be 10 regions
                    print 'Cropped'
                else:
                    print "Select correct number of ROIs !!!!"
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print 'Saving'
                rect_coordinate = self.convert_rectcoordinates()
                cv2.imwrite(os.path.join(self.ResultFolder, self.ROIfilename + '.tif'),
                            self.frame)  # Save cropped areas
                np.save(os.path.join(self.ResultFolder, self.ROIfilename + '.npy'),
                        rect_coordinate)  # Save regions to numpy file
                return rect_coordinate

    def displayROIorder(self):
        if self.updateROIoffirstframe:
            print 'Update where the fish is in the first frame'
            print '1. Location of Fish B'
            print '2. Location of Fish A'

        else:

            print '1. FishTank B on the left [0, 4]'
            print '2. FishTank A on the right [2, 3]'
            print '3. Location of Fish B [4, 5]'
            print '4. Location of Fish A [6, 7]'
            print '5. LED ROI [8, 9]'

    def click_and_crop(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt.append((x, y))
            print self.refPt
            self.cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))
            self.cropping = False
            print 'Finished %d Number of ROI' % (len(self.refPt) - 1)

            cv2.rectangle(self.frame, self.refPt[-2], self.refPt[-1], (0, 0, 255), 2)  # Draw rectangle on the image
            cv2.imshow(self.selectroiwindowname, self.frame)
            print self.refPt
            # Make sure the fish tank boxes are same height
            if not self.updateROIoffirstframe:
                if len(self.refPt) == 4:
                    self.refPt[2] = (self.refPt[1][0], self.refPt[0][1])
                    self.refPt[3] = (self.refPt[3][0], self.refPt[1][1])
                    cv2.rectangle(self.frame, self.refPt[0], self.refPt[1], 0.5, 5)
                    cv2.rectangle(self.frame, self.refPt[2], self.refPt[3], 0.5, 5)

    def convert_rectcoordinates(self):
        # All rectangular coordinates are saved as x,y,w, h
        newrefpt = []
        count = 0
        for ii in xrange(0, len(self.refPt), 2):
            x = self.refPt[ii][0]
            y = self.refPt[ii][1]
            w = np.abs(self.refPt[ii][0] - self.refPt[ii + 1][0])
            h = np.abs(self.refPt[ii][1] - self.refPt[ii + 1][1])
            newrefpt.append((x, y, w, h))
            print self.refPt[ii:ii + 2], newrefpt[count]
            count += 1
        if self.updateROIoffirstframe:
            ##Open the previous ROI and gather all the other Rois and save in new file.
            refpt_oldfile = np.load(os.path.join(self.ResultFolder, 'ROIs.npy'))
            refpt_newfile = refpt_oldfile.copy()
            refpt_newfile[2:4, :] = newrefpt

            print refpt_oldfile
            print refpt_newfile
            return refpt_newfile
        else:
            return newrefpt
