import re
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


class Aiming:
    def __init__(self, filePath):
        """ Initialize Aiming """
        self._data = self.extractData(filePath)
        self._radii = self.getAllRadii()

    @staticmethod
    def extractData(filePath):
        """ Used to extract aiming data from a file """
        lines = np.array([])
        numberOfRows = int()
        with open(filePath, 'r') as f:
            line = f.readline()
            numberOfColumns =  len(line.split(','))
            while line:
                numberOfRows += 1
                lines = np.append(lines, line.split(','))
                line = f.readline()

        lines.resize((numberOfRows, numberOfColumns))

        data = dict()
        for i in range(0, numberOfColumns-1, 3):
            xRow = list(filter(None, lines[:, i]))
            yRow = list(filter(None, lines[:, i+1]))
            sessionID = int(re.findall(r'TrainingSession\[(\d+)\]', xRow[0])[0])
            if not sessionID in data:
                data[sessionID] = {'X': list(), 'Y': list()}
            data[sessionID]['X'].append([float(j)*100 for j in xRow[1:]])
            data[sessionID]['Y'].append([float(j)*100 for j in yRow])
        return data

    def getMeanRadius(self, sessionId):
        """ Used to calculate the mean radius for all hits in one session """
        radii = list()
        for xs, ys in zip(self._data[sessionId]['X'], self._data[sessionId]['Y']):
            longestDistance = int()
            for x1, y1 in zip(xs, ys):
                for x2, y2 in zip(xs, ys):
                    xDiff = x2 - x1
                    yDiff = y2 - y1
                    distance = sqrt(yDiff**2 + xDiff**2)
                    longestDistance = distance if longestDistance < distance else longestDistance
            radii.append(longestDistance/2)
        return np.mean(radii)/2

    def getAllRadii(self):
        """ Used to get all radii """
        radii = list()
        for k, v in self._data.items():
            radii.append(self.getMeanRadius(k))
        return radii

    def plotMovementPaths(self):
        """ Used to plot the paths """
        for k, v in self._data.items():
            for i in range(len(v['X'])):
                plt.gcf().canvas.set_window_title('Session{} Hit{}'.format(k, i+1))
                plt.title('Session{} Hit{}'.format(k, i+1))
                plt.plot(v['X'][i], v['Y'][i])
                plt.scatter([-25, 0, 25, 0], [0, 25, 0, -25])
                plt.scatter(0, 0, c='r')
                plt.show()

    def barRadii(self):
        """ Used to create a histogram of the radii """
        plt.bar([i for i in range(len(self._radii))], self._radii)
        plt.show()

    def exportData(self, filename='aiming'):
        """ Used to export the data """
        with open(filename, 'w') as f:
            for radius in self.getAllRadii():
                f.write(str(radius) + '\n')


if __name__ == '__main__':
    aiming = Aiming('Data/aiming1sec.csv')
    # aiming.exportData('Output/aiming')
    # aiming.plotMovementPaths()
    # aiming.barRadii()
