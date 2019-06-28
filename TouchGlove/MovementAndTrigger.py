from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

figureNumber = 0

class MovementAndTrigger:
    def __init__(self, pathToFile):
        self._time = list()
        self._pressure = list()
        self._acceleration = list()

        self._rawAcc = list()
        self._rawGyro = list()
        self._rawPressure = list()

        self.extractRawDataFromFile(pathToFile)
        self._shotTimes = self.getShotTimes()
        self.extractDataBeforeShots()

    @staticmethod
    def setFigureName(figureName):
        plt.gcf().canvas.set_window_title(figureName)
        plt.title(figureName)

    @staticmethod
    def readFile(filePath):
        with open(filePath, 'r') as f:
            line = f.readline()
            lines = list()
            while line:
                lines.append(list(map(float, line.split(' '))))
                line = f.readline()
        return lines


    def extractRawDataFromFile(self, filePath):
        rawData = np.array(self.readFile(filePath))
        self._time = rawData[:, 0]
        self._rawPressure = rawData[:, 10]

        # Offset values is set based from data when the weapon is static
        accXOffset = 11908 #rawData[0, 1]
        accYOffset = 7354  #rawData[0, 2]
        accZOffset = 7354  #rawData[0, 3]
        for accX, accY, accZ in zip(rawData[:, 1], rawData[:, 2], rawData[:, 3]):
            self._rawAcc.append((accX-accXOffset, accY-accYOffset, accZ-accZOffset))

    def getShotTimes(self, threshold=2000):
        shotTimes = list()
        foundShot = False

        for i in range(len(self._rawAcc)-1):
            time = self._time[i]
            roll = self._rawAcc[i][1]
            nextRoll = self._rawAcc[i+1][1]
            if roll > threshold or roll < -threshold:
                if roll > 0:
                    if roll > nextRoll and not foundShot:
                        if len(shotTimes) == 0:
                            shotTimes.append(time)
                            foundShot = True
                        elif shotTimes[len(shotTimes)-1] + 1000 < time:
                            shotTimes.append(time)
                            foundShot = True
                else:
                    if roll < nextRoll and not foundShot:
                        if len(shotTimes) == 0:
                            shotTimes.append(time)
                            foundShot = True
                        elif shotTimes[len(shotTimes)-1] + 1000 < time:
                            shotTimes.append(time)
                            foundShot = True
            else:
                foundShot = False
        return shotTimes

    def extractDataBeforeShots(self, msBeforeShot=1000):
        shotIndex = int()
        self._pressure.append(list())
        self._acceleration.append(list())
        for pressure, acc, time in zip(self._rawPressure, self._rawAcc, self._time):
            if self._shotTimes[shotIndex] - msBeforeShot < time < self._shotTimes[shotIndex]:
                self._pressure[shotIndex].append({'P':pressure, 'T':time})
                self._acceleration[shotIndex].append({'X':acc[0], 'Y':acc[1], 'Z':acc[2], 'T':time})
            elif len(self._pressure[shotIndex]) != 0:
                if len(self._shotTimes)-1 == shotIndex:
                    return
                self._pressure.append(list())
                self._acceleration.append(list())
                shotIndex += 1

    def getAccCurveLengths(self):
        meanLength = list()

        for acc in self._acceleration:
            length = {'X':0, 'Y':0, 'Z':0}
            for j in range(len(acc)-2):
                deltaX = acc[j+1]['X'] - acc[j]['X']
                deltaY = acc[j+1]['Y'] - acc[j]['Y']
                deltaZ = acc[j+1]['Z'] - acc[j]['Z']
                deltaT = acc[j+1]['T'] - acc[j]['T']

                length['X'] += sqrt(deltaX**2 + deltaT**2)
                length['Y'] += sqrt(deltaY**2 + deltaT**2)
                length['Z'] += sqrt(deltaZ**2 + deltaT**2)

            meanLength.append((length['X'] + length['Y'] + length['Z'])/3)
        return meanLength

    def getMeanPressureDerivatives(self):
        meanPressureDerivative = list()
        for pressure in self._pressure:
            pressureDerivatives = list()
            for i in range(len(pressure)-2):
                deltaP = pressure[i+1]['P'] - pressure[i]['P']
                deltaT = pressure[i+1]['T'] - pressure[i]['T']
                derivative = deltaP / deltaT      

                if derivative > 0:
                    pressureDerivatives.append(derivative)

            meanPressureDerivative.append(np.mean(pressureDerivatives))
        return meanPressureDerivative

    def createGraph(self, pressure=False, shotTimes=False):
        global figureNumber
        figure = plt.figure(figureNumber)
        figureNumber += 1
        self.setFigureName('Data Graph')

        legendNames = list()
        if pressure:
            legendNames.append('Pressure')
            times = list()
            pressures = list()
            for pressureX in self._pressure:
                for pressure in pressureX:
                    times.append(pressure[1])
                    pressures.append(pressure[0])
            plt.plot(times, pressures)

        if shotTimes:
            legendNames.append('Shot times')
            plt.scatter(self._shotTimes, [12000 for i in range(len(self._shotTimes))])

        plt.legend(legendNames, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=int(len(legendNames)/2))
        plt.subplots_adjust(bottom=0.20, top=0.9, right=0.99, left=0.1)
        figure.show()

    def visualizeMovement(self):
        global figureNumber
        figure = plt.figure(figureNumber)
        figureNumber += 1
        self.setFigureName('Movement')
        acc = self.getAccCurveLengths()
        plt.bar([i+0.25 for i in range(len(acc))], acc, 0.25)
        plt.legend(['AccScope'])
        figure.show()

    def visualizePressure(self):
        global figureNumber
        figure = plt.figure(figureNumber)
        figureNumber += 1
        self.setFigureName('Pressure')
        pressure = self.getMeanPressureDerivatives()
        plt.bar([i for i in range(len(pressure))], pressure, 0.25)
        plt.legend(['Pressure'])
        figure.show()

    def exportData(self, outputDirectory, first=False):
        openAs = 'w' if first else 'a'

        pressures = self.getMeanPressureDerivatives()
        with open('{}/trigger'.format(outputDirectory), openAs) as f:
            f.write( str(np.mean(pressures)) + '\n')

        acc = self.getAccCurveLengths()
        with open('{}/movement'.format(outputDirectory), openAs) as f:
            f.write( str(np.mean(acc)) + '\n')

def exportAllData(dataDirectoryName, outputDirectory):
    for i in range(1, 9):
        for j in range(1, 6):
            data = MovementAndTrigger('{}/{}{}.txt'.format(dataDirectoryName, i, j))
            data.exportData(outputDirectory, first=(i == 1 and j == 1))


if __name__ == '__main__':
    exportAllData('Data', 'Output')
