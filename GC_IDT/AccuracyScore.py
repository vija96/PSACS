import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, atan2, sqrt


class AccuracyScore:
    def __init__(self, filePath, data=None):
        """ Initialize AccuracyScore """
        self._allData = self.extractDataFromFile(filePath) if data is None else data

        self._plexiHits = list(map(bool, map(int, self._allData["PlexiHit"])))
        self._trainingSessionIds = list(map(int, self._allData["TrainingSessionId"]))

        self._impactPos = {'X': list(map(float, self._allData["ImpactPosRelativeX"])),
                           'Y': list(map(float, self._allData["ImpactPosRelativeY"]))}

        self._hitData = self.extractHitData()

        self.getCircleCenter(2)

    @staticmethod
    def extractDataFromFile(filePath):
        """ Used to extract data from a file """
        with open(filePath, 'r') as f:
            lines = list()
            line = f.readline()
            while line:
                lines.append(line.split(','))
                line = f.readline()

            data = dict()
            for i in range(len(lines[0])):
                aRow = [l[i] for l in lines]
                data.update({aRow[0]: aRow[1:]})
        return data

    def extractHitData(self):
        """ Used to extract data about the hits """
        hitData = {'X': dict(), 'Y': dict()}
        allXHits = dict()
        allYHits = dict()
        for sessionId, plexiHit, xHitPos, yHitPos in zip(self._trainingSessionIds, self._plexiHits, self._impactPos['X'], self._impactPos['Y']):
            if not plexiHit:
                if sessionId in allXHits:
                    allXHits[sessionId].append(-100 * xHitPos)
                    allYHits[sessionId].append((yHitPos - 1.3) * 100)
                else:
                    allXHits[sessionId] = [-100 * xHitPos]
                    allYHits[sessionId] = [(yHitPos - 1.3) * 100]

        for x, y in zip(allXHits.items(), allYHits.items()):
            # x/y[0] = session id, x/y[1] = list with hits
            hitData['X'][x[0]] = x[1]
            hitData['Y'][y[0]] = y[1]

        return hitData

    def getHit(self, sessionId=None):
        """ Used to get the hit data """
        return self._hitData if sessionId is None else self._hitData['X'][sessionId], self._hitData['Y'][sessionId]

    def plotHits(self, sessionId):
        """ Used to plot the hist in a specific session """
        x, y = self.getHit(sessionId)
        plt.scatter(x, y)
        plt.show()

    def getLongestDistance(self, sessionId):
        """ Used to get the distance between the two farthest hits """
        longestDistance = int()
        xs = self._hitData['X'][sessionId]
        ys = self._hitData['Y'][sessionId]
        for x1, y1 in zip(xs, ys):
            for x2, y2 in zip(xs, ys):
                xDiff = abs(x2 - x1)
                yDiff = abs(y2 - y1)
                distance = sqrt(yDiff**2 + xDiff**2)
                longestDistance = distance if longestDistance < distance else longestDistance
        return longestDistance

    def getRadius(self, sessionId):
        """ Used to get the radius of the hit group """
        return self.getLongestDistance(sessionId)/2

    def getCircleCenter(self, sessionId):
        """ Used to get the center of the hit group """
        centerPoint = list()
        longestDistance = int()
        xs = self._hitData['X'][sessionId]
        ys = self._hitData['Y'][sessionId]
        for x1, y1 in zip(xs, ys):
            for x2, y2 in zip(xs, ys ):
                xDiff = abs(x2-x1)
                yDiff = abs(y2-y1)
                distance = sqrt(yDiff**2 + xDiff**2)
                if longestDistance < distance:
                    longestDistance = distance
                    centerPoint = ((x1 + x2)/2, (y1 + y2)/2)
        return centerPoint

    def getMeanCenter(self, sessionId):
        """ Used to get the mean center of the hit group """
        return np.mean(self._hitData['X'][sessionId]), np.mean(self._hitData['Y'][sessionId])

    def getDistanceToCenter(self, sessionId=None, center=None, centerType='mean'):
        """ Used to get the distance to the group center """
        if center is None:
            center = self.getMeanCenter(sessionId) if centerType == 'mean' else self.getCircleCenter(sessionId)
        return sqrt(center[0]**2 + center[1]**2)

    def getAngle(self, sessionId=None, center=None):
        """ Used to get the angle of the group """
        if center is None:
            center = self.getMeanCenter(sessionId)
        return 2*pi+atan2(center[1], center[0])

    @staticmethod
    def getScore(r, d, a, numberOfHits):
        """ Used to calculate the score """
        r = 0 if r < 5 else r
        if d >= 5:
            if abs(sin(a)) < abs(cos(a)):
                return 2*(50-r-d)*abs(cos(a)) - 20*(5-numberOfHits)
            else:
               return 2*(50-r-d)*abs(sin(a)) - 20*(5-numberOfHits)
        else:
            return 2*(50-r) - 20*(5-numberOfHits)

    def getAllScores(self):
        """ Used to get all scores for all sessions """
        scores = list()
        for sessionId in list(dict.fromkeys(self._trainingSessionIds)):
            radius = self.getRadius(sessionId)
            distanceToCenter = self.getDistanceToCenter(sessionId)
            angle = self.getAngle(sessionId)
            numberOfHits = len(self._hitData['X'][sessionId])
            scores.append(self.getScore(radius, distanceToCenter, angle, numberOfHits))
        return scores

    def exportAllScores(self, fileName='score'):
        """ Used to export all scores """
        with open(fileName, 'w') as f:
            for score in self.getAllScores():
                f.write(str(score) + '\n')


if __name__ == "__main__":
    AccuracyScore('Data/Hit.csv').exportAllScores('Output/score')
