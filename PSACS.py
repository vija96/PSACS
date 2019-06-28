import numpy as np
from numpy import arange
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from GC_IDT.Movement import *
from GC_IDT.AccuracyScore import *
from TouchGlove.MovementAndTrigger import *
from Tools.ProgressBar import ProgressBar

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


class PSACS:
    @staticmethod
    def setUpGraphFont(font, size, fontFamily='sans-serif'):
        matplotlib.rc('font', **{'family'     : fontFamily,
                                 'sans-serif' : font,
                                 'size'       : size})

    def __init__(self, outputFolder='Output', shouldExportData=True):
        """ Initialize PSACS """
        self.setUpGraphFont('Times new roman', 18)

        if shouldExportData:
            prgBar = ProgressBar('Exporting data:', 3)

            exportAllData('TouchGlove/Data', outputFolder)
            print(prgBar, end='')

            Aiming('GC_IDT/Data/aiming1sec.csv').exportData(outputFolder + '/aiming')
            print(prgBar, end='')

            AccuracyScore('GC_IDT/Data/Hit.csv').exportAllScores(outputFolder + '/score')
            print(prgBar, end='')

        self._targetData = np.array([])
        self._featureData = np.array([])

        self._targetNames = np.array(["Accuracy score"])
        self._featureNames = np.array(["Heart rate", "Trigger pull force", "Rifle movement"])

        self._mlModel = RandomForestRegressor(bootstrap=True, max_depth=80,
                                              max_features=2, min_samples_leaf=5,
                                              min_samples_split=12, n_estimators=100)
        self._movements = list()
        self._heartRates = list()
        self._triggerPulls = list()
        self.loadFeatureData(outputFolder+'/heartRate', outputFolder+'/trigger', 
                             outputFolder+'/movement', outputFolder+'/aiming')

        self.loadTargetData(outputFolder+'/score')

        self._explainer = self.createExplainer()

    def loadFeatureData(self, heartRateFilePath, triggerFilePath, movementFilePath, aimingFilePath):
        """ Used to extract feature data from four given files """
        prgBar = ProgressBar('Loading feature data:', 5)

        with open(heartRateFilePath, 'r') as heartRateFile:
            self._heartRates = list(map(float, heartRateFile.read().splitlines()))
        print(prgBar, end='')

        with open(triggerFilePath, 'r') as triggerFile:
            self._triggerPulls = list(map(float, triggerFile.read().splitlines()))
        print(prgBar, end='')

        with open(movementFilePath, 'r') as movementFile:
            accelerations = list(map(float, movementFile.read().splitlines()))
        print(prgBar, end='')

        with open(aimingFilePath, 'r') as aimingFile:
            aimings = list(map(float, aimingFile.read().splitlines()))
        print(prgBar, end='')

        self._movements = [(movement + aiming)/2 for movement, aiming in zip(accelerations, aimings)]

        if len(self._heartRates) != len(self._triggerPulls) != len(self._movements):
            raise Exception("The lists is not equal in size!")

        for heartRate, trigger, movement in zip(self._heartRates, self._triggerPulls, self._movements):
            self._featureData = np.append(self._featureData, [heartRate, trigger, movement])
        self._featureData.resize((len(self._heartRates), 3))
        print(prgBar, end='')

    def loadTargetData(self, targetFilePath):
        """ Used to extract target data from a given file """
        prgBar = ProgressBar('Loading target data:', 1)
        
        with open(targetFilePath, 'r') as targetFile:
            self._targetData = np.array(list(map(float, targetFile.read().splitlines())))
        print(prgBar, end='')

    def trainModel(self):
        """ Used to train the model on feature and target data """
        self._mlModel.fit(self._featureData, self._targetData)

    def crossEvaluate(self, numOfFolds):
        """ Used to evaluate the model, returns MAE and MSE """
        MEA = cross_val_score(self._mlModel, self._featureData, self._targetData, 
                                cv=numOfFolds, scoring='neg_mean_absolute_error')
        MSE = cross_val_score(self._mlModel, self._featureData, self._targetData,
                                cv=numOfFolds, scoring='neg_mean_squared_error')
        return MEA, MSE
       
    def predict(self, inputData):
        """ Used to predict all data points in given list """
        return [self._mlModel.predict([[i[0], i[1], i[2]]])[0] for i in inputData]

    def setUp3DGraph(self):
        """ Used to setup a 3D graph """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(self._featureNames[0])
        ax.set_ylabel(self._featureNames[1])
        ax.set_zlabel(self._featureNames[2])

        # Create color bar
        cm = LinearSegmentedColormap.from_list('BlackToGreen', [(1, 0, 0), (0, 1, 0)])
        fig.colorbar(ax.plot_surface([0], [0], np.array([[0]]), cmap=cm), shrink=0.5, aspect=10)

        return ax

    def scatterRawData(self):
        """ Used to scatter the raw feature data in a 3D graph"""
        ax = self.setUp3DGraph()
        maxScore = max(self._targetData)
        colorList = list()

        for i in self._targetData:
            green = (i if i > 0 else 0.1) / maxScore
            colorList.append((1 - green, green, 0))
        
        ax.scatter(self._heartRates, self._triggerPulls, self._movements, c=colorList)
        plt.show()

    def histogramOfRawData(self, featureName):
        """ Used to display a histogram of the given feature """
        if featureName.lower() in {'heart rate', 'heartrate', 'hr'}:
            plt.ylabel('Heart rate')
            feature = self._heartRates
        elif featureName.lower() in {'trigger pull', 'triggerpull', 'tp'}:
            plt.ylabel('Trigger pull force')
            feature = self._triggerPulls
        elif featureName.lower() in {'rifle movement', 'riflemovement', 'rm'}:
            plt.ylabel('Rifle movement')
            feature = self._movements
        else:
            raise Exception('{} is not a valid feature name!'.format(featureName))

        plt.xlabel('Shooting exercise')
        plt.bar([i for i in range(len(feature))], feature, color='black')
        plt.show()

    def getMaxMinAndSteps(self, numberOfSteps):
        """ Used to get the min/max values of the features and a step """
        maximum = {'hr': max(self._heartRates),
                   'tp': max(self._triggerPulls),
                   'rm': max(self._movements)}

        minimum = {'hr': min(self._heartRates),
                   'tp': min(self._triggerPulls),
                   'rm': min(self._movements)}

        step = {'hr': (maximum['hr']-minimum['hr'])/numberOfSteps,
                'tp': (maximum['tp']-minimum['tp'])/numberOfSteps,
                'rm': (maximum['rm']-minimum['rm'])/numberOfSteps}

        return minimum, maximum, step

    def plotIsolatedFeaturePrediction(self, featureName, steps=1000):
        """ Used to illustrate how a single feature affect the predicted value  """
        plt.ylabel('Predicted accuracy score')

        minimum, maximum, step = self.getMaxMinAndSteps(steps)

        label = {'hr': 'Heart rate',
                 'tp': 'Trigger pull force',
                 'rm': 'Rifle movement'}

        lower = {'hr': [minimum['hr']]*steps,
                 'tp': [minimum['tp']]*steps,
                 'rm': [minimum['rm']]*steps}

        upper = {'hr': [maximum['hr']]*steps,
                 'tp': [maximum['tp']]*steps,
                 'rm': [maximum['rm']]*steps}

        if featureName.lower() in {'heart rate', 'heartrate', 'hr'}:
            isolatedFeature, unit, feature1, feature2 = 'hr', '%', 'tp', 'rm'
        elif featureName.lower() in {'trigger pull', 'triggerpull', 'tp'}:
            isolatedFeature, unit, feature1, feature2 = 'tp', '', 'hr', 'rm'
        elif featureName.lower() in {'rifle movement', 'riflemovement', 'rm'}:
            isolatedFeature, unit, feature1, feature2 = 'rm', '', 'hr', 'tp'
        else:
            raise('"{}" is not a valid feature name!'.format(featureName))
        
        plt.xlabel(label[isolatedFeature] + (' ({})'.format(unit) if unit!='' else ''))
        lower[isolatedFeature] = upper[isolatedFeature] = list(arange(minimum[isolatedFeature],
                                                                      maximum[isolatedFeature],
                                                                      step[isolatedFeature]))

        lowerLegend  = '{} = {}\n'.format(label[feature1], int(round(minimum[feature1])))
        lowerLegend += '{} = {}'.format(label[feature2],   int(round(minimum[feature2])))

        upperLegend  = '{} = {}\n'.format(label[feature1], int(round(maximum[feature1])))
        upperLegend += '{} = {}'.format(label[feature2],   int(round(maximum[feature2])))

        lowerPredictedScores = list()
        upperPredictedScores = list()
        prgBar = ProgressBar('Predicting lower bound:', steps)

        for lHeartRate, lTriggerPull, lRifleMovement, uHeartRate, uTriggerPull, uRifleMovement in zip(lower['hr'], lower['tp'], lower['rm'], upper['hr'], upper['tp'], upper['rm']):
            lowerPredictedScores.append( self.predict([ [lHeartRate, lTriggerPull, lRifleMovement] ]) )
            upperPredictedScores.append( self.predict([ [uHeartRate, uTriggerPull, uRifleMovement] ]) )
            print(prgBar, end='')

        plt.plot(lower[isolatedFeature], lowerPredictedScores, c='black')
        plt.plot(upper[isolatedFeature], upperPredictedScores, c='black')
        plt.legend([lowerLegend, upperLegend])
        plt.show()

    def plotAllIncreasingFeaturePredictions(self, steps=1000):
        """ Used to illustrate how the predicted value behave when all features are increasing"""
        plt.ylabel('Predicted accuracy score')

        fig = plt.figure()
        axM = fig.add_subplot(111)
        axH = axM.twiny()
        axT = axM.twiny()

        axM.set_ylabel('Predicted accuracy score')
        axM.set_xlabel('Movement')
        axH.set_xlabel('Heart rate')
        axT.set_xlabel('Trigger pull force')

        axH.spines['top'].set_position(('outward', -500))
        axT.spines['top'].set_position(('outward', -580))    

        minimum, maximum, step = self.getMaxMinAndSteps(steps)

        heartRates = list(arange(minimum['hr'], maximum['hr'], step['hr']))
        triggers   = list(arange(minimum['tp'], maximum['tp'], step['tp']))
        movements  = list(arange(minimum['rm'], maximum['rm'], step['rm']))

        predictedScores = list()
        pgBar = ProgressBar('Predicting data:', steps)
        for pulse, trigger, movement in zip(heartRates, triggers, movements):
            predictedScores.append( self.predict([ [pulse, trigger, movement] ]) )
            print(pgBar, end='')

        axM.plot(movements,  predictedScores, c='black')
        axH.plot(heartRates, predictedScores, c='black')
        axT.plot(triggers,   predictedScores, c='black')

        fig.subplots_adjust(bottom=0.39, top=1)
        fig.show()
        input('Press ENTER to continue')

    def getFeatureImportance(self):
        """ Returns a dictionary with the all feature importance in percent """
        featureImportance = self._mlModel.feature_importances_
        return {name: importance*100 for name, importance in zip(self._featureNames, featureImportance)}

    def performRandomizedSearch(self):
        """ Returns the results from the randomized search """
        from sklearn.model_selection import RandomizedSearchCV
        
        # Number of trees in the forest
        nEstimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 10)]

        # Number of features to consider at every split
        maxFeatures = ['auto', 'sqrt', 3]

        # Maximum number of levels in tree
        maxDepth = [int(x) for x in np.linspace(10, 110, num = 11)]
        maxDepth.append(None)

        # Minimum number of samples required to split a node
        minSamplesSplit = [2, 5, 10]

        # Minimum number of samples required at each leaf node
        minSamplesLeaf = [1, 2, 4]

        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        randomGrid = {'n_estimators': nEstimators,
                       'max_features': maxFeatures,
                       'max_depth': maxDepth,
                       'min_samples_split': minSamplesSplit,
                       'min_samples_leaf': minSamplesLeaf,
                       'bootstrap': bootstrap}

        rf = RandomForestRegressor()
        randomSearch = RandomizedSearchCV(estimator=rf, param_distributions=randomGrid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

        randomSearch.fit(self._featureData, self._targetData)
        return randomSearch.best_params_

    def performGridSearch(self):
        """ Returns the results from the grid search """
        from sklearn.model_selection import GridSearchCV
        
        # Create the parameter grid based on the results of random search 
        param_grid = {
            'bootstrap': [True, False],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3, 'auto'],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }
        # Instantiate the grid search model
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                   cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

        grid_search.fit(self._featureData, self._targetData)
        return grid_search.best_params_

    def createExplainer(self):
        """ Creates the LIME explainer """
        from lime.lime_tabular import LimeTabularExplainer
        
        return LimeTabularExplainer(self._featureData,
                                    mode="regression",
                                    feature_names=self._featureNames,
                                    class_names=self._targetNames)

    def explain(self, inputData, showInNotebook=False):
        """ Prints explanations on the inputData """

        if type(inputData) is list:
            inputData = np.array(inputData)
            
        for t in inputData:
            # Create an explanation on the specific instance
            exp = self._explainer.explain_instance(t, self._mlModel.predict, num_features=3)
            if showInNotebook:
                exp.show_in_notebook(show_table=True)
            else:
                print('Input: {}'.format(t))
                print('Output: {}\n'.format(self.predict([t])))

                print('LIME explanation:')
                for group in exp.as_list():
                    print(group[0])

                print('\n______________________________\n')

    def explainAllExperimentCases(self, showInNotebook=False):
        self.explain([[min(self._heartRates), min(self._triggerPulls), min(self._movements)],
                      [min(self._heartRates), min(self._triggerPulls), max(self._movements)],
                      [min(self._heartRates), max(self._triggerPulls), min(self._movements)],
                      [min(self._heartRates), max(self._triggerPulls), max(self._movements)],
                      [max(self._heartRates), min(self._triggerPulls), min(self._movements)],
                      [max(self._heartRates), min(self._triggerPulls), max(self._movements)],
                      [max(self._heartRates), max(self._triggerPulls), min(self._movements)],
                      [max(self._heartRates), max(self._triggerPulls), max(self._movements)]],
                    showInNotebook=showInNotebook)

if __name__ == '__main__':
    psacs = PSACS(shouldExportData=False)
    psacs.trainModel()

    psacs.histogramOfRawData('hr')
    psacs.histogramOfRawData('tp')
    psacs.histogramOfRawData('rm')

    psacs.plotAllIncreasingFeaturePredictions()

    psacs.plotIsolatedFeaturePrediction('hr')
    psacs.plotIsolatedFeaturePrediction('tp')
    psacs.plotIsolatedFeaturePrediction('rm')

    print(psacs.crossEvaluate(5))
    print(psacs.explain([[50,300,10000]]))


