from scipy.stats.mstats import *
from readDataFiles import *

from getAnomalyFeatures import *
import numpy as np
from sklearn.linear_model import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import utils
from scipy import stats
from math import sqrt
import pandas
from statistics import mean
from sklearn.metrics import roc_auc_score
from statistics import mean

from pyod.models.pca import PCA
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.combination import *
from pyod.utils.utility import standardizer
from sklearn import preprocessing
from numpy import percentile


from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

from itertools import combinations

import importlib

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class IndirectSupervisor:
    def __init__(self, filename, validationfilename, num_evals, num_trials):
   
        self.detectorcombolist = [("PCA",), ("IForest",), ("KNN",), ("LOF",), ("OCSVM",)]
        self.filename = filename
        self.validationfilename = validationfilename
        self.num_evals = num_evals
        self.num_trials = num_trials

        self.markedrawdata = self.getSimpleDataFrame(filename)
        self.featureGenerator = getAnomalyFeatures(self.markedrawdata)

        self.results = []
        

    def getSimpleDataFrame(self, filename):
        smart_home_dataframe = pandas.read_csv(filename)
        parts = smart_home_dataframe["datetime"].str.split()
        dates = []
        times = []
        for part in parts:
            dates.append(part[0])
            times.append(part[1])
        smart_home_dataframe["Date"] = dates
        smart_home_dataframe["Time"] = times
        smart_home_dataframe["datetime"] = pandas.to_datetime(smart_home_dataframe["datetime"])
        smart_home_dataframe = smart_home_dataframe.set_index(["datetime"])
        smart_home_dataframe = smart_home_dataframe.sort_index()
        return smart_home_dataframe

    def gmean(self, TP, FP, FN):
        if TP + FN == 0 or TP + FP == 0:
            print("divide by 0")
            return 0
        precision = TP / (TP + FP)
        
        recall = TP / (TP + FN)
    

        return sqrt(precision * recall)

    def get_rates(self, y, y_pred):
        FP = 0
        FN = 0
        TP = 0
        TN = 0

        for i in range(len(y)):
            truth = y[i]
            pred = y_pred[i]
            if(truth == 1 and pred == 1):
                TP += 1
            if(truth == 0 and pred == 1):
                FP += 1
            if(truth == 1 and pred == 0):
                FN += 1
            if(truth == 0 and pred == 0):
                TN += 1

        return TP, TN, FP, FN

    def getDetectors(self, parameterdict):
        detectors = []
  
        windowSize = [value for key,value in parameterdict.items() if "windowsize_" in key.lower()][0]
        featureset = [value for key,value in parameterdict.items() if "featureset_" in key.lower()][0]
        detectortype = [value for key,value in parameterdict.items() if "type_" in key.lower()][0]

        detectorlist = detectortype.split()



        features = self.featureGenerator.getSlidingWindowFeaturesEvents(int(windowSize), int(featureset))

        X = features[0]
        y = features[1]
    

        for detector in detectorlist:

            if(detector.rstrip() == "OCSVM"):
                kernel = [value for key,value in parameterdict.items() if "kernel_" in key.lower()][0]
                nu = [value for key,value in parameterdict.items() if "nu_" in key.lower()][0]
                clf = OCSVM(kernel=kernel, nu=nu, max_iter=100)

            if(detector.rstrip()  == "IForest"):
                num_estimators = [value for key,value in parameterdict.items() if "num_estimators_" in key.lower()][0]
                max_samples = [value for key,value in parameterdict.items() if "max_samples_" in key.lower()][0]
                clf = IForest(n_estimators=int(num_estimators), max_samples=int(max_samples))

            if(detector.rstrip() == "PCA"):
                clf = PCA()
            
            if(detector.rstrip() == "LOF"):
                n_neighbors = [value for key,value in parameterdict.items() if "lof_n_neighbors_" in key.lower()][0]
                clf = LOF(n_neighbors=int(n_neighbors))

            if(detector.rstrip() == "KNN"):
                n_neighbors = [value for key,value in parameterdict.items() if "knn_n_neighbors_" in key.lower()][0]
                clf = KNN(n_neighbors=int(n_neighbors))

            with HiddenPrints():
                clf.fit_predict_score(X, y,  scoring='roc_auc_score')

            detectors.append(clf)
        return detectors, detectorlist, y, X

    def getGmean(self, parameterdict):
         
        anomaly_algorithm_predictions = []
        anomaly_scores = []
        detectors, detectorlist, y, X = self.getDetectors(parameterdict)

        simpleaveragescores = detectors[0].decision_scores_

        outliers_fraction = 0.1
        threshold = percentile(simpleaveragescores, 100 * (1 - outliers_fraction))
    
        combined_pred = []
        for i in range(len(simpleaveragescores)):
            if(simpleaveragescores[i] > threshold):
                combined_pred.append(1)
            else:
                combined_pred.append(0)
    

        TP, TN, FP, FN = self.get_rates(y, combined_pred)

  
        g_mean = self.gmean(TP, FP, FN)

        auc_score = roc_auc_score(y, combined_pred)

        self.results.append([parameterdict, g_mean, auc_score, detectors[0], X, y])

        print("Detector: ", detectors[0])
        print("Gmean: ", g_mean)
        return -g_mean

    def generatespace(self):
        spacelist = []
        detectorcountdict = {}
        for combo in self.detectorcombolist:
 
            combodict = {}
            combotypestr = ""

            for detector in combo:
                combotypestr += detector + " "
                count = detectorcountdict.get(detector, 0)
                count += 1
                count_str = str(count)
                detectorcountdict[detector] = count

                if(detector == "PCA"):
                    pass
                if(detector == "KNN"):
                    combodict["knn_n_neighbors_"+count_str] = hp.uniform('knn_n_neighbors_'+count_str, 5, 20)

                if(detector == "OCSVM"):
                    combodict['kernel_'+count_str] =  hp.choice('svm_kernel_'+count_str, ["linear", "rbf", "poly", "sigmoid"])
                    combodict["nu_"+count_str] = hp.uniform('nu_'+count_str, 0, 1)

                if(detector == "LOF"):
                    combodict["lof_n_neighbors_"+count_str] = hp.uniform('lof_n_neighbors_'+count_str, 5, 50)

                if(detector == "IForest"):
                    combodict["max_samples_"+count_str] = hp.choice('max_samples_'+count_str, list(range(256, len(self.markedrawdata))))
                    combodict["num_estimators_"+count_str] = hp.choice('num_estimators_'+count_str, list(range(30,200)))

            combodict['type_'+detector+"_"+count_str] = combotypestr
            combodict['windowsize_'+detector+"_"+count_str] = hp.uniform('windowsize_'+combotypestr, 10, 70)
            combodict['featureset_'+detector+"_"+count_str] = hp.choice('featureset_'+combotypestr, [0,1,2,3,4,5])
            spacelist.append(combodict)

        space = hp.choice('classifier_type', spacelist)
      
        return space


        
    def runIndirect(self):

        resultsfile = open(self.filename+".resultsonedetector", "w")
        resultsfile.write("bestparams,gmean\n")

        validationresultsfilename = self.validationfilename + "resultsvalidation"
        validationresultsfile = open(validationresultsfilename, "w")

        space = self.generatespace()

    
        for i in range(self.num_trials):

            
            tpe_algo = tpe.suggest
            tpe_trials = Trials()
            tpe_best = fmin(fn=self.getGmean, space=space, 
                    algo=tpe_algo, trials=tpe_trials, 
                    max_evals=self.num_evals)


            #sort the list based on the gmean score
  
            self.results.sort(key=lambda x: x[1])
           
     
            best = self.results[-1]
            bestparams = best[0]
            bestgmean = best[1]
            resultsfile.write(str(bestparams) + "," + str(bestgmean) + "\n")
        
            

            validationgmean = self.runValidation(self.validationfilename)
            validationresultsfile.write(str(validationgmean)+"\n")


            self.results = []


    def runValidation(self, validationfilename):
        bestresult = self.results[-1]
        
        clf = bestresult[3]
   
        parameterdict = bestresult[0]
        featuresetkey = ""
        windowsizekey = ""

        for key in parameterdict.keys():
            if("featureset" in key):
                featuresetkey = key
            if("windowsize" in key):
                windowsizekey = key

        featureset = parameterdict[featuresetkey]
        windowSize = parameterdict[windowsizekey]
       


        #get validaton data
        validationrawdata = self.getSimpleDataFrame(validationfilename)
        validationFeatureGenerator = getAnomalyFeatures(validationrawdata)

        #run detector on validation data
        features = validationFeatureGenerator.getSlidingWindowFeaturesEvents(int(windowSize), int(featureset))

        X = features[0]
        y = features[1]

        with HiddenPrints():
            clf.fit_predict_score(X, y,  scoring='roc_auc_score')

        simpleaveragescores = clf.decision_scores_

        outliers_fraction = 0.1
        threshold = percentile(simpleaveragescores, 100 * (1 - outliers_fraction))
        combined_pred = []
        for i in range(len(simpleaveragescores)):
            if(simpleaveragescores[i] > threshold):
                combined_pred.append(1)
            else:
                combined_pred.append(0)
    

        TP, TN, FP, FN = self.get_rates(y, combined_pred)

        #get the gmean measure
        g_mean = self.gmean(TP, FP, FN)
     
        auc_score = roc_auc_score(y, combined_pred)
        return g_mean



 