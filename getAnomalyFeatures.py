import pandas as pd
from readDataFiles import *
import datetime as datetime
import numpy as np


class getAnomalyFeatures:

    def __init__(self, markeddataframe):
        self.rawDataFrame = markeddataframe
     
        
        self.sensorlist = self.getSensorList()
        self.activitylist = self.getActivityList()
        self.sensorstatelist = self.getSensorStateList()

        self.sensorNameMap = self.getSensorNameMap()
        self.sensorStateMap = self.getSensorStateMap()
        self.activityNameMap = self.getActivityNameMap()

     
        self.bathroomnumlist = self.getBathroomSensorNums()
        self.chairnumlist = self.getChairSensorNums()
        self.doornum = self.getMainDoorNum()
        self.bedroomnumlist = self.getBedroomSensorNums()
   
        self.windowfeaturevectordict = {}
        self.previousDominantSensor = -1
        self.timeElapsedForSameSensor = 0

        self.activitycountslist = []
        self.sensorcountlist = []


        self.rawDataFrame["Activity_Num"] = self.rawDataFrame["Activity"].replace(self.activityNameMap)
        
        self.rawDataFrame["Sensor_Name_Num"] = self.rawDataFrame["Sensor_Name"].replace(self.sensorNameMap)
        self.rawDataFrame["Sensor_State_Num"] = self.rawDataFrame["Sensor_State"].replace(self)
        
        self.rawDataFrame["Hour"] = self.rawDataFrame["Time"].apply(self.getHour)
        self.rawDataFrame["Minutes"] = self.rawDataFrame["Time"].apply(self.getMinute)
        self.rawDataFrame["Seconds"] = self.rawDataFrame["Time"].apply(self.getSeconds)
        self.rawDataFrame["DayofWeek"] = self.rawDataFrame["Date"].apply(self.getDayOfWeek)
        self.rawDataFrame["datetime"] = self.rawDataFrame.index

        self.rawDataFrame["isnightbathroom"] = self.rawDataFrame.apply(self.getisnightbathroom, axis = 1)

        self.rawDataFrame["next_time"] = self.rawDataFrame["datetime"].shift(
            -1)

        self.rawDataFrame["SensorDelay"] = self.rawDataFrame.apply(self.getDelayToThisSensorEvent, axis = 1)

        self.rawDataFrame["TimeOfDay"] = self.rawDataFrame["datetime"].apply(self.getTimeOfDay)

        
    def getBathroomSensorNums(self):
        bathroomsensornums = []
        for key, value in self.sensorNameMap.items():
            if("Bathroom" in key):
                bathroomsensornums.append(value)

        return bathroomsensornums

    def getBedroomSensorNums(self):
        bedroomsensornums = []
        for key,value in self.sensorNameMap.items():
            if("Bedroom" in key):
                bedroomsensornums.append(value)
        return bedroomsensornums

    def getChairSensorNums(self):
        chairsensornums = []
        for key, value in self.sensorNameMap.items():
 
            if("Chair" in key):
                chairsensornums.append(value)

        return chairsensornums

    def getMainDoorNum(self):
        for key, value in self.sensorNameMap.items():
            if("MainDoor" in key):
                return value


    def getFeaturesForWindowSize(self, windowSize, featureSet):
        X, groups = self.getLearnedWindowFeatures(windowSize, featureSet)
        y = self.getLearnedWindowGroundTruth(groups)
        return X, y, groups

    def getHour(self, time):
        hour = 0
        hour = float(time.split(":")[0])
        return hour
    
    def getMinute(self, time):
        minutes = 0
        minutes = float(time.split(":")[1])
        return minutes
    
    def getSeconds(self, time):
        seconds = 0
        seconds = float(time.split(":")[2])
        return seconds
    
    def getDayOfWeek(self, date):
        dayofweek = 0
 
        year, month, day = map(int, date.split("-"))
        date = datetime.date(year, month, day)
        dayofweek = date.weekday()

        return dayofweek

    def getDelayToThisSensorEvent(self, time):
        currenttime = time['datetime']
        nexttime = time["next_time"]
        duration = (nexttime-currenttime).total_seconds()
        return duration

    def getisnightbathroom(self, row):
        time = row["datetime"]
        activitynum = row["Sensor_Name_Num"]
        if(activitynum not in self.bathroomnumlist):
            return 0
        else:
            date = time.strftime("%Y %m %d")
            nocturiatimeframenight = [datetime.datetime.strptime(date + " 22:00:00.000000",'%Y %m %d %H:%M:%S.%f'), datetime.datetime.strptime(date + " 23:59:00.000000",'%Y %m %d %H:%M:%S.%f')]
            nocturiatimeframemorning = [datetime.datetime.strptime(date + " 00:00:00.000000",'%Y %m %d %H:%M:%S.%f'), datetime.datetime.strptime(date + " 08:00:00.000000",'%Y %m %d %H:%M:%S.%f')]
            if(time >= nocturiatimeframemorning[0] and time <= nocturiatimeframemorning[1]):
                return 1
            if(time >= nocturiatimeframenight[0] and time <= nocturiatimeframenight[1]):
                return 1
            return 0

    def getTimeOfDay(self, time):
        timeoflastsenorevent = time
        date = time.strftime("%Y %m %d")

        morning = [datetime.datetime.strptime(date + " 00:00:00.000000",'%Y %m %d %H:%M:%S.%f'), datetime.datetime.strptime(date + " 11:59:00.000000",'%Y %m %d %H:%M:%S.%f')]
        afternoon = [datetime.datetime.strptime(date + " 12:00:00.000000",'%Y %m %d %H:%M:%S.%f'), datetime.datetime.strptime(date + " 16:59:00.000000",'%Y %m %d %H:%M:%S.%f')]
        evening = [datetime.datetime.strptime(date + " 17:00:00.000000",'%Y %m %d %H:%M:%S.%f'), datetime.datetime.strptime(date + " 23:59:00.000000",'%Y %m %d %H:%M:%S.%f')]

        timeofday  = 0

        if(timeoflastsenorevent >= morning[0] and timeoflastsenorevent <= morning[1]):
            timeofday = 0
        if(timeoflastsenorevent >= afternoon[0] and timeoflastsenorevent <= afternoon[1]):
            timeofday = 1
        if(timeoflastsenorevent >= evening[0] and timeoflastsenorevent <= evening[1]):
            timeofday = 2
        return timeofday


    def getSlidingWindowFeaturesEvents(self, windowSize, featureSet):

        X = []

        if(featureSet == 0): #basic sensor
            self.rawDataFrame["numsensorevents"] = self.rawDataFrame["Activity_Num"].rolling(windowSize, min_periods=1).apply(self.getNumSensorEvents)

            self.rawDataFrame["numuniquesensors"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getNumUniqueSensorsSliding)
  

            self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getCountOfEachSensorSliding)
            self.rawDataFrame["counteachsensor"] = self.sensorcountlist

            self.rawDataFrame["lastseconds"] = self.rawDataFrame["Seconds"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lasthour"] = self.rawDataFrame["Hour"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lastminutes"] = self.rawDataFrame["Minutes"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            
            self.rawDataFrame["lastdayofweek"] = self.rawDataFrame["DayofWeek"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)


            self.rawDataFrame["dominantsensorid"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getDominantSensorIDSliding)

            self.rawDataFrame["avgsensordelay"] = self.rawDataFrame["SensorDelay"].rolling(windowSize, min_periods=1).apply(self.getAvgSensorDelaySliding)

            self.rawDataFrame["timeofday"] = self.rawDataFrame["TimeOfDay"].rolling(windowSize, min_periods=1).apply(self.getTimeOfDayOfLastSensorEventSliding)


            for index, row in self.rawDataFrame.iterrows():
                features = []
                features.append(row["numsensorevents"])
                features.append(row["numuniquesensors"])
                features.extend(row["counteachsensor"])

                features.append(row["lastseconds"])
                features.append(row["lasthour"])
                features.append(row["lastminutes"])
                features.append(row["lastdayofweek"])

                features.append(row["dominantsensorid"])
                features.append(row["avgsensordelay"])


                features.append(row["timeofday"])

                if(np.isnan(features).any()):
                    features = np.nan_to_num(features)

                X.append(features)

        if(featureSet == 1): #activity
    

            self.rawDataFrame["numuniqueactivities"] = self.rawDataFrame["Activity_Num"].rolling(windowSize, min_periods=1).apply(self.getNumUniqueActivtiesSliding)
     

            self.rawDataFrame["Activity_Num"].rolling(windowSize, min_periods=1).apply(self.getCountOfEachActivitySliding)


            self.rawDataFrame["counteachactivity"] = self.activitycountslist


            self.rawDataFrame["lastseconds"] = self.rawDataFrame["Seconds"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lasthour"] = self.rawDataFrame["Hour"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lastminutes"] = self.rawDataFrame["Minutes"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            
            self.rawDataFrame["lastdayofweek"] = self.rawDataFrame["DayofWeek"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)

            self.rawDataFrame["avgsensordelay"] = self.rawDataFrame["SensorDelay"].rolling(windowSize, min_periods=1).apply(self.getAvgSensorDelaySliding)

            self.rawDataFrame["timeofday"] = self.rawDataFrame["TimeOfDay"].rolling(windowSize, min_periods=1).apply(self.getTimeOfDayOfLastSensorEventSliding)


       
            for index, row in self.rawDataFrame.iterrows():
                features = []

                features.append(row["numuniqueactivities"])
                features.extend(row["counteachactivity"])
                features.append(row["lastseconds"])
                features.append(row["lasthour"])
                features.append(row["lastminutes"])
                features.append(row["lastdayofweek"])
                features.append(row["avgsensordelay"])


                features.append(row["timeofday"])
     

                if(np.isnan(features).any()):
                    features = np.nan_to_num(features)

                X.append(features)
            

        if(featureSet == 2): #all features
            self.rawDataFrame["numsensorevents"] = self.rawDataFrame["Activity_Num"].rolling(windowSize, min_periods=1).apply(self.getNumSensorEvents)

            self.rawDataFrame["numuniquesensors"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getNumUniqueSensorsSliding)
            self.rawDataFrame["numuniqueactivities"] = self.rawDataFrame["Activity_Num"].rolling(windowSize, min_periods=1).apply(self.getNumUniqueActivtiesSliding)

            self.rawDataFrame["Activity_Num"].rolling(windowSize, min_periods=1).apply(self.getCountOfEachActivitySliding)
            self.rawDataFrame["counteachactivity"] = self.activitycountslist

            self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getCountOfEachSensorSliding)
            self.rawDataFrame["counteachsensor"] = self.sensorcountlist

            self.rawDataFrame["lastseconds"] = self.rawDataFrame["Seconds"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lasthour"] = self.rawDataFrame["Hour"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lastminutes"] = self.rawDataFrame["Minutes"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            
            self.rawDataFrame["lastdayofweek"] = self.rawDataFrame["DayofWeek"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)

            self.rawDataFrame["lastsensorlocation"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)

            self.rawDataFrame["dominantsensorid"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getDominantSensorIDSliding)

            self.rawDataFrame["avgsensordelay"] = self.rawDataFrame["SensorDelay"].rolling(windowSize, min_periods=1).apply(self.getAvgSensorDelaySliding)

            self.rawDataFrame["timeofday"] = self.rawDataFrame["TimeOfDay"].rolling(windowSize, min_periods=1).apply(self.getTimeOfDayOfLastSensorEventSliding)

  
            for index, row in self.rawDataFrame.iterrows():
                features = []
                features.append(row["numsensorevents"])
                features.append(row["numuniquesensors"])
                features.append(row["numuniqueactivities"])
                features.extend(row["counteachactivity"])
                features.extend(row["counteachsensor"])
                features.append(row["lastseconds"])
                features.append(row["lasthour"])
                features.append(row["lastminutes"])
                features.append(row["lastdayofweek"])
                features.append(row["lastsensorlocation"])
                features.append(row["dominantsensorid"])
                features.append(row["avgsensordelay"])


                features.append(row["timeofday"])


                if(np.isnan(features).any()):
                    features = np.nan_to_num(features)

                X.append(features)


        if(featureSet == 3): #bathroom usage
            self.rawDataFrame["numsensorevents"] = self.rawDataFrame["Activity_Num"].rolling(windowSize).apply(self.getNumSensorEvents)

            self.rawDataFrame["numuniquesensors"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getNumUniqueSensorsSliding)

            self.rawDataFrame["lastseconds"] = self.rawDataFrame["Seconds"].rolling(windowSize).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lasthour"] = self.rawDataFrame["Hour"].rolling(windowSize).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lastminutes"] = self.rawDataFrame["Minutes"].rolling(windowSize).apply(self.getLastEventValueSliding)
            
            self.rawDataFrame["lastdayofweek"] = self.rawDataFrame["DayofWeek"].rolling(windowSize).apply(self.getLastEventValueSliding)


            self.rawDataFrame["dominantsensorid"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getDominantSensorIDSliding)

            self.rawDataFrame["avgsensordelay"] = self.rawDataFrame["SensorDelay"].rolling(windowSize).apply(self.getAvgSensorDelaySliding)

            self.rawDataFrame["timeofday"] = self.rawDataFrame["TimeOfDay"].rolling(windowSize).apply(self.getTimeOfDayOfLastSensorEventSliding)
            
            self.rawDataFrame["bathroomentries"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getBathroomEntryCountSliding)
            self.rawDataFrame["nightbathroomentries"] = self.rawDataFrame["isnightbathroom"].rolling(windowSize, min_periods=1).apply(self.getNightBathroomEntryCountSliding)


            for index, row in self.rawDataFrame.iterrows():
                features = []
                features.append(row["numsensorevents"])
                features.append(row["numuniquesensors"])

                features.append(row["lastseconds"])
                features.append(row["lasthour"])
                features.append(row["lastminutes"])
                features.append(row["lastdayofweek"])

                features.append(row["dominantsensorid"])
                features.append(row["avgsensordelay"])


                features.append(row["timeofday"])
                features.append(row["bathroomentries"])
                features.append(row["nightbathroomentries"])

                if(np.isnan(features).any()):
                    features = np.nan_to_num(features)

                X.append(features)

        if(featureSet == 4): #location one place

            self.rawDataFrame["numsensorevents"] = self.rawDataFrame["Activity_Num"].rolling(windowSize, min_periods=1).apply(self.getNumSensorEvents)

            self.rawDataFrame["numuniquesensors"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getNumUniqueSensorsSliding)
  

            self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getCountOfEachSensorSliding)
            self.rawDataFrame["counteachsensor"] = self.sensorcountlist

            self.rawDataFrame["lastseconds"] = self.rawDataFrame["Seconds"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lasthour"] = self.rawDataFrame["Hour"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            self.rawDataFrame["lastminutes"] = self.rawDataFrame["Minutes"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)
            
            self.rawDataFrame["lastdayofweek"] = self.rawDataFrame["DayofWeek"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)


            self.rawDataFrame["dominantsensorid"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getDominantSensorIDSliding)

            self.rawDataFrame["avgsensordelay"] = self.rawDataFrame["SensorDelay"].rolling(windowSize, min_periods=1).apply(self.getAvgSensorDelaySliding)

            self.rawDataFrame["timeofday"] = self.rawDataFrame["TimeOfDay"].rolling(windowSize, min_periods=1).apply(self.getTimeOfDayOfLastSensorEventSliding)

            self.rawDataFrame["lastlocation"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getLastEventValueSliding)

            self.rawDataFrame["timeinoneplace"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize, min_periods=1).apply(self.getTimeElaspedInOnePlaceSliding, raw=False)


            for index, row in self.rawDataFrame.iterrows():
                features = []
                features.append(row["numsensorevents"])
                features.append(row["numuniquesensors"])
                features.extend(row["counteachsensor"])

                features.append(row["lastseconds"])
                features.append(row["lasthour"])
                features.append(row["lastminutes"])
                features.append(row["lastdayofweek"])

                features.append(row["dominantsensorid"])
                features.append(row["avgsensordelay"])


                features.append(row["timeofday"])

                features.append(row["lastlocation"])

                features.append(row["timeinoneplace"])

                if(np.isnan(features).any()):
                    features = np.nan_to_num(features)

                X.append(features)


        if(featureSet == 5): #social and time in chair
           
            self.rawDataFrame["numsensorevents"] = self.rawDataFrame["Activity_Num"].rolling(windowSize).apply(self.getNumSensorEvents, raw=True)

            self.rawDataFrame["numuniquesensors"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getNumUniqueSensorsSliding, raw=True)

            self.rawDataFrame["lastseconds"] = self.rawDataFrame["Seconds"].rolling(windowSize).apply(self.getLastEventValueSliding, raw=True)
            self.rawDataFrame["lasthour"] = self.rawDataFrame["Hour"].rolling(windowSize).apply(self.getLastEventValueSliding, raw=True)
            self.rawDataFrame["lastminutes"] = self.rawDataFrame["Minutes"].rolling(windowSize).apply(self.getLastEventValueSliding, raw=True)
            
            self.rawDataFrame["lastdayofweek"] = self.rawDataFrame["DayofWeek"].rolling(windowSize).apply(self.getLastEventValueSliding, raw=True)


            self.rawDataFrame["dominantsensorid"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getDominantSensorIDSliding, raw=True)

            self.rawDataFrame["avgsensordelay"] = self.rawDataFrame["SensorDelay"].rolling(windowSize).apply(self.getAvgSensorDelaySliding, raw=True)

            self.rawDataFrame["timeofday"] = self.rawDataFrame["TimeOfDay"].rolling(windowSize).apply(self.getTimeOfDayOfLastSensorEventSliding, raw=True)
            
          
            self.rawDataFrame["timeinchair"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getTimeInChairSliding, raw=False)

            self.rawDataFrame["timeoutofhomecount"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getOutOfHomeCount, raw=False)

            self.rawDataFrame["totaltimeoutofhome"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getTotalTimeOutOfHome, raw=False)


            self.rawDataFrame["upatnight"] = self.rawDataFrame["Sensor_Name_Num"].rolling(windowSize).apply(self.getUpAtNight, raw=False)
           


            for index, row in self.rawDataFrame.iterrows():
                features = []
                features.append(row["numsensorevents"])
                features.append(row["numuniquesensors"])

                features.append(row["lastseconds"])
                features.append(row["lasthour"])
                features.append(row["lastminutes"])
                features.append(row["lastdayofweek"])

                features.append(row["dominantsensorid"])
                features.append(row["avgsensordelay"])


                features.append(row["timeofday"])

                features.append(row["timeinchair"])
                features.append(row["timeoutofhomecount"])
                features.append(row["totaltimeoutofhome"])
                features.append(row["upatnight"])

                if(np.isnan(features).any()):
                    features = np.nan_to_num(features)

                X.append(features)
    
            
        self.rawDataFrame["AnomalyGroundTruth"] = self.rawDataFrame["Is_Anomaly"].rolling(windowSize, min_periods=1).apply(self.getAnomalyGroundTruthSliding)

        y = self.rawDataFrame["AnomalyGroundTruth"].tolist()

        self.resetFeatureLists()

        return X, y

    def resetFeatureLists(self):
        self.windowfeaturevectordict = {}
        self.previousDominantSensor = -1
        self.timeElapsedForSameSensor = 0

        self.activitycountslist = []
        self.sensorcountlist = []



    def getRawDataFrameTM(self, filename, startdate, enddate):
        return getDataFrameDateRange(filename, startdate, enddate)

    def getRawDataFrameTMMarked(self, filename, startdate, enddate):
        return getAnomalyDataFrameDateRange(filename, startdate, enddate)

    #get list of sensors present in data
    def getSensorList(self):
        return self.rawDataFrame["Sensor_Name"].unique()

    def getSensorStateList(self):
        return self.rawDataFrame["Sensor_State"].unique()

    def getActivityList(self):
        return self.rawDataFrame["Activity"].unique()

    #get map for sensor name to ID
    def getSensorNameMap(self):
        sensornamemap = {}
        for i in range(len(self.sensorlist)):
            sensornamemap[self.sensorlist[i]] = i
        return sensornamemap

    def getSensorStateMap(self):
        sensorstatemap = {}
        for i in range(len(self.sensorstatelist)):
            sensorstatemap[self.sensorstatelist[i]] = i
        return sensorstatemap    

    def getActivityNameMap(self):
        activitynamemap = {}
        for i in range(len(self.activitylist)):
            activitynamemap[self.activitylist[i]] = i
        return activitynamemap

    

    def getAnomalyGroundTruthSliding(self, window):

            anomalylist = list(window)
            
            if (1.0 in anomalylist): 
                #get length of anomaly list
                #get number of 1's
                #check if 1's comprise % or more of list
                length = len(anomalylist)
                num1 = anomalylist.count(1.0)
                fraction = num1/length

                if(fraction >= 0.10):
                    
                    return 1
                else:
                    return 0
            else:
                return 0


    def getNumSensorEvents(self, window):
        return len(window)

    def getNumUniqueSensorsSliding(self, window):
        return len(np.unique(window))

    def getNumUniqueSensors(self, window):
        return window["Sensor_Name"].nunique()

    def getNumUniqueActivtiesSliding(self, window):
        return len(np.unique(window))

    def getNumUniqueActivties(self, window):
        return window["Activity"].nunique()

    def getCountOfEachActivity(self, window):
        activitycounts = []
        for i in range(len(self.activitylist)):
            activitycounts.append(0)
        valuecounts = window["Activity"].value_counts()
        for name, count in valuecounts.iteritems():
            activitynameindex = self.activityNameMap[name]
            activitycounts[activitynameindex] = count
        return activitycounts

    def getCountOfEachActivitySliding(self, window):
        current = []
        for i in range(len(self.activitylist)):
            current.append(0)
        unique, counts = np.unique(window, return_counts=True)
        for i in range(len(counts)):
            name = unique[i]
            current[int(name)] = counts[i]
        self.activitycountslist.append(current)
        return 1
    
    def getBedToiletCount(self, window):
        count = 0
        valuecounts = window["Sensor_Name"].value_counts()
        try:
            count = valuecounts["Bathroom"]
        except KeyError:
            count = 0  
        return count

    def getBedToiletCountSliding(self, window):
        unique, counts = np.unique(window, return_counts=True)
        for i in range(len(counts)):
            if(unique[i] in self.bathroomnumlist):
           
                return counts[i]
        return 0

    def getBathroomEntryCountSliding(self, window):
        startflag = 0
        entrycount = 0
        for i in range(len(window)):
            if(int(window[i]) in self.bathroomnumlist and startflag == 0):
                startflag = 1
                entrycount += 1
            if(int(window[i]) not in self.bathroomnumlist):
                startflag = 0
        #go through window each time new bathroom entry occur add 1 to count
        return entrycount

    def getNightBathroomEntryCountSliding(self, window):
        startflag = 0
        entrycount = 0
        for i in range(len(window)):
            if(int(window[i]) == 1 and startflag == 0):
                startflag = 1
                entrycount += 1
            if(int(window[i]) != 1):
                startflag = 0
   
        return entrycount

    #how much time in a row has dom sensor been in same area
    #need something keep track of last dom sensor
    #if current the same keep adding to the time using duration
    #if different reset the time
    def getTimeElaspedInOnePlace(self, window):
     
        dominantsensor = self.getDominantSensorID(window)

        windowduration = (window.index[len(window) - 1] - window.index[0]).total_seconds()


        if(self.previousDominantSensor != -1):#-1 means just starting
           
            if(dominantsensor != self.previousDominantSensor):
                #update new elasped time
                self.timeElapsedForSameSensor = windowduration
                self.previousDominantSensor = dominantsensor
            else:#if they are equal add to elapsed time
                self.timeElapsedForSameSensor += windowduration

        else:
            self.previousDominantSensor = dominantsensor
            self.timeElapsedForSameSensor = windowduration

        return self.timeElapsedForSameSensor

    def getWindowDurationSliding(self, window):
        return np.sum(window)

    
    def getOutOfHomeCount(self, window):
        count = 0
        for value in window:
            if(value == self.doornum):
                count += 1
        return count

    def getTotalTimeOutOfHome(self, window):
        timeelapsed = 0
        startflag = 0
        previoustimestamp = 0
        for index, value in window.iteritems():
            if(value == self.doornum):
                if(startflag == 0):
                    startflag = 1
                    previoustimestamp = index
                else:
                    currentimestamp = index
                    timeelapsed = timeelapsed + (currentimestamp - previoustimestamp).total_seconds()
                    previoustimestamp = currentimestamp

        return timeelapsed

    def getUpAtNight(self, window):
        
        nonbedroomcount = 0
        startimestamp = window.index[0]
       
        #check within night time range
        date = startimestamp.strftime("%Y %m %d")
        nocturiatimeframenight = [datetime.datetime.strptime(date + " 22:00:00.000000",'%Y %m %d %H:%M:%S.%f'), datetime.datetime.strptime(date + " 23:59:00.000000",'%Y %m %d %H:%M:%S.%f')]
        nocturiatimeframemorning = [datetime.datetime.strptime(date + " 00:00:00.000000",'%Y %m %d %H:%M:%S.%f'), datetime.datetime.strptime(date + " 08:00:00.000000",'%Y %m %d %H:%M:%S.%f')]
        if(startimestamp >= nocturiatimeframemorning[0] and startimestamp <= nocturiatimeframemorning[1]):
            #get number of non bedroom activations
            for value in window:
                if(value not in self.bedroomnumlist):
                    nonbedroomcount += 1
        if(startimestamp >= nocturiatimeframenight[0] and startimestamp <= nocturiatimeframenight[1]):
            #get number of non bedroom activations
            for value in window:
                if(value not in self.bedroomnumlist):
                    nonbedroomcount += 1

       
        return nonbedroomcount

    def getTimeElaspedInOnePlaceSliding(self, window):
       
        #I need the most frequent sensor and also the duration of the window
        #get first and last time stamps
        
        dominantsensor = self.getDominantSensorIDSliding(window)

        windowduration = (window.index[-1] - window.index[0]).total_seconds()


        if(self.previousDominantSensor != -1):#-1 means just starting
           
            if(dominantsensor != self.previousDominantSensor):
                #update new elasped time
                self.timeElapsedForSameSensor = windowduration
                self.previousDominantSensor = dominantsensor
            else:#if they are equal add to elapsed time
                self.timeElapsedForSameSensor += windowduration

        else:
            self.previousDominantSensor = dominantsensor
            self.timeElapsedForSameSensor = windowduration

        return self.timeElapsedForSameSensor


    def getCountOfEachSensor(self, window):
        sensorcounts = []
        for i in range(len(self.sensorlist)):
            sensorcounts.append(0)
        valuecounts = window["Sensor_Name"].value_counts()
        for name, count in valuecounts.iteritems():
            sensornameindex = self.sensorNameMap[name]
            sensorcounts[sensornameindex] = count
        return sensorcounts

    def getCountOfEachSensorSliding(self, window):
        current = []
        for i in range(len(self.sensorlist)):
            current.append(0)

        unique, counts = np.unique(window, return_counts=True)
        for i in range(len(counts)):
            name = unique[i]
            current[int(name)] = counts[i]
        self.sensorcountlist.append(current)
        return 1


    def getLastSensorEventSeconds(self, window):
        seconds = 0
        lastevent = window.tail(1)
        time = lastevent.iloc[0]["Time"]
        seconds = float(time.split(":")[2])
        return seconds

    def getLastEventValueSliding(self, window):
        last = window[-1]
        return last

    def getLastSensorEventHour(self, window):
        hour = 0
        lastevent = window.tail(1)
        time = lastevent.iloc[0]["Time"]
        hour = float(time.split(":")[0])
        return hour

    def getLastSensorEventMinutes(self,window):
        minutes = 0
        lastevent = window.tail(1)
        time = lastevent.iloc[0]["Time"]
        minutes = float(time.split(":")[1])
        return minutes

    def getLastSensorLocation(self, window):
        location = 0
        lastevent = window.tail(1)
        location = lastevent.iloc[0]["Sensor_Name"]
        location = self.sensorNameMap[location]

        return location

    def getDayOfWeekLastSensor(self, window):
        dayofweek = 0
        lastevent = window.tail(1)
        date = lastevent.iloc[0]["Date"]
        year, month, day = map(int, date.split("-"))

        date = datetime.date(year, month, day)
        dayofweek = date.weekday()
        return dayofweek

    def getTimeOfDayOfLastSensorEvent(self, window):
        timeofday  = 0

        morning = [datetime.datetime.strptime("00:00:00.000000",'%H:%M:%S.%f'), datetime.datetime.strptime("11:59:00.000000",'%H:%M:%S.%f')]
        afternoon = [datetime.datetime.strptime("12:00:00.000000",'%H:%M:%S.%f'), datetime.datetime.strptime("16:59:00.000000",'%H:%M:%S.%f')]
        evening = [datetime.datetime.strptime("17:00:00.000000",'%H:%M:%S.%f'), datetime.datetime.strptime("23:59:00.000000",'%H:%M:%S.%f')]

        lastevent = window.tail(1)
        timeoflastsenorevent = lastevent.iloc[0]["Time"]
        timeoflastsenorevent = datetime.datetime.strptime(timeoflastsenorevent,'%H:%M:%S.%f')

        if(timeoflastsenorevent >= morning[0] and timeoflastsenorevent <= morning[1]):
            timeofday = 0
        if(timeoflastsenorevent >= afternoon[0] and timeoflastsenorevent <= afternoon[1]):
            timeofday = 1
        if(timeoflastsenorevent >= evening[0] and timeoflastsenorevent <= evening[1]):
            timeofday = 2
        return timeofday

    def getTimeOfDayOfLastSensorEventSliding(self, window):
        return int(window[-1])

    def getTimeInChairSliding(self, window):
        count = 0
        for index, value in window.iteritems():
          
            if(value in self.chairnumlist):
                count += 1
        return count

    def getSlowWalkSpeed(self, window):
        avgdelay = self.getAvgSensorDelaySliding(window)
        

    def getDominantSensorID(self, window):
        valuecounts = window["Sensor_Name"].value_counts()

    
        dominantsensor = self.sensorNameMap[valuecounts.idxmax()]
        return dominantsensor

    def getDominantSensorIDSliding(self, window):
        
        unique, counts = np.unique(window, return_counts=True)
        countsdict = dict(zip(unique, counts))
        #get value counts of window
        #return one with max value

        dominantsensor = max(countsdict, key=countsdict.get)
 
        return int(dominantsensor)

  
    def getAvgSensorDelay(self, window):
        delaysum = 0
        #get durations between each sensor event
        if(len(window) > 1):
            for i in range(len(window) - 1):
                currenttime = window.index[i]
                nexttime = window.index[i+1]
                delaysum += (nexttime-currenttime).total_seconds()

            #sum and then divide by num sensorevents
            length = len(window) - 1
            return delaysum / length
        else:
   
            return 0

    def getAvgSensorDelaySliding(self, window):
        return np.mean(window)




