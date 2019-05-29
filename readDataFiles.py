import pandas as pd

def getDataFrameDateRange(filename, datestart, dateend):

    headers = ["date", "time", "sensor_name", "sensor_state", "activity_name"]
    shcategoricaldata = pd.read_csv(filename, header=None, names=headers, delim_whitespace=True)
    shcategoricaldata['timestamp'] = shcategoricaldata["date"] + " " + shcategoricaldata["time"]
    shcategoricaldata['timestamp'] = pd.to_datetime(shcategoricaldata['timestamp'])
    shcategoricaldata.index = shcategoricaldata['timestamp']


    mask = (shcategoricaldata['timestamp'] >datestart) & (shcategoricaldata['timestamp'] <= dateend)
    shcategoricaldata = shcategoricaldata.loc[mask]
    del shcategoricaldata['timestamp']

    return shcategoricaldata

def getAnomalyDataFrameDateRange(filename, datestart, dateend):

    headers = ["date", "time", "sensor_name", "sensor_state", "activity_name", "anomaly"]
    shcategoricaldata = pd.read_csv(filename, header=None, names=headers, delim_whitespace=True)
    shcategoricaldata['timestamp'] = shcategoricaldata["date"] + " " + shcategoricaldata["time"]
    shcategoricaldata['timestamp'] = pd.to_datetime(shcategoricaldata['timestamp'])
    shcategoricaldata.index = shcategoricaldata['timestamp']


    mask = (shcategoricaldata['timestamp'] >datestart) & (shcategoricaldata['timestamp'] <= dateend)
    shcategoricaldata = shcategoricaldata.loc[mask]
    del shcategoricaldata['timestamp']

    return shcategoricaldata