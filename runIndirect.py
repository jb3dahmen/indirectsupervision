from IndirectSupervisor import *

#Data should be in format:
#datetime,Sensor_Name,Sensor_State,Activity,Is_Anomaly
#2017-03-12 00:08:28.191062,Kitchen,ON,Other_Activity,0
#2017-03-12 00:08:29.320621,Kitchen,OFF,Other_Activity,1

filename = "/path/to/file"
validationfilename =  validationfilename = "/path/to/file"
number_of_bayesian_evaluations = 30
number_of_trials = 10

indirectsupervisor = IndirectSupervisor(filename, validationfilename, number_of_bayesian_evaluations, number_of_trials)
indirectsupervisor.runIndirect()