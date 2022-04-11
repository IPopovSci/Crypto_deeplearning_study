from utility import resample
'''Entry-point for data resampling
Accepts:
interval_from: interval from which to convert
interval_to: to which time interval to convert the data
Creates folder interval_to and populates it with resampled .csv files'''
resample('1m','1h')