#!/usr/bin/env python
#coding=utf-8

import os
from sys import *
from os import *
import math
import cmath
import re
import scipy
import string
import numpy
from numpy import *
from scipy.io import read_array, write_array
from pylab import *
import shutil
import ConfigParser
from scipy.linalg import *
from scipy import io  as s_io
import random as rdm
from pylab import *
import pickle as pp
from scipy.integrate import simps as simps

import  Scientific.IO.NetCDF as sioncdf
import time
    
from lxml import etree as xet
import pymseed



import pickle as pp
import numpy.random as Rand  
import time


    
cur_dir      = path.abspath(path.realpath( os.curdir ) )
in_data_dir  = path.abspath(path.realpath( cur_dir))
out_data_dir = path.abspath(path.realpath(path.join( cur_dir,'mseed_data_new')))

if not path.isdir(out_data_dir):
    makedirs(out_data_dir)   


event_year  = 2001
event_month = 5
event_day   = 7
event_hour  = 9
event_minute = 43
event_seconds = 33.88

event_time_in_epoch  = time.mktime((event_year, event_month, event_day, event_hour, event_minute,int(round(event_seconds)),0,0,0)) - time.timezone 

event_time_tuple = time.gmtime(event_time_in_epoch)
event_julianday = event_time_tuple[7]

lo_datafiles = []
lo_stations = []
lo_channels = []
lo_receivers = []

network = 'ek'
location = ''
search_string = re.compile(r'(?<!\d)([A-Z]{5})\.(?P<station>\w+)\.(BH)(?P<chan>[NEZ]{1})(?!\d)')



for root,dirs,files in os.walk(in_data_dir):
    for name in files:
        fn = os.path.join(root,name)
        kk = search_string.search(name)
        if kk:
            print fn
            lo_datafiles.append(fn)
            kk_dict=kk.groupdict()
            file_station = kk_dict['station'].upper()
            file_channel = 'BH'+kk_dict['chan'].upper()
            component    = file_channel[-1] 
            
            in_FH = file(fn,'r')
            in_array = loadtxt(in_FH)
            in_FH.close()
            time_axis = in_array[:,0]
            data_axis = in_array[:,1]
            deltat = (time_axis[-1] - time_axis[0])/(len(time_axis)-1)
            abs_time_axis = time_axis + event_time_in_epoch
            t_min = abs_time_axis[0]
            t_max = abs_time_axis[-1]
            start_time = event_time_in_epoch - 60 
            if t_min > start_time:
                new_data_axis = zeros(( len(time_axis) + ( (t_min - start_time)/deltat  )  ))
                t_min = start_time
                new_data_axis[-len(data_axis):] = data_axis
                data_axis = new_data_axis
                
            out_filename = '%s.%s.%s.%s.%s.%s' %(network,file_station,location,file_channel,event_year,event_julianday)
            out_file     = path.abspath(path.realpath(path.join( out_data_dir,out_filename )))

            trtup = (network,file_station,location,file_channel,int(t_min*pymseed.HPTMODULUS), int(t_max*pymseed.HPTMODULUS), 1./deltat, data_axis)

            pymseed.store_traces([trtup], out_file)

            
#raise SystemExit
