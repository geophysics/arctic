#!/usr/bin/env ipython
# coding=utf-8
#
#-*- coding: utf-8 -*-
#----------------------------------------------------------------------
#  version 0.14    -   12.02.11   -   LK   -
#----------------------------------------------------------------------
#
#
#  Functions for setting up a database of Green's functions for a
#   - circular shaped configuration of receivers
#   - rectangular setup of source-grid
#  around a given centre (point, given in (lat,lon)-coordinates
#
#  Function for setting up synthetic data-set for arbitrarily chosen source-point from source-grid  
#
#  All functions needed for SETUP_GF and ARCTIC
#  
#  contains:
#  
#  - read_in_config_file
#
#  - set_source_grid
#  - set_receiver
#  - setup_db
#  - source_coords 
#  - receiver_coords 
#  - s_r_azi_dist
#  - calc_distance 
#  - neighbouring_distances 
#
#  - setup_synth_data
#.
#.
#----------------------------------------------------------------------

# to do
#.
#.
#  korrektur der koordinatenberechung im rechtwinkligen Gitter
# rausschreiben des xml files
# einlesen der benoetigten daten bzgl der topographie
# neue GF datenbank
# "automatisches" erstellen eines config-files
# "konsistentes" aufrufen des codes (zeitschleife nicht noetig, wenn von conny aufgerufen)
#
#----------------------------------------------------------------------
#
#  'indirect output' : the result is written to the global config dictionary
#
#----------------------------------------------------------------------


import sys
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
import shutil
import ConfigParser
from scipy.linalg import *
from scipy import io  as s_io
import random as rdm
import pickle as pp
import cPickle as cP
from scipy.integrate import simps as simps

import Scientific.IO.NetCDF as sioncdf
import time
import calendar

from lxml import etree as xet
import pymseed
from pyrocko import util

from pylab import *

#----------------------------------------------------------------------

#complex_unit = complex(0,1)
pi           = math.pi

#conversion from radians to/from degrees
rad_to_deg   = 180./pi

#Earth's radius
R            = 6371000
radius       = R

# numerical precision    
numeric_epsilon = 1e-15
#----------------------------------------------------------------------

global _debug

_debug      = 1
_debug_plot = 0
_debug_print_on_screen  = 1

import code
import pdb

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def printf(format, *args):
    sys.stdout.write(format % args)
    
#----------------------------------------------------------------------
#----------------------------------------------------------------------

def read_in_config_file(filename):
    """Reads given configuration file.
    This has to have the structure
    
    [section]
    key     =   value
    ...

    Names of sections do not matter - only for human reading.

    input:
    -- name of configuration file

    output:
    -- config dictionary

    """

    print 'reading config_file...'

    configfile    = filename

    #set up configuration-dictionary
    config           = ConfigParser.ConfigParser()

    #read in configuration-file
    config.read(configfile)

    cfg = dict()

    #read in key-value pairs
    sec = config.sections()
    for s in sec:
        o = config.options(s)
        for oi in o:
            oi_v         = config.get(s,oi)
            cfg[oi]      = oi_v

        
    #setting paths:
    base_dir  = path.abspath(path.realpath(cfg['base_dir']))
    gf_dir    = path.abspath(path.realpath(path.join(cfg['base_dir'], cfg['gf_dir']) ))
    data_dir  = path.abspath(path.realpath(path.join(cfg['base_dir'], cfg['data_dir']) ))
    temp_dir  = path.abspath(path.realpath(path.join(cfg['base_dir'], 'temp' )))
    plot_dir  = path.abspath(path.realpath(path.join(cfg['base_dir'], 'plots' )))


    if (not path.exists( base_dir ) ):
        os.makedirs(base_dir)
    if (not path.exists(gf_dir)):
        os.makedirs(gf_dir)
    if (not path.exists(data_dir) ):
        os.makedirs(data_dir)
    if (not path.exists(temp_dir)):
        os.makedirs(temp_dir)            
    if (not path.exists(plot_dir)):
        os.makedirs(plot_dir)            

    cfg['gf_dir']                =  gf_dir
    cfg['GF_directory']          =  gf_dir
    cfg['data_dir']              =  data_dir
    cfg['parent_data_directory'] =  data_dir
    cfg['temp_dir']              =  temp_dir
    cfg['temporary_directory']   =  temp_dir
    cfg['plot_dir']              =  plot_dir

    if not cfg.has_key('network'):
        cfg['network'] = 'LK'
    
    if _debug:
        for i in sort(cfg.keys()):
            print '%s\t \t= \t%s'%(i, cfg[i])
            
    return cfg

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def bp_butterworth(data_in,sampling, cfg):
    """
    Filters given 1-D data array in frequency domain with numpy inherited butterworth filter. Corner
    frequencies are given in config dictionary.

    input:
    -- data
    -- sampling in Hz
    -- config dictionary

    output:
    -- frequency filtered data of same length
    """

    # sampling in Hz
    sampling_rate = sampling
    
    deltat = 1./sampling_rate
    length = float(cfg['time_window_length'])
    lower_corner_freq = float(cfg['bp_lower_corner'])
    upper_corner_freq  = float(cfg['bp_upper_corner'])
    order = int(float(cfg['bp_order']))

    if cfg.has_key('butterworth_dict') and cfg['butterworth_dict']['data_length'] == len(data_in) \
           and cfg['butterworth_dict']['sampling'] == sampling:
        (b,a) = cfg['butterworth_dict']['ba']
    else:
        # setting the filter coefficients
        (b,a ) = scipy.signal.butter(order,[corner*2.0*deltat for corner in (lower_corner_freq, upper_corner_freq) ], btype='band' )
        cfg['butterworth_dict'] = {}
        cfg['butterworth_dict']['ba'] = (b,a )
        cfg['butterworth_dict']['data_length'] = len(data_in)
        cfg['butterworth_dict']['sampling'] = sampling
        
    data   = data_in.astype(numpy.float64)
    datamean = numpy.mean(data)
    #data -= datamean

    # apply scipy filter
    data_out = scipy.signal.lfilter(b,a,data)
     
    return data_out#+datamean

#----------------------------------------------------------------------

def bp_boxcar( in_data,sampling, cfg):
    """real-valued rectangular frequency filter for data and Green's functions.

    TODO rename to bp_boxcar
    
    input:
    -- sampling in Hz
    -- data as numpy array of floats
    -- config dictionary

    output:
    -- frequency filtered data of same length
    """

    f_up       = float(cfg['bp_upper_corner'])
    f_low      = float(cfg['bp_lower_corner'])

    data       = in_data.astype(float64)
    n          = len(data)
    fdata      = numpy.fft.rfft(data)
    nf         = len(fdata)
    df         = 1./(n*sampling)
    freqs      = arange(nf)*df
    fdata     *= logical_and(f_low < freqs, freqs < f_up)
    data       = numpy.fft.irfft(fdata,n)
    assert len(data) == n
    out_data   = real(data)

    
    #check dimension of result:
    if ( len(in_data) != len(out_data) ):
        print "ERROR in filter-function !!!!"
        raise SystemExit

 
    #output of filtered data:
    return out_data

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def setup_A(cfg):
    """Gives array of correlation matrices A_(i,j) for combination of [source, station, component].

    input:
    -- config dictionary

    indirect output:
    -- correlation matrix array A[gp,s,c] in config dictionary

    direct output:
    -- control parameter '1'
    """


    # read in all data via config dict:
    
    gf                       = cfg['GF']
    time_axis                = cfg['time_axis_of_GF']

    station_index_dict       = cfg['station_index_dictionary']
    stations_array           = cfg['station_coordinates']
    
    lo_chans                 = cfg['list_of_channels']
    channel_index_dict       = cfg['channel_index_dictionary']

    lo_stations              = cfg['list_of_stations']
    
    lo_all_gridpoints        = cfg['list_of_all_gridpoints']
    if cfg.has_key('list_of_gridpoint_section'):
        current_lo_gridpoints    = cfg['list_of_gridpoint_section']
    else:
        current_lo_gridpoints    = lo_all_gridpoints
        
    n_gridpoints             = len(current_lo_gridpoints)
    n_stations               = len(lo_stations)

    filter_flag = int(cfg['filter_flag'])

    # set filenames -- storing A for the current setting for later use:
    # filename is coded by stations.

    #If too many stations shortening of filename:
    lo_stats                 = lo_stations
    if len(lo_stats) > 10:
        lo_stats = ['many','stations']
    try:
        filter_flag = int(cfg['filter_flag'])
        bp_lower    = cfg['bp_lower_corner']
        bp_upper    = cfg['bp_upper_corner']
        window_length = cfg['time_window_length']
        
    except:
        filter_flag = 0
    if not filter_flag in [0,1,2]:
        filter_flag = 0    

    if filter_flag == 1 or filter_flag == 2 :
        corr_mat_filename        = 'bandpass_filtered_current_%s_%s_%ss_correlation_matrix_A_for_stations_'%(bp_lower,bp_upper,window_length)+'_'.join(lo_stats)+'.pp'
    else:
        corr_mat_filename        = 'correlation_matrix_A_for_stations_'+'_'.join(lo_stats)+'.pp'

    corr_mat_file_abs          = path.realpath(path.abspath(path.join(cfg['temporary_directory'],corr_mat_filename)))
    corr_mat_file = path.realpath(path.abspath(corr_mat_file_abs ))


    # reset internal variable to original value
    if len(lo_stats) == 2:
       lo_stats                 = lo_stations
 
    # calculate A and save to file:

    try:
        #not possible, if gridpoint section changed
        #raise
        # if aleready existing just read in A from file

        if int(float(cfg['changed_setup'])) == 0:
        
            File1b = file(corr_mat_file,'r')
            A = pp.load(File1b)
            
            File1b.close()
            print 'read in  correlation matrix A...'
        else:
            raise

    except:
        print 'Calculating correlation matrix A...'

        summation_key = int(cfg.get('summation_key',0))
        if not (summation_key == 0 or summation_key == 1):
            summation_key = 0

        if _debug:
            lo_added_receivers = []


        if summation_key == 0:
            #sum over all receivers (station-component-combinations)
            A   = zeros((6,6,n_gridpoints),float)

            #loop over all sources, stations and components
            for gp in arange(n_gridpoints):
                dummy_matrix = zeros((6,6))

                for station in lo_stats:
                    station_index  = int(station_index_dict[station])-1

                    station_weight = stations_array[station_index,4]
                    
                    for channel in lo_chans:
                        channel_index   =  int(channel_index_dict[channel])
                        
                        dummy_matrix   +=  calc_corr_mat(gp,station_index,channel_index,station_weight, time_axis, cfg)

                        if _debug:
                            receiver = '%s.%s'%(station,channel)
                            lo_added_receivers.append(receiver)

                A[:,:,gp]    = dummy_matrix
                sys.stdout.write( 'A for gridpoint %4i of %4i set \r'%(gp+1,n_gridpoints))
            
        if summation_key == 1:
            A   = zeros((6,6,n_gridpoints,3),float)
            pass

        if _debug:
            set_of_added_receivers = sort(list(set(lo_added_receivers)))


        #d_break(locals(),'AAAA')
        File1b = file(corr_mat_file,'w')
        pp.dump(A,File1b)
        File1b.close()
            

    # put matrix A into config dictionary
    cfg['A'] = A

    #print A[:,:,0],'\n'
    #print A[:,:,-1],'\n'
    #exit()

    #return control value
    return 1

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def calc_corr_mat(gp_idx, station_idx, channel_idx,station_weight, time_axis, cfg):
    """Calculate correlation matrix of different Green functions.

    Indizes are given w.r.t. the Greens functions tensor GF for the chosen selection of stations and gridpoints, all starting with '0'

    input:
    -- index of source (grid point index)
    -- index of station 
    -- index of component
    -- time axis
    -- config dictionary
    sampling_in_s
    output:
    -- correlation matrix  A 
    """

    
    gf = cfg['GF']

    A_gp_s_c          = matrix(zeros((6,6), float))
    sampling_in_Hz    = float(cfg['gf_sampling_rate'])
    sampling_in_s     = 1./sampling_in_Hz

    for i in xrange(6):
        for j in xrange(i,6):

            #calculate integrand by reading in pair of greens functions with indizes i,j
            tmp2  = gf[station_idx, gp_idx, channel_idx, i,:] 
            tmp4  = gf[station_idx, gp_idx, channel_idx, j,:]

            integrand = tmp2 * tmp4
  
            #set up matrix-elements symmetrically: 
            A_gp_s_c[i,j] = station_weight * simps(integrand, dx=sampling_in_s)
            A_gp_s_c[j,i] = A_gp_s_c[i,j] 

    # return whole matrix A[i=1...6,j=1...6] for fixed set (grid point (gp), station (s), component (c) )
    return A_gp_s_c

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def calc_inv_A(cfg):

    """Gives an array with the  inverses 'inv_A[gp,s,c]' of the correlation matrices 'A[gp,s,c]'.

    input:
    -- config dictionary 

    indirect output:
    Depending of mode of inversion ('summation_key') either a simple matrix or array of matrices
    -- 0: summation over all receivers and all components - obtain matrix inv_A
    TODO:
    (-- 1: summation separated by components (k) - obtain array of 3 matrices inv_A[k])

    direct output:
    -- control parameter '1'
    """

    A                        = cfg['A']
    station_index_dict       = cfg['station_index_dictionary']
    channel_index_dict       = cfg['channel_index_dictionary']
    lo_chans                 = cfg['list_of_all_channels']
    lo_stations              = cfg['list_of_stations']

    filter_flag = int(cfg['filter_flag'])

    lo_stats = lo_stations
    # filename gets too long for too many stations
    if len(lo_stats) > 10:
        lo_stats = ['many','stations']

    try:
        filter_flag = int(cfg['filter_flag'])
        bp_lower    = cfg['bp_lower_corner']
        bp_upper    = cfg['bp_upper_corner']
        window_length = cfg['time_window_length']

    except:
        filter_flag = 0
    if not filter_flag in [0,1,2]:
        filter_flag = 0    


    #set filename for saving output
    if filter_flag == 1 or filter_flag == 2:
        inv_corr_mat_filename        = 'bandpass_filtered_current_%s_%s_%ss_inverted_matrix_A_for_stations_'%(bp_lower,bp_upper,window_length)+'_'.join(lo_stats)+'.pp'
    else:
        inv_corr_mat_filename        = 'inverted_matrix_A_for_stations_'+'_'.join(lo_stats)+'.pp'
    
    inv_corr_mat_file_abs        = path.realpath(path.abspath(path.join(cfg['temporary_directory'],inv_corr_mat_filename)))
    inv_corr_mat_file = path.realpath(path.abspath(inv_corr_mat_file_abs ))

    #if too many stations: reload stations list
    if len(lo_stats) == 2:
        lo_stats = lo_stations

    # calculate inv_A if not existing:

    try:
        #only possible, if gridpoint section has changed
        #allow only, if completely sure, that NOTHING has changed in the setup
        #raise
        # if aleready existing just read in inv_ A from file
        if int(float(cfg['changed_setup'])) == 0:
            File1b = file(inv_corr_mat_file,'r')
            inv_A  = pp.load(File1b)
            File1b.close()
            print 'read in inverted correlation matrix inv_A...'
        else:
            raise

    except:
        print 'Inversion of matrices A:'

        summation_key = int(cfg.get('summation_key',0))
        if not (summation_key == 0 or summation_key == 1):
            summation_key = 0
        lo_all_gridpoints        = cfg['list_of_all_gridpoints']

        if cfg.has_key('list_of_gridpoint_section'):
            current_lo_gridpoints= cfg['list_of_gridpoint_section']
        else:
            current_lo_gridpoints    = lo_all_gridpoints
        n_gridpoints             = len(current_lo_gridpoints)
 
        if summation_key == 0:
            
                   
            a_tmp2 = matrix(zeros((6,6),float))
            a_tmp3 = matrix(zeros((6,6),float))
            inv_A  = zeros((6,6,n_gridpoints),float)
            
            for gp in arange(n_gridpoints):
                #inversion of the respective correlation matrix for each grid point
                a_tmp2         =  matrix(A[:,:,gp])
                a_tmp3         =  inv(a_tmp2)
                inv_A[:,:,gp]  =  a_tmp3

        if summation_key == 1:
            a_tmp2 = matrix(zeros((6,6),float))
            a_tmp3 = matrix(zeros((6,6),float))
            inv_A  = zeros((6,6,n_gridpoints,3),float)

            pass

        #save inv_A for further use in file
        File1c = file(inv_corr_mat_file,'w')
        pp.dump(inv_A,File1c)
        File1c.close()
    

    #save inv_A in config dictionary
    cfg['inv_A'] = inv_A

    #print inv_A[:,:,0],'\n'
    #print inv_A[:,:,-1],'\n'
    #exit()
    
    if _debug:
        print shape(inv_A)

        
    #return control parameter
    return 1
   

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def calc_corr_vec(cfg):
    """Calculate array of correlation vectors of data and Green's function 'b[time_window,M_component,grid point]'.

    input:
    -- config dictionary
    
    indirect output:
    -- array of correlation vectors b

    direct output:
    -- control parameter
    """

    
    data_array          = cfg['data_array']
    gf_array            = cfg['GF']

    print 'size of GF, data: ' ,shape(gf_array),shape(data_array)
    time_array          = cfg['data_time_axes_array']
    
    
    lo_stations_w_data  =  cfg['list_of_stations_with_data']

    filter_flag         = int(cfg['filter_flag'])

    lo_stats = lo_stations_w_data
    # filename gets too long for too many stations
    if len(lo_stats) > 10 :
        lo_stats = ['many','stations']
    try:
        filter_flag = int(cfg['filter_flag'])
        bp_lower    = cfg['bp_lower_corner']
        bp_upper    = cfg['bp_upper_corner']
        window_length = cfg['time_window_length']

    except:
        filter_flag = 0
    if not filter_flag in [0,1,2]:
        filter_flag = 0    

    #set filename for saving output
    if filter_flag == 1 or filter_flag == 2 :
        corr_vec_filename           = path.realpath(path.abspath(path.join(cfg['temporary_directory'], 'bandpass_filtered_current_%s_%s_%ss_corr_vec_for_stations_'%(bp_lower,bp_upper,window_length)+'_'.join(lo_stats)+'.pp')))
    elif int(cfg['filter_flag']) == 0:
        corr_vec_filename           = path.realpath(path.abspath(path.join(cfg['temporary_directory'], 'current_corr_vec_for_stations_'+'_'.join(lo_stats)+'.pp')))
    else:
        print 'ERROR!!! filter_flag must be "0" or "1" or "2"'
        raise SystemExit

    #if too many stations: reload stations list
    if len(lo_stats) == 2 :
        lo_stats    = lo_stations_w_data
  

    #load b from file if existing        
    try:

        #HUMBUG - muss ueberarbeitet werden!!!!
        raise
    
        File1d = file(corr_vec_filename,'r')
        b      = pp.load(File1d)
        File1d.close()

        print 'Reading array of correlation vectors "b" from file: ', corr_vec_filename


    # otherwise calculation of b  
    except:      
        print 'Calculation of correlation vector array "b":...'

        key              =  int(cfg['summation_key'])
        filter_flag      =  int(cfg['filter_flag'])
        sampling_in_s    =  1./float(cfg['data_sampling_rate'])

        if cfg.has_key('list_of_gridpoint_section'):
            lo_gp  = cfg['list_of_gridpoint_section']
        else:
            lo_gp  = cfg['list_of_gridpoints']
#         print lo_gp
#         exit()

        n_gridpoints     =  len(lo_gp)
         
        lo_receivers     = cfg['list_of_receivers_with_data']
        n_receivers      =  len(lo_receivers)

        stations_array            = cfg['station_coordinates']
        channel_index_dict        = cfg['channel_index_dictionary']
        stat_idx_dict             = cfg['station_index_dictionary']
        cont_stat_index_dict      = cfg['ContributingStation_index_dictionary']
        receiver_index_dict       = cfg['receiver_index_dictionary']
        list_of_moving_window_starttimes  = cfg['list_of_moving_window_starttimes']
        
        
        number_of_window_steps    = len(list_of_moving_window_starttimes)

        # time axis
        n_time      =  int(float(cfg['time_window_length'])*float(cfg['data_sampling_rate']) + 1 )
        
        #temporary values, vector and integrand:
        b_tmp       =  zeros((number_of_window_steps, n_gridpoints, n_receivers,  6),float)
        int_tmp     =  zeros((n_gridpoints, n_receivers,  6, n_time),float)

        if _debug:
            lo_added_receivers = []

        #d_break(locals(),'bbbb1')

        # loop over all time windows
        for window_idx in arange(number_of_window_steps):
            best_data_start_idx  = cfg['list_of_moving_window_startidx'][window_idx]

            
            #loop over all receivers - combination of station and channel  
            for receiver in lo_receivers:

                receiver_index  =  int(float( receiver_index_dict[receiver]  ))

                station_name    = receiver.split('.')[0]
                station_index   = int(stat_idx_dict[station_name])-1
                abs_station_index = int(stat_idx_dict[station_name])-1
                station_weight  = cfg['station_weights_dict'][station_name]#stations_array[abs_station_index,4]

                channel         = receiver.split('.')[1]
                regex_check     = re.compile(r'\w$')
                channel_letter  = regex_check.findall(channel)[0]

                if _debug:
                    pass
                    #print receiver, station_name, channel_letter
                
                channel_index   = int(channel_index_dict[channel_letter])

                #tmp5            = station_weight * data_array[window_idx, receiver_index, :][:]
                #print cfg['data_trace_samples']
                tmp5            = station_weight * data_array[receiver_index,best_data_start_idx:best_data_start_idx + cfg['data_trace_samples'] ].copy()

                #loop over source points
                for gp in  arange(n_gridpoints):
                    gp_index = lo_gp[gp]-1
                    # print shape(gf_array)
#                     print  gp_index
#                     exit()
                    

                    #loop over all M components
                    for m_component in xrange(6):

                        #read in green function and data:
                        #print station_index, gp_index, channel_index, m_component
                        tmp2                   = gf_array[station_index, gp, channel_index, m_component,:len(tmp5)][:]
                        #setup integrand:
                        #print shape(tmp2),shape(tmp5)
                        int_tmp                = tmp2 * tmp5

                        #
                        #calculate correlation vector by integrating with scipy internal Simpson rule:
                        #print sampling_in_s
                        #exit()
                        
                        b_tmp[window_idx, gp, receiver_index, m_component]  = simps(int_tmp ,dx=sampling_in_s)
                        #visual check
                        if _debug_plot:
                            if gp == 50 and ( m_component == 4):
                            
                                figure(  10 * (station_index+1) + channel_index )
                                plot(tmp2)
                                figure(100 +  10 * (station_index+1) + channel_index)
                                plot(tmp5)
                                
                                print 'plotted gf(4) for station ',station_name,' and channel  ',channel_letter, ' in window ',  10 * (station_index+1) + channel_index
                                print 'plotted data for station ',station_name,' and channel  ',channel_letter, ' in window ',100 +  10 * (station_index+1) + channel_index,'\n'
                                print sum(tmp2-tmp5)


                                
                #                            print "prepared element (s,r,c_m)=",idx_s,idx_r,idx_k,j," of b"

                if _debug:
                    lo_added_receivers.append(receiver)

        # calculate sum over all stations for each time window, source point, and M component 
        if key == 0:
            b     =  zeros((number_of_window_steps,6, n_gridpoints),float)
            for window_idx in arange(number_of_window_steps):
                for gp in xrange(n_gridpoints):
                    for m_component in xrange(6):
                        b[window_idx, m_component, gp]   =  sum(b_tmp[window_idx,gp, :, m_component])
                            

        #save array 'b' into file
        File1d = file(corr_vec_filename,'w')
        pp.dump(b,File1d)
        File1d.close()

    #write array 'b' into config dictionary
    cfg['b'] = b

    #print '\n\n BBBBBBBBBBBBBBBBBBBBBBBB:\n\n', b[0,:,0],'\n\n\n'
#     print b[0,:,-1],'\n'
    #exit()
    
    if _debug:
        set_of_added_receivers = sort(list(set(lo_added_receivers)))

    #d_break(locals(),'bbbb2')

    if _debug:
        print shape(b)

    #control parameter
    return 1

#----------------------------------------------------------------------
def corr_vec_temp(data_array, cfg):


    gf_array            = cfg['GF']

    lo_stations_w_data  =  cfg['list_of_stations_with_data']

    sampling_in_s    =  1./float(cfg['data_sampling_rate'])

    if cfg.has_key('list_of_gridpoint_section'):
        lo_gp  = cfg['list_of_gridpoint_section']
    else:
        lo_gp  = cfg['list_of_gridpoints']


    n_gridpoints     =  len(lo_gp)
     
    lo_receivers     = cfg['list_of_receivers_with_data']

    
    n_receivers      =  len(lo_receivers)

    stations_array            = cfg['station_coordinates']
    channel_index_dict        = cfg['channel_index_dictionary']
    stat_idx_dict             = cfg['station_index_dictionary']
    cont_stat_index_dict      = cfg['ContributingStation_index_dictionary']
    receiver_index_dict       = cfg['receiver_index_dictionary']

    n_time      =  int(float(cfg['time_window_length'])*float(cfg['data_sampling_rate']) + 1 )

    b_tmp       =  zeros((n_gridpoints, n_receivers,  6),float)

    

    for receiver in lo_receivers:

        receiver_index  =  int(float( receiver_index_dict[receiver]  ))

        station_name    = receiver.split('.')[0]
        station_index   = int(stat_idx_dict[station_name])-1
        abs_station_index = int(stat_idx_dict[station_name])-1
        station_weight  = cfg['station_weights_dict'][station_name]#stations_array[abs_station_index,4]

        channel         = receiver.split('.')[1]
        regex_check     = re.compile(r'\w$')
        channel_letter  = regex_check.findall(channel)[0]
        channel_index   = int(channel_index_dict[channel_letter])

        tmp5            = station_weight * data_array[receiver_index,:]

        for gp in  arange(n_gridpoints):
            gp_index = lo_gp[gp]-1
            
            for m_component in xrange(6):


                tmp2             = gf_array[station_index, gp, channel_index, m_component,:len(tmp5)][:]
                int_tmp          = tmp2 * tmp5
                b_tmp[ gp, receiver_index, m_component]  = simps(int_tmp ,dx=sampling_in_s)

    b_4window  =  zeros((6, n_gridpoints),float)
    for gp in xrange(n_gridpoints):
        for m_component in xrange(6):
            b_4window[m_component, gp]   =  sum(b_tmp[gp, :, m_component])


    #d_break(locals(),'continuous - current b ')

    return b_4window

#----------------------------------------------------------------------
def read_in_data_and_find_source_continuous(traces,cfg):

    #neeed:
    # station_index_dictionary
    #set up data array in corrected order

    filter_flag = int(cfg['filter_flag'])
    time_stamp  =  cfg['current_window_starttime']
    lo_stations            = cfg['list_of_stations']
    
    list_of_ContributingStations = []
    list_of_receivers            = []
    list_of_channels             = []

    
    for kk in traces:
        if kk.station in lo_stations:
            list_of_ContributingStations.append(kk.station)
            list_of_channels.append(kk.channel)
            receiver_name = str(kk.station)+'.'+str(kk.channel)
            list_of_receivers.append(receiver_name)

    list_of_ContributingStations = list(set(list_of_ContributingStations ))
    list_of_receivers            = list(set(list_of_receivers))
    list_of_channels             = list(set(list_of_channels))

    cfg['list_of_stations_with_data']     = list(sort(list_of_ContributingStations))
    cfg['list_of_receivers_with_data']    = list(sort(list_of_receivers))
    cfg['list_of_channels_with_data']     = list(sort(list_of_channels))
    
    make_dict_stations_indices_complete(cfg)
    make_dict_ContributingStations_indices(cfg)
    make_dict_receivers_indices(cfg)
    if not correct_inv_A_4_missing_traces_and_weights(cfg):
        print 'ERROR !! Updating inv_A not poosible'
        raise SystemExit


    rec_idx_dict = cfg['receiver_index_dictionary']

    tr1 = traces[0]
    gf_sampling = float(cfg['gf_sampling_rate'])
    t_min       = tr1.tmin
    d_samp_in_s = tr1.deltat
    cfg['data_sampling_rate'] =  1./d_samp_in_s
    gf_samp_in_s= 1./float(cfg['gf_sampling_rate'])
    rounded_samp_d_ms  = int(round(d_samp_in_s*1000  , -2) )
    rounded_samp_gf_ms = int(round(gf_samp_in_s*1000 , -2))
    if not rounded_samp_d_ms == rounded_samp_gf_ms:
        if rounded_samp_d_ms%rounded_samp_gf_ms == 0:
            cfg['data_sampling_rate'] = 1./gf_samp_in_s
            tr1.downsample_to(gf_samp_in_s)
    n_samples  = len(tr1.ydata)
    
    ta           = tr1.get_xdata()
    ta_internal  = ta - time_stamp


    data_array   = zeros(( len(list_of_receivers), n_samples),float32)
    for tr_n in traces:
        channel   = tr_n.channel
        station   = tr_n.station
        receiver  = station+'.'+channel
        if rec_idx_dict.has_key(receiver):
            rec_idx   = rec_idx_dict[receiver]
        else:
            continue
        #tr_n.downsample_to(float(1./cfg['data_sampling_rate']))

        #correcting data for unphysical/meaningless entries
        model_tminmax   =  cfg['stations_tmin_tmax'][station]
        section_t_min   = model_tminmax[0]
        section_t_max   = min([model_tminmax[1],ta_internal[-1] ])
        idx_tmin  = (abs(ta_internal-section_t_min)).argmin()
        idx_tmax  = (abs(ta_internal-section_t_max)).argmin()
        
        in_data_raw  =  tr_n.get_ydata()
        in_data      =  zeros((len(in_data_raw)))
        in_data[idx_tmin:idx_tmax]=taper_data(in_data_raw[idx_tmin:idx_tmax])

        if int(cfg['filter_flag']) == 2:
            in_data = taper_data(bp_butterworth(in_data,cfg['data_sampling_rate'],cfg))
        
        data_array[rec_idx,:] = in_data


    cfg['data_array'] = data_array
    cfg['optimal_time_window_index'] = 0
    cfg['list_of_moving_window_startidx'] = {'0':0,0:'0'}
    cfg['data_trace_samples'] = n_samples
    cfg['optimal_source_time'] =  t_min

    
    #d_break(locals(),'continuous - data array read in -- start %s'%(util.time_to_str(ta[0])))

    #build correlation vector b from GF and data
    
    
    try:
        b_4window = corr_vec_temp(data_array,cfg)
    except:
        sys.stderr.write(   '\nERROR! could not set up b!\n')
        sys.stderr.write(  util.time_to_str(t_min))
        raise 

    if not estimate_m(b_4window, cfg):
        sys.stderr.write(   '\nERROR! could not estimate M!\n')
        sys.stderr.write(  util.time_to_str(t_min))
        raise
    m = cfg['M_array'].copy()

    #d_break(locals(),'')
            
    if not  estimate_res(data_array, cfg['M_array'], cfg):
        sys.stderr.write(   '\nERROR! could not estimate RES!\n')
        sys.stderr.write(  util.time_to_str(t_min))
        raise
    if not estimate_vr(data_array, cfg):
        sys.stderr.write(   '\nERROR! could not estimate VR!\n')
        sys.stderr.write(  util.time_to_str(t_min))
        
       
    vr = cfg['VR_array']
    #set arrays for storing parameters of optimal source
    m_opt     = zeros((6),float)
    source    = zeros((1),float)
    vr_opt    = zeros((1),float)

    #find element with maximal vr:
    idx_max      = argmax(vr)
    source_index = idx_max
    lo_gridpoint_section     = cfg['list_of_gridpoint_section']
    source       = lo_gridpoint_section[source_index]
    m_opt[:]     = m[idx_max, :]
    vr_opt       = vr[idx_max]
    #put array M of optimal source into 3x3 matrix shape
    m_opt_mat    = matrix(array((m_opt[0],m_opt[3],m_opt[4],m_opt[3],m_opt[1],m_opt[5],m_opt[4],m_opt[5],m_opt[2])).reshape(3,3))

    temp_solutions_dict = {}
    #put solution for current time window into sub dictionary
    temp_solutions_dict['M']      = m
    temp_solutions_dict['M_opt']  = m_opt
    temp_solutions_dict['source_index'] = source_index
    temp_solutions_dict['source'] = source
    temp_solutions_dict['VR_opt'] = vr_opt
    temp_solutions_dict['VR_array'] = vr.copy()
    temp_solutions_dict['source_time'] = t_min 

    #put    solution dictionary into cfg
    cfg['temp_solution_dictionary']  =   temp_solutions_dict    
    cfg['optimal_source_dictionary'] = temp_solutions_dict

    cfg['optimal_source_index']  = source_index
    cfg['optimal_M']  = m_opt

    decomp_m(m_opt_mat, cfg)
    
    #d_break(locals(),'end of run for step in continuous mode')
    
    return vr_opt

#----------------------------------------------------------------------

def read_in_data(cfg):
    """Reading data files into pure ascii time-series.

    input:
    -- config dictionary

    indirect output:
    -- data array into config dictionary
    -- time axes  into config dictionary
    
    direct output:
    -- control parameter
    """

    print 'Reading the current data-set:...'
    if cfg['change_z_up_down']:
        print "data Z changed from 'down' to 'up'" 
    
    if _debug:
        lo_added_receivers =[]
 
    temp_datetime   = cfg['datetime']  
    lo_datetime     = temp_datetime.split(':')
    datetime_new    = '_'.join(lo_datetime)

    lo_stations     = cfg['list_of_stations_with_data']

    filter_flag = int(cfg['filter_flag'])

    lo_stats = lo_stations
    # filename gets too long for too many stations
    if len(lo_stats) > 10:
        lo_stats = ['many','stations']

    #set filename for saving output    
    if filter_flag == 1 or filter_flag == 2 :
        data_filename = path.realpath(path.abspath(path.join(cfg['temporary_directory'], 'bandpass_filtered_data_array_v%i_event%s_stations_%s.pp'% (int(cfg['data_version']),datetime_new,'_'.join(lo_stats) )  )))
        data_time_filename = path.realpath(path.abspath(path.join(cfg['temporary_directory'], 'bandpass_filtered_current_data_time_axis_event%s_stations_%s.pp'% (datetime_new,'_'.join(lo_stats) ))))

    elif filter_flag == 0:
        data_filename = path.realpath(path.abspath(path.join(cfg['temporary_directory'], 'data_array_v%i_event%s_stations_%s.pp'% (int(cfg['data_version']),datetime_new,'_'.join(lo_stats) )  )))
        data_time_filename = path.realpath(path.abspath(path.join(cfg['temporary_directory'], 'current_data_time_axis_event%s_stations_%s.pp'% (datetime_new,'_'.join(lo_stats) ))))

    else:
        print 'ERROR!!! filter_flag must be "0" or "1" or "2"  '
    
    
    #if too many stations: reload stations list 
    if len(lo_stats)  ==  2:
        lo_stats = lo_stations

    #d_break(locals(),'filename datafile')
    
    #TODO: nur zum debuggen
    #load data from file if existing        
    if 0:#path.isfile(data_filename) and path.isfile(data_time_filename):

        File8a = file(data_filename,'r')
        print 'read from ',data_filename, 'and ', data_time_filename
        data_array       = pp.load(File8a)
        File8a.close()
        File9a = file(data_time_filename,'r')
        time_axes_array  = pp.load(File9a)
        File9a.close()
        print 'data ... re-read from file\n'
        
        data_pile_for_main_window = cfg['data_pile_for_main_window']
        data_sampling = (1./data_pile_for_main_window[0].deltat)
        #cfg['sampling_rate']      = data_sampling
        cfg['data_sampling_rate'] = data_sampling


    # otherwise read in data    
    else:
        
        data_pile_for_main_window = cfg['data_pile_for_main_window']

        lo_receivers   = cfg['list_of_receivers_with_data']
        lo_cont_stats  = cfg['list_of_stations_with_data']
        lo_channels    = cfg['list_of_channels_with_data']

#         print lo_receivers
#         print lo_cont_stats
#         print lo_channels
#         exit()
        
#         length      = int(cfg['time_window_length'])
        length      = (float(cfg['main_window_length']))

        gf_sampling = float(cfg['gf_sampling_rate'])
                
                
        data_sampling             = (1./data_pile_for_main_window[0].deltat)
        cfg['data_sampling_rate'] = data_sampling
        
        n_time      = int(length * min(data_sampling,gf_sampling)) + 1
        cfg['data_trace_samples'] =  int( float(cfg['time_window_length'])* min(data_sampling,gf_sampling)) + 1

        parent_data_directory            = cfg['parent_data_directory']
        receiver_index_dictionary        = cfg['receiver_index_dictionary']
        channel_index_dict               = cfg['channel_index_dictionary']
        station_index_dict               = cfg['station_index_dictionary']
        window_tmin_tmax_dict            = cfg['stations_tmin_tmax']
        list_of_moving_window_starttimes = cfg['list_of_moving_window_starttimes']
        number_of_window_steps           = len(list_of_moving_window_starttimes)
        
#         data_array      = zeros((number_of_window_steps, len(lo_receivers), n_time),float32)
#         time_axes_array = zeros((number_of_window_steps,n_time),float32)
        data_array      = zeros(( len(lo_receivers), n_time),float32)
        time_axes_array = zeros((n_time),float32)


        # loop over time windows
        for window_idx in arange(1):#number_of_window_steps):

            window_starttime = (list_of_moving_window_starttimes[window_idx])

            #loop over receivers 
            for current_rec in lo_receivers:
                receiver_index = int(receiver_index_dictionary[current_rec])
                dummy1 = current_rec.split('.')
                current_station = dummy1[0]
                current_channel = dummy1[1]
                station_index   = int(station_index_dict[current_station])


                #loop for finding the appropriate data pile for the respective station/component combination
                for fff in arange(len(cfg['list_of_datafiles'])):
                    dummy2 = data_pile_for_main_window[fff]

                    #searching for the correct station.channel combination in the unsorted set; breaking, if found:
                    if dummy2.channel == current_channel and dummy2.station == current_station:
                        current_data_pile_idx = fff
                        break

                #extract correct data pile for the given receiver 
                current_data_pile_element = data_pile_for_main_window[current_data_pile_idx]
                   
#TODO !!! auch in andere richtung resampling ermoeglichen
                if (not int(1./current_data_pile_element.deltat) == int(gf_sampling)):
                    if data_sampling%gf_sampling == 0:
                        downsample_factor    = int(data_sampling/gf_sampling)
                        current_data_pile_element.downsample(downsample_factor)
                        cfg['sampling_rate'] = data_sampling/downsample_factor
                        cfg['data_sampling_rate'] = data_sampling/downsample_factor
                        #n_time               =  int(length * data_sampling/downsample_factor) + 1


                    #error, if downsampling not possible
                    else:
                        exit( 'ERROR : incompatible sampling rates [Hz] of data and GF !!!!\n ',data_sampling, gf_sampling)

                #endtime must be one sample later than time of interest
                window_endtime = window_starttime + length + current_data_pile_element.deltat
                #print window_starttime,length,current_data_pile_element.deltat,window_endtime

                #cut out time interval
                current_pile_section = current_data_pile_element.chop(window_starttime,window_endtime )

                #
                # use only  section of data which is physically meaningful:
                #
#                 model_tminmax   = window_tmin_tmax_dict[current_station]+window_starttime

#                 section_t_min   = model_tminmax[0]
#                 if model_tminmax[0] > window_endtime:
#                     section_t_min = window_starttime
                    
#                 section_t_max = min([model_tminmax[1],window_endtime ])
                
#                 ta_min    = current_pile_section.tmin
#                 deltat    = current_pile_section.deltat
#                 ta_len    = len(current_pile_section.ydata)
#                 ta        = arange(ta_len)*deltat+ta_min
#                 idx_tmin  = (abs(ta-section_t_min)).argmin()
#                 idx_tmax  = (abs(ta-section_t_max)).argmin()

#                 # copy  out interesting data
#                 current_data2taper = current_pile_section.ydata[idx_tmin:idx_tmax].copy()
#                 # set trace to 0
#                 current_pile_section.ydata = 0
#                 # fill tapered interesting part as only signal 
#                 current_pile_section.ydata[idx_tmin:idx_tmax] = taper_data(current_data2taper)
                

                #check, if right interval length has been taken
                #print current_rec,len(current_pile_section.ydata),window_starttime,window_endtime
                if (not len(current_pile_section.ydata) == n_time ):
                    exit_string = 'ERROR !!!! Chosen section has wrong length: ', len(current_pile_section.ydata), ' -- must have ', n_time
                    #exit(exit_string )
                    #print exit_string

                #option: filter here with butterworth bandpass...   
                #if (int(cfg['filter_flag']) == 2) :
                #    current_pile_section.bandpass( int(cfg['bp_order']),float(cfg['bp_lower_corner']) , float(cfg['bp_upper_corner']))

                #datalonger = cmp(len(current_pile_section.ydata),len(data_array[window_idx,receiver_index,:]))
                datalonger = cmp(len(current_pile_section.ydata),len(data_array[receiver_index,:]))
                if datalonger == 1 :
                    #                     data_trace_to_store = current_pile_section.ydata[:len(data_array[window_idx,receiver_index,:])]
                    data_trace_to_store = current_pile_section.ydata[:len(data_array[receiver_index,:])]
                else:
                    data_trace_to_store = current_pile_section.ydata

                if cfg.has_key('restitute') and cfg['restitute'].upper() in ['TRUE', '1']:
                    tmp_trace = data_trace_to_store - mean(data_trace_to_store)
                    dat_out    = cumsum(tmp_trace)
                    dat_out    *= 1./gf_sampling
                    data_trace_to_store = dat_out
                    
                #d_break(locals(), 'data trace to store')
                #exit()               


                #d_break(locals(),'')
                if 0:#current_rec == 'HM20.BHZ':
                    d_break(locals(),'')
                    plot(data_trace_to_store)
                    show()
                    raw_input()
                    exit()
        
                    #print current_rec,data_trace_to_store

                if cfg['change_z_up_down'] == 1 and channel_index_dict[current_channel] == 2 :
                    #data_array[window_idx,receiver_index,:len(data_trace_to_store)] = -data_trace_to_store
                    data_array[receiver_index,:len(data_trace_to_store)] = -data_trace_to_store

                else:
                    #data_array[window_idx,receiver_index,:len(data_trace_to_store)] = data_trace_to_store
                    data_array[receiver_index,:len(data_trace_to_store)] = data_trace_to_store

                if _debug:
                    lo_added_receivers.append(current_rec)
 
                #option 2: ... or here with boxcar filter
                if 0:#(int(cfg['filter_flag']) == 1):
                    temp_in_data= current_pile_section.ydata
                    temp_filtered = bp_boxcar(taper_data(temp_in_data),float(cfg['data_sampling_rate']),  cfg)
                    #data_array[window_idx,receiver_index,:] = temp_filtered
                    data_array[receiver_index,:] = temp_filtered

                # current_time_axis = current_pile_section.make_xdata()
#                 #print current_time_axis
#                 #read time axis from data pile object
#                 try:
#                     if len(current_time_axis) < len(temp_time_axis):
#                         dummy27 = zeros((len(temp_time_axis)))
#                         dummy27[:len(current_time_axis)] = current_time_axis
#                         temp_time_axis += dummy27
#                     else:
#                         temp_time_axis += current_time_axis
#                 except:
#                     temp_time_axis += current_time_axis


                #visualisation for debugging
#                 if _debug_plot:
#                     station_index = station_index_dict[current_station]
#                     channel_index = channel_index_dict[current_channel]
#                     figure(70 +  10 * (station_index+1) + channel_index)
#                     print 70 +  10 * (station_index+1) + channel_index , current_rec

#                     plot(data_array[window_idx,receiver_index,:])

#                     print 'plotted data for station ',current_station,' and channel  ',current_channel, ' in window ',70 +  10 * (station_index+1) + channel_index,'\n'


            #building average time axis for case of slight time shifts
            #print len(temp_time_axis),len(time_axes_array[window_idx,:])

            temp_time_axis = arange(n_time)*current_data_pile_element.deltat +  window_starttime
            
            #time_axes_array[window_idx,:] = temp_time_axis
            time_axes_array[:] = temp_time_axis

        #save data to file for later reprocessing
        #print data_filename    
        #File8a = file(data_filename,'w')
        #pp.dump(data_array,File8a)
        #File8a.close()
        #File9 = file(data_time_filename,'w')
        #pp.dump(time_axes_array,File9)
        #File9.close()
        #print 'data-array written to file\n'
    
    
    if _debug:
        print 'data_db and time set: \n data shape:',shape(data_array),'\n time-axis shape:',shape(time_axes_array),'\n '

    if int(cfg['filter_flag']) == 1 or int(cfg['filter_flag']) == 2 :
        print '\n\n     Data are bandpass filtered !!!!\n'

    if cfg.has_key('restitute') and cfg['restitute'].upper() in ['TRUE', '1']:
        print '\n\n     Data are restituted (integrated) !!!!\n'

    #put data and time axes into config dictionary
    cfg['data_time_axes_array']  = time_axes_array
    cfg['data_array'] = data_array

    if _debug:
        print shape(data_array),data_array.max(),data_array.min()
        set_of_added_receivers = sort(list(set(lo_added_receivers)))

    #d_break(locals(),'DATAaaaaa')
    #exit()
    
    #control parameter
    return 1
    
#----------------------------------------------------------------------
#----------------------------------------------------------------------

def read_in_GF(cfg):
    """Reading gf-files into combined array.

    input:
    -- config dictionary

    indirect output:
    -- time axis, array of Greens functions 'gf'

    direct output:
    -- control parameter
    """

    #pdb.set_trace()


    gf_version = int(cfg['gf_version'])

    #list of chosen stations
    lo_stations     = cfg['list_of_stations']
    n_stations      = len(lo_stations)
    print 'current number of stations: ',n_stations,':\n (',lo_stations,')'

    window_tmin_tmax_dict        = cfg['stations_tmin_tmax']
   

    #list of source points, according to chosen section
    try:
        gs_section_key  = int(cfg.get('gs_type_of_section'))
    except:
        gs_section_key  = 0

    if gs_section_key == 0:
        current_lo_gridpoints    = cfg['list_of_all_gridpoints'] 
    else:
        current_lo_gridpoints    = cfg['list_of_gridpoint_section']

    n_gridpoints_full       = len( cfg['list_of_all_gridpoints'])
    n_gridpoints_section    = len(current_lo_gridpoints)

    print 'number of current gridpoints: %i\n'%(n_gridpoints_section)
    #time parameters
    length_of_orig_gf = int(cfg['gf_length'])
    #length           = int(cfg['gf_length'])
    length           = float(cfg['time_window_length'])
    gf_sampling      = float(cfg['gf_sampling_rate'])
    n_time           = int(float(length) * gf_sampling) + 1

    #filtering
    try:
        filter_flag = int(cfg['filter_flag'])
        bp_lower    = cfg['bp_lower_corner']
        bp_upper    = cfg['bp_upper_corner']
        window_length = cfg['time_window_length']
        
    except:
        filter_flag = 0
    if not filter_flag in [0,1,2]:
        filter_flag = 0

    try:
        n_rdm_stations = int(cfg.get('number_of_random_stations',0))
    except:
        n_rdm_stations = 0
    
    #dummy list for naming file
    lo_stats  = lo_stations

    # temporary filename for storing array gets too long for too many stations - restrict to 10    
    if len(lo_stats) > 10:        
        lo_stats = ['many','stations']
    if n_rdm_stations >= 10:
        lo_stats = ['many','random','stations']


    #set filename for saving output            
    if filter_flag == 1 or filter_flag == 2:
        GF_pickle_filename          = 'bandpass_filtered_current_%s_%s_%ss_GF_pickle_file_for_stations_'%(bp_lower,bp_upper,window_length)+'_'.join(lo_stats)+'.pp'
    else:
        GF_pickle_filename          = 'current_GF_pickle_file_for_stations_'+'_'.join(lo_stats)+'.pp'


    GF_pickle_file_abs          = path.realpath(path.abspath(path.join(cfg['temporary_directory'],GF_pickle_filename)))    
    current_time_axis_file_abs  = path.realpath(path.abspath(path.join(cfg['temporary_directory'],'current_GF_time_axis_file.pp')))


    #if too many stations: reset stations list    
    if len(lo_stats) in [2,3]:
        lo_stats                    = lo_stations


    #load data from file if existing  (gridpoint/station setup unchanged)
 
    if path.exists(GF_pickle_file_abs) and path.exists(current_time_axis_file_abs) and n_rdm_stations < 10:
        print 'read in  Greens Functions from local pickle-file... \n(%s)\n\n'%(GF_pickle_file_abs)

        File6a = file(GF_pickle_file_abs,'rb')
        #read successively from binary file - important - keep right order
        #first 'type'...
        GF_type = pp.load(File6a)
        #...then 'shape' of array...
        GF_shape = pp.load(File6a)

        #calculate total size of data string
        number_of_elements = 1
        for ee in GF_shape:
            number_of_elements *= ee
        #...finally read the data
        temp_GF = fromfile(File6a, GF_type, number_of_elements)

        #reshape data array according to saved structure
        gf_full = temp_GF.reshape(GF_shape)

        File6a.close()

        print 'Greens Functions array re-read from file\n shape: ' ,shape(gf_full),'\n'

        print 'read in  current time axis from local file... \n'
        File7a = file(current_time_axis_file_abs,'r')
        time_axis = pp.load(File7a)
        File7a.close()
        print 'time axis re-read from file\n'
        
    # otherwise read in Greens functions from NetCDF files 
    else:
        print 'reading in  Greens Functions from station specific source files... \n'
        print 'looking up files with GF in path ', path.abspath(cfg['gf_dir']),'  ...\n'

        print 'approx size of GF tensor to keep in memory: %i MB \n'%(int(4*n_stations* n_gridpoints_full*18*n_time/1024**2 ))
        #setup array for holding Greens functions
        gf_full              = zeros((n_stations, n_gridpoints_full,  3,6, n_time),float32)

        if _debug:
            print 'GF dimensions:', shape(gf_full),'\n'
            print 'approx size of file: %i MB '%(int(4 * n_stations * n_gridpoints_full * 18 * n_time/1024**2 ))

        station_index_dict = cfg['station_index_dictionary']

        stationcounter = 1
        for current_stat in lo_stats:
            
            
            idx_stat = int(station_index_dict[current_stat])-1
            abs_index = int( cfg['station_index_dictionary_complete'][current_stat])
            print 'Station %.5s - Index: %2i (no. %3i of %3i) ' %(current_stat,abs_index,stationcounter,n_stations)
            stationcounter += 1
            
            #build input filename:
            input_filename          = 'gf_v%i_length%i_sampling%.2f_station_%s.nc' % (gf_version,length_of_orig_gf,float(gf_sampling),current_stat)
            input_filename_total    =  path.realpath(path.abspath(path.join( cfg['gf_dir'],input_filename))) 
            print 'reading from: ',input_filename_total,'\n'

            #set the NetCDF handler/object
            tmp_ncfile_gf        = sioncdf.NetCDFFile(input_filename_total,'r')
                
            #read in gf for current station
            #dictionary with content of file
            contained_vars_dict  = tmp_ncfile_gf.variables
            #list with dictionary keys
            contained_vars       = contained_vars_dict.keys()
            #name of the first dictionary entry
            gf_var               = contained_vars[0]
            #grab data from the first value in the dictionary
            tmp_data             = tmp_ncfile_gf.variables[gf_var]
            #convert data structure into float64 array
            gf_raw               = array(tmp_data).astype('float64')

            #close input file
            tmp_ncfile_gf.close()

            #transpose GF raw data 
            tmp_gf2         =  gf_raw.transpose()

            #set time axis, cut, if too long
            time_axis       =  tmp_gf2[0,:n_time]
            #time_axis       =  time_axis[:n_time]

            model_tminmax   = window_tmin_tmax_dict[current_stat]
            section_t_min   = model_tminmax[0]
            section_t_max   = min([model_tminmax[1],time_axis[-1] ])

            #loop over grid points, components, M_components to set array of Greens functions
            for gp in xrange(n_gridpoints_full):
                
                #for every gridpoint
                idx_s_real = int(cfg['list_of_all_gridpoints'][gp])-1
                for idx_k in xrange(3):
                    #for every station component
                    for idx_m in xrange(6):
                        #for every moment tensor component
                        #set right column index
                        #data is sorted in 18 columns for each station, therein the order is (N,E,Zup), within these 6 columns for M
                        #adding '1' for the time axis as the first entry in the array
                        current_column = 18*(idx_s_real) + 6*idx_k + idx_m + 1
                        try:
                            in_data_raw    = tmp_gf2[current_column,:n_time ]
                        except:
                            d_break(locals(),' size of NetCDF array wrong')
                            exit()
                            
                        #cut  section identically to data section
                        ta_gf = arange(len(in_data_raw))/gf_sampling
                        idx_tmin  = (abs(ta_gf-section_t_min)).argmin()
                        idx_tmax  = (abs(ta_gf-section_t_max)).argmin()

                        in_data = zeros((len(in_data_raw)))

                        #double tapering ONLY for analysis of synthetic data
                        # which is built by already taperd GFs

                        #in_data[idx_tmin:idx_tmax]=taper_data(taper_data(in_data_raw[idx_tmin:idx_tmax]))
                        in_data[idx_tmin:idx_tmax]=(taper_data(in_data_raw[idx_tmin:idx_tmax]))
                      

                        #option 2: filter data with butterworth...
                        if filter_flag == 2 :

                            out_data                      =  taper_data(bp_butterworth(in_data,gf_sampling,cfg))
                            gf_full[idx_stat,gp,idx_k,idx_m,:] = out_data
                            
                        #option 1:...or boxcar function...
                        elif filter_flag == 1 :

                            out_data = taper_data(bp_boxcar(in_data,gf_sampling, cfg))
                            gf_full[idx_stat,gp,idx_k,idx_m,:] = out_data

                        #option 3: ...or not at all
                        else:                            
                            gf_full[idx_stat,gp,idx_k,idx_m,:] = in_data

                print_string= 'Greens functions for station %.6s and source index %4i read in! \r'%(current_stat,idx_s_real+1)
                sys.stdout.write(print_string)
            print '\n\n Greens Functions for station ' + current_stat +' ...set !\n'
            
            if filter_flag == 1 or filter_flag == 2 :
                print '\n GFs are bandpass filtered !!\n'


        #write GF array into binary pickle file 
        File6 = file(GF_pickle_file_abs,'wb')
        #important - keep right order:
        GF_type  = gf_full.dtype
        GF_shape = gf_full.shape

        #first 'type'..
        pp.dump(GF_type,File6)
        #second 'shape'
        pp.dump(GF_shape,File6)
        #finally 'GF'
        File6.write(gf_full)

        File6.close()        

        #dump time axis into pickle file
        File7 = file(current_time_axis_file_abs,'w')
        pp.dump(time_axis,File7)
        File7.close()  

        print 'dimensions of GF array: ', shape(gf_full)
        print 'length of time axis: ', shape(time_axis)
        print 'all Greens Functions are written to file ',GF_pickle_file_abs,' as single array\n'

    ###################
    #build current gf from full gf:
    #.
    #.
    gf          = zeros((n_stations, n_gridpoints_section,  3,6, n_time),float32)

    
    #print shape(gf),shape(gf_full)
    #idx192 = abs( array( current_lo_gridpoints) -192 ).argmin()
    
    for gp_section_idx,gp_section_gp in enumerate(current_lo_gridpoints):
        gf[:,gp_section_idx,:,:,:] = gf_full[:,gp_section_gp-1,:,:,:].copy()
        
    print 'GF for section of %i gridpoints set up \n '%n_gridpoints_section

    del gf_full
    #put GF and time axis into config dictionary
    cfg['GF'] = gf#.copy()
    cfg['time_axis_of_GF'] = time_axis   

    #print idx192
    #print gf[1,idx192,2,0,:10]
    #d_break(locals(),'gf ')
    
    #exit()

    #control parameter
    return 1
            
#----------------------------------------------------------------------
#----------------------------------------------------------------------

def decomp_m(m_in, cfg):
    """Decomposition of Moment-Tensor into different parts (following Jost & Hermann).
    
    Style of partition ruled by 'decomp_key':
    0 isotropic + DC + CLVD (default)
    1 isotropic + major DC + minor DC
    2 isotropic + 3 DC

    input:
    -- moment-tensor as 3x3 matrix
    -- config dictionary
    
    indirect output (depends of partition option):
    -- 9 element list 'decomposed_M_opt' in config dictionary
    -- 1: original M
    -- 2: decomp_key
    -- 3-9:
       -- for partition = 0
          isotropic part, double couple (DC) part, clvd part, percentage of DC, seism. moment of DC, seismic moment of isotropic part , total seism. moment m_0

       -- for partition = 1
          isotropic part, major DC part, minor DC part, 'not used', seismic moment of isotropic part ,seism. moment of major DC, total seism. moment m_0

       -- for partition = 2
          isotropic part, DC1 part, DC2 part,DC3 part, seism. moment of DCs, seismic moment of isotropic part , total seism. moment m_0

    direct output:
    -- control parameter
    """


    decomp_key =int( cfg.get('decomp_key',0))
        
    #find eigenvalues "eigenw1" and eigenvectors "eigenv1": 
    eigenw1,eigenv1  = eig(m_in)

    #find trace and new eigenvs. of trace-reduced matrix
    trace_in     = sum(eigenw1)
    red_eigenw1  = eigenw1 - 1/3.*trace_in

    #in ascending order:
    eigenw           = real( take( red_eigenw1,argsort(abs(red_eigenw1)) ) )    
    eigenv           = real( take( eigenv1,argsort(abs(red_eigenw1)) ,1 ) )

    #named according to Jost & Herrmann:
    a1 = eigenv[:,0]
    a2 = eigenv[:,1]
    a3 = eigenv[:,2]
    

    F           = -eigenw[0]/eigenw[2]
    epsilon     = abs(F)

    #calculate isotropic part of moment-tensor:
    tmp3        = [trace_in,0,0,0,trace_in,0,0,0,trace_in]
    tmp_mat_iso = matrix(tmp3).reshape(3,3)
    part_iso    = real(1/3. * tmp_mat_iso)

    #scalar moment of this part after D.Bowers & A.Hudson (1999):
    m_0_iso     = 1./(sqrt(6)) * (eigenw[0]+eigenw[1]+eigenw[2]+trace_in)

    #d_break(locals(),'decomp')

    if decomp_key == 0:
        
        part_dc     = matrix(zeros((9),float)).reshape(3,3)
        part_clvd   = matrix(zeros((9),float)).reshape(3,3)
               
        part_dc     = eigenw[2]*(1-2*F)*( outer(a3,a3) - outer(a2,a2) )
        part_clvd   = eigenw[2]*F*( 2*outer(a3,a3) - outer(a2,a2) - outer(a1,a1))
        perc_dc     = ( 1 - 2*epsilon )*100
        
        m_0_dc      = abs(eigenw[2])
        m_0         = m_0_iso + m_0_dc
        
        out1 = part_iso
        out2 = part_dc
        out3 = part_clvd
        out4 = perc_dc
        out5 = m_0_dc
        out6 = m_0_iso
        out7 = m_0
        
    if decomp_key == 1:
        
        part_maj = matrix(zeros((3,3),float)).reshape(3,3)
        part_min = matrix(zeros((3,3),float)).reshape(3,3)
        
        part_major = (eigenw[1]+eigenw[2])/2. * ( outer(a3,a3) - outer(a2,a2) )
        part_minor = eigenw[0] * ( outer(a1,a1) - outer(a2,a2) ) 

        # Jost & Herrmann:
        m_0_total = sqrt( eigenw[0]**2/2 + eigenw[1]**2/2 + eigenw[2]**2/2 )

        # ???????  :
        m_0_major = 5.  
            
        out1 = part_iso
        out2 = part_major
        out3 = part_minor
        out4 = "not used"
        out5 = m_0_iso  
        out6 = m_0_major
        out7 = m_0_total
        
    if decomp_key == 2:
                
        part_dc1 = matrix(zeros((3,3),float)).reshape(3,3)
        part_dc2 = matrix(zeros((3,3),float)).reshape(3,3)
        part_dc3 = matrix(zeros((3,3),float)).reshape(3,3)
        
        
        part_dc_1 = 1/3.*( eigenw[0] - eigenw[1] ) * ( outer(a1,a1) - outer(a2,a2) ) 
        part_dc_2 = 1/3.*( eigenw[1] - eigenw[2] ) * ( outer(a2,a2) - outer(a3,a3) )  
        part_dc_3 = 1/3.*( eigenw[2] - eigenw[0] ) * ( outer(a3,a3) - outer(a1,a1) ) 
        
        m_0_total = sqrt( (eigenw[0]+1./3.*trace_in)**2/2 + (eigenw[1]+1./3.*trace_in)**2/2 + (eigenw[2]+1./3.*trace_in)**2/2 )
        m_0_dev   = m_0_total - m_0_iso
        
        out1 = part_iso
        out2 = part_dc_1
        out3 = part_dc_2
        out4 = part_dc_3
        out5 = m_0_dev
        out6 = m_0_iso  
        out7 = m_0_total

    #put components of M partition into config dictionary
    cfg['decomposed_M_opt'] =  [m_in, decomp_key, out1, out2, out3, out4, out5, out6, out7]

    #control parameter
    return 1

#----------------------------------------------------------------------
#----------------------------------------------------------------------

  
def find_source(cfg):
    """Gives index with highest Variance Reduction (VR).

    Finds maximal VR after calculating the optimal M. Therefrom one
    obtains the necessary index/indices. Depending on chosen summation
    key either global maximum of VR or one best VR for each
    Signal-Component (N,E,Z).

    input:
    -- config dictionary

    [inv_A,b, data_array,GF,data_time_axes_array,summation_key,list_of_moving_window_starttimes,list_of_gridpoint_section]

    indirect output:
    -- index of best fittinf time window
    -- index of optimal grid point
    -- koordinates of optimal grid point
    -- best M  as 6-array
    -- best variance reduction in \%
    -- best fitting source time in epoch seconds
    -- dictionary with source parameters 

    [optimal_time_window_index,optimal_source_index,optimal_source,optimal M,optimal VR,optimal_source_time,optimal_source_dictionary]


    direct output:
    -- control paramter
    """


    inv_A                   = cfg['inv_A']
    #b                       = cfg['b']
    data_array              = cfg['data_array']
    gf                      = cfg['GF']
    time_axis               = cfg['data_time_axes_array']      
    receiver_index_dictionary        = cfg['receiver_index_dictionary']


    #if _debug:
    #    print 'invA, b, data, GF, time : ',shape(inv_A),shape(b),shape(data_array),shape(gf),shape(time_axis)

    list_of_moving_window_starttimes  = cfg['list_of_moving_window_starttimes']
    list_of_moving_window_idxs        = cfg['list_of_moving_window_startidx']
    number_of_window_steps            = len(list_of_moving_window_starttimes)
    station_index_dict                = cfg['station_index_dictionary']
    window_tmin_tmax_dict             = cfg['stations_tmin_tmax']

    key                               =  int(cfg['summation_key'] )

    filter_flag = int(cfg['filter_flag'])

    #only search in given geometrical subsection of total source grid 
    lo_gridpoint_section              = cfg['list_of_gridpoint_section']
    n_gridpoints                      =  len(lo_gridpoint_section)


    #put solutions into dictionary
    solutions_for_all_time_windows_dict = {}


    #write VR-history into file
    VR_time_file = path.realpath(path.abspath(path.join(cfg['temporary_directory'],'VR_history.dat')))
    file(VR_time_file,'w').close()

    print 'searching most probable source location and time:...'

    #loop over all possible start time windows 
    for window_idx in arange(number_of_window_steps):

        #sub dictionary with solution for respective time window
        temp_solutions_dict = {}

        window_idx_start = list_of_moving_window_idxs[window_idx]
        
        print '\n    Moving time window - step no. %i (of %i)\n'%(window_idx+1,number_of_window_steps)        #, shape(b)
        
        #setting data trace:
        data_array_tmp = data_array[:,window_idx_start:window_idx_start+cfg['data_trace_samples']].copy() 
        
        for receiver,data in enumerate(data_array_tmp):

            data_raw = data.copy()
            data_section = zeros((len(data_raw)))
            current_rec     = receiver_index_dictionary[str(receiver)]
            dummy1          = current_rec.split('.')
            
            current_station = dummy1[0]
            current_channel = dummy1[1]
            station_index   = int(station_index_dict[current_station])
            
            model_tminmax   = window_tmin_tmax_dict[current_station]
            
            section_t_min   = model_tminmax[0]                    
            deltat          = 1./cfg['data_sampling_rate'] 
            ta              = arange(len(data) )*deltat
            
            section_t_max = min( [model_tminmax[1] ,ta[-1] ])
            
            idx_tmin  = (abs(ta-section_t_min)).argmin()
            idx_tmax  = (abs(ta-section_t_max)).argmin()
            

            # fill tapered interesting part as only signal 
            data_section[idx_tmin:idx_tmax]=taper_data(data_raw[idx_tmin:idx_tmax])
            
            if filter_flag == 2:
                data_array_tmp[receiver,:] = taper_data( bp_butterworth(data_section,cfg['data_sampling_rate'] ,cfg ))
            else:
                data_array_tmp[receiver,:] = data_section

            #d_break(locals(),'data_array_tmp')
            #exit()

    
            
        b_4window = corr_vec_temp(data_array_tmp,cfg)

        #estimation of moment-tensor:
        #if not estimate_m(b[window_idx,:,:].copy(), cfg):
        if not estimate_m(b_4window, cfg):
            exit( 'ERROR! could not estimate M!')
        m = cfg['M_array'].copy()
        
        #print 'm, time window',m,window_idx
        #exit()
        
        #if _debug:
        #print 'shape of M:', shape(m)

        #print sum(m[121,:])
        #exit()
        #estimation of deviation(residuum):
        #if not  estimate_res(data_array[window_idx,:,:], time_axis[window_idx], cfg['M_array'], cfg):
        #if not  estimate_res(data_array[:,window_idx_start:window_idx_start+cfg['data_trace_samples']], cfg['M_array'], cfg):

        if not  estimate_res(data_array_tmp, cfg['M_array'], cfg):
            exit( 'ERROR! could not estimate RES!')
        if _debug:
            res = cfg['RES_array']
            #print 'shape of res',shape(res)
            #print (res)
            print 'best residual value found'

        #calculation of variance-reduction (vr):
        #if not estimate_vr(data_array[window_idx,:,:], cfg):
        #if not estimate_vr(data_array[:,window_idx_start:window_idx_start+cfg['data_trace_samples']], cfg):
        if not estimate_vr(data_array_tmp, cfg):
            exit( 'ERROR! could not estimate VR!')
        vr = cfg['VR_array']

        if _debug:
            print '\n Shape of VR-array: ',shape(vr)

        #set arrays for storing parameters of optimal source
        m_opt     = zeros((6),float)
        source    = zeros((1),float)
        vr_opt    = zeros((1),float)

        #find element with maximal vr:
        idx_max      = argmax(vr)
        source_index = idx_max
        source       = lo_gridpoint_section[source_index]
        m_opt[:]     = m[idx_max, :]
        vr_opt       = vr[idx_max]
        #put array M of optimal source into 3x3 matrix shape
        m_opt_mat    = matrix(array((m_opt[0],m_opt[3],m_opt[4],m_opt[3],m_opt[1],m_opt[5],m_opt[4],m_opt[5],m_opt[2])).reshape(3,3))

        #put solution for current time window into sub dictionary
        temp_solutions_dict['M']      = m
        temp_solutions_dict['M_opt']  = m_opt
        temp_solutions_dict['source_index'] = source_index
        temp_solutions_dict['source'] = source
        temp_solutions_dict['VR_opt'] = vr_opt
        temp_solutions_dict['VR_array'] = vr.copy()
        temp_solutions_dict['starttime'] = list_of_moving_window_starttimes[window_idx]
        
        #put whole sub dictionary into solution dict

        solutions_for_all_time_windows_dict[str(window_idx)] = temp_solutions_dict      

        VR_vector = zeros((1,2))
        VR_vector[0,0] = cfg['list_of_moving_window_starttimes'][window_idx]
        VR_vector[0,1] = vr_opt

        #write VR-history into file
        F4 = file(VR_time_file,'a')
        savetxt(F4,VR_vector)
        F4.close()

    print 'best VR for the resp. time steps:'
    for ii in arange(number_of_window_steps):
        temp_solutions_dict = solutions_for_all_time_windows_dict[str(ii)]
        
        print ii,temp_solutions_dict['VR_opt'],temp_solutions_dict['source']

    #add extra key VR - results will be ordered/searched by this key   
    list_of_idx = solutions_for_all_time_windows_dict.keys()
    list_of_vr = []
    for jjj in solutions_for_all_time_windows_dict.values():
        list_of_vr.append(jjj['VR_opt'])

    #look for best VR    
    best_time_window_idx = int(list_of_idx[argmax(list_of_vr)])

    #take respective sub dictionary as final solution dict
    best_dict            = solutions_for_all_time_windows_dict[str(best_time_window_idx)]
         

    # put results from final solutions sub dict into config dictionary        
    cfg['optimal_time_window_index'] = best_time_window_idx
    cfg['optimal_source_index']      = best_dict['source_index']
    cfg['optimal_source']            = best_dict['source']
    cfg['optimal_M']                 = best_dict['M_opt']
    cfg['optimal_VR']                = best_dict['VR_opt']
    cfg['optimal_source_time']       = list_of_moving_window_starttimes[best_time_window_idx]
    cfg['optimal_source_dictionary'] = best_dict

    # dump 4D-grid of VR-values into temp-directory for later analysis (robustness of solution):
    vr_array_total_fn =  path.abspath(path.realpath(path.join(cfg['temporary_directory'],'dictionary_w_all_solutions_4D.pp')))


    #d_break(locals(),'write solution dict')
    
    FH = file(vr_array_total_fn,'w')
    cP.dump(solutions_for_all_time_windows_dict,FH)
    FH.close()


    #if not rescale_M_opt(cfg):
    #    print 'ERROR!! - Could not rescale tensor M'
    #    raise SystemExit

    #d_break(locals(),'find source ende - rescaled M in cfg vorhanden')

    #d_break(locals(),'m, vr, source')

    #control parameter:
    return 1

#----------------------------------------------------------------------

def estimate_res( data, M_array, cfg):
    """Calculates the residuum for each M - Normalised deviation of the internally calculated artificial traces from the observed data. Using the L2-Norm - including weightings !

    input:
    -- raw data trace
    -- time axis
    -- M_array   
    -- config dictionary

    indirect output:
    -- Array with Residua for each source (and component) for given M - 'res[<gridpoint>]'
    [Res_array]

    direct output:
    -- control parameter
    """

    print '    estimate best residual value...'

    m    = M_array
    gf   = cfg['GF']


    lo_gridpoint_section             = cfg['list_of_gridpoint_section']
    lo_receivers                     = cfg['list_of_receivers_with_data']
    receiver_index_dictionary        = cfg['receiver_index_dictionary']
    station_index_dict               = cfg['station_index_dictionary']
    channel_index_dict               = cfg['channel_index_dictionary']
    sampling_in_Hz                   = float(cfg['data_sampling_rate'])
    sampling_in_s                    = 1./sampling_in_Hz


    n_gridpoints   =  len(lo_gridpoint_section)
    n_receivers    =  len(lo_receivers)
    key            =  int(cfg['summation_key'] )

    n_time         = int(float(cfg['time_window_length']) *  sampling_in_Hz    ) + 1

    tmp_array_data_and_synth = zeros((n_gridpoints,n_receivers,2, n_time ))

    #summation over all components:
    if key == 0:

        #set dimensions of dummies
        res           = zeros((n_gridpoints),float)
        tmp_data      = zeros((n_time),float)
        tmp_gf        = zeros((6, n_time),float)
        tmp_synth     = zeros((n_time),float)
        tmp_dummy     = zeros((n_time),float)
        tmp_m         = zeros((6),float)
        tmp_integrand = zeros((n_time),float)

        norm_facs     = zeros((n_gridpoints,n_receivers))

        testsum  = 0
        
        #loop over all gridpoints in chosen section:
        for gp in xrange(n_gridpoints):

            #allocate original index of total grid
            gp_index   = gp#int(lo_gridpoint_section[gp])
            tmp_m      = m[gp,:].copy()

            
            #loop over all receivers:
            tmp_integral_collection = 0
            
            for current_receiver in lo_receivers:
                #get index and split into station and channel
                if receiver_index_dictionary.has_key(current_receiver):
                    receiver_index  = int(receiver_index_dictionary[current_receiver])
                else:
                    continue
                dummy1          = current_receiver.split('.')
                current_station = dummy1[0]
                current_channel = dummy1[1]
                regex_check     = re.compile(r'\w$')
                current_channel_letter = regex_check.findall(current_channel)[0]
                
                station_index = int(station_index_dict[current_station])-1
                channel_index = int(channel_index_dict[current_channel_letter])

                #extract data for receiver-source combination:
                current_data = data[receiver_index,:].copy()
                
#                 if _debug_plot:
#                     if gp_index == 1:
#                         figure(30 + 10 * (station_index+1) +channel_index )
#                         plot(current_data)
#                         print 'data for station, channel... ',current_station,current_channel, ' in window ', 30 + 10 * (station_index+1) +channel_index


                    
                #set temporary GF with 6 components of M:
                for m_component in arange(6):
                    #print shape(gf),station_index, gp_index, channel_index, m_component
                    #d_break(locals(),'shape...')
                    tmp_gf[m_component,:]       =  gf[station_index, gp_index, channel_index, m_component, :].copy()
                    
                    #print 'gp: %i - station: %i -- channel: %i -- m_comp: %i'%(gp_index,station_index,channel_index,m_component)
                    # if station_index==1 and gp_index==6 and channel_index==2 and m_component==0:
                    #                         print tmp_gf[m_component,:10] 

                    #exit()
                #generate synth. data d(t) = G*m 
                for t in xrange(n_time):
                    tmp_synth[t] = dot( tmp_gf[:,t], tmp_m )

                #d_break(locals(),'time axis ')


                if (len(current_data) != len(tmp_synth)):
                    exit( "ERROR ... falsche Laenge der Zeitachse fuers Residuum!!")

                #normalisation by receiver/station/channel/...
                #HUMBUG - nicht nötig, wenn kein BUG!!!
                #alle M werden durch Inversion bestimmt, also auch die absolute Amplitude!!!
                
                norm_factor   =  1# max(current_data) / max(tmp_synth)
                #tmp_synth     = tmp_synth * norm_factor
                norm_facs[gp,receiver_index]     =  norm_factor

                #set integrand and carry out integration
                # including weight of station
                station_weight = cfg['station_weights_dict'][current_station]
                    
                tmp_dummy                   =  station_weight * (current_data - tmp_synth)
                
                tmp_integrand               =  array(tmp_dummy) * array(tmp_dummy)

                tmp_integral                =  simps( tmp_integrand, dx=sampling_in_s )

                tmp_integral_collection     += tmp_integral
                  

                #sum over all integrals according to S.Sipkin

                #print current_receiver,station_index,channel_index

            res[gp] = tmp_integral_collection

           #  if 110<gp<130:
#                 print gp, abs(res[gp] ),'\t',sum(current_data),'\t',sum(tmp_synth),'\n'
#     print res[110:130]                  
        #     print testsum
    #exit()

    #d_break(locals(),'RES')

    #put result into config dictionary:
    cfg['RES_array'] = res
    #control parameter:
    return 1

#----------------------------------------------------------------------

def estimate_vr(data_array, cfg):
    """Calculates the 'Variance Reduction' for each M.

    input:
    -- raw data trace of measured data
    -- config dictionary

    indirect output:
    -- Array with variance reduction for each source (and component) for given moment tensor M - VR[<gridpoint>]
    [VR_array]

    direct output:
    -- control parameter
    """

    print  'Estimation of best variance reduction "vr" : ...'

    res                       = cfg['RES_array']


    #d_break(locals())

    lo_gridpoint_section      = cfg['list_of_gridpoint_section']
    lo_receivers              = cfg['list_of_receivers_with_data']
    
    receiver_index_dictionary = cfg['receiver_index_dictionary']
    station_index_dict        = cfg['station_index_dictionary']
    channel_index_dict        = cfg['channel_index_dictionary']
    
    n_gridpoints              =  len(lo_gridpoint_section)
    n_receivers               =  len(lo_receivers)
    key                       =  int(cfg['summation_key'] )
    n_time                    =  int(float(cfg['time_window_length']) * float(cfg['data_sampling_rate'])) + 1

    data_sampling_in_s        = 1./float(cfg['data_sampling_rate'])

    if key == 0:

        #set dimensions for dummies
        tmp_data      = zeros((n_time),float)
        tmp_integrand = zeros((n_time),float)
        tmp_integral  = zeros((n_receivers),float)
        vr            = zeros((n_gridpoints),float)

        #loop over all grid points in chosen geometrical section
        for gp in xrange(n_gridpoints):

            #loop over all receivers
            for current_receiver in lo_receivers:

                #get index:
                receiver_index                = int(receiver_index_dictionary[current_receiver])

                #set integrand and carry out integration:
                #including station weight
                current_station               =  current_receiver.split('.')[0]
                station_weight                = cfg['station_weights_dict'][current_station]
                tmp_data                      = station_weight * data_array[receiver_index, :][:]
                tmp_integrand                 =  array(tmp_data) * array(tmp_data)
                tmp_integral[receiver_index]  =  simps( tmp_integrand, dx=data_sampling_in_s)
                   
            #sum up over all receivers:
            tmp_data_sum =  sum( tmp_integral[:] )
            tmp_res      =  res[gp]

            try: #calculate VR:
                vr_tmp       =  100 * (1 - (tmp_res/tmp_data_sum))
            except:
                vr_tmp       =  0
                
            if vr_tmp < 0 or isnan(vr_tmp):
                vr[gp]       = 0
            else:
                vr[gp]       = vr_tmp

#             if 110<gp<130:
#                 print gp, abs(res[gp] ),vr[gp],'\n'
            
#     exit()        
            
            
    #d_break(locals(),'vr')

   
    #put result into config dictionary
    cfg['VR_array'] = vr

    #control parameter
    return 1

#----------------------------------------------------------------------

def estimate_m(b, cfg):
    """Calculates the moment tensor M.

    input:
    -- Correlation vector b
    -- Config dictionary

    indirect output:
    -- Array with Moment-Tensors for each source (and component, if summation_key is set) - M[<gridpoint>,<component>]

       Order of entries:
       m_xx, m_yy, m_zz, m_xy, m_xz, m_yz

    [M_array]

    direct output:
    -- control parameter
    """


    print 'Estimation of (best fitting) moment tensor "m" : ...'

    inv_A                  = cfg['inv_A']
    lo_gridpoint_section   = cfg['list_of_gridpoint_section']
    lo_receivers           = cfg['list_of_receivers_with_data']   
    
    n_gridpoints           =  len(lo_gridpoint_section)
    n_receivers            =  len(lo_receivers)
    key                    =  int(cfg['summation_key']) 

    #print inv_A[:,0,70]
    #exit()

    # mmm = transpose(matrix(dot(inv_A[:,:,121], transpose(matrix(b[:, 121]).reshape(1,6)))))
#     print mmm
#     exit()


    m_cmp = cfg.get('m_cmp',[1,2,3,4,5,6])
    
    if key == 0:
        m  =  zeros((n_gridpoints, 6),float)
        for gp in xrange(n_gridpoints):
            #find original index of grid point
            gp_index = gp# lo_gridpoint_section[gp]

            #extract matrix inv_A for given gp
            #print gp_index
            iAs      = inv_A[:, :, gp_index]
            
            if _debug:
                #print gp,shape(inv_A),matrix(iAs),'\n'
                pass
            bs       = transpose(matrix(b[:, gp]).reshape(1,6))

            #get m by simple multiplication inv_A * b
            ms       = transpose(matrix(dot(iAs, bs)))
            m[gp, :] = ms#m_cmp.copy()#ms 

            # if gp ==  36:
#                 d_break(locals(),'sdfsdfsdfsdf')
#                 exit()

    #put result into config dictionary
    cfg['M_array'] = m

    #d_break(locals(),'mmmmmmmm')

    #control parameter
    return 1

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def grid_to_array(cfg):
    """Gives a bijection of the 3D-indices of the sources to a 1D-array.

    Continuous indexing by looping over 3D-coordinates in the order N, E, Z !! 

    input:
    -- config dictionary
    [numbers of gridpoints in every dimension]

    output:
    -- array in dimension [number of sources, 3] of total index  and grid-indices of all gridpoints
    """

    print 'read grid into array ... '

    N_N              = int(cfg['northdim'])*2 +1
    N_E              = int(cfg['eastdim'])*2 +1
    N_Z              = int(cfg['depthdim'])
    cfg['n_source_n']= N_N
    cfg['n_source_e']= N_E
    cfg['n_source_z']= N_Z

    #TODO :check this : 
    n_source = cfg['n_source']
                          
    index_array     = zeros((n_source, 3), float)

    tot = 0 
    for z1 in xrange(N_Z):
        for e2 in xrange(N_E):
            for n3 in xrange(N_N):
                index_array[tot, 0] = n3
                index_array[tot, 1] = e2
                index_array[tot, 2] = z1
                tot                += 1

    #index_array gives relation total_index <-> north_index,east_index,depth_index
    return index_array

#----------------------------------------------------------------------

def grid_coords_rect(cfg):

    #TODO check ob noch aktuell !!

    """Sets a symmetric rectangular grid around given centre. Dimensions given in config dictionary  a bijection of the 3D-coordinates of the sources to a 1D-array.

    Continuous indexing by looping over 3D-coordinates in the order N, E, Z !! 

    input:
    -- config dictionary
    [numbers of gridpoints in every dimension]

    output:
    -- array with coordinates of grid points in (latitude in deg, longitude in deg, depth in meter)
    """

    print 'build rectangular grid of source points'

    N_N         = int(cfg['n_source_n']) 
    N_E         = int(cfg['n_source_e'])
    N_Z         = int(cfg['n_source_z'])
    N_tot       = int(cfg['n_source'])
    lat0_d_m_s  = cfg['lat0'] # centre point (deg, min, sec - or decimal degree)
    len0_d_m_s  = cfg['len0'] # centre point (deg, min, sec - or decimal degree)
    depth0      = cfg['depth0']
    latradius   = cfg['latrange']
    lenradius   = cfg['lenrange']


    lat0        = lat0_d_m_s
    len0        = len0_d_m_s

    #reshape deg,min,sec into decimal degree:
    if (len(lat0_d_m_s) == 3):
        lat0 = lat0_d_m_s[0] + (1./60. * (lat0_d_m_s[2] + (1./60. * lat0_d_m_s[3])))
    if (len(len0_d_m_s) == 3):
        len0 = len0_d_m_s[0] + (1./60. * (len0_d_m_s[2] + (1./60. * len0_d_m_s[3])))


    latmin      = lat0 - latradius 
    latmax      = lat0 + latradius
    lenmin      = len0 - lenradius 
    lenmax      = len0 + lenradius
    depthmax    = cfg['depthmax']

    latstep     = ((latmax - latmin)/(N_N-1) ) 
    lenstep     = ((lenmax - lenmin)/(N_E-1) ) 
    depthstep   = ((depthmax - depth0)/(N_Z-1))

    coord_array = zeros((N_tot, 3),float)

    tot = 0 
    for n1 in xrange(N_Z):
        depth = depthmin + (n1 * depthstep)
        for n2 in xrange(N_E):
            lon = lonmin + (n2 * lonstep)
            for n3 in xrange(N_N):
                lat = latmin + (n3 * latstep)
                
                coord_array[tot, 0] = lat
                coord_array[tot, 1] = lon
                coord_array[tot, 2] = depth

                tot += 1
                
    #array in dimensions [<number of total points>,3]
    return coord_array

#----------------------------------------------------------------------

def find_receiver_idx(coords, cfg):

    # TODO check ob noch notwendig
    
    """Finds the best indices of the given receiver-coordinates needed for the
    internal calculations.

    Coordinates of the grid must be stored in the 'receiver_coordinats' file

    input:
    -- coordinate grid
    -- config dictionary

    output:
    -- array with index of the receivers w.r.t. the total grid (not the section)
    """

    print 'find_receiver_idx'
    
    n_source = int(cfg['n_source'])
    n_source_horizontal = int(n_source / int(cfg['n_source_z']))
    N_rec    = int(cfg['n_receiver'])
    filename = cfg['base_dir']+cfg['receiver_co_file']
    File2    = file(filename,'r')

    rec_co   = zeros((N_rec,2),float)
    
    #read in receiver_coordinates:
    temp3    = loadtxt(File2, usecols=tuple(range(0,3)))
    
    #test if number of receivers is read in correctly:
    if ( len(temp3) < N_rec):
        exit( ' number of receiver-coordinates too small ')

        
    #set up temporary array with coordinates:
    for ii in xrange(len(temp3)):
        bb           = temp3[ii,:]
        rec_co[ii,0] = bb[0]
        rec_co[ii,1] = bb[1]

    rec_idxs   = zeros((N_rec),float)
    tmp_grid   = zeros((n_source_horizontal,3),float)  

    #section from coord-grid with only horizontal coordinates (lat,lon,index):
    tmp_idx    = 0
    for grid_idx in xrange(n_source):
        if (coords[grid_idx, 2] == 0):
            tmp_grid[tmp_idx, :2] = coords[grid_idx,:2]
            tmp_grid[tmp_idx, 2]  = grid_idx
            tmp_idx += 1

    #finding best fitting index for each receiver by comparing coordinates: 
    for i in xrange(N_rec):
        tmp_co = zeros((n_source_horizontal, 3),float)
        lat1 = rec_co[i][0]
        lon1 = rec_co[i][1]

        tmp_co[:,1] = lat1
        red_co      = tmp_grid - tmp_co
        tmp_co[:,1] = 0
        tmp_co[:,2] = lon1
        red_co      = coords - tmp_co
    
#        print red_co        
        abs_vec     = zeros((n_source_horizontal),float)
        abs_vec[:]  = (red_co[:,1])**2 +(red_co[:,2])**2  

        #getting most probable grid_index for receiver:
        min_idx = min((n, j) for j, n in enumerate(abs(abs_vec)))[1]
        rec_idxs[i] = tmp_grid[min_idx, 2]

    return rec_idxs

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def test_threshold(vr, cfg):
    """Test whether a given threshold for variance reduction value is exceeded.

    input:
    -- VR in percentage
    -- config dictionary

    output:
    boolean answer
    -- 0 = no
    -- 1 = yes
    """

    pass

    # TODO !!!
    print 'test if worth an alert:...'
    
    limit  = float(cfg['vr_threshold'])
    vr_cur = vr

    print 'vr:  ',vr
    
    answer = 0

    if vr_cur > limit:
        answer = 1    

    return answer

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def compare_results(source_coords, m, m_0,cfg):

    #TODO

    """ Function for comparing results of ARTMOTIV with stochastic modelling"""

    pass



    filename    =  cfg['base_dir']+cfg['compare_file']
    File4       =  file(filename,'r')
    fc          =  read_array(File4, columns=tuple(range(0,1)))

    # in der liste nach M_0 suchen und ausgeben
    # aus  "    "  koordinaten  einlesen
    
    aa = abs(float(fc)/m_0 - 1)

    print 'compare with results from other method:...'
    
    File4.close()

    relativeDeviation = aa

    return relativeDeviation

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def taper_data(data):
    """ Taper data with Tukey window function. 
    

    input:
    -- data trace of length N

    output
    -- tapered data trace
    """


    datamean = mean(data)
    data -= datamean

    N                    = len(data)

    #restriction to data with more than 50 samples
    if 0:# (N <= 50 ):
        print ' useful tapering impossible !\n Returned original data'
        return data
    
    else:

        steepness = 0.85
        
        taper                = ones((N),float)
        x_axis               = arange(N)-int(N/2) 

        for i,val in enumerate(x_axis):
            if N/2.*steepness <= abs(val) <= N/2.:
               taper[i] = 1./2.*(1+cos(pi*(abs(val) - N/2.*steepness)/((1-steepness)/2.*N))) 
                                    
            


        tapered_data         = data*taper

#         plot(taper)
#         show()
#         raw_input()
#         exit()
        return tapered_data#+datamean

#----------------------------------------------------------------------
def left_taper_data(data):
    """ Taper data with Tukey window function.
    Only on left side. If unfiltered data is given, this coincides with the standard generation of synthetics!!!
    

    input:
    -- data trace of length N

    output
    -- tapered data trace
    """


    datamean = mean(data)
    data -= datamean

    N                    = len(data)

    #restriction to data with more than 50 samples
    if (N <= 50 ):
        print ' useful tapering impossible !\n Returned original data'
        return data
    
    else:

        steepness = 0.85
        
        taper                = ones((N),float)
        x_axis               = arange(N)-int(N/2) 

        for i,val in enumerate(x_axis):
            if N/2.*steepness <= abs(val) <= N/2. and val < 0:
               taper[i] = 1./2.*(1+cos(pi*(abs(val) - N/2.*steepness)/((1-steepness)/2.*N))) 
                                    
            
        tapered_data         = data*taper

        return tapered_data

#----------------------------------------------------------------------

def plot_result(cfg):
    """Evokes the plotting routine for visual control of the results. Calls plotting routines for different sections of interest and plot styles.

    Yet implemented:

    -- 'gridplot_z'
    -- ...

    output:
    -- control parameter
    
    """

    m_cmp = array([-24,76,-52,6,51,53])/90.887358 *10**(1.5*15.7)#cfg.get('m_cmp',[1,2,3,4,5,6])

    #gridplot_2pages(cfg,m_cmp)

    if (int(cfg['plotflag_z']) == 1 or int(cfg['plotflag_z'])== 2 or int(cfg['plotflag_z'])== 3 ) :
        try:
            gridplot_2pages(cfg,m_cmp)
            pass
        
        except:
            print 'PLOT not possible '
 
            
    #control parameter
    return 1

#----------------------------------------------------------------------

def gridplot_1page(cfg):
    """Plots results in .pdf-file (optimized for A4); contains 3 subplots: 
    A) geographical overview over stations and source grid
    B) horizontal section of the source grid in most probable depth, 'Beachballs' are shown at every gridpoint, coded in size(moment) and colour(VR)
    C) traces of real and best synthetic data for each station, channel accoring to entry in configuration file


    Plot is produced with help of GMTPY and GMT

    input:
    -- config dictionary

    output into file:
    -- 1x .pdf file in base directory named according to configuration file ('grd_plot_plotfile')

    direct output:
    -- control parameter
    """
    

    from gmtpy import GMT,cm,GridLayout,FrameLayout,CenterLayout,golden_ratio, aspect_for_projection

    #build final filename for plotfile
    gridplotdatafile_z  = path.realpath(path.abspath(path.join(cfg['temporary_directory'],cfg['grd_plot_datafile'])))

    coord_array         = cfg['source_point_coordinates'] 
    gf                  = cfg['GF']

    lo_all_gridpoints   = cfg['list_of_all_gridpoints']
    if cfg.has_key('list_of_gridpoint_section'):
        current_lo_gridpoints    = cfg['list_of_gridpoint_section']
    else:
        current_lo_gridpoints    = lo_all_gridpoints
    sourcelist          = current_lo_gridpoints
    n_gridpoints        = len(current_lo_gridpoints)
    coords_current_lo_gridpoints = coord_array[list(array(current_lo_gridpoints)-1)]
    latsrc              = array(coords_current_lo_gridpoints[:,0])
    lonsrc              = array(coords_current_lo_gridpoints[:,1])
    
    idx_s               = int(cfg['optimal_source_index'])
    idx_max             = sourcelist[idx_s]
    m_opt               = cfg['optimal_M']
    vr                  = cfg['VR_array']
    best_time_window_idx= cfg['optimal_time_window_index'] 
    data_array_total    = cfg['data_array']
    data                = data_array_total[best_time_window_idx,:,:]

    #find effective depthstep in coordinate file
    depthstep           = 0
    dummy_idx3          = 0
    while depthstep == 0:
        depthstep = int(coord_array[dummy_idx3,2] - coord_array[0,2])
        dummy_idx3 += 1
    cfg['depthstep'] = depthstep

    #open source parameters for optimal solution
    optimal_source_dictionary = cfg['optimal_source_dictionary'] 


    source_coords       = coord_array[idx_max-1,:]
    m                   = optimal_source_dictionary['M']
    sourcedepth         = float(source_coords[2])

    
    #plot the n-th layer beneath the optimal source, if given so by 'plot_devi_layer' !=0 in config file
    dev_n               = int(cfg['plot_devi_layer'])
    
    plotdepth           = float(sourcedepth + ( dev_n * int(cfg['depthstep']) ) ) 


    if _debug:
        print dev_n, int(cfg['depthstep']),plotdepth



    #fill 'plotlist' with the indices to plot
    plotlist            = []
    for pot_plot_idx in arange(len(sourcelist)):
        if ( float(coord_array[sourcelist[pot_plot_idx]- 1,2] ) == plotdepth ):
             plotlist.append(pot_plot_idx)
    

    #prepare array containing GMT input for psmeca
    plotlist            = array(plotlist)
    n_plot              = len(plotlist)
    gridplotdata_z      = zeros((n_plot,12),float)

    if _debug:
        print 'plot layer in depth: ',plotdepth,'m'


    m_opt_mat           = matrix(array((m[idx_s,0],m[idx_s,3],m[idx_s,4],m[idx_s,3],m[idx_s,1],m[idx_s,5],m[idx_s,4],m[idx_s,5],m[idx_s,2])).reshape(3,3))

    #build psmeca input for every plottable source point
    
    count  = 0 
    for idx_plot in plotlist:
        m_mat=matrix(array((m[idx_plot,0],m[idx_plot,3],m[idx_plot,4],m[idx_plot,3],m[idx_plot,1],m[idx_plot,5],m[idx_plot,4],m[idx_plot,5],m[idx_plot,2])).reshape(3,3))

        #longitude in degree
        gridplotdata_z[count,0] = coord_array[sourcelist[idx_plot]-1,1]
        #latitude in degree
        gridplotdata_z[count,1] = coord_array[sourcelist[idx_plot]-1,0]
        #vr for colour of beachball
        gridplotdata_z[count,2] = vr[idx_plot]
        # TODO : complete info
        #moment tensor input for psmeca is in ()coordinates - for relation to (N,E,Z) see Aki \& Richards S.... 
        #m_zz of deviatoric part
        gridplotdata_z[count,3] = m_mat[2,2]-1./3.*(m_mat[0,0]+m_mat[1,1]+m_mat[2,2])#remove trace
        #m_nn of deviatoric part       
        gridplotdata_z[count,4] = m_mat[0,0]-1./3.*(m_mat[0,0]+m_mat[1,1]+m_mat[2,2])#remove trace
        #m_ee of deviatoric part
        gridplotdata_z[count,5] = m_mat[1,1]-1./3.*(m_mat[0,0]+m_mat[1,1]+m_mat[2,2])#remove trace
        #m_nz
        gridplotdata_z[count,6] = m_mat[0,2]
        #-m_ez
        gridplotdata_z[count,7] = -m_mat[1,2]
        #-m_ne
        gridplotdata_z[count,8] = -m_mat[0,1]

        # TODO scale !!!!!!!!!!
        #scale of symbol according to moment - relative to optimal 
        gridplotdata_z[count,9] = 10#log(decomp_m(m_mat,cfg)[8]/decomp_m(m_opt_mat,cfg)[8])+10 
        # 2 dummy variables arbitraryly set to zero
        gridplotdata_z[count,10] = 0
        gridplotdata_z[count,11] = 0
        count += 1

    #d_break(locals(),'plot')

    #build ascii file with plot parameter table - to be piped into psmeca
    if (int(cfg['plotflag_z']) == 2 or int(cfg['plotflag_z']) == 3) :
        grplodatafi_z           = file(gridplotdatafile_z,'w')
        write_array(grplodatafi_z, gridplotdata_z, separator='    ',linesep='\n')
        grplodatafi_z.close()


        #TODO ???????
        if (int(cfg['plotflag_z']) == 2):
            return 1

   
    lo_CS = cfg['list_of_stations_with_data']
    lo_CR = cfg['list_of_receivers_with_data']

    #set up list of receivers incl. coordinates whose data is been used in the inversion
    tmp_used_rec = []
    for ii in lo_CR:
        station = ii.split('.')[0]
        station_dict = cfg['station_coordinate_dictionary']
        lat_lon_dict = station_dict[station]
        lat_tmp = float(lat_lon_dict['lat'])
        lon_tmp = float(lat_lon_dict['lon'])
        
        tmp_used_rec.append([lat_tmp,lon_tmp,0])

    # set array with coordinates of the receivers in GMT order with dummy variable '1' for plot size of symbols        
    tmp_used_rec = array(tmp_used_rec)
    coo_used_rec = zeros((len(lo_CR),3))
    coo_used_rec[:,2] = 1
    coo_used_rec[:,1]  = tmp_used_rec[:,0]
    coo_used_rec[:,0]  = tmp_used_rec[:,1]

    latrec       = coo_used_rec[:,1] 
    lonrec       = coo_used_rec[:,0] 


    #m_0          = gridplotdata_z[:,9]

    #find maximal geographical extension of the setup 

    minlatrec    = min(latrec)
    maxlatrec    = max(latrec)
    minlatsrc    = min(latsrc)
    maxlatsrc    = max(latsrc)
    minlonrec    = min(lonrec)
    maxlonrec    = max(lonrec)
    minlonsrc    = min(lonsrc)
    maxlonsrc    = max(lonsrc)
    minlat       = min([minlatrec,minlatsrc])
    maxlat       = max([maxlatrec,maxlatsrc])
    minlon       = min([minlonrec,minlonsrc])
    maxlon       = max([maxlonrec,maxlonsrc])

    #spatial dimensions for plotting the overview A) -- including receiver- and station geometry
    minlat_eff   = minlat-(0.3*(maxlat-minlat))
    maxlat_eff   = maxlat+(0.3*(maxlat-minlat))
    minlon_eff   = minlon-(0.3*(maxlon-minlon))
    maxlon_eff   = maxlon+(0.3*(maxlon-minlon))
    westbound    = minlon_eff
    eastbound    = maxlon_eff
    southbound   = minlat_eff
    northbound   = maxlat_eff


    #########
    # setup page for plots
    
    n_traces     = len(lo_CR)

    #if more than 36 stations, just plot data of 36 randomly chosen ones
    if ( len(lo_CR) > 36):
        n_traces                = 36
        rec_idxs_plotting_traces= sort(rdm.sample( arange(len(lo_CR)) , n_traces ))
    else:
        rec_idxs_plotting_traces=arange(len(lo_CR))
     
    n_traceplots_per_column = int(round(n_traces/2.))

    #open gmt file handler of gmtpy type            
    gmtplot      = GMT(config={'PAPER_MEDIA':'a3+','PLOT_DEGREE_FORMAT':'D'},version='4.2.1')

    #set outer layout 1 column, 2 rows
    outerlayout  = GridLayout(1,2)
    #find width w and height hin points
    w,h = gmtplot.page_size_points()
    outerlayout.set_policy((w,h))

    #collect widgets to be plotted
    allwidgets   = []
    
    #set up upper row

    #set inner layout 1 row, 2 columns
    inner_layout_up = GridLayout(2,1)

    #positioning of inner window at position (0,0) of outer layout
    outerlayout.set_widget(0, 0, inner_layout_up)

    #set first part of plot - first row, left column   
    #set layout type of first part 
    inner_layout_up_left = FrameLayout()
    #put the layout into the widget
    inner_layout_up.set_widget(0,0,inner_layout_up_left)
    #set the widget handler - to be filled with plot content
    widget       = inner_layout_up_left.get_widget('center')
    #set GMT plot parameter
    rng          = (westbound, eastbound, southbound, northbound)
    prj          = 'M1000p'
    # automatically calculate plot optimal aspect for chosen projection
    asp_rat      = aspect_for_projection('-R%g/%g/%g/%g'%rng, '-J%s'%prj )
    #set physical extension of plot 
    widget.set_horizontal( 8*cm )
    widget.set_vertical( 8*cm *asp_rat )
    #    inner_layout_up.set_min_margins( 2*cm,2*cm,2*cm,2*cm )
    #append the completely configured widget to list of all widgets
    allwidgets.append( widget )
    
    #set second part of plot - first row, right column   
    #set layout type   
    inner_layout_up_right = FrameLayout()
    #put layout into widget
    inner_layout_up.set_widget(1,0,inner_layout_up_right)
    #set the widget handler
    widget               = inner_layout_up_right.get_widget('center')
    #rectangular plot expected - no calculation of aspects needed, neither prj, rng
    widget.set_horizontal( 8*cm )
    widget.set_vertical( 8*cm )
    #append the completely configured widget to list of all widgets
    allwidgets.append( widget )
  
    #set third part of plot - second row (lower half)
    #set layout type (subgrid)
    inner_layout_low   = GridLayout(2,n_traceplots_per_column)
    #put grid layout into outer widget
    outerlayout.set_widget(0, 1, inner_layout_low)
    #set sub widget for every trace
    for icol in range(2):
        for irow in range(n_traceplots_per_column):
            #set layout type
            inner_inner_layout_low =  FrameLayout()
            #put layout type into current sub widget
            inner_layout_low.set_widget(icol,irow,inner_inner_layout_low)
            #set widget handler - positioned according to trace index
            widget              = inner_inner_layout_low.get_widget('center')
            #set hardcoded borders for each plot for have sufficient space between traces
            #inner_inner_layout_low.set_fixed_margins( 1*cm,0.5*cm,0.5*cm,0.5*cm )
            #widget.set_size((8*cm,4*cm),(2*cm,2*cm))
            inner_inner_layout_low.set_min_margins( 2.5*cm,0.5*cm,0.25*cm,1*cm )
            allwidgets.append( widget )

    if _debug:
        #show widget structure in colours
        gmtplot.draw_layout(outerlayout)


    ######### now fill widgets with content:
    ## first widget 
    plotwidget = allwidgets[0]

    if _debug:
        print plotwidget.get_params()

    # first part: general plot setup (axis, title,etc.) and map of region

    # font size according to gmtdefaults - to be set when initiating the "GMT" instance
    plottitle    = 'EKOFISK (source depth %i m)'%(int(plotdepth))

    #set plot parameters for pscoast
    #set range
    rng          = (westbound, eastbound, southbound, northbound)
    #automatic calculation of effective plotwidth
    plotwidth    = float(plotwidget.get_params()['width'])
    #Mercator projection - size is calculated as 'plotwidth' and added in call of 'pscoast'
    prj          = 'M%gp'
    #xof          = '0c'
    #set details of rivers
    clr          = ('1',52,101,164)
    #set details of borders/boundaries
    bnd          = '1'
    #set resolution
    res          = 'f'
    #set pencil
    pcl          = 'thinnest' # ('0.25p',0,0,0)
    #set colour of land
    cll          = (233,185,110)
    #set colour of sea
    cls          = (52,101,164)
    #set annotation steps
    lonannstep   = (maxlon_eff - minlon_eff)/3.
    latannstep   = (maxlat_eff - minlat_eff)/5.

    ann          = ('a%.3ff%.3f:"Longitude"::,"deg":/a%.3ff%.3f:"Latitude"::,"deg"::."%s":WSen')%(lonannstep,lonannstep/10.,latannstep,latannstep/10.,plottitle)

    #plot a pscoast element into the active widget ('plotwidget') 
    gmtplot.pscoast( R=rng, J=prj%(plotwidth), B=ann, D=res, S=cls, G=cll, W=pcl, I=clr, N=bnd, *plotwidget.XY() )

    
    #prepare next part of overlayplot: beachball grid
    
    #set plot parameters for beachballs (psmeca)
    #TODO put into config file!!
    #set scaling factor for beachballs
    bbs          = 0.104
    #set pencil
    pcl          = '0.8pt' 
    #set colour table
    ctb          = path.realpath(path.abspath(path.join(cfg['base_dir'],cfg['colourtable'])))
    #set dummy content for psmeca parameter 'M' (same size for all magnitudes)
    ssm          = ' '

    print 'plot beachbaelle'
    
    #calculation of effective beachball size and plotting for each gridpoint 
    size_of_optimum_in_plot = exp(-(  ((100. - max(gridplotdata_z[:,2]))/100.)**2/0.001))
    #loop over all gridpoints in chosen horizontal layer - 
    for plotline in arange(len(gridplotdata_z[:,0])):
        scale  = size_of_optimum_in_plot#exp(-(  ((100. - gridplotdata_z[plotline,2])/100.)**2/0.001))/size_of_optimum_in_plot
        #set effective size for current beachball
        psm          = 'm'+str(bbs*scale)
        #plot psmeca element into the active widget ('plotwidget') 
        gmtplot.psmeca(in_rows=[gridplotdata_z[plotline,:].tolist()], M=ssm, S=psm, W=pcl, Z=ctb, R=True, J=True,  *plotwidget.XY() )


    #set receiver colour
    sfg      = 'black'
    #set receiver symbol size
    rcs_size = 0.15
    #set receiver type
    rcs      ='i'+str(rcs_size)
    #plot receiver symbols into the active widget ('plotwidget')  - coordinates in array (lon,lat,1) 
    gmtplot.psxy(in_rows=coo_used_rec, S=rcs, G=sfg,  R=True, J=True , *plotwidget.XY() )


    #set circles around receivers whose traces are plotted
    #set pencil
    pen      = '2p,red'
    #set symbol and size
    sym      = 'c0.2'

    
    #set coordinate array for respective receivers
    plot_rec_sym_coo = zeros((n_traces,3))
    jjj = 0
    for iii in rec_idxs_plotting_traces:
        plot_rec_sym_coo[jjj,2] = 1
        plot_rec_sym_coo[jjj,0] = lonrec[iii]
        plot_rec_sym_coo[jjj,1] = latrec[iii]
        jjj                    += 1

    #plot circles into the active widget ('plotwidget')   
    gmtplot.psxy(in_rows=plot_rec_sym_coo, S=sym, W=pen,  R=True, J=True , *plotwidget.XY() )

    #plot names for the stations
    #set size of font
    namesize = 13
    #set font
    fontno   = 4
    #set position w.r.t. receiver symbol
    position = 'TL'
    #set angle of text
    angle    = 0
    #set distance to receiver symbol
    sym_dist = 'j0.2c'
    #set colour
    color    = 'green'

    #set list of annotations
    pstext_in_data = []
    for iii in rec_idxs_plotting_traces:
        receivername  = lo_CR[iii]
        stationname   = receivername.split('.')[0]
        tmp_listentry = ['%g %g %g %g %g %s %s'%(lonrec[iii],latrec[iii],namesize, angle, fontno, position, stationname )]
        pstext_in_data.append(tmp_listentry)

    #plot station names into the active widget ('plotwidget')   
    gmtplot.pstext(in_rows=pstext_in_data, G=color, D=sym_dist, R=True, J=True , *plotwidget.XY() )


    
    #########
    ## second widget - plot only source grid (zoomed in) 

    plotwidget = allwidgets[1]

    #set subtitle
    plottitle    = 'Horizontal section of source grid in depth: '+str(int(plotdepth))+' m)'


    
    #set spatial extension
    minlat_eff   = minlatsrc-(0.2*(maxlatsrc-minlatsrc))
    maxlat_eff   = maxlatsrc+(0.2*(maxlatsrc-minlatsrc))
    minlon_eff   = minlonsrc-(0.2*(maxlonsrc-minlonsrc))
    maxlon_eff   = maxlonsrc+(0.2*(maxlonsrc-minlonsrc))

    westbound    = minlon_eff
    eastbound    = maxlon_eff
    southbound   = minlat_eff
    northbound   = maxlat_eff



    #set annotation steps
    lonannstep   = (maxlon_eff - minlon_eff)/4.
    latannstep   = (maxlat_eff - minlat_eff)/4.

    #set annotations
    ann          = ('a%.3ff%.3f:Lon:/a%.3ff%.3f:Lat:WSen')%(lonannstep,lonannstep/10.,latannstep,latannstep/10.)

    #automatic calculation of effevctive plotwidth 
    plotwidth    = plotwidget.get_params()['width']

    #set colour of sea/background
    cls =(255,255,255)

    #set legend
    lgd = 'f%f/%f/%f/50k:km:'%((maxlon_eff+minlon_eff)/2.,maxlat_eff+latannstep/2,(maxlat_eff+minlat_eff)/2.)

    #take former information for rest of colours

    #plot grid with beach balls into active widget 
    gmtplot.pscoast( R=(westbound, eastbound, southbound, northbound), J=prj%(plotwidth), B=ann, L=lgd,  D=res, S=cls, G=cll, W=pcl, I=clr, N=bnd, *plotwidget.XY())

    
    #loop over all gridpoints in chosen horizontal layer - 
    for plotline in arange(len(gridplotdata_z[:,0])):
        #set effective size for current beachball
        scale  = 2#*exp(-(  ((100. - gridplotdata_z[plotline,2])/100.)**2/0.001))/size_of_optimum_in_plot
        psm    = 'm'+str(bbs*scale)
        #plot psmeca element into the active widget ('plotwidget') 
        gmtplot.psmeca(in_rows=[gridplotdata_z[plotline,:].tolist()], M=ssm, S=psm, W=pcl, Z=ctb, R=True, J=True , *plotwidget.XY() )

    #plot blue square around best solution
    #set pencil
    pen      = '2p,blue'
    #set symbol
    sym      = 's%g'%(1.5*bbs*scale)
    #input for psxy (must be  iterable set of rows)
    plotinput= [['%g %g %g'%(source_coords[1],source_coords[0],1) ]]
    #plot psmeca element into the active widget ('plotwidget') 
    gmtplot.psxy(in_rows=plotinput, S=sym, W=pen,  R=True, J=True , *plotwidget.XY() )

    #plot colour scale right of grid
    pos = '%f/%f/%f/%f'%(maxlon_eff+1.5*lonannstep/2,(maxlat_eff+minlat_eff)/2.,0.6*cm,0.5*cm)
    ann = '%i'%(25)
    skp = ''
    fil = '%s'%(path.realpath(path.abspath(path.join(cfg['base_dir'],cfg['colourtable']))))

    gmtplot.psscale(D=pos,B=ann,S=skp,C=fil, *plotwidget.XY())
    

    #d_break(locals(),'gridplot-lsdöksldkfmn')

    #########
    ## third widget - plot traces of real and synthetic data

    # plot traces for receivers in list 'lo_CR'
    # indizes for plottable traces are in 'rec_idxs_plotting_traces'

    #setup array for 2 traces in each plot: 'data' and 'synthetic'
    gf_sampling_in_Hz   =   float(cfg['gf_sampling_rate'])
    data_sampling_in_Hz =   float(cfg['data_sampling_rate'])   
    data_sampling_in_s  =   1./data_sampling_in_Hz
    
    #set time axis 
    time_axis  = arange(int(float(cfg['time_window_length'])) * int(gf_sampling_in_Hz ) +1 ) /float(gf_sampling_in_Hz)        

    plottraces_data  = zeros((len(rec_idxs_plotting_traces),3,len(time_axis)))

    for idx_plottrace in arange(len(rec_idxs_plotting_traces)):      
        #run once for read out data and GF:
        
        #set current station
        idx_lo_CR_entry        = rec_idxs_plotting_traces[idx_plottrace]
        current_receiver       = lo_CR[idx_lo_CR_entry]
        current_station        = current_receiver.split('.')[0]
        current_channel        = current_receiver.split('.')[1]

        #set indizes
        station_index_dict     = cfg['station_index_dictionary']
        receiver_index_dict    = cfg['receiver_index_dictionary']
        channel_index_dict     = cfg['channel_index_dictionary']
        receiver_index         =  int(receiver_index_dict[current_receiver])
        station_index          =  int(station_index_dict[current_station])-1
        channel_index          =  int(channel_index_dict[current_channel[-1]])


        #read in GF and build synthetic - no convolution, just multiplication - delta peak stf needeed for this !!!
        print  station_index,current_station, idx_s,channel_index,current_channel

        n_time         = int(float(cfg['time_window_length']) * data_sampling_in_Hz ) + 1
        tmp_gf         = zeros((6, n_time),float)
        tmp_synth      = zeros((n_time),float)
        gp_index       = idx_s
        
        for m_component in arange(6):
            tmp_gf[m_component,:]    =  gf[station_index, gp_index, channel_index, m_component, :]
        for t in xrange(n_time):
            tmp_synth[t]             = dot( tmp_gf[:,t], m_opt )

        synth_data             = tmp_synth #dot(gf_cur.transpose(),m_opt)

        real_data              = data[receiver_index,:]

        Res_tmp                = simps( (real_data - synth_data)**2, dx=data_sampling_in_s  )
        norm_temp              = simps( (real_data)**2, dx=data_sampling_in_s  )
        VR_tmp                 = (1 - (Res_tmp/norm_temp) )*100

        plottraces_data[idx_plottrace,0,:] = real_data
        plottraces_data[idx_plottrace,1,:] = synth_data
        plottraces_data[idx_plottrace,2,:] = VR_tmp


    #
    #
    #
    #normalisation by maximum absolute value of real data - within this trace, absolute maximal amplitude of synthetic is the same
    #TODO: schalter einbauen für andere normierungen 
    #
    #1. normalise by best fitting trace:
    # - estimate all residuums for the synthetic data made by best M
    # - find trace with lowest residuum and take this as reference
    # - find stretch-factor from raw GF to data (mean of maximum and minimum ratio respectively)
    # - apply factor to all GF

    #2. scale by weighting factor 

    stations_array            = cfg['station_coordinates']

    dummy_maxval_data = []
    dummy_maxval_synt = []
    dummy_vr          = []

    for i in arange(len(plottraces_data[:,1,0])):

        dummy_maxval_data.append( max  ( abs( plottraces_data[i,0,:] ) ) )
        dummy_maxval_synt.append( max  ( abs( plottraces_data[i,1,:] ) ) )
        dummy_vr.append(plottraces_data[i,2,0])

    arg_max_data    = array(dummy_maxval_data).argmax()
    max_val_data    = max(array(dummy_maxval_data))
    max_val_synth   = max(abs(plottraces_data[arg_max_data,1,:]))   

    best_trace_idx = array(dummy_vr).argmax()
    opt_data       = plottraces_data[best_trace_idx,0,:]  
    opt_synth      = plottraces_data[best_trace_idx,1,:]  


    #find scaling factor
    #scale_for_synth_by_opt_source_and_trace  = mean([max(opt_data)/max(opt_synth),min(opt_data)/min(opt_synth) ]) 
    #scale_for_synth_by_opt_source_and_trace  = mean([max(opt_data)/max(opt_synth),min(opt_data)/min(opt_synth) ]) 

    #for i in arange(len(plottraces_data[:,1,0])):
    #    plottraces_data[i,1,:] *= scale_for_synth_by_opt_source_and_trace
    #    plottraces_data[i,1,:] *= scale_for_synth_by_opt_source_and_trace


    for i in arange(len(plottraces_data[:,1,0])):
        scaling = max_val_data#max(abs(plottraces_data[i,0,:]))
        plottraces_data[i,0,:] /= scaling
        print i , scaling/max(abs(plottraces_data[i,1,:]))
        plottraces_data[i,1,:] /= scaling
    
    #for i in arange(len(plottraces_data[:,1,0])):
    #    plottraces_data[i,0,:] /= max_val_data
    #    plottraces_data[i,1,:]  = plottraces_data[i,1,:]/max_val_synth

    #
    #
    #

    #d_break(locals(),'m_abs')



    for idx_plottrace in arange(len(rec_idxs_plotting_traces)):      
        #run second time for plotting
        
        #set current widget index - including two already used widgets for upper part of plot 
        idx_plotwidget         = idx_plottrace + 2
        plotwidget             = allwidgets[idx_plotwidget]
 
        #set current station
        idx_lo_CR_entry        = rec_idxs_plotting_traces[idx_plottrace]
        current_receiver       = lo_CR[idx_lo_CR_entry]
        current_station        = current_receiver.split('.')[0]
        current_channel        = current_receiver.split('.')[1]
        

        #set indizes
        station_index_dict     = cfg['station_index_dictionary']
        receiver_index_dict    = cfg['receiver_index_dictionary']
        channel_index_dict     = cfg['channel_index_dictionary']
        receiver_index         =  int(receiver_index_dict[current_receiver])
        station_index          =  int(station_index_dict[current_station])-1
        channel_index          =  int(channel_index_dict[current_channel[-1]])

        station_weight         = stations_array[station_index,4]
        
        #set data to plot
        #scaled by weighting factor 
        yd = plottraces_data[idx_plottrace,0,:]#* station_weight
        ys = plottraces_data[idx_plottrace,1,:]#* station_weight

        print 'station',current_receiver,station_weight,max(abs(plottraces_data[idx_plottrace,0,:])),max(abs(plottraces_data[idx_plottrace,1,:])),max(abs(plottraces_data[idx_plottrace,0,:]))/max(abs(plottraces_data[idx_plottrace,1,:]))

        #yd[:] = 0.
        #ys[:] = 0.
        
        #set parameters for plot
        #set range
        rng        = (float(min(time_axis)),float(max(time_axis)),-1,1 )
        #set annotations
        ann        = '%g/%g:%s:SW' % (100,0.5, current_receiver)
        current_gmtdefaults = {}
        current_gmtdefaults['ANNOT_FONT_SIZE_SECONDARY'] = '12p'
        current_gmtdefaults['ANNOT_FONT_SIZE_PRIMARY'] = '9p'

        #plot psbasemap element into the active widget ('plotwidget') - setup of the coordinate system
        gmtplot.psbasemap( R=rng,B=ann,*plotwidget.XYJ()  ) 

        #et pencil for data trace
        pen_data  = '1p,red'
        #set pencil for synthetic
        pen_synth = '0.65p,blue,x'
        #plot data trace in red
        gmtplot.psxy(R=True,W=pen_data,in_columns=(time_axis,yd),*plotwidget.XYJ( ))
        #plot synthetic in blue
        gmtplot.psxy(R=True,W=pen_synth,in_columns=(time_axis,ys) ,*plotwidget.XYJ())
        
    #if _debug:
    #    gmtplot.draw_layout(outerlayout)


    #set filename for total output 
    plotfilename = path.join(cfg['plot_dir'],cfg['model_name']+'_depth_'+str(int(plotdepth))+'m_'+cfg['grd_plot_plotfile'])

    if _debug:
        print 'name of plotfile: ', plotfilename

    #save complete plot according to file suffix - should be .pdf - including bounding box    
    gmtplot.save(plotfilename, bbox=outerlayout.bbox())

    
#----------------------------------------------------------------------
def gridplot_2pages(cfg,m_cmp):
    """Plots results in 2 .pdf-files (optimized for A3); contains 2 and N  subplots: 
    first file
    A) geographical overview over stations and source grid
    B) horizontal section of the source grid in most probable depth, 'Beachballs' are shown at every gridpoint, coded in size(moment) and colour(VR)
    second file
    C) traces of real and best synthetic data for each station, channel accoring to entry in configuration file


    Plot is produced with help of GMTPY and GMT

    input:
    -- config dictionary

    output into file:
    -- 2 x .pdf file in base directory named according to configuration file ('grd_plot_plotfile')

    direct output:
    -- control parameter
    """
    

    from gmtpy import GMT,cm,GridLayout,FrameLayout,CenterLayout,golden_ratio, aspect_for_projection

    #build final filename for plotfile
    gridplotdatafile_z  = path.realpath(path.abspath(path.join(cfg['temporary_directory'],cfg['grd_plot_datafile'])))

    coord_array         = cfg['source_point_coordinates'] 
    gf                  = cfg['GF']

    lo_all_gridpoints   = cfg['list_of_all_gridpoints']
    if cfg.has_key('list_of_gridpoint_section'):
        current_lo_gridpoints    = cfg['list_of_gridpoint_section']
    else:
        current_lo_gridpoints    = lo_all_gridpoints

    sourcelist          = current_lo_gridpoints
    n_gridpoints        = len(current_lo_gridpoints)
    coords_current_lo_gridpoints = coord_array[list(array(current_lo_gridpoints)-1)]

    latsrc              = array(coords_current_lo_gridpoints[:,0])
    lonsrc              = array(coords_current_lo_gridpoints[:,1])
    
    idx_s               = int(cfg['optimal_source_index'])
    idx_max             = sourcelist[idx_s]
    m_opt               = cfg['optimal_M']

    #vr                  = cfg['VR_array']
    best_time_window_idx= cfg['optimal_time_window_index'] 
    sol_dict            = cfg['optimal_source_dictionary']
    vr                  = sol_dict['VR_array']

    best_vr             = vr[idx_s]


    filter_flag = int(cfg['filter_flag'])


    #d_break(locals(),'filtered data - eingelesen fuer plot')
    #find effective depthstep in coordinate file
    depthstep           = 0
    dummy_idx3          = 0
    while depthstep == 0:
        depthstep = int(coord_array[dummy_idx3,2] - coord_array[0,2])
        dummy_idx3 += 1
    cfg['depthstep'] = depthstep

    #open source parameters for optimal solution
    optimal_source_dictionary = cfg['optimal_source_dictionary'] 


    source_coords       = coord_array[idx_max-1,:]
    m                   = optimal_source_dictionary['M']
    sourcedepth         = float(source_coords[2])
    
    
    #plot the n-th layer beneath the optimal source, if given so by 'plot_devi_layer' !=0 in config file
    dev_n               = int(cfg['plot_devi_layer'])
    
    plotdepth           = float(sourcedepth + ( dev_n * int(cfg['depthstep']) ) ) 

    #d_break(locals(),'plot vr')
    
    if _debug:
        print dev_n, int(cfg['depthstep']),plotdepth



    #fill 'plotlist' with the indices to plot
    plotlist            = []
    for pot_plot_idx in arange(len(sourcelist)):
        if ( plotdepth - 500  < float(coord_array[sourcelist[pot_plot_idx]- 1,2] ) <  plotdepth + 500 ):
             plotlist.append(pot_plot_idx)
    

    #prepare array containing GMT input for psmeca (each point has 12 entries)
    plotlist            = array(plotlist)
    n_plot              = len(plotlist)
    gridplotdata_z      = zeros((n_plot,12),float)

    if _debug:
        print 'plot layer in depth: ',plotdepth,'m'


    m_opt_mat           = matrix(array((m[idx_s,0],m[idx_s,3],m[idx_s,4],m[idx_s,3],m[idx_s,1],m[idx_s,5],m[idx_s,4],m[idx_s,5],m[idx_s,2])).reshape(3,3))

    #build psmeca input for every plottable source point
    
    count  = 0 
    for idx_plot in plotlist:
        m_mat=matrix(array((m[idx_plot,0],m[idx_plot,3],m[idx_plot,4],m[idx_plot,3],m[idx_plot,1],m[idx_plot,5],m[idx_plot,4],m[idx_plot,5],m[idx_plot,2])).reshape(3,3))

        #longitude in degree
        gridplotdata_z[count,0] = coord_array[sourcelist[idx_plot]-1,1]
        #latitude in degree
        gridplotdata_z[count,1] = coord_array[sourcelist[idx_plot]-1,0]
        #vr for colour of beachball
        gridplotdata_z[count,2] = vr[idx_plot]
        # TODO : complete info
        #moment tensor input for psmeca is in ()coordinates - for relation to (N,E,Z) see Aki \& Richards S.... 
        #m_zz of deviatoric part
        gridplotdata_z[count,3] = m_mat[2,2]-1./3.*(m_mat[0,0]+m_mat[1,1]+m_mat[2,2])#remove trace
        #m_nn of deviatoric part       
        gridplotdata_z[count,4] = m_mat[0,0]-1./3.*(m_mat[0,0]+m_mat[1,1]+m_mat[2,2])#remove trace
        #m_ee of deviatoric part
        gridplotdata_z[count,5] = m_mat[1,1]-1./3.*(m_mat[0,0]+m_mat[1,1]+m_mat[2,2])#remove trace
        #m_nz
        gridplotdata_z[count,6] = m_mat[0,2]
        #-m_ez
        gridplotdata_z[count,7] = -m_mat[1,2]
        #-m_ne
        gridplotdata_z[count,8] = -m_mat[0,1]

        # TODO scale !!!!!!!!!!
        #scale of symbol according to moment - relative to optimal 
        gridplotdata_z[count,9] = vr[idx_plot]/best_vr*30#log(decomp_m(m_mat,cfg)[8]/decomp_m(m_opt_mat,cfg)[8])+10 
        # 2 dummy variables arbitraryly set to zero
        gridplotdata_z[count,10] = 0
        gridplotdata_z[count,11] = 0
        count += 1

    #d_break(locals(),'plot bbs ')
    
    #build ascii file with plot parameter table - to be piped into psmeca
    if (int(cfg['plotflag_z']) == 2 or int(cfg['plotflag_z']) == 3) :
        grplodatafi_z           = file(gridplotdatafile_z,'w')
        write_array(grplodatafi_z, gridplotdata_z, separator='    ',linesep='\n')
        grplodatafi_z.close()


        #TODO ???????
        if (int(cfg['plotflag_z']) == 2):
            return 1

   
    lo_CS = cfg['list_of_stations_with_data']
    lo_CR = cfg['list_of_receivers_with_data']

    #set up list of receivers incl. coordinates whose data is been used in the inversion
    tmp_used_rec = []
    for ii in lo_CR:
        station = ii.split('.')[0]
        station_dict = cfg['station_coordinate_dictionary']
        lat_lon_dict = station_dict[station]
        lat_tmp = float(lat_lon_dict['lat'])
        lon_tmp = float(lat_lon_dict['lon'])
        
        tmp_used_rec.append([lat_tmp,lon_tmp,0])

    # set array with coordinates of the receivers in GMT order with dummy variable '1' for plot size of symbols        
    tmp_used_rec = array(tmp_used_rec)
    coo_used_rec = zeros((len(lo_CR),3))
    coo_used_rec[:,2] = 1
    coo_used_rec[:,1]  = tmp_used_rec[:,0]
    coo_used_rec[:,0]  = tmp_used_rec[:,1]

    latrec       = coo_used_rec[:,1] 
    lonrec       = coo_used_rec[:,0] 



    #find maximal geographical extension of the setup 

    minlatrec    = min(latrec)
    maxlatrec    = max(latrec)
    minlatsrc    = min(latsrc)
    maxlatsrc    = max(latsrc)
    minlonrec    = min(lonrec)
    maxlonrec    = max(lonrec)
    minlonsrc    = min(lonsrc)
    maxlonsrc    = max(lonsrc)
    minlat       = min([minlatrec,minlatsrc])
    maxlat       = max([maxlatrec,maxlatsrc])
    minlon       = min([minlonrec,minlonsrc])
    maxlon       = max([maxlonrec,maxlonsrc])

    #spatial dimensions for plotting the overview A) -- including receiver- and station geometry
    minlat_eff   = minlat-(0.3*(maxlat-minlat))
    maxlat_eff   = maxlat+(0.3*(maxlat-minlat))
    minlon_eff   = minlon-(0.3*(maxlon-minlon))
    maxlon_eff   = maxlon+(0.3*(maxlon-minlon))
    westbound    = minlon_eff
    eastbound    = maxlon_eff
    southbound   = minlat_eff
    northbound   = maxlat_eff

    #d_break(locals(),'plot1')

   
    n_traces_w_data     = len(lo_CR)
    n_traces_in_plot    = n_traces_w_data
    #if more than 36 stations, just plot data of 36randomly chosen ones
    #n_traces_in_plot = min(int(cfg.get('n_traces_in_plot','36')), n_traces_w_data )

    #if ( n_traces_in_plot > 36):
    #    print 'only 36 traces plottable - chosing  channels randomly'
    #    n_traces_in_plot             = 36
    #if n_traces_in_plot < n_traces_w_data:
    #    rec_idxs_plotting_traces = sort( rdm.sample( arange(n_traces_w_data) ,n_traces_in_plot ))
    #else:
    rec_idxs_plotting_traces = arange(n_traces_w_data)
     
    #n_traceplots_per_column = int(round(n_traces_in_plot/2.))

    #d_break(locals(),'plot2')

    #########
    # setup first page for plots
 
    #open gmt file handler of gmtpy type            
    page1      =  GMT(config={'PAPER_MEDIA':'a3','PLOT_DEGREE_FORMAT':'D'},version='4.2.1')

    #d_break(locals(),'plot2 - 1. page setup')

    #set outer layout 1 column, 2 rows
    outerlayout  = GridLayout(1,2)
    #find width w and height hin points
    w,h = page1.page_size_points()
    outerlayout.set_policy((w,h))

    #collect widgets to be plotted
    allwidgets_p1   = []
    
    #set up upper row

    #set layout type of first part 
    inner_layout_up = FrameLayout()

    #positioning of inner window at position (0,0) of outer layout
    outerlayout.set_widget(0, 0, inner_layout_up)

    widget       = inner_layout_up.get_widget('bottom')
    widget.set_vertical( 1*cm )
    widget       = inner_layout_up.get_widget('top')
    widget.set_vertical( 1*cm )

    #set the widget handler - to be filled with plot content
    widget       = inner_layout_up.get_widget('center')

    #set GMT plot parameter
    rng          = (westbound, eastbound, southbound, northbound)

        
    prj          = 'M1000p'
    # automatically calculate plot optimal aspect for chosen projection
    asp_rat      = aspect_for_projection('-R%g/%g/%g/%g'%rng, '-J%s'%prj )
    #set physical extension of plot 
    
    if (eastbound-westbound) > (northbound-southbound):
        widget.set_vertical( 8*cm *asp_rat )
    else:
        widget.set_vertical( 8*cm )
        widget.set_horizontal( 8*cm /asp_rat  )

        
    #    inner_layout_up.set_min_margins( 2*cm,2*cm,2*cm,2*cm )
    #append the completely configured widget to list of all widgets
    allwidgets_p1.append( widget )

    
    #set second part of plot - second row    
    #set layout type   
    inner_layout_low = FrameLayout()
    #put layout into widget
    outerlayout.set_widget(0,1,inner_layout_low)
    #set the widget handler
    widget               = inner_layout_low.get_widget('center')
    #rectangular plot expected - no calculation of aspects needed, neither prj, rng
    widget.set_horizontal( 11*cm )
    widget.set_vertical( 11*cm )
    #append the completely configured widget to list of all widgets
    allwidgets_p1.append( widget )

    if  _debug:
        #show widget structure in colours
        #page1.draw_layout(outerlayout)
        pass
    #d_break(locals(),'page 1 - widgets set')


    ######### now fill widgets with content:
    ## first widget 
    plotwidget = allwidgets_p1[0]

    #if _debug:
        #print plotwidget.get_params()

    # first part: general plot setup (axis, title,etc.) and map of region

    # font size according to gmtdefaults - to be set when initiating the "GMT" instance
    plottitle    = '%s (source depth %i m)'%(cfg['model_name'],int(plotdepth))

    #set plot parameters for pscoast
    #set range
    rng          = (westbound, eastbound, southbound, northbound)
    #automatic calculation of effective plotwidth
    plotwidth    = float(plotwidget.get_params()['width'])
    #Mercator projection - size is calculated as 'plotwidth' and added in call of 'pscoast'
    prj          = 'M%gp'
    #xof          = '0c'
    #set details of rivers
    clr          = ('1',52,101,164)
    #set details of borders/boundaries
    bnd          = '1'
    #set resolution
    res          = 'f'
    #set pencil
    pcl          = 'thinnest' # ('0.25p',0,0,0)
    #set colour of land
    cll          = (233,185,110)
    #set colour of sea
    cls          = (52,101,164)
    #cls          = cll #(52,101,164)
    #set annotation steps
    lonannstep   = (maxlon_eff - minlon_eff)/3.
    latannstep   = (maxlat_eff - minlat_eff)/5.

    ann          = ('a%.3ff%.3f:"Longitude"::,"deg":/a%.3ff%.3f:"Latitude"::,"deg"::.%s:WSen')%(lonannstep,lonannstep/10.,latannstep,latannstep/10.,plottitle)

    #d_break(locals(),'page 1 - plot coast')

    #plot a pscoast element into the active widget ('plotwidget') 
    page1.pscoast( R=rng, J=prj%(plotwidth), B=ann, D=res, S=cls, G=cll, W=pcl, I=clr, N=bnd, *plotwidget.XY() )

    
    #prepare next part of overlayplot: beachball grid
    
    #set plot parameters for beachballs (psmeca)
    #TODO put into config file!!
    #set scaling factor for beachballs
    bbs          = 0.104
    #set pencil
    pcl          = '1.1pt' 
    #set colour table
    ctb          = path.realpath(path.abspath(path.join(cfg['base_dir'],cfg['colourtable'])))
    #set dummy content for psmeca parameter 'M' (same size for all magnitudes)
    ssm          = ' '

    print 'plot beachbaelle'
    
    #calculation of effective beachball size and plotting for each gridpoint 
    size_of_optimum_in_plot = 3.#exp(-(  ((100. - max(gridplotdata_z[:,2]))/100.)**2/0.001))
    #loop over all gridpoints in chosen horizontal layer - 
    for plotline in arange(len(gridplotdata_z[:,0])):
        scale  = size_of_optimum_in_plot*gridplotdata_z[plotline,2]/best_vr#exp(-(  ((100. - gridplotdata_z[plotline,2])/100.)**2/0.001))/size_of_optimum_in_plot
        #set effective size for current beachball
        psm          = 'm'+str(bbs*scale)
        #plot psmeca element into the active widget ('plotwidget') 
        page1.psmeca(in_rows=[gridplotdata_z[plotline,:].tolist()], M=ssm, S=psm, W=pcl, Z=ctb, R=True, J=True,  *plotwidget.XY() )

    #set receiver colour
    sfg      = 'red'
    #set receiver symbol size
    rcs_size = 0.25
    #set receiver type
    rcs      ='i'+str(rcs_size)
    #set offset
    off      ='0/0.125'
    
    #plot receiver symbols into the active widget ('plotwidget')  - coordinates in array (lon,lat,1) 
    page1.psxy(in_rows=coo_used_rec, S=rcs, G=sfg,D=off,  R=True, J=True , *plotwidget.XY() )


    #set circles around receivers whose traces are plotted
    #set pencil
    pen      = '2p,red'
    #set symbol and size
    sym      = 'c0.2'

    
    #set coordinate array for respective receivers
    plot_rec_sym_coo = zeros((n_traces_in_plot,3))
    jjj = 0
    for iii in rec_idxs_plotting_traces:
        plot_rec_sym_coo[jjj,2] = 1
        plot_rec_sym_coo[jjj,0] = lonrec[iii]
        plot_rec_sym_coo[jjj,1] = latrec[iii]
        jjj                    += 1

    #plot circles into the active widget ('plotwidget')   
    #page1.psxy(in_rows=plot_rec_sym_coo, S=sym, W=pen,  R=True, J=True , *plotwidget.XY() )

    #plot names for the stations
    #set size of font
    namesize = 13
    #set font
    fontno   = 4
    #set position w.r.t. receiver symbol
    position = 'BC'
    #set angle of text
    angle    = 0
    #set distance to receiver symbol
    sym_dist = '0./0.2.'#'j2.5'
    #set colour
    color    = 'red'
    #set colour of background quader
    clq    = 'black'

    #set list of annotations
    pstext_in_data = []
    for iii in rec_idxs_plotting_traces:
        receivername  = lo_CR[iii]
        stationname   = receivername.split('.')[0]
        tmp_listentry = ['%g %g %g %g %g %s %s'%(lonrec[iii],latrec[iii],namesize, angle, fontno, position, stationname )]
        pstext_in_data.append(tmp_listentry)

    #plot station names into the active widget ('plotwidget')   
    page1.pstext(in_rows=pstext_in_data, G=color, D=sym_dist,W=clq, R=True, J=True , *plotwidget.XY() )


    #d_break(locals(),'plot2 - page 1 - first grid done')
   
    #########
    ## second widget - plot only source grid (zoomed in) 

    plotwidget = allwidgets_p1[1]

    #set subtitle
    plottitle    = 'Horizontal section of source grid in depth: '+str(int(plotdepth))+' m)'


    
    #set spatial extension
    minlat_eff   = minlatsrc-(0.2*(maxlatsrc-minlatsrc))
    maxlat_eff   = maxlatsrc+(0.2*(maxlatsrc-minlatsrc))
    minlon_eff   = minlonsrc-(0.2*(maxlonsrc-minlonsrc))
    maxlon_eff   = maxlonsrc+(0.2*(maxlonsrc-minlonsrc))
    lat0         = (minlat_eff+maxlat_eff)/2.
    lon0         = (minlon_eff+maxlon_eff)/2.

    if minlat_eff==maxlat_eff:
        minlat_eff = lat0 - (  100000 * rad_to_deg /R)
        maxlat_eff = lat0 + (  100000 * rad_to_deg /R)

    if maxlon_eff == minlon_eff:
        maxlon_eff = lon0 + ( 100000 * rad_to_deg / R / sin( (90-lat0)/rad_to_deg ) ) 
        minlon_eff = lon0 - ( 100000 * rad_to_deg / R / sin( (90-lat0)/rad_to_deg ) ) 


        
    westbound    = minlon_eff
    eastbound    = maxlon_eff
    southbound   = minlat_eff
    northbound   = maxlat_eff
    
    

    #set annotation steps
    lonannstep   = (eastbound - westbound)/4.
    latannstep   = (northbound -southbound )/4.

    #set annotations
    ann          = ('a%.3ff%.3f:Lon:/a%.3ff%.3f:Lat:WSen')%(lonannstep,lonannstep/10.,latannstep,latannstep/10.)

    #automatic calculation of effevctive plotwidth 
    plotwidth    = plotwidget.get_params()['width']

    #set colour of sea/background
    cls =cll#(255,255,255)

    #set legend
    lgd = 'f%f/%f/%f/20k:km:'%(lon0,maxlat_eff+latannstep/2,lat0)

    #take former information for rest of colours

    #plot grid with beach balls into active widget 
    page1.pscoast( R=(westbound, eastbound, southbound, northbound), J=prj%(plotwidth), B=ann, L=lgd,  D=res, S=cls, G=cll, W=pcl, I=clr, N=bnd, *plotwidget.XY())

    
    #loop over all gridpoints in chosen horizontal layer - 
    size_of_optimum_in_plot=3.

    for plotline in arange(len(gridplotdata_z[:,0])):
        #set effective size for current beachball
        scale  = size_of_optimum_in_plot *gridplotdata_z[plotline,2]/best_vr   #*exp(-(  ((100. - gridplotdata_z[plotline,2])/100.)**2/0.001))/size_of_optimum_in_plot
        psm    = 'm%g'%(bbs*scale)
        #plot psmeca element into the active widget ('plotwidget') 
        page1.psmeca(in_rows=[gridplotdata_z[plotline,:].tolist()], M=ssm, S=psm, W=pcl, Z=ctb, R=True, J=True , *plotwidget.XY() )

    #plot red square around best solution
    #set pencil
    pen      = '2p,red'
    #set symbol
    sym      = 's%g'%(size_of_optimum_in_plot*bbs*1.5)
    #input for psxy (must be  iterable set of rows)
    plotinput= [['%g %g %g'%(source_coords[1],source_coords[0],1) ]]
    #plot psmeca element into the active widget ('plotwidget') 
    page1.psxy(in_rows=plotinput, S=sym, W=pen,  R=True, J=True , *plotwidget.XY() )

    #plot colour scale right of grid
    pos = '%f/%f/%f/%f'%(5,2,3,0.3)#(maxlon_eff+1.5*lonannstep/2,(maxlat_eff+minlat_eff)/2.,3*cm,0.5*cm)
    ann = '%i:VR:/:%s:'%(25,'%')
    skp = ''
    fil = '%s'%( path.realpath(path.abspath(path.join(cfg['base_dir'],cfg['colourtable']))))

    page1.psscale(D=pos,B=ann,S=skp,C=fil, *plotwidget.XY())
    

    #d_break(locals(),'gridplot-lsdöksldkfmn')

    #set filename for first output 
    if filter_flag == 2:  
        plot_fn1  = 'overview_'+cfg['model_name']+'_depth_%im_%is_%.1f_%.1fHz_%s' %( int(plotdepth) ,int(float(cfg['time_window_length'])), float(cfg['bp_lower_corner']), float(cfg['bp_upper_corner']),cfg['grd_plot_plotfile'])
    else:
        plot_fn1  = 'overview_'+cfg['model_name']+'_depth_%im_%is_%s' %( int(plotdepth) ,int(float(cfg['time_window_length'])),cfg['grd_plot_plotfile']) 

    plotfile1 = path.join(cfg['plot_dir'],plot_fn1)

    if _debug:
        print 'name of plotfile: ', plotfile1

    #save complete plot according to file suffix - should be .pdf - including bounding box    
    page1.save(plotfile1, bbox=outerlayout.bbox())


    ##################
    #
    # setup second page for plots
    #

    #
    #open gmt file handler of gmtpy type            
#     page2      =  GMT(config={'PAPER_MEDIA':'a3+','PLOT_DEGREE_FORMAT':'D'},version='4.2.1')

#     #set layout type 
#     outerlayout   = GridLayout(2,n_traceplots_per_column)
#     #find width w and height hin points
#     w,h = page2.page_size_points()
#     outerlayout.set_policy((w,h))

#     #collect widgets to be plotted
#     allwidgets_p2   = []
        
#     #set sub widget for every trace
#     for icol in range(2):
#         for irow in range(n_traceplots_per_column):
#             #set layout type
#             inner_layout_trace =  FrameLayout()
#             #put layout type into current sub widget
#             outerlayout.set_widget(icol,irow,inner_layout_trace)
#             #set widget handler - positioned according to trace index
#             widget              = inner_layout_trace.get_widget('center')
#             #set hardcoded borders for each plot for have sufficient space between traces
#             #inner_inner_layout_low.set_fixed_margins( 1*cm,0.5*cm,0.5*cm,0.5*cm )
#             #widget.set_size((8*cm,4*cm),(2*cm,2*cm))
#             inner_layout_trace.set_min_margins( 2.5*cm,0.5*cm,0.25*cm,1*cm )
#             allwidgets_p2.append( widget )

#     if 0:#_debug:
#         #show widget structure in colours
#         page2.draw_layout(outerlayout)


    ##########################################
    # build folder with traces of real data and synthetics - to be compared with 'snuffler'
    # naming : separate real data and synthetic by setting 'location' to 'RD' or 'SY'

    traces_folder          = path.realpath(path.abspath(path.join(cfg['base_dir'],'DB','traces')))
    traces_folder_compare  = path.realpath(path.abspath(path.join(cfg['base_dir'],'DB','traces_compare')))

    if  path.exists(traces_folder):
        #print traces_folder,'already exists \n'
        shutil.rmtree(traces_folder)
    os.makedirs(traces_folder)
    if  path.exists(traces_folder_compare):
        #print traces_folder,'already exists \n'
        shutil.rmtree(traces_folder_compare)
    os.makedirs(traces_folder_compare)
    print '\ndirectory %s set up'%traces_folder

    #########
    ## content of second page  widget - plot traces of real and synthetic data

    # plot traces for receivers in list 'lo_CR'
    # indizes for plottable traces are in 'rec_idxs_plotting_traces'

    #setup array for 2 traces in each plot: 'data' and 'synthetic'
    gf_sampling_in_Hz   =   float(cfg['gf_sampling_rate'])
    data_sampling_in_Hz =   float(cfg['data_sampling_rate'])   
    data_sampling_in_s  =   1./data_sampling_in_Hz
    
    #set time axis 
    time_axis  = arange(int( float(cfg['time_window_length']) * gf_sampling_in_Hz ) +1 ) /float(gf_sampling_in_Hz)        

    plottraces_data  = zeros((len(rec_idxs_plotting_traces),3,len(time_axis)))

    location_rd = 'RD'
    location_sy = 'SY'
    location_cp = 'CP'
    
    network          = cfg['network']
    t_min            = cfg['optimal_source_time']
    #t_min = 0
    t_max            = t_min + len(time_axis) * data_sampling_in_s
    event_time_tuple = time.gmtime(t_min)
    event_date       = '%02i%02i%4i'%(int(event_time_tuple[2]),int(event_time_tuple[1]),int(event_time_tuple[0]) )

    print '\n build synthetic data with M =\n', m_cmp,'\n'

    for idx_plottrace in arange(len(rec_idxs_plotting_traces)):      
        #run once for read out data and GF:
        
        #set current station
        idx_lo_CR_entry        = rec_idxs_plotting_traces[idx_plottrace]
        current_receiver       = lo_CR[idx_lo_CR_entry]
        current_station        = current_receiver.split('.')[0]
        current_channel        = current_receiver.split('.')[1]

        #set indizes
        station_index_dict     = cfg['station_index_dictionary']
        receiver_index_dict    = cfg['receiver_index_dictionary']
        channel_index_dict     = cfg['channel_index_dictionary']
        receiver_index         =  int(receiver_index_dict[current_receiver])
        station_index          =  int(station_index_dict[current_station])-1
        channel_index          =  int(channel_index_dict[current_channel[-1]])


        #read in GF and build synthetic - no convolution, just multiplication - delta peak stf needeed for this !!!
        #print 'build synthetic - ', station_index,current_station, idx_s,channel_index,current_channel

        n_time         = int( float(cfg['time_window_length']) * data_sampling_in_Hz ) + 1
        tmp_gf         = zeros((6, n_time),float)
        tmp_synth      = zeros((n_time),float)
        tmp_synth_comp = zeros((n_time),float)
        gp_index       = idx_s


        # print m_opt
        #print cfg['M_scaled'][0]
        
        for m_component in arange(6):
            tmp_gf[m_component,:]    =  gf[station_index, gp_index, channel_index, m_component, :]
        for t in xrange(n_time):
            tmp_synth[t]             = dot( tmp_gf[:,t], m_opt )
            tmp_synth_comp[t]        = dot( tmp_gf[:,t], m_cmp )
            #tmp_synth_comp[t]        = dot( tmp_gf[:,t], cfg['M_scaled'][0]/1000./4 )

        synth_data             = tmp_synth #dot(gf_cur.transpose(),m_opt)
        synth_data_compare     = tmp_synth_comp #dot(gf_cur.transpose(),m_opt)

        
    
        data_array_total    = cfg['data_array']
        #data                = data_array_total[best_time_window_idx,:,:]
        best_data_start_idx = int(cfg['list_of_moving_window_startidx'][best_time_window_idx])
        data_section        = data_array_total[:,best_data_start_idx:best_data_start_idx + cfg['data_trace_samples']].copy()

        
        
        real_data_tmp           = data_section[receiver_index,:]
        real_data           = zeros((n_time))       
        
        #finding correct indices for tapering
        ta_internal     =  arange(n_time)/data_sampling_in_Hz
        model_tminmax   =  cfg['stations_tmin_tmax'][current_station]
        section_t_min   = model_tminmax[0]
        section_t_max   = min([model_tminmax[1],ta_internal[-1] ])
        idx_tmin  = (abs(ta_internal-section_t_min)).argmin()
        idx_tmax  = (abs(ta_internal-section_t_max)).argmin()

        real_data[idx_tmin:idx_tmax]=taper_data(real_data_tmp[idx_tmin:idx_tmax])
        
        if filter_flag == 2:            
            real_data = taper_data( bp_butterworth(real_data,cfg['data_sampling_rate'] ,cfg ))
        
        

        # d_break(locals(),'dfdf')
#         print m_opt
#         exit()

        # weighting the residuum, also for plotting
        station_weight = cfg['station_weights_dict'][current_station]
        
        
        Res_tmp                = simps( (station_weight*real_data - station_weight*synth_data)**2, dx=data_sampling_in_s  )
        norm_temp              = simps( (station_weight*real_data)**2, dx=data_sampling_in_s  )
        VR_tmp                 = (1 - (Res_tmp/norm_temp) )*100

        plottraces_data[idx_plottrace,0,:] = real_data
        plottraces_data[idx_plottrace,1,:] = synth_data
        plottraces_data[idx_plottrace,2,:] = VR_tmp


        #write data and synthetic into respective miniSeed files in folder base_dir/DB/traces:

        #d_break(locals(),'rausschreib check')
        
        fn_data     = '%s.%s.%s.%s.%s'%(network,current_station,location_rd, current_channel, event_date )
        fn_synth    = '%s.%s.%s.%s.%s'%(network,current_station,location_sy, current_channel, event_date )
        fn_synth_cp = '%s.%s.%s.%s.%s'%(network,current_station,location_cp, current_channel, event_date )


        m_seed_tuple_data     = (network,current_station,location_rd,current_channel,int(t_min*pymseed.HPTMODULUS),\
                                 int(t_max*pymseed.HPTMODULUS),1./data_sampling_in_s,real_data)
        m_seed_tuple_synth    = (network,current_station,location_sy,current_channel,int(t_min*pymseed.HPTMODULUS), \
                                 int(t_max*pymseed.HPTMODULUS),1./data_sampling_in_s,synth_data)
        m_seed_tuple_synth_cp = (network,current_station,location_cp,current_channel,int(t_min*pymseed.HPTMODULUS), \
                                 int(t_max*pymseed.HPTMODULUS),1./data_sampling_in_s,synth_data_compare)

        outfile_data  = path.join(traces_folder,fn_data)
        outfile_synth = path.join(traces_folder,fn_synth)
        outfile_data2  = path.join(traces_folder_compare,fn_data)
        outfile_synth_cp = path.join(traces_folder_compare,fn_synth_cp)

        pymseed.store_traces([m_seed_tuple_data],outfile_data)
        pymseed.store_traces([m_seed_tuple_synth],outfile_synth)
        pymseed.store_traces([m_seed_tuple_data],outfile_data2)
        pymseed.store_traces([m_seed_tuple_synth_cp],outfile_synth_cp)


    print '\ntraces written to files in %s'%traces_folder
    print 'synthetics of "correct"/comparison  solution  written to files in %s\n'%traces_folder_compare

    #
    #
    #
    #normalisation by maximum absolute value of real data - within this trace, absolute maximal amplitude of synthetic is the same
    #TODO: schalter einbauen für andere normierungen 
    #
    #1. normalise by best fitting trace:
    # - estimate all residuums for the synthetic data made by best M
    # - find trace with lowest residuum and take this as reference
    # - find stretch-factor from raw GF to data (mean of maximum and minimum ratio respectively)
    # - apply factor to all GF

    #2. scale by weighting factor 

    stations_array            = cfg['station_coordinates']

    dummy_maxval_data = []
    dummy_maxval_synt = []
    dummy_vr          = []

    for i in arange(len(plottraces_data[:,1,0])):

        dummy_maxval_data.append( max  ( abs( plottraces_data[i,0,:] ) ) )
        dummy_maxval_synt.append( max  ( abs( plottraces_data[i,1,:] ) ) )
        dummy_vr.append(plottraces_data[i,2,0])

    arg_max_data    = array(dummy_maxval_data).argmax()
    max_val_data    = max(array(dummy_maxval_data))
    max_val_synth   = max(abs(plottraces_data[arg_max_data,1,:]))   

    best_trace_idx = array(dummy_vr).argmax()
    opt_data       = plottraces_data[best_trace_idx,0,:]  
    opt_synth      = plottraces_data[best_trace_idx,1,:]  


    #find scaling factor
    #scale_for_synth_by_opt_source_and_trace  = mean([max(opt_data)/max(opt_synth),min(opt_data)/min(opt_synth) ]) 
    #scale_for_synth_by_opt_source_and_trace  = mean([max(opt_data)/max(opt_synth),min(opt_data)/min(opt_synth) ]) 

    #for i in arange(len(plottraces_data[:,1,0])):
    #    plottraces_data[i,1,:] *= scale_for_synth_by_opt_source_and_trace
    #    plottraces_data[i,1,:] *= scale_for_synth_by_opt_source_and_trace


    for i in arange(len(plottraces_data[:,1,0])):

        scaling_data            = max_val_data# max(abs(plottraces_data[i,0,:]))
        plottraces_data[i,0,:] /= scaling_data

        #print 'trace number scaling factor - ',i , scaling_data #/max(abs(plottraces_data[i,1,:]))

        #scaling_synth           = max_val_data# max(abs(plottraces_data[i,1,:]))
        scaling_synth           = max_val_synth
        #scaling_synth           = max(abs(plottraces_data[i,1,:]))
        plottraces_data[i,1,:] /= scaling_synth
    
    #for i in arange(len(plottraces_data[:,1,0])):
    #    plottraces_data[i,0,:] /= max_val_data
    #    plottraces_data[i,1,:]  = plottraces_data[i,1,:]/max_val_synth

    #
    #
    #

    #d_break(locals(),'plot: plottraces_data ,  m_abs')

    print '\n'

#     for idx_plottrace in arange(len(rec_idxs_plotting_traces)):      
#         #run second time for plotting
        
#         #set current widget index - including two already used widgets for upper part of plot 
#         idx_plotwidget         = idx_plottrace
#         plotwidget             = allwidgets_p2[idx_plotwidget]
 
#         #set current station
#         idx_lo_CR_entry        = rec_idxs_plotting_traces[idx_plottrace]
#         current_receiver       = lo_CR[idx_lo_CR_entry]
#         current_station        = current_receiver.split('.')[0]
#         current_channel        = current_receiver.split('.')[1]
        

#         #set indizes
#         station_index_dict     = cfg['station_index_dictionary']
#         receiver_index_dict    = cfg['r            tmp_synth_comp[t]        = dot( tmp_gf[:,t], m_cmp )
#eceiver_index_dictionary']
#         channel_index_dict     = cfg['channel_index_dictionary']
#         receiver_index         =  int(receiver_index_dict[current_receiver])
#         station_index          =  int(station_index_dict[current_station])-1
#         channel_index          =  int(channel_index_dict[current_channel[-1]])

#         station_weight         = cfg['station_weights_dict'][current_station]  #stations_array[station_index,4]
        
#         #set data to plot
#         #scaled by weighting factor 
#         yd = plottraces_data[idx_plottrace,0,:]#* station_weight
#         ys = plottraces_data[idx_plottrace,1,:]#* station_weight

#         #print 'station, weight, maxdata, maxsynth, ratio --  ',current_receiver,station_weight,max(abs(plottraces_data[idx_plottrace,0,:])),max(abs(plottraces_data[idx_plottrace,1,:])),max(abs(plottraces_data[idx_plottrace,0,:]))/max(abs(plottraces_data[idx_plottrace,1,:]))

#         #yd[:] = 0.
#         #ys[:] = 0.
        
#         #set parameters for plot
#         #set range
#         rng        = (float(min(time_axis)),float(max(time_axis)),-1,1 )

#         #set annotations
#         ann        = '%g/%g:%s:SW' % (int(float(max(time_axis))/5+1),0.5, current_receiver)
#         current_gmtdefaults = {}
#         current_gmtdefaults['ANNOT_FONT_SIZE_SECONDARY'] = '12p'
#         current_gmtdefaults['ANNOT_FONT_SIZE_PRIMARY'] = '9p'

#         #plot psbasemap element into the active widget ('plotwidget') - setup of the coordinate system
#         page2.psbasemap( R=rng,B=ann,*plotwidget.XYJ()  ) 

#         #set pencil for data trace
#         pen_data  = '1p,red'
#         #set pencil for synthetic
#         pen_synth = '0.65p,blue,x'
#         #plot data trace in red
#         page2.psxy(R=True,W=pen_data,in_columns=(time_axis,yd),*plotwidget.XYJ( ))
#         #plot synthetic in blue
#         page2.psxy(R=True,W=pen_synth,in_columns=(time_axis,ys) ,*plotwidget.XYJ())
        

#     #set filename for second output
#     plot_fn  = 'traces_'+cfg['model_name']+'_depth_%im_%is_%.1f_%.1fHz_%s' %(int(plotdepth),int(float(cfg['time_window_length'])),float(cfg['bp_lower_corner']),float(cfg['bp_upper_corner']),cfg['grd_plot_plotfile'])
#     plotfile = path.join(cfg['plot_dir'],plot_fn)

#     if _debug:
#         print 'name of plotfile: ', plotfile

#     #save complete plot according to file suffix - should be .pdf - including bounding box    
#     page2.save(plotfile, bbox=outerlayout.bbox())



#----------------------------------------------------------------------
    
def setup_time(cfg):
    
    #TODO check ob nötig

    """Set up time axis from <length in seconds> and <sampling in Hz>.

    input:
    -- config dictionary
    [time_window_length,sampling_rate]
    
    output:
    -- time axis
    """

    if _debug:
        print 'setup_time'

    n_timesteps   = int( float(cfg['time_window_length']) / float(cfg['sampling_rate']))
    time_axis     = numpy.arange(n_timesteps+1)*cfg['sampling_rate']

    return time_axis

#----------------------------------------------------------------------

def set_sourcepoint_configuration(cfg):
    """
    Setting geographical coordinates of source locations. 

    Indexing is done by looping over all source-points in the order (N,E,Z)


    Input:
    -- Configuration-dictionary
    -- expansion in north-direction
    -- expansion in east-direction

    indirect output:
    -- File with coordinate-array (lat,lon,depth) in in base directory 
    -- array with source soordinates

    direct output:
    -- control paramteter
    """

    print 'setting grid_coords...'

    #read in coordinates of central point from config dictionary
    #if given in (degree,minute,second), coordinates are transformed into decimal degrees

    central_latitude_raw_string     = cfg['central_latitude']
    central_latitude_raw = central_latitude_raw_string.split(',')

    #check if given coordinate is of correct length and made of valid  numerical characters 
    if not (len(central_latitude_raw) == 3 or len(central_latitude_raw) == 1) :
        exit('ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!')
    else:
        for central_latitude_raw_element in central_latitude_raw:
            dummy5 = central_latitude_raw_element.split('.')
            if not ( len(dummy5) in [1,2]):
                exit( 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!')
            for dummy5_element in dummy5:
                if (not ( dummy5_element.isdigit() )) and (not(dummy5_element[0]=='-' and dummy5_element[1:]   ) ):
                    exit( 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!')
                
                        
        if len(central_latitude_raw) == 3:
            central_latitude_deg = float( float( central_latitude_raw[0]) + (1./60. * ( float(central_latitude_raw[1]) + (1./60. * float(central_latitude_raw[2]) ))))
        else:
            central_latitude_deg = float(central_latitude_raw[0])
    
    
    
    central_longitude_raw_string = cfg['central_longitude']
    central_longitude_raw = central_longitude_raw_string.split(',')
    #check if given coordinate is of correct length and made of valid  numerical characters 
    if not(len(central_longitude_raw) == 3 or len(central_longitude_raw) == 1 ):
        exit('ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!')
    else:
        for central_lonitude_raw_element in central_longitude_raw:
            dummy5 = central_lonitude_raw_element.split('.')
            if not ( len(dummy5) in [1,2]):
                exit( 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!')
            for dummy5_element in dummy5:
                if (not ( dummy5_element.isdigit() )) and (not(dummy5_element[0]=='-' and dummy5_element[1:]   ) ):
                    exit( 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!')
                        
        if len(central_longitude_raw) == 3:
            central_longitude_deg = float( float( central_longitude_raw[0]) + (1./60. * ( float(central_longitude_raw[1]) + (1./60. * float(central_longitude_raw[2]) ))))
        else:
            central_longitude_deg = float(central_longitude_raw[0])

        
    #read in and set parameters for the source grid - centre, spatial extensions and spatial increments
    lat0        = central_latitude_deg
    lon0        = central_longitude_deg
    northdim    = int(cfg['northdim'])
    eastdim     = int(cfg['eastdim'])
    depthdim    = int(cfg['depthdim'])
    northstep   = int(cfg['northstep'])
    eaststep    = int(cfg['eaststep'])
    depthstep   = int(cfg['depthstep'])

    #set number of gridpoints
    N_N         = int(2 * northdim + 1)
    N_E         = int(2 * eastdim + 1)
    N_Z         = int(depthdim)         
    N_tot       = int(N_N * N_E * N_Z)


    list_of_source_points = []
    
    coord_array_temp = zeros((N_N,N_E,N_Z, 3),float)
    coord_array      = zeros((N_tot, 3),float)

    #set grid coordinates by looping over the rectangular grid 
    for z1 in (arange(N_Z)):
        depth    = (z1+1) * depthstep
        rad_act  = R - depth
        
        for n3 in (arange(N_N) - northdim):
            #approximation: no latitude change with depth
            lat = lat0 + ( n3 * northstep * rad_to_deg / rad_act) 

            for e2 in (arange(N_E)-eastdim):
                #longitude differs depending on latitude - grid is equidistant
                lon = lon0 + ( e2 * eaststep * rad_to_deg / rad_act / sin( (90-lat)/rad_to_deg ) ) 

                coord_array_temp[n3+int(northdim),e2+int(eastdim),z1, 0] = lat
                coord_array_temp[n3+int(northdim),e2+int(eastdim),z1, 1] = lon
                coord_array_temp[n3+int(northdim),e2+int(eastdim),z1, 2] = depth


    #give indices to grid points and save in list [total index,north index, east index, depth index] -- running N->E->Z
    tot = 0
    for z1 in arange(N_Z):
        for e2 in arange(N_E):
            for n3 in arange(N_N):
                coord_array[tot, :] =  coord_array_temp[n3,e2,z1,:]
                source_list_entry = [tot,n3,e2,z1]
                list_of_source_points.append(source_list_entry)
                tot += 1  

    #sorting array after indices
    list_of_source_points.sort()

    #save ascii table into file in temporary directory - for visual check or external use
    temp_dir       = cfg['temporary_directory']
    filename       = path.realpath(path.abspath(path.join(temp_dir , cfg['source_co_file'])))
    File7          = file(filename,'w')
    write_array(File7, coord_array, separator='    ',linesep='\n')
    File7.close()

    #put source grid coordinates into config dictionary
    cfg['list_of_source_points']    = list_of_source_points
    cfg['source_point_coordinates'] = coord_array


    #control parameter
    return 1

    print '...done\n'

#----------------------------------------------------------------------

# def set_receiver(cfg):
#     """
#     Sets up a set of receiver locations. (for generation of synthetic data)

#     Centre of the circular geometry is the epicentre of source locations. 

#     Input:
#     -- config dictionary

#     Output:
#     -- File with receiver coordinates (lat,lon,depth) in base directory  
#     """

#     print 'setting receiver_coords...'

#     #TODO - set as default...but allow input ( try except)

#     #set dummy values
#     networks = [ 'xx' ]
#     stations = ['AAA','BBB','CCC','DDD','EEE']
#     locations =  ['ww']  
#     channels = ['N','E','Z']

#     #put values into config dictionary
#     cfg['setup_gf_list_of_stations'] = stations
#     cfg['setup_gf_list_of_networks'] = networks
#     cfg['setup_gf_list_of_locations'] = locations
#     cfg['setup_gf_list_of_channels'] = channels
#     cfg['list_of_ContributingStations'] = stations

#     #make dictionary 'station_index_dict' within config dictionary
#     make_dict_stations_indices_complete(cfg)

#     #set parameters and file
#     number      = len(stations)
#     distmin     = int(cfg['rec_dist_min'])
#     diststep    = int(cfg['rec_dist_step'])
#     temp_dir    = cfg['temporary_directory']
#     filename    = path.realpath(path.abspath(path.join(temp_dir,cfg['rec_co_file'])))
    
#     #build coordinate array around given central point - coordinates in (deg) or (deg,min,sec)
#     coo         = zeros((number,3),float)
#     central_latitude_raw_string = cfg['central_latitude']
#     central_latitude_raw = central_latitude_raw_string.split(',')
#     if len(central_latitude_raw) == 3 or len(central_latitude_raw) == 1 :
#         for central_latitude_raw_element in central_latitude_raw:
#             dummy5 = central_latitude_raw_element.split('.')
#             if not ( len(dummy5) in [1,2]):
#                 exit( 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!')
#             for dummy5_element in dummy5:
#                 if not ( dummy5_element.isdigit() ):
#                     exit( 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!')
                        
#         if len(central_latitude_raw) == 3:
#             central_latitude_deg = float( float( central_latitude_raw[0]) + (1./60. * ( float(central_latitude_raw[1]) + (1./60. * float(central_latitude_raw[2]) ))))
#         else:
#             central_latitude_deg = float(central_latitude_raw[0])
    
    
#     central_longitude_raw_string = cfg['central_longitude']
#     central_longitude_raw = central_longitude_raw_string.split(',')
#     if len(central_longitude_raw) == 3 or len(central_longitude_raw) == 1 :
#         for central_lonitude_raw_element in central_longitude_raw:
#             dummy5 = central_lonitude_raw_element.split('.')
#             if not ( len(dummy5) in [1,2]):
#                 exit( 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!')                
#             for dummy5_element in dummy5:
#                 if not ( dummy5_element.isdigit() ):
#                     exit( 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!')
                        
#         if len(central_longitude_raw) == 3:
#             central_longitude_deg = float( float( central_longitude_raw[0]) + (1./60. * ( float(central_longitude_raw[1]) + (1./60. * float(central_longitude_raw[2]) ))))
#         else:
#             central_longitude_deg = float(central_longitude_raw[0])

    
#     lat0        = central_latitude_deg
#     lon0        = central_longitude_deg
  
#     #build spiral geometry with regular steps in azimuth and distance
#     for idx_r in xrange(number):
#         azimuth    = (idx_r + 1) * 350./float(number)
#         dist       = float(distmin + idx_r * diststep)
#         dist_north = dist * cos(azimuth / rad_to_deg)
#         dist_east  = dist * sin(azimuth / rad_to_deg)
#         lat_shift  = dist_north * rad_to_deg /R
#         lat_rec    = lat0 + lat_shift 
#         lon_shift  = dist_east * rad_to_deg / (R * sin((90-lat_rec)/rad_to_deg))
#         lon_rec    = lon0 + lon_shift 

#         coo[idx_r,0] = lat_rec
#         coo[idx_r,1] = lon_rec
#         coo[idx_r,2] = 0.

#     #write coordinate array as ascii to file     
#     File5 = file(filename,'w')
#     write_array(File5, coo, separator='    ',linesep='\n')
#     File5.close()

#     print '...done\n'

#----------------------------------------------------------------------

# def setup_db_white_noise(cfg):
#     """Build database with arbitrary greens function traces consisting of white noise.

#     input:
#     -- config dictionary

#     output:
#     NetCDF files with 'Greens funtions'; orderd by receiver name

#     """

#     #read in parameters
#     sourcelist    = cfg['list_of_source_points']
#     stations      = cfg['setup_gf_list_of_stations']
#     networks      = cfg['setup_gf_list_of_networks'] 
#     locations     = cfg['setup_gf_list_of_locations']
#     channels      = cfg['setup_gf_list_of_channels'] 
#     #window length in seconds
#     length        = int(cfg['time_window_length'])
#     #data sampling in Hz
#     gf_sampling   = float(cfg['gf_sampling_rate'])
#     #version of dataset 
#     gf_version    = int(cfg['gf_version'])

#     n_stations    = len(stations)
#     n_source      = len(sourcelist)

#     #set path for output
#     gf_dir_out    = path.realpath(path.abspath(path.join(cfg['base_dir'],cfg['gf_dir'])))
#     if (not path.exists(gf_dir_out) ):
#         os.makedirs(gf_dir_out)
            
#     #sampling in seconds
#     gf_sampling_in_s  = 1./gf_sampling
#     #number of samples 
#     n_time            = int(float(length) * gf_sampling) + 1
#     #set up dimensions of Greens functions array
#     gf                = zeros((n_stations, n_source,  3,6, n_time),float32)
    
#     #loop over all stations
#     for idx_stat in arange(len(stations)):

#         #set output filename
#         outfilename = path.realpath(path.abspath(path.join( gf_dir_out, 'gf_v%(ver)i_length%(twl)i_sampling%(samp)i_station_%(stat)s.nc' %{'ver':gf_version,'twl':length,'samp':gf_sampling,'stat':stations[idx_stat]})))

#         #set time axis
#         time_array  = arange(0,n_time)*gf_sampling_in_s

#         #provide empty list for collecting the traces
#         alldataout = []

#         #loop over all source points, components, and M_components    
#         for idx_s in arange(n_source):
#             for idx_comp in arange(3):
#                 for idx_m_comps in arange(6):
#                     #generate standfard distributed values
#                     dummy_data = numpy.random.randn(n_time)*10.
#                     #put 'data trace' into data list
#                     alldataout.append(dummy_data)

#         print 'writing to file:  ',outfilename,'\n'
        
#         #set file handler for output
#         outfile          = sioncdf.NetCDFFile(outfilename, 'w')#, 'Created ' + time.ctime(time.time())) 
#         #set NetCDF parameters
#         outfile.title    = "GF for station"+stations[idx_stat]
#         outfile.version  = int(cfg['gf_version'])
#         outfile.createDimension('GF_idx', len(alldataout)+1)
#         outfile.createDimension('t', n_time)

#         #create mandatory variable name under which the data array will be stored within the file
#         GF_for_act_rec   = outfile.createVariable('GF_for_act_rec', 'f', ('t', 'GF_idx'))

#         #put list into array
#         temp             = zeros((len(time_array),len(alldataout)+1), dtype=float32)
#         temp[:,0]        = time_array.astype('float32')
#         temp[:,1:]       = array(alldataout, dtype=float32).transpose()

#         #write array into the NetCDF variable
#         GF_for_act_rec[:] = temp
        
#         GF_for_act_rec.units = "'seconds' and 'arbitrary'"

#         #close file handler
#         outfile.close()
        
#         print 'gf for station  '+ stations[idx_stat] + ' ...  ok !!\n\n'

    
#----------------------------------------------------------------------

# def setup_db_qseis(cfg):
#     """Build database with arbitrary greens function calculated with 'QSEIS'.

#     input:
#     -- config dictionary

#     output:
#     NetCDF files with Greens funtions; orderd by receiver name

#     """
    
#     s_coo      =  source_coords(cfg)
#     r_coo      =  receiver_coords(cfg)
    
#     n_s        =  int(cfg['n_source'])
#     n_r        =  int(cfg['n_receiver'])

#     rec_list   = arange(n_r)
#     if (int(cfg['rec_names_given']) == 1):
#         tempfile   = file(cfg['rec_names_file'],'r')
#         rec_list_1 = tempfile.readlines()
#         rec_list   = [x.strip() for x in rec_list_1]

#     if ( int(cfg['partial_rec']) == 1 and int(cfg['rec_names_given']) == 0):
#         rec_max    = int(cfg['rec_max'])
#         rec_min    = int(cfg['rec_min'])
#         rec_list   = arange(rec_min,rec_max+1)-1

#     gf_dir_out = path.realpath(path.abspath(path.join(cfg['base_dir'],cfg['gf_dir'])))
#     if (not path.exists(gf_dir_out) ):
#         os.makedirs(gf_dir_out)
#     gf_dir_in  = cfg['gf_dir_input']

#     dist_step  = int(cfg['dist_step'])
#     anz_steps  =  1000/dist_step-1 
#     steps_array=  (arange(anz_steps)+1)*dist_step

#     rad_to_deg = 180./pi
#     order      = [1,1,3,1,2,2,4,4,1,4,5,5,6,6,8,6,7,7]

#     version    = int(cfg['gf_version'] )

#     print rec_list
#     for idx_r in rec_list:

#         files      = []
#         count      = 0   
    
#         outfilename = (gf_dir_out + 'gf_v%(ver)i_rec%(idx_r)i.nc') %{'ver':version,'idx_r':idx_r}
        

#         co_rec_act = r_coo[idx_r,:]


#         for idx_s in arange(n_s):

#             co_src_act         = s_coo[idx_s,:]
#             depth              = int(co_src_act[2])
#             azimuth, distance  = s_r_azi_dist(co_rec_act,co_src_act,cfg)
#             phi                = azimuth  / rad_to_deg
            
#             factors            = [cos(phi)**2,sin(phi)**2,1.,sin(2*phi),cos(phi),sin(phi),-0.5*sin(2*phi),0.5*sin(2*phi),0.,cos(2*phi),-sin(phi),cos(phi),cos(phi)**2,sin(phi)**2,1.,sin(2*phi),cos(phi),sin(phi)]
            

#             if (distance%1000 in steps_array):

#                 for gf_idx in arange(8):
                    
#                     filetemp           = (gf_dir_in +'gf%(idx1)i.dep.%(idx2)i.dist.%(idx3)i') %{'idx1':gf_idx+1,'idx2':depth,'idx3':distance} 
#                     files.append(filetemp)
#                     count             += 1 

#             else:

#                 distances,weights  = neighbouring_distances(distance,dist_step)
#                 distance1          = int(distances[0])
#                 distance2          = int(distances[1])

# #                print 'aaaaaaaaaaaaaaaa\n',idx_r,co_rec_act,idx_s,co_src_act,distance,azimuth,distances, weights

#                 for gf_idx in arange(8):
                
#                     tempfile1          = (gf_dir_in +'gf%(idx1)i.dep.%(idx2)i.dist.%(idx3)i') %{'idx1':gf_idx+1,'idx2':depth,'idx3':distance1}
#                     tempfile2          = (gf_dir_in +'gf%(idx1)i.dep.%(idx2)i.dist.%(idx3)i') %{'idx1':gf_idx+1,'idx2':depth,'idx3':distance2}
#                     t_FILE1            = file(tempfile1,'r')
#                     t_FILE2            = file(tempfile2,'r')
                    
#                     t_gf1              = loadtxt(t_FILE1)
#                     t_gf2              = loadtxt(t_FILE2)

#                     t_time             = 1/2.*(t_gf1[:,0]+t_gf2[:,0])

#                     t_data             = weights[0]*t_gf1[:,1]+ weights[1]*t_gf2[:,1]
#                     t_array            = zeros((len(t_time),2),float)
#                     t_array[:,0]       = t_time
#                     t_array[:,1]       = t_data
                    
#                     tempfile_out       = (gf_dir_in +'gf%(idx1)i.dep.%(idx2)i.dist.%(idx3)i') %{'idx1':gf_idx+1,'idx2':depth,'idx3':distance}

#                     t_FILE_OUT         = file(tempfile_out,'w')
#                     write_array(t_FILE_OUT,t_array, separator='    ',linesep='\n')
# #                    print 'aaaaaaaaa  ',t_array.shape
                    
#                     t_FILE1.close()
#                     t_FILE2.close()
#                     t_FILE_OUT.close()

#                     files.append(tempfile_out)
#                     count +=1

#                 print 'reading  source '+str(idx_s+1)+'  -  rec '+str(idx_r+1)+'     ( depth '+str(depth)+'  -  dist '+str(distance)+' )'


                
            
#         time1      = file(files[0],'r').readlines()
#         time_list  = [x.split()[0] for x in time1]

# #        print 'time:  ',time

#         alldatain=[]
#         alldataout=[]
#         gf_idx = 0

# #        print 'files', files,'\n'
        
#         for filename in files:
#             data1 = file(filename,'r').readlines()
#             data2 = [float(x.split()[1]) for x in data1]
#             alldatain.append(data2)
#             gf_idx += 1
#             if (gf_idx == 8):
#                 gf_idx = 0
#                 for tmp_idx in xrange(18):
#                     data_tmp = array(alldatain[order[tmp_idx]-1]) *array(factors[tmp_idx])
# #                    print len(data_tmp),filename,tmp_idx+1
                    
# #                     if (int(lp_flag) ==1 ):
# #                         periode      = 2.
# #                         dauer        = 12
# #                         length       = 15
# #                         step         = 0.5
# #                         zeit         = arange(length/step +1 ) * step
# #                         source       = stfff(zeit,periode,dauer)
# #                         data_tmp     = convolve(data_tmp,source)[:len(data_tmp)]

#                     alldataout.append(data_tmp)

#                 alldatain=[]


#         time_array = zeros((len(time_list)),float)
#         t_idx = 0                   
#         for lll in time_list:
#             time_array[t_idx]=float(lll)
#             t_idx +=1

# #         dataout_array = zeros((len(time_list),len(alldataout)+1),float)
# #         dataout_array[:,0] = time_array
        
# #         for ll in arange(len(alldataout)):
# #             dataout_array[:,ll+1] = array(alldataout)[ll,:]

#         print 'writing to file:  ',outfilename,'\n'
        
#         outfile          = sioncdf.NetCDFFile(outfilename, 'w')#, 'Created ' + time.ctime(time.time())) 
#         outfile.title    = "GF for receiver "+str(idx_r+1)
#         outfile.version  = int(cfg['gf_version'])

#         outfile.createDimension('GF_idx', len(alldataout)+1)
#         outfile.createDimension('t', len(time_list))

#         GF_for_act_rec   = outfile.createVariable('GF_for_act_rec', 'f', ('t', 'GF_idx'))

#         temp = zeros((len(time_list),len(alldataout)+1), dtype=float32)
    
#         temp[:,0] = time_array.astype('float32')
#         temp[:,1:]  = array(alldataout, dtype=float32).transpose()
            
#         GF_for_act_rec[:] = temp
#         #GF_for_act_rec[:,0]  = time_array.astype('float32')

#  #       for ll in arange(len(alldataout)):
#  #           print "xxx"
#   #          GF_for_act_rec[:,ll+1]  = array(alldataout)[ll,:].astype('float32')
        
#         GF_for_act_rec.units = "'seconds' and 'arbitrary'"
        
#         outfile.close()
        
#         print 'gf for receiver  '+str(idx_r+1)+ ' ...  ok !!'


#     print 'Green`s functions - database for current source-receiver-combinations ready !'
            
#----------------------------------------------------------------------

def source_coords(cfg):
    """Reads in the file with coordinates of source-points.

    Input:
    -- Configuration-dictionary

    Output:
    -- array in dimension [number of sources, 3] of coordinates of all gridpoints
       indexing in order (N,E,Z)
    """

    print 'reading grid_coords into memory...'

    n_source    =  10#cfg['n_source']
    filename    =  cfg['base_dir']+cfg['source_co_file']
    File3       =  file(filename,'r')
    #read in source_coordinates:
    temp4       =  loadtxt(File3, usecols=tuple(range(0,3)))
    File3.close()
    
    #test if number of sources is read in correctly:
    if ( len(temp4) < n_source):
        print ' number of source-coordinates in file too small '
        exit(0)
        #    if (len(temp4) != n_source):
        #        print 'number of usable source-points set to ', len(temp4)

#    temp4 = temp4[2::10,:]

    cfg['n_source'] = len(temp4)
    print 'n_source is: ', len(temp4)

    source_co   = temp4


    print '...done\n'

    return source_co

#----------------------------------------------------------------------

def receiver_coords(cfg):
    """Reads in the file with coordinates of receivers.

    Input:
    -- Configuration-dictionary

    Output:
    -- array in dimension [number of sources, 3] of coordinates of all receivers
    """

    print 'reading receiver_coords into memory ...'

    n_rec       =  4#int(cfg['n_receiver'])

    filename    =  cfg['base_dir']+cfg['rec_co_file']
    File2a       =  file(filename,'r')
    #read in receiver_coordinates:
    temp2a       =  loadtxt(File2a, usecols=tuple(range(0,3)))
    File2a.close()
    
    #test if number of receivers is read in correctly:
    if ( len(temp2a) < n_rec):
        print ' number of receivers too small '
        exit(0)
#    print 'number of used receivers set to ', len(temp2a)

    cfg['n_receiver'] = len(temp2a)
    print 'n_receiver set to ', len(temp2a)


    rec_co  = temp2a

    print ' ...done\n'

    return rec_co

#----------------------------------------------------------------------

def distance_azi_backazi(coord_s,coord_r):
    """
    Calculating the (back-)azimuth (clockwise) and the distance between the
    surface-projection of the source and the receiver.

    Input:
    -- array with source coordinates in (lat,lon,depth)
    -- array with receiver coordinates in (lat,lon,depth)
    -- Configuration-dicotionary

    Output:
    -- (back-)azimuth (in degree) -- as seen from 2nd argument (receiver)
    -- distance between surface-projection of the source and receiver (in m)
    """

    lat1  =  float(coord_s[0])/rad_to_deg
    lon1  =  float(coord_s[1])/rad_to_deg
    lat2  =  float(coord_r[0])/rad_to_deg
    lon2  =  float(coord_r[1])/rad_to_deg

    distance_in_m = R * arccos( sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1) )
    ddd = distance_in_m/R

    def calc_backazi(s_lat,s_lon,r_lat,r_lon,sr_dist):
        
        if sin(r_lon-s_lon) < 0:       
            tc1=arccos((sin(r_lat)-sin(s_lat)*cos(sr_dist))/(sin(sr_dist)*cos(s_lat)))    
        else:       
            tc1=2*pi-arccos((sin(r_lat)-sin(s_lat)*cos(sr_dist))/(sin(sr_dist)*cos(s_lat)))  

        if (cos(s_lat) < 0.000001) or (cos(r_lat) < 0.000001):
            if (s_lat > 0):
                tc1 = pi # starting from N pole
            elif (s_lat < 0):
                tc1 = 0.   #  starting from S pole

        backazi = -tc1*rad_to_deg%360
        return backazi

    azimuth_s_r      = calc_backazi(lat1,lon1,lat2,lon2,ddd)
    back_azimuth_r_s = calc_backazi(lat2,lon2,lat1,lon1,ddd)

    ba_d    = int(back_azimuth_r_s)
    ba_m    = (back_azimuth_r_s-ba_d)*60.
    ba_fm   = int(ba_m)
    ba_s    =  (ba_m-ba_fm)*60
    ba_dsm  = [ba_d,ba_fm,ba_s]
    a_d     = int(azimuth_s_r)
    a_m     = (azimuth_s_r-a_d)*60.
    a_fm    = int(a_m)
    a_s     =  (a_m-a_fm)*60
    a_dsm   = [a_d,a_fm,a_s]
    
    #print ba_dsm,a_dsm, [back_azimuth_r_s,azimuth_s_r],distance_in_m/1000,'\n'    

    return  distance_in_m, azimuth_s_r, back_azimuth_r_s



#----------------------------------------------------------------------

def calc_distance(coord1,coord2):
    """ 
    Calculates the surface-distance between two points on earth.

    Input:
    -- array with coordinates of point 1 in (lat,lon)
    -- array with coordinates of point 2 in (lat,lon)

    Output:
    -- Distance on earth-surface between the the points (in m)
    """


    lat1  =  coord1[0]/rad_to_deg
    lon1  =  coord1[1]/rad_to_deg
    lat2  =  coord2[0]/rad_to_deg
    lon2  =  coord2[1]/rad_to_deg

    distance = R * arccos( sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1) )

    return distance

#----------------------------------------------------------------------

# def neighbouring_distances(distance,dist_step):
#     """ 
#     Calculating the two best fitting distances for setting the respective interval, containing the actual distance.   


#     Input:
#     -- acvtual distance in m
#     -- step size of distance-grid/mesh

#     Output:
#     -- array (lower_corner_distance,upper_corner_distance) in m
#     -- array (weighting factor for lower corner, weighting factor for upper corner) 
#     """
    
#     anz   =  1000/dist_step - 1 
#     steps =  (arange(anz)+1) * dist_step
    
#     km    = int(distance/1000)
#     m     = distance%1000
#     km1  = km
#     km2  = km

    
#     if sum(m/steps)==0:
#         ind1 = 0
#         ind2 = 1
#     else:
#         aa = 0
#         bb = 0
#         while aa < anz:
#             if m/steps[aa] == 0:
#                 ind1 = aa
#                 ind2 = aa+1
#                 aa   = anz
#                 bb   = 1
#             else:
#                 aa += 1
#         if bb == 0:
#             ind1 = anz 
#             ind2 = 0
#             km2  = km+1

#     stepstotal  =  (arange(anz+1))*dist_step


#     dist1   = int(1000*km1+stepstotal[ind1])
#     dist2   = int(1000*km2+stepstotal[ind2])


#     weight1 = 1- (m- stepstotal[ind1])/float(dist_step)
#     weight2 = 1-weight1
    
#     distances = array([dist1,dist2])    
#     weights   = array([weight1,weight2])    

#     return distances, weights

#----------------------------------------------------------------------

# def setup_synth_data(cfg):

#     """ Beispieldaten fuer die gegebenen receiver fuer ein event an gegebener Quelle ('source'-Index)
    
#     """

#     import pickle as pp
#     import numpy.random as Rand  

#     s_coo      =  source_coords(cfg)
#     r_coo      =  receiver_coords(cfg)


#     gf_dir     = cfg['base_dir']+cfg['gf_dir']

#     date_dir   = cfg['base_dir']+cfg['date_dir']

#     dat_subdir = cfg['model_name']

#     lp_flag    = int(cfg['lp_flag'])
#     noiselevel = float(cfg['noiselevel'])            
#     n_r        = int(cfg['n_receiver'])
#     reclist    = xrange(n_r)

#     format_key = int(cfg['format_key'])
#     print 'format_key', format_key ,'\n'

#     chandged_data_setup = int(cfg['chandged_data_setup'])
                              
#     s_idx      = int(cfg['source_idx'])

#     print 'synthetic data for source no. '+ str(s_idx) + ' is being built ... \n'

#     if (int(lp_flag) == 1): 
#         date_dir = date_dir + "lp_"+  dat_subdir+"-noise-" + str(int(noiselevel*100)) +"/"
#     else:
#         date_dir = date_dir + dat_subdir +"-noise-" + str(int(noiselevel*100)) +"/"

#     if path.exists(date_dir):
#         shutil.rmtree(date_dir)       
#     mkdir(date_dir)
 
#     gf_version = int(cfg['gf_version']) 
#     version    = int(cfg['data_version']) 

   
#     column_min = (s_idx)*18+1
#     column_max = (s_idx+1)*18+1

#     print 'Spalten: ',column_min, ' -- ', column_max
    
#     M = array([1,-2,4,6,0,-1]) #bsp aus jost und hermann mit dominantem vertical strike-slip
# #    M = array([0,0,0,1,0,0])


#     for r_idx in reclist :
#         print 'and receiver  no. '+ str(r_idx) + ' ... \n'

#         if (format_key == 0):
#             t_infilename = (gf_dir + 'gf_v%(ver)i_rec%(idx_r)i.dat') %{'ver':gf_version,'idx_r':r_idx}
#             t_infile = file(t_infilename,'rb')
#         if (format_key == 1):
#             t_infilename = (gf_dir + 'gf_v%(ver)i_rec%(idx_r)i.dat') %{'ver':gf_version,'idx_r':r_idx}
#             t_infile = file(t_infilename,'r')
#         if ( format_key == 2) :
#             t_infilename  = (gf_dir + 'gf_v%(ver)i_rec%(idx_r)i.nc') %{'ver':gf_version,'idx_r':r_idx}
#             tmp_ncfile_gf = sioncdf.NetCDFFile(t_infilename,'r')
#             contained_vars_dict = tmp_ncfile_gf.variables
#             contained_vars = contained_vars_dict.keys()
#             gf_var         = contained_vars[0]
#             tmp_data       = tmp_ncfile_gf.variables[gf_var]
#             gf_raw         = array(tmp_data).astype('float64')
#             tmp_ncfile_gf.close()
 
           

#         if (lp_flag ==1):
#             t_outfilename = (date_dir + 'data_v%(ver)i_rec%(idx_r)i.dat') %{'ver':version,'idx_r':r_idx}
#         else:
#             t_outfilename = (date_dir + 'data_v%(ver)i_rec%(idx_r)i.dat') %{'ver':version,'idx_r':r_idx}


#         t_outfile = file(t_outfilename,'w')
#         print 'data for receiver:  ',r_idx,'\n into file: ', t_outfilename,'\n'

#         count = 0

#         if ( int(chandged_data_setup) == 1):

#             if (format_key == 0 or format_key == 1 ):
#                 print 'lese zeitachse von ', t_infilename
#                 time  = loadtxt(t_infile, usecols=((0,)) ) 
#                 print time
#                 print '... done\n'

#                 print 'lese GF-Matrix von ', t_infilename
#                 GF    = loadtxt(t_infile, usecols=tuple(range(column_min,column_max)) )
#                 print GF
#                 print '... done\n'

#                 print 'transponiere GF-Matrix\n'
#                 GF = transpose(matrix(GF))
#                 print '... done\n'

#             if (format_key == 2 ):
#                 print 'lese zeitachse von ', t_infilename
#                 time        = gf_raw[:,0]
#                 print '... done\n'
#                 print 'lese GF-Matrix von ', t_infilename
#                 tmp_gf1     = gf_raw[:,column_min:column_max] 
#                 print '... done\n'
#                 print 'transponiere GF-Matrix\n'
#                 GF          =  transpose(matrix(tmp_gf1))
#                 print '... done\n'


#             print 'Groesse GF: ', [size(GF,i) for i in range(2)]

#             File1a = file('GF_synth_data_act.dat','w')
#             File2a = file('time_synth_data_act.dat','w')
#             pp.dump(GF,File1a)
#             pp.dump(time,File2a)
#             File1a.close()
#             File2a.close()


#         elif ( int(chandged_data_setup) == 0):
#             print 'Einlesen von gespeicherten GF und time...\n'

#             File1a = file('GF_synth_data_act.dat','r')
#             GF     = pp.load(File1a)
#             File1a.close()
 
#             File2a = file('time_synth_data_act.dat','r')
#             time   = pp.load(File2a)
#             File2a.close()

# #        print [size(M,i) for i in range(1)]
            

#         G1 = (GF[0:6])
#         G2 = (GF[6:12])
#         G3 = (GF[12:18])

            

       
#         tmpD1 = array(transpose(matrix(dot (M, G1))))
#         tmpD1 = array([float(tmpD1[i]) for i in range(len(tmpD1))] )

#         tmpD2 = array(transpose(matrix(dot (M, G2))))
#         tmpD2 = array([float(tmpD2[i]) for i in range(len(tmpD2))] )

#         tmpD3 = array(transpose(matrix(dot (M, G3))))
#         tmpD3 = array([float(tmpD3[i]) for i in range(len(tmpD3))] )


#         if (int(lp_flag) == 1):
            
#             periode      = 2.
#             dauer        = 12
#             length       = 15
#             step         = 0.25
#             zeit         = arange(length/step +1 ) * step
#             source       = stfff(zeit,periode,dauer)

#             tmpD1        = convolve(tmpD1,source)[:len(tmpD1)]
#             tmpD2        = convolve(tmpD1,source)[:len(tmpD2)]
#             tmpD3        = convolve(tmpD1,source)[:len(tmpD3)]

#             #        print [size(tmpD1,i) for i in range(2)]

#         if (not int(noiselevel) == 0 ):

#             print 'addiere noise\n'
 
#             tmpD1  += noiselevel*max(abs(tmpD1))*Rand.randn(len(tmpD1))
#             tmpD2  += noiselevel*max(abs(tmpD2))*Rand.randn(len(tmpD2)) 
#             tmpD3  += noiselevel*max(abs(tmpD3))*Rand.randn(len(tmpD3)) 
            

# #        print len(time)
        
#         for i in xrange(len(time)):
# #            print 'itime = '+str(i)
#             t_outfile.write(str(time[i]))
#             t_outfile.write('    ')
#             t_outfile.write(str(float(tmpD1[i]))) # radial-component (R)
#             t_outfile.write('    ')
#             t_outfile.write(str(float(tmpD2[i]))) # tangential-component (phi)
#             t_outfile.write('    ')
#             t_outfile.write(str(float(tmpD3[i]))) # vertical-component (Z)
#             t_outfile.write('\n')
                
#         t_outfile.close()
#         print 'daten geschrieben \n'

#     print '....set !!!\n'

#----------------------------------------------------------------------

# def stfff(relt,per,durex):
#     """ Building a source-time-function for lp-equivalent event.

#     Input:
#     -- time-axis
#     -- main-periode of lp
#     -- duration of excitation
#     """

    
#     t1 = 2
#     t2 = t1 + durex - 5
#     t3 = t2/4.

#     funk = e**(-(relt-t3)**2/(2*pi*durex))*1./(1+e**(-2*(relt-t1))) *1./(1+e**(0.5*(relt-t2)))
#     #*sin(2*pi/per*relt)  

#     indmax = list(funk).index(max(funk))
#     indone = int(indmax*0.7)
#     newfun = funk[indone:]
    
#     return newfun

#----------------------------------------------------------------------

def set_reclist(rec_coo,cfg):


    if (int(cfg['changedsetup']) == 1):

        key = int(cfg['reclist_key'])

        if key == 0 :

            reclist = arange( len(rec_coo[:,0]))
            print 'take all (',len(rec_coo[:,0]),') receivers for analysis\n' 


        if key == 1:

            reclist = arange(int(cfg['rec_min']),int(cfg['rec_max'])+1)
            print 'take only ',int(cfg['rec_max'])-int(cfg['rec_min'])+1,' receivers ',list(reclist),' for analysis\n' 


        if key == 2 :

            number = int(cfg['n_rdm_rec'])
        
            reclist = rdm.sample(arange(int( cfg['n_receiver'] )) , number)
            print 'take only ',number,' randomly chosen receivers ',list(reclist),' for analysis'

        tempfile_rl = file(cfg['base_dir']+'reclist.dat','w')
        pp.dump(reclist,tempfile_rl)
        tempfile_rl.close()
        print 'wrote list of receivers  to file reclist.dat \n '

    elif (int(cfg['changedsetup']) == 0):
        tempfile_rl = file(cfg['base_dir']+'reclist.dat','r')
        reclist = list(pp.load(tempfile_rl))
        tempfile_rl.close()
        print 'list of receivers read from file reclist.dat: \n ',reclist


    else:
        print 'Key "changedsetup" wrong --- must be 0 or 1 '
        exit()
                        
    cfg['reclist']  = reclist
    
#----------------------------------------------------------------------

def set_sourcelist(coords,cfg):
    
    aa = 0
    while int(coords[aa,2] == coords[0,2]):
        aa +=  1
        
    ds  =  int(coords[aa,2] - coords[0,2])                    
    cfg['depthstep']   = ds



    if (int(cfg['changedsetup']) == 1):

        key = int(cfg['sourcelist_key'])

        if key == 0 :

            sourcelist = arange( len(coords[:,0]))
            print 'take all',len(coords[:,0]),' source_points for analysis\n' 


        if key == 1:

            sourcedepthmin   =   int(cfg['sourcedepthmin'])
            sourcedepthmax   =   int(cfg['sourcedepthmax'])

            numsourcetot     = len ( coords[:,0] )
            sourcelist = []
            for idx_s in arange(numsourcetot):
                if (coords[idx_s,2] >=sourcedepthmin and coords[idx_s,2] <=sourcedepthmax):
                    sourcelist.append(idx_s)


            sourcelist       = array(sourcelist)
            print 'take only ',int(len(sourcelist)),' sources for analysis (',min(sourcelist),' - ',max(sourcelist),' ) \n' 

        tempfile_sl = file(cfg['base_dir']+'sourcelist.dat','w')
        pp.dump(sourcelist,tempfile_sl)
        tempfile_sl.close()
        print 'wrote list of sources to file sourcelist.dat \n '

    elif (int(cfg['changedsetup']) == 0):

        tempfile_sl = file(cfg['base_dir']+'sourcelist.dat','r')
        sourcelist = list(pp.load(tempfile_sl))
        tempfile_sl.close()
        print 'list of sources read from file sourcelist.dat :\n NRs', min(sourcelist),' - ',max(sourcelist),'\n'

    else:
        print 'Key "changedsetup" wrong --- must be 0 or 1 '
        exit()
                        

    cfg['sourcelist']  = sourcelist
    cfg['n_source']    = len(sourcelist)


#----------------------------------------------------------------------

def rotate_traces(data_x,data_y,angle):
    """Rotating incoming traces (x,y) around the given angle in degrees (!) into (x',y'). 
    
    Input data is given w.r.t. the basis (e_x,e_y), whereas the output
    data is presented w.r.t. new basis (e_x',e_y'). The rotation
    angle is positive from e_y to e_y' (counter clockwise)

    Example:
    x is channel 1 of a seismometer pointing to the source
    in backazimuth 80 and y is channel 2, pointing to 170 degree. Then a
    rotation around 80 degree gives the channels north and east:

    east,north = rotate_traces(channel2,channel1,80)

    Assuming orthogonal base and canonical naming: e.g. input 'x'
    (tranversal) ist turned to be 'east' and input 'y' (radial) to
    'north'
    
    Input:
    -- data in positive x-direction
    -- data in positive y-direction
    -- angle in degrees
    
    Output: 
    -- x'-data , y'-data
    
    """
    
    rad_to_deg  = 180./pi
    rad_angle   = angle/rad_to_deg 
    cphi        = math.cos(rad_angle)
    sphi        = math.sin(rad_angle)
              
    if 0:# ( len(data_x) != len(data_y) ):
        print 'ERROR! Rotation not possible!  Length of first trace : %i samples -- Length of second trace : %i samples'%(len(data_x),len(data_y))
        print 'ERROR! Invoke function only with data traces of equal length!'
        raise SystemExit

    new_x, new_y  =  (data_x * cphi + data_y * sphi) ,  - data_x * sphi + data_y * cphi
    
    return new_x, new_y

#----------------------------------------------------------------------

def make_solution_dict(cfg):

    sol_dict = {}
    solution_parameters = ['event_ID' , 'eventType','lat','lat_err_down','lat_err_up','lon','lon_err_down','lon_err_up','height','height_err_down','height_err_up','datetime','datetime_err_down','datetime_err_up','M_0','M_0_unit','VR','clvd','DC','tensorWeighted','weighting_type', 'weighted_M_0','weighted_M_0_unit','weighted_VR','weighted_clvd','weighted_DC','M_xx','M_yy','M_zz','M_xy','M_xz','M_yz','stationinformationfiles','datafiles','GF_files','topography_model_files','velocity_model_files','stf_file','inversion_grid_file' ]


#     solution_values = [cfg['event_ID'] , cfg['eventType'] , cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg[''], cfg['']


    
    for ii in solution_parameters :
        sol_dict[ii] = 0.#solution_values[ii]
 
 
 

    cfg['solution_dictionary'] = sol_dict
    
    return 1

#----------------------------------------------------------------------
def make_parameter_dict(cfg):

    params_dict = {}
    general_parameters = ['GF_files', 'topography_model_files','velocity_model_files','stf_file','inversion_grid_file']

    
    for ii in general_parameters:
        params_dict[ii] = []

    cfg['parameter_dictionary'] = params_dict
    
    return 1

#----------------------------------------------------------------------
def put_xml_to_database(cfg):


    import httplib
    import os
    import sys
    import glob
    import base64
    
    xml_filename   = path.realpath(path.abspath(path.join(cfg['temporary_directory'], cfg['xml_file_name'])))

    
    servername = '193.174.161.15:8080'
    serverpath = '/xml/seismology/event'

    error_log    = ''
    upload_user  = 'data_upload'
    upload_pw    = 'little-Prince'
    auth         = 'Basic ' + (base64.encodestring(upload_user+':'+ upload_pw) ).strip()
    
    temp_FH = file(xml_filename,'r')
    data = temp_FH.read()
    temp_FH.close()
    
    #construct and send the header
    webservice = httplib.HTTP(servername)

    # mode: 'PUT' overwrites old file with the same name
    # mode: 'POST' appends xml file to database, if not already existing
    #webservice.putrequest("PUT", serverpath + '/' + cfg['xml_file_name'])
    webservice.putrequest("POST", serverpath + '/' + cfg['xml_file_name'])
    webservice.putheader("Authorization",auth)
    webservice.putheader("Host", "localhost")
    webservice.putheader("User-Agent", "put_xml_to_database")
    webservice.putheader("Content-type", "text/xml; charset=\"UTF-8\"")
    webservice.putheader("Content-length", "%d" % len(data))
    webservice.endheaders()
    webservice.send(data)
    
    # get the response
    statuscode, statusmessage, header = webservice.getreply()
    # handle errors
    if statuscode==201:
        
        # CREATED - resources has been created successfully
        pass
    elif statuscode==403:
        # FORBIDDEN - usually means resource already exists - we skip it
        # Use a POST request to modify a existing resource!
        pass
        #continue
        # there was some error
    print "Server: %s%s" % (servername,serverpath)
    print "Response: %s - %s" % (statuscode, statusmessage)
    print "Headers: %s" % (header)


    
    
    return 1   
#----------------------------------------------------------------------
 
def present_results(cfg):

    opt_source_dict        = cfg['optimal_source_dictionary']
    source_coordinates     = cfg['source_point_coordinates']
    #lo_gridpoints          = cfg['list_of_gridpoints']
    lo_stats               = cfg['list_of_stations_with_data']
    lo_receivers           = cfg['list_of_receivers_with_data']

    #n_gridpoints           =  len(lo_gridpoints)
    n_receivers            =  len(lo_receivers)
    n_stats                =  len(lo_stats)

    #correction factor for synthetic data, generated with wrong M0
    m_opt                  = opt_source_dict['M_opt']#*10**(10.7/2.)
    
    vr_opt                 = opt_source_dict['VR_opt']
    source_opt             = opt_source_dict['source']
    
    source_opt_coords      = source_coordinates[source_opt-1]
    source_time            = cfg['optimal_source_time']
    source_time_tuple      = time.gmtime(source_time)

    
    if (int(cfg['summation_key']) == 0):
        m_temp             = [m_opt[0], m_opt[3], m_opt[4], m_opt[3], m_opt[1], m_opt[5], m_opt[4], m_opt[5], m_opt[2]]
        m_opt_mat          = matrix(m_temp).reshape(3,3)
        
        if not decomp_m(m_opt_mat, cfg):
            print 'ERROR! could not decompose MT!'
            exit()
        print 'decomposition of moment tensor ...done \n'
    
        decomp_out         = cfg['decomposed_M_opt'] 

    #d_break(locals(),'solution')

    
    print  '\n Event am %2.2i.%2.2i.%4.4i um %2.2i h %2.2i min %2.2i sec \n'%( source_time_tuple[2],source_time_tuple[1],source_time_tuple[0],source_time_tuple[3],source_time_tuple[4],source_time_tuple[5])

    print '(epoch:',source_time,')'
    
    norm_value = abs(m_opt_mat.max() )
    #M0 = sqrt(abs(decomp_out[8]*norm_value))*(1e-9)

    M_iso            = diag( array([1./3*trace(m_opt_mat),1./3*trace(m_opt_mat),1./3*trace(m_opt_mat)] ) )
    M0_iso           = abs(1./3*trace(m_opt_mat))
    M_devi           = m_opt_mat - M_iso
    eigenw1,eigenv1  = linalg.eig(M_devi)
    eigenw_devi      = real( take( eigenw1,argsort(abs(eigenw1)) ) )
    M0_devi          = max(abs(eigenw_devi))
    M0               = M0_iso + M0_devi
    #M_orig           = array([-24,76,-52,6,51,53])/90.887358 *10**(1.5*15.7)
    

    MW = 2./3.* log10(M0*10**7)  - 10.7
    print '\n ... mit Staerke M_0 = %e  (equiv. M_w = %.2f)' %(M0 ,MW)

    print '\n bei Index %i  ( Lat = %.3f deg, Lon = %.3f deg, z = %.1f m )' %(source_opt,source_opt_coords[0],source_opt_coords[1],source_opt_coords[2])

    print '\n...DC:  \t%3.2f  \n...CLVD:  \t%3.2f  \n...VR:   \t%3.2f ' %(int(round(decomp_out[5])),  int(round(100-decomp_out[5])) ,vr_opt)

    print '\n...  full Moment Tensor:'
    expo = int(log10(M0))
    mred = array(m_opt)/10.**(expo-1)
    print '\nMT:\n ( %.1f,%.1f,%.1f,%.1f,%.1f,%.1f )  x 10 **%i\n'%(mred[0],mred[1],mred[2],mred[3],mred[4],mred[5],expo-1)

    print '\n...  full Moment Tensor:\npython mopad.py p  %i,%i,%i,%i,%i,%i -D -I -s 8 -f ~/BB.svg&' %(int(mred[0]),int(mred[1]),int(mred[2]),int(mred[3]),int(mred[4]),int(mred[5]) ) 
    print 'python mopad.py c  %f,%f,%f,%f,%f,%f -t sdr -y ' %(mred[0],mred[1],mred[2],mred[3],mred[4],mred[5])
    print 'python mopad.py d  %f,%f,%f,%f,%f,%f -p full,iso,devi,iso_perc,clvd_perc,dc_perc -y ' %(mred[0],mred[1],mred[2],mred[3],mred[4],mred[5])


    print '\n\n',m_opt_mat/norm_value
    print '\nscaling factor %e\n'%(norm_value)
    print '\n...  MT - deviatoric part: \n\n',(decomp_out[3]+decomp_out[4])/norm_value    

    print '\n...  MT - isotropic part: \n\n',decomp_out[2]/norm_value    
    print '\n...  MT - DC part: \n\n',decomp_out[3]/norm_value    
    print '\n...  MT - CLVD part: \n\n',decomp_out[4]/norm_value    #

    print '\n\n  Solution for analysis with data from stations \n'
    
    print cfg['list_of_stations_with_data']
    print '\n   ( with weights: \n'
    print cfg['station_weights_dict'], '  ) \n\n'

    cfg['decomposed_M_opt'] = decomp_out
    
    
    return 1   

#----------------------------------------------------------------------

def build_xml_hardcode(cfg):



    opt_source_dict        = cfg['optimal_source_dictionary']
    source_coordinates     = cfg['source_point_coordinates']
    decomp_out             = cfg['decomposed_M_opt'] 
    m_opt                  = opt_source_dict['M_opt']
    vr_opt                 = opt_source_dict['VR_opt']
    source_opt             = opt_source_dict['source']

    #---------------------------


    source_coordinates     = cfg['source_point_coordinates']
    lo_stats               = cfg['list_of_stations_with_data']
    lo_receivers           = cfg['list_of_receivers_with_data']

    n_receivers            =  len(lo_receivers)
    n_stats                =  len(lo_stats)
    
    source_opt_coords      = source_coordinates[source_opt-1]
    source_time            = cfg['optimal_source_time']
    source_time_tuple      = time.gmtime(source_time)

    
    m_temp             = [m_opt[0], m_opt[3], m_opt[4], m_opt[3], m_opt[1], m_opt[5], m_opt[4], m_opt[5], m_opt[2]]
    m_opt_mat          = matrix(m_temp).reshape(3,3)
      
    decomp_m(m_opt_mat, cfg)
    decomp_out         = cfg['decomposed_M_opt']
    
    norm_value = abs(m_opt_mat.max() )
    M_iso            = diag( array([1./3*trace(m_opt_mat),1./3*trace(m_opt_mat),1./3*trace(m_opt_mat)] ) )
    M0_iso           = abs(1./3*trace(m_opt_mat))
    M_devi           = m_opt_mat - M_iso
    eigenw1,eigenv1  = linalg.eig(M_devi)
    eigenw_devi      = real( take( eigenw1,argsort(abs(eigenw1)) ) )
    M0_devi          = max(abs(eigenw_devi))
    M0               = M0_iso + M0_devi
    
    MW               = 2./3.*( log10(M0*10**7)  - 10.7)
    
    #--------------------------

    #solution_dict  = cfg['solution_dictionary']
    #params_dict  = cfg['parameter_dictionary']

    #for kkk in params_dict.keys():
    #    solution_dict[kkk] = params_dict[kkk]

    root = xet.Element('Seismic_Data', volcano_id = str(cfg['volcano_id']), project_id = str(cfg['project_id']))
    doc = xet.ElementTree(root)
    
    event = xet.Element('Event')
    origin = xet.Element('Origin')
    inv_params = xet.Element('InversionParam2200eter')
    root.append(event)
    root.append(origin)   
    root.append(inv_params)   

    ev_ID = xet.SubElement(event,'detectionID')
    ev_ID.text = str(cfg['event_ID'])
    
    ev_type = xet.SubElement(event,'eventType')
    ev_type.text = str(cfg['event_type'])
    
    
    or_lat = set_valueTag(origin,'latitude',source_opt_coords[0],source_opt_coords[0],source_opt_coords[0])
    or_len = set_valueTag(origin,'longitude',source_opt_coords[1],source_opt_coords[1],source_opt_coords[1])
    or_height = set_valueTag(origin,'height',-source_opt_coords[2],-source_opt_coords[2],-source_opt_coords[2])
   
    or_datetime = set_valueTag(origin,'datetime',cfg['event_datetime_in_epoch'],cfg['event_datetime_in_epoch'],cfg['event_datetime_in_epoch'])

    or_M = xet.SubElement(origin,'MomentTensor')

    #    set_tensor_tag(or_M,'tensor',solution_dict)
    
    MT  = xet.SubElement(or_M,'tensor') 
    set_standard_tag(MT,'Mnn',str(m_opt[0]))
    set_standard_tag(MT,'Mee',str(m_opt[1]))
    set_standard_tag(MT,'Mzz',str(m_opt[2]))
    set_standard_tag(MT,'Mne',str(m_opt[3]))
    set_standard_tag(MT,'Mnz',str(m_opt[4]))
    set_standard_tag(MT,'Mez',str(m_opt[5]))

    set_standard_tag(or_M, 'M0',  decomp_out[8])
    set_standard_tag(or_M, 'scalarMoment_unit','Nm')
    
    set_standard_tag(or_M, 'VR',vr_opt)
    
    set_standard_tag(or_M, 'clvd',str(int(round(100-decomp_out[5])))  )
    set_standard_tag(or_M, 'DC', str(int(round(decomp_out[5]) ) ))
    
    or_weighted_M = xet.SubElement(or_M,'tensorWeighted')
    set_standard_tag(or_weighted_M,'weighting',str(0))

    w_MT  = xet.SubElement(or_weighted_M,'tensor')
    set_standard_tag(w_MT,'Mnn',str(m_opt[0]))
    set_standard_tag(w_MT,'Mee',str(m_opt[1]))
    set_standard_tag(w_MT,'Mzz',str(m_opt[2]))
    set_standard_tag(w_MT,'Mne',str(m_opt[3]))
    set_standard_tag(w_MT,'Mnz',str(m_opt[4]))
    set_standard_tag(w_MT,'Mez',str(m_opt[5]))
 
    set_standard_tag(or_weighted_M, 'scalarMoment',  decomp_out[8])
    set_standard_tag(or_weighted_M, 'scalarMoment_unit','10e17 Nm')
    set_standard_tag(or_weighted_M, 'VR',vr_opt)
    set_standard_tag(or_weighted_M, 'clvd',str(int(round(100-decomp_out[5])))  )
    set_standard_tag(or_weighted_M, 'DC',str(int(round(decomp_out[5]))) )

    ip_contstats = set_listtag(inv_params,'ContributingStations',cfg['list_of_stations_with_data'])
    ip_statinfo = set_filetag(inv_params,'StationInformation',[])
    #ip_data = set_filetag(inv_params,'SeismicData',cfg['list_of_datafiles'])

    gf_filelist  = []
    gf_version   = (cfg['gf_version'])
    length       = (cfg['time_window_length'])
    gf_sampling  = (cfg['gf_sampling_rate'])
    for current_stat in cfg['list_of_stations_with_data']:
        fn_tmp   =  'gf_v%s_length%s_sampling%s_station_%s.nc' % (gf_version,length,gf_sampling,current_stat)
        filename =  path.realpath(path.abspath(path.join( cfg['base_dir'], cfg['gf_dir'],fn_tmp))) 
        gf_filelist.append(filename)
        
    ip_GF = set_filetag(inv_params,'GreensFunctions',gf_filelist)

    ip_topo = set_filetag(inv_params,'TopographyModel', [cfg['topography_model_file'] ])
    ip_vel = set_filetag(inv_params,'VelocityModel',[cfg['velocity_model_file']])
    ip_stf = set_filetag(inv_params,'SourceTimeFunction',[cfg['sourcetimefunction_file']])
    ip_invgrid = set_filetag(inv_params,'InversionGrid',[path.realpath(path.abspath(path.join(cfg['base_dir'],cfg['grid_coordinates_filename']  )))] )
    
    ip_twindow = xet.SubElement(inv_params,'TimeWindowLength')
    ip_twindow.text  = str(cfg['time_window_length'])
    
    ip_smplrate = xet.SubElement(inv_params,'SamplingRate')
    ip_smplrate.text  = str(cfg['data_sampling_rate'])

    ip_bw = xet.SubElement(inv_params,'Bandwidth')
    ip_bw_low = xet.SubElement(ip_bw,'LowerFrequency')
    ip_bw_low.text = cfg['bp_lower_corner']
    ip_bw_up = xet.SubElement(ip_bw,'UpperFrequency')
    ip_bw_up.text = cfg['bp_upper_corner']


    author = xet.Element('Author')
    root.append(author)
    author.text = cfg['author']
    
    
    docform       = xet.tostring(doc,encoding='utf-8',xml_declaration=True,pretty_print=True)
    outfilename   = path.realpath(path.abspath(path.join(cfg['temporary_directory'], cfg['xml_file_name'])))
    outf          = file(outfilename,'w')
#    print docform ,'\nwrite to file ',outfilename
    outf.write(docform)
    outf.close()
    
    return 1   

#----------------------------------------------------------------------
def set_standard_tag(parentelement, tagname, value):
    act_tag  =xet.SubElement(parentelement, tagname)
    val_act = xet.SubElement(act_tag, 'value')
    val_act.text = str(value)
#----------------------------------------------------------------------
   
def set_tensor_tag(parentelement, tagname, sol_dict):
    tensor_tag = xet.SubElement(parentelement,tagname)
    M1 = xet.SubElement(tensor_tag,'M_xx')
    M1.text = str(sol_dict['M_xx'])
    M2 = xet.SubElement(tensor_tag,'M_yy')
    M2.text = str(sol_dict['M_yy'])
    M3 = xet.SubElement(tensor_tag,'M_zz')
    M3.text = str(sol_dict['M_zz'])
    M4 = xet.SubElement(tensor_tag,'M_xy')
    M4.text = str(sol_dict['M_xy'])
    M5 = xet.SubElement(tensor_tag,'M_xz')
    M5.text = str(sol_dict['M_xz'])
    M6 = xet.SubElement(tensor_tag,'M_yz')
    M6.text = str(sol_dict['M_yz'])
    
#----------------------------------------------------------------------

def set_listtag(parentelement,elementname,in_list ):

    act_tag = xet.SubElement(parentelement, elementname)
    list_of_files = in_list
    joined_list_of_files =[]
    for ii in list_of_files:
        if ii == list_of_files[0]:
            joined_list_of_files.append(str(ii))
        else:
            joined_list_of_files.append('\n'+str(ii))
        act_tag.text =  (''.join(joined_list_of_files))
    return act_tag
        

#----------------------------------------------------------------------
def set_filetag(parentelement,elementname,list_of_files ):

    act_tag = xet.SubElement(parentelement, elementname)
    act_file = xet.SubElement(act_tag, 'files')

#    act_file.text  = str(list_of_files)
#   return act_tag

    joined_list_of_files =[]
    for ii in list_of_files:
        if ii == list_of_files[0]:
            joined_list_of_files.append(str(ii))
        else:
            joined_list_of_files.append('\n'+str(ii))
        act_file.text =  (''.join(joined_list_of_files))
    return act_tag
        
#----------------------------------------------------------------------

def set_valueTag(parent,name, num_value, lowerUncertainty='', upperUncertainty=''):
    
    act_tag = xet.SubElement(parent,name)
    
    act_value = xet.SubElement(act_tag,'value')
    sigma_down = xet.SubElement(act_tag,'lowerUncertainty')
    sigma_up = xet.SubElement(act_tag,'upperUncertainty')
    act_value.text=str(num_value)
    sigma_down.text=str(lowerUncertainty)
    sigma_up.text=str(upperUncertainty)
    
    return act_tag

#----------------------------------------------------------------------

def build_xml(solution_dict):

    from lxml import etree as xet

    mandatory_contents =['detectionID','eventType']#,latitude,longitude,height, datetime,MomentTensor,]
    
    list_maintags =['Event', 'Origin', 'InversionParameter', 'Author']
    list_maintag_variables = ['event','origin','inv_params','author']
    
    list_event=['ID', 'type']
    list_origin =['lat', 'lon', 'height', 'date', 'M']
    list_inv_params = ['contstats', 'statinfo', 'data', 'twindow', 'smplrate', 'bw', 'stf', 'invgrid', 'GF', 'topo', 'vel']
    list_M = ['M_0', 'M_0_unit','VR','clvd','DC','bp_up','bp_down','sigma_down','sigma_up']
    
    for m_c_i in mandatory_contents:
        if (not m_c_i in solution_dict):
            print 'missing part of solution: %s'%(str(m_c_i))
            exit ()
    

    root = xet.Element('Seismic_Data', volcano_id='234234', project_id='ExuperyV1.0' )

    event = xet.Element('Event')
    origin = xet.Element('Origin')
    inv_params = xet.Element('InversionParameter')
    author = xet.Element('Author')

    ev_ID = xet.SubElement(event,'detectionID')
    ev_type = xet.SubElement(event,'eventType')
    
    or_lat = xet.SubElement(origin,'latitude')
    or_lon = xet.SubElement(origin,'longitude')
    or_height = xet.SubElement(origin,'height')
    or_date = xet.SubElement(origin,'datetime')
    or_M = xet.SubElement(origin,'MomentTensor')

    ip_contstats = xet.SubElement(inv_params,'ContributingStations')
    ip_statinfo = xet.SubElement(inv_params,'StationInformation')
    ip_data = xet.SubElement(inv_params,'SeismicData')
    ip_twindow = xet.SubElement(inv_params,'TimeWindow')
    ip_smplrate = xet.SubElement(inv_params,'SamplingRate')
    ip_bw = xet.SubElement(inv_params,'Bandwidth')
    ip_stf = xet.SubElement(inv_params,'SourceTimeFunction')
    ip_invgrid = xet.SubElement(inv_params,'InversionGrid')
    ip_GF = xet.SubElement(inv_params,'GreensFunctions')
    ip_topo = xet.SubElement(inv_params,'TopographyModel')
    ip_vel = xet.SubElement(inv_params,'VelocityModel')


    M = xet.Element('tensor')
    M_0 = xet.Element('scalarMoment')
    M_0_unit = xet.Element('unit')
    VR  = xet.Element('varianceReduction')
    clvd = xet.Element('clvd')
    DC = xet.Element('doubleCouple')
    bp_down = xet.Element('LowerFrequency')
    bp_up = xet.Element('UpperFrequency')
    sigma_down = xet.Element('lowerUncertainty')
    sigma_up = xet.Element('upperUncertainty')


    for solution_variable in  mandatory_contents:
        temp_value = xet.SubElement(solution_variable,'value')
        temp_value.text(float(cfg[str(solution_variable)]))

        

    roottree = xet.ElementTree(root)

    doc = xet.tostring(roottree, xml_declaration=True, pretty_print=True)
    xml_file_handle = file(xml_filename,'w')
    xml_file_handle.write(doc)
    xml_file_handle.close()

    if _debug:
        print doc
        
#-#----------------------------------------------------------------------

def set_general_parameters(cfg):
    

    # SET TIME FRAMES:
    
    # eventtime
    eventtime =  read_datetime_to_epoch(cfg['datetime'])
    #print eventtime
    #exit()
    cfg['event_datetime_in_epoch'] = eventtime
    current_year = time.gmtime(eventtime)[0]
    #todo: sonderfall silvester noch einbauen !!!!

    #time windows
    length_of_moving_window   = float(cfg['time_window_length'])
    length_of_main_window     = float(cfg['main_window_length'])
    stepsize_of_moving_window = float(cfg['window_steps_in_s'])
    stepsize_of_moving_window_in_idx = round(stepsize_of_moving_window*float(cfg['gf_sampling_rate']))

    print stepsize_of_moving_window
    print 1./float(cfg['gf_sampling_rate'])
    print stepsize_of_moving_window%(1./float(cfg['gf_sampling_rate']))
    
    if not abs( (stepsize_of_moving_window/(1./float(cfg['gf_sampling_rate']))%1) ) < numeric_epsilon  :
        exit_string = 'sampling of GF (%.3f sec) and shift of time window (%.3f sec) are in no integer ratio'%(1./float(cfg['gf_sampling_rate']),stepsize_of_moving_window)
        exit(exit_string)

    if length_of_main_window <  length_of_moving_window :
        length_of_main_window = length_of_moving_window
        cfg['main_window_length'] = length_of_main_window
        print 'PROBLEM! Length of main window shorter than analysis window! - resized to ',length_of_main_window,' seconds'
    number_of_window_steps = int(int((length_of_main_window - length_of_moving_window )/stepsize_of_moving_window ) + 1. )
        
    starttime_of_main_window  =  eventtime
    endtime_of_main_window    =  eventtime + length_of_main_window
    cfg['starttime_of_main_window']   =  starttime_of_main_window
    cfg['endtime_of_main_window']     =  endtime_of_main_window

    #show information about time configuration
    print 'Analysing time interval from ', starttime_of_main_window,' to ',endtime_of_main_window-length_of_main_window,' in ',  number_of_window_steps,' step(s) of %.3f seconds \n '%(stepsize_of_moving_window)

    #provide list of successive moving time window startttimes
    cfg['list_of_moving_window_starttimes'] = (arange(number_of_window_steps)*stepsize_of_moving_window +starttime_of_main_window).tolist()
    #provide list of successive moving time window startttimes
    cfg['list_of_moving_window_startidx']   = [int(x) for x in arange(number_of_window_steps)*stepsize_of_moving_window_in_idx]

    #SET LISTS OF STATIONS AND CHANNELS (manually set and/or to_skip)

    #build list with contributing stations, if provided in config file   
    dummy_station_list_in_raw = cfg.get('list_of_contributing_stations','0')
    dummy_station_list_in     = dummy_station_list_in_raw.split(',')        
    provided_list_of_stations = []
    for ii in dummy_station_list_in:
        ii2 = ii.strip()
        if not ii2[0].isdigit():
            provided_list_of_stations.append(ii2.upper())
    if provided_list_of_stations:
        ordered_list_of_stations = list(sort(list(set(provided_list_of_stations))))
        cfg['provided_list_of_stations'] = ordered_list_of_stations


    #build list with stations to skip in analysis, if provided in config file 
    dummy_stations2skip_in_raw = cfg.get('list_of_stations_to_skip','')
    dummy_stations2skip_in     = dummy_stations2skip_in_raw.split(',')        
    stations2skip_list         = []
    for ii in dummy_stations2skip_in:
        ii2 = ii.strip()
        stations2skip_list.append(ii2.upper())
    if stations2skip_list:
        ordered_list_of_s2skip = list(sort(list(set(stations2skip_list))))
        cfg['stations2skip_list'] = ordered_list_of_s2skip
        
    cfg['list_of_all_channels'] = ['BHN','BHE','BHZ']

    #build list with contributing channels, if provided in config file   
    dummy_channel_list_in_raw = cfg.get('list_of_contributing_channels','0')
    dummy_channel_list_in     = dummy_channel_list_in_raw.split(',')        
    provided_list_of_channels = []
    for ii in dummy_channel_list_in:
        ii2 = ii.strip()
        if not ii2[0].isdigit():
            provided_list_of_channels.append(ii2.upper())
    if provided_list_of_channels:
        cfg['provided_list_of_channels'] = list(sort(list(set(provided_list_of_channels))))

    #build list with channels to skip in analysis, if provided in config file 
    dummy_channels2skip_in_raw = cfg.get('list_of_channels_to_skip','')
    dummy_channels2skip_in     = dummy_channels2skip_in_raw.split(',')        
    channels2skip_list         = []
    for ii in dummy_channels2skip_in:
        ii2 = ii.strip()
        channels2skip_list.append(ii2.upper())
    if channels2skip_list:
        cfg['channels2skip_list'] = channels2skip_list

    make_dict_channels_indices(cfg)

    
    #read source point coordinate file to array
    if not read_sourcepoint_configuration(cfg):
        print 'ERROR! Could not read in grid configuration!'
        exit()

    #read station coordinate file to array
    if not read_station_coordinates(cfg):
        print 'ERROR! Could not read in station configuration!'
        exit()
        
    #check, if Z channel of data is given in 'down' instead of default 'up':
    z_up_down = cfg.get('change_z_up_down',0)
    if int(round(float(z_up_down))) == 1:
        cfg['change_z_up_down'] = 1
    else:
        cfg['change_z_up_down'] = 0


        
    return 1

#---------------------------------------------------------------------
    
def read_station_coordinates(cfg):

    station_coords_filename = cfg['station_coordinates_filename']
    station_coords_file = path.realpath(path.abspath(path.join(cfg['base_dir'], station_coords_filename  )))
    
    if not path.isfile(station_coords_file):
        print 'file with station coordinates not found !'
        print station_coords_file
        exit()            

    config_filehandler = ConfigParser.ConfigParser()
    config_filehandler.read(station_coords_file)
       
            
    list_of_all_stations     = []
    station_index_dict       = {}    
    station_coordinate_dict  = {}
    station_distance_dict    = {}
    lo_dist                  = []
    
    coord_array      = cfg['source_point_coordinates']
    midpoint_coords  = [mean(coord_array[:,0]), mean(coord_array[:,1])] 
     
    sections = config_filehandler.sections()
    #loop over all stations in the provided file
    for sec in sections:
        list_of_all_stations.append(sec)        
        temp_lat_lon_dict ={}
        options = config_filehandler.options(sec)
        #mandatory entries lat,lon, index are read out
        for opt in options:
            value = config_filehandler.get(sec,opt)
            if opt == 'lat' or opt == 'latitude' or opt == 'lon' or opt == 'longitude':
                if opt == 'latitude':
                    opt = 'lat'
                if opt == 'longitude':
                    opt = 'lon'
                    
                lat_lon =  value.split(',')
                if not (len(lat_lon) == 3 or len(lat_lon) == 1) :
                    print 'ERROR!! Wrong coordinate format for ',opt, ' of station ', sec,'!!!'
                    exit()
                else:
                    for lat_lon_element in lat_lon:
                        dummy3 = lat_lon_element.split('.')
                        if not ( len(dummy3) in [1,2]):
                            print 'ERROR!! Wrong coordinate format for ',opt, ' of station ', sec,'!!!'
                            exit()
                        for dummy3_element in dummy3:
                            try:
                                int(dummy3_element)
                            except:
                                print 'ERROR!! Wrong coordinate format for ',opt, ' of station ', sec,'!!!'
                                raise SystemExit
                                
                        
                    if len(lat_lon) == 3:
                        lat_lon_deg = float( float( lat_lon[0]) + (1./60. * ( float(lat_lon[1]) + (1./60. * float(lat_lon[2]) ))))
                    else:
                        lat_lon_deg = float(lat_lon[0])
                        
                    
                temp_lat_lon_dict[opt] = lat_lon_deg

            if  opt == 'index' or opt == 'idx':
                station_index              = int(float(value))
                temp_lat_lon_dict['index'] = station_index
            
        if not len(temp_lat_lon_dict) == 3:
            print 'coordinates of station ',sec, ' are wrong - latitude and/or longitude and/or index missing  !!!!!'
            raise SystemExit

        #building dictionaries, having current station 'sec'
        station_coordinate_dict[sec]           = temp_lat_lon_dict
        station_index_dict[sec]                = station_index
        station_index_dict[str(station_index)] = sec
        distance,azi,backazi                   = distance_azi_backazi(midpoint_coords,[temp_lat_lon_dict['lat'], temp_lat_lon_dict['lon']])
        station_distance_dict[sec]             = distance
        lo_dist.append(distance)
        
    #end loop over stations

    cfg['station_coordinate_dictionary']      = station_coordinate_dict
    cfg['list_of_all_stations']               = list(sort(list(set(list_of_all_stations))))
    cfg['list_of_stations']                   = list(sort(list(set(list_of_all_stations))))
    station_index_dictionary_complete         = station_index_dict
    cfg['station_index_dictionary_complete']  = station_index_dictionary_complete
    cfg['station_index_dictionary']           = station_index_dict




    
    #provided list overrides  complete list
    if cfg.has_key('provided_list_of_stations'):
        print 'take provided list of stations'
        cfg['list_of_stations']         = cfg['provided_list_of_stations']
        dummy_idx                       = 1

        subsection_station_index_dict   = {}        
        for kk in cfg['provided_list_of_stations']:
            subsection_station_index_dict[kk]             = dummy_idx
            subsection_station_index_dict[str(dummy_idx)] = kk
            dummy_idx += 1
        cfg['station_index_dictionary'] = subsection_station_index_dict

    #randomly chosen sample of stations is built up 
    try:
        n_stations = int(cfg.get('number_of_random_stations',0))
    except:
        n_stations = 0
        
    if n_stations == 0:
        pass
    else:
        lo_stats  =   cfg['list_of_all_stations']   

        if n_stations > len(lo_stats):
            n_stations = len(lo_stats)
        if n_stations < 1:
            n_stations = 1
        
        random_station_list     = rdm.sample(lo_stats , n_stations)
        cfg['list_of_stations'] = list(sort(list(set(random_station_list))))
        dummy_idx = 1
        random_station_index_dict = {}
        for kk in random_station_list:
            random_station_index_dict[kk] = dummy_idx
            random_station_index_dict[str(dummy_idx)] = kk
            dummy_idx += 1
        cfg['station_index_dictionary'] = random_station_index_dict


    #stations which are marked to skip are eliminated
    if cfg.has_key('stations2skip_list'):
        lo_stats           = cfg['list_of_stations']   
        lo_stations2skip   = cfg['stations2skip_list']
        print 'skipped stations:'
        print lo_stations2skip
        station_index_dict = {}
        for stat in lo_stats:
            if stat in lo_stations2skip:
                lo_stats.remove(stat)
        count = 1
        for stat in lo_stats:
            station_index_dict[stat]         = count
            station_index_dict[str(count)] = stat
            count += 1
        cfg['station_index_dictionary'] = station_index_dict
        cfg['list_of_stations']         = lo_stats
    

    
    #filling array 'stat_coords'
    lo_stats    = cfg['list_of_all_stations']   
    s_number    = len(lo_stats)
    stat_coords = zeros((s_number,7),float)
    distmax     = max(lo_dist)

    #give weights to stations - if nothing is given, weights are '1'
    weights_dict= find_weights_for_stations(cfg,station_distance_dict,lo_dist)

    # find minimum and maximum arrival times for each station from model's velocity parameters:
    vp_max      = float(cfg.get('vpmax',-1))
    vs_min      = float(cfg.get('vsmin',-1))
    if (vs_min > vp_max) or (vs_min < 0) or  (vp_max < 0) :
        exit('ERROR in config-file !! Provide proper velocities in m/s with vpmax > vsmin!')

    tmin_tmax_dict ={}
    
    for hh in xrange(s_number):
        stationname       = station_index_dictionary_complete[str(hh+1)]
        temp_co_dict      = station_coordinate_dict[stationname]
        lat               = float(temp_co_dict['lat'])
        lon               = float(temp_co_dict['lon'])
        idx               = int(temp_co_dict['index'])
        dist              = station_distance_dict[stationname]
        weight            = weights_dict[stationname]
        print stationname,  idx, dist,weight

        stat_coords[hh,0] = int(idx)
        stat_coords[hh,1] = lat
        stat_coords[hh,2] = lon
        stat_coords[hh,3] = dist
        stat_coords[hh,4] = weight

        # calculate earliest and latest signal arrival at this station, factor 2 is arbitrarily set !! 
        # for to do so, expand times including half flank length of tapering window, which overall flank length is 15%
        # so the whole useful data is within FWHM of tapering function with steepness factor 0.85 
        # (flank length = 15% = 0.15 => half flank = 0.075 => stretch needed = 1/(1-0.075) = 40/37
        #  => extension to every side 1/2*3/37 = 3/74

        model_tmin = dist/vp_max
        
        largest_depth = max(abs( cfg['source_point_coordinates'][:,2]))

        stretch_factor = float(cfg.get('stretch_factor',1.5))
        model_tmax = stretch_factor*sqrt(dist**2+largest_depth**2)/vs_min

        effective_window = model_tmax - model_tmin

        # expansion of section
        model_tmin_eff   = model_tmin - 3./74.*effective_window

        #window cannot start before t=0:
        if model_tmin_eff < 0:
            model_tmin_eff = 0 

        model_tmax_eff   = model_tmax + 3./74.*effective_window

            
        #set dictionary-entry for curent station 
        tmin_tmax_dict[stationname] = [model_tmin_eff ,model_tmax_eff]
            
            
    cfg['stations_tmin_tmax']              = tmin_tmax_dict
    cfg['station_coordinates']             = stat_coords

    if _debug:
        print '\n\nusing %2i stations for analysis:\n'%(len(cfg['list_of_stations']))
        print sort(cfg['list_of_stations']),'\n'#.sort()
    
    print 'Station coordinates ok!\n\n'

    return 1

#---------------------------------------------------------------------
def read_sourcepoint_configuration(cfg):

    grid_coords_filename   = cfg['grid_coordinates_filename']
    grid_coords_file       = path.realpath(path.abspath(path.join(cfg['base_dir'], grid_coords_filename  )))

    
    if not path.isfile(grid_coords_file):
        print 'file with grid coordinates not found ! :\n',grid_coords_file
        raise SystemExit

    try:
        grid_fh     = file(grid_coords_file,'r')
        coord_array = loadtxt(grid_fh)
        grid_fh.close()
    except:
        print 'ERROR !! Provided file %s not readable\n'%(grid_coords_file)
        raise SystemExit

    #indizes of source grid points start at '1' for human readability
    lo_gridpoints = list( arange( len(coord_array[:,0]) )+1 )

    #entries to config dictionary
    cfg['list_of_all_gridpoints']     = lo_gridpoints
    cfg['source_point_coordinates']   = coord_array

    #find centre coordinate

    #PREPARE SECTION OF GRID AS GIVEN IN CONFIGURATION FILE

    #start with a copy of the gridpoint list:
    list_of_gridpoint_section = lo_gridpoints

    gs_type_of_section = int(cfg.get('gs_type_of_section',0))

    if not int(gs_type_of_section) == 0 :        
        list_of_gridpoint_section = []
        depth_boundaries = [min(abs(coord_array[:,2])),max(abs(coord_array[:,2]))]
        try:
            depth_boundaries[0] = int(cfg['gs_depth_upper_boundary'])                                      
        except:
            pass
        try:
            depth_boundaries[1] = int(cfg['gs_depth_lower_boundary'])
        except:
            pass                                                              
        
        if int(gs_type_of_section) in [11,12]:
            #centre of section is centre of grid:
            midpoint_coords  = [mean(coord_array[:,0]), mean(coord_array[:,1])]
            
            if int(gs_type_of_section) == 11:
                #type 'circle', so calculate points within radius
                try:
                    circle_radius = float(cfg['gs_radius_of_circle'])
                except:
                    print 'no radius provided - sectioning skipped\n\n'
                    print 'Grid coordinates ok!\n\n'
                    return 1
                
                for ii in lo_gridpoints:
                    coords2 = coord_array[ii-1]
                    distance,azi,back_azi = distance_azi_backazi(midpoint_coords,coords2)
                    
                    if distance <= circle_radius and (depth_boundaries[0] <= coords2[2] <= depth_boundaries[1] ):
                        list_of_gridpoint_section.append(ii)
            #d_break(locals(),'grid')
            
            if int(gs_type_of_section) == 12:
                #type 'lat/lon range', so calculate points in given range
                if cfg.has_key('gs_lat_range_in_m'):
                    dist_in_m = float(cfg['gs_lat_range_in_m'])
                    new_lat_n, new_lon = distance2coordinates(midpoint_coords,dist_in_m,0)
                    new_lat_s, new_lon = distance2coordinates(midpoint_coords,dist_in_m,180)
                    new_lat_range      = 0.5*(new_lat_n - new_lat_s)
                    cfg['gs_lat_range']= new_lat_range
                if cfg.has_key('gs_lon_range_in_m'):
                    dist_in_m = float(cfg['gs_lon_range_in_m'])
                    new_lon, new_lon_e = distance2coordinates(midpoint_coords,dist_in_m,90)
                    new_lat, new_lon_w = distance2coordinates(midpoint_coords,dist_in_m,270)
                    new_lon_range      = 0.5*(new_lon_e - new_lon_w)
                    cfg['gs_lon_range']= new_lon_range
                try:
                    lat_range  = float(cfg['gs_lat_range']) 
                    lon_range  = float(cfg['gs_lon_range'])
                except:
                    print 'no lat/lon range provided - sectioning skipped\n\n'
                    print 'Grid coordinates ok!\n\n'
                    return 1
                    
                
                lat_lower = float(   midpoint_coords[0] - lat_range )
                lat_upper = float(   midpoint_coords[0] + lat_range )
                lon_lower = float(   midpoint_coords[1] - lon_range )
                lon_upper = float(   midpoint_coords[1] + lon_range )

                for ii in lo_gridpoints:
                    if  lat_lower <= coord_array[ii-1,0] <= lat_upper and lon_lower <= coord_array[ii-1,1] <= lon_upper and (depth_boundaries[0] <= coord_array[ii-1,2] <= depth_boundaries[1] ):
                        list_of_gridpoint_section.append(ii)
   
                    
        if int(gs_type_of_section) in [21,22]:

            #centre of section is given in config file:

            try:
                lat0_raw_string = cfg['gs_section_centre_lat']
                lat0_raw        = lat0_raw_string.split(',')
            except:
                print 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!  - skipping sectioning\n\n'
                print 'Grid coordinates ok!\n\n'
                return 1
                    
            if len(lat0_raw) == 3 or len(lat0_raw) == 1 :
                for lat0_raw_element in lat0_raw:
                    dummy3 = lat0_raw_element.split('.')
                    if not ( len(dummy3) in [1,2]):
                        print 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!  - skipping sectioning\n\n'
                        print 'Grid coordinates ok!\n\n'
                        return 1
                    for dummy3_element in dummy3:
                        try:
                            int(dummy3_element)
                        except:
                            print 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!  - skipping sectioning\n'
                            print 'Grid coordinates ok!\n\n'
                            return 1
                if len(lat0_raw) == 3:
                    lat0_raw_deg = float( float( lat0_raw[0]) + (1./60. * ( float(lat0_raw[1]) + (1./60. * float(lat0_raw[2]) ))))
                else:
                    lat0_raw_deg = float(lat0_raw[0])


            try:
                lon0_raw_string = cfg['gs_section_centre_lon']
                lon0_raw        =lon0_raw_string.split(',')
            except:
                print 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!  - skipping sectioning\n'
                print 'Grid coordinates ok!\n\n'
                return 1
                    
            if len(lon0_raw) == 3 or len(lon0_raw) == 1 :
                for lon0_raw_element in lon0_raw:
                    dummy3 = lon0_raw_element.split('.')
                    if not ( len(dummy3) in [1,2]):
                        print 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!  - skipping sectioning\n'
                        break
                    for dummy3_element in dummy3:
                        try:
                            int(dummy3_element)
                        except:
                            print 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!  - skipping sectioning\n'
                            print 'Grid coordinates ok!\n\n'
                            return 1
                if len(lon0_raw) == 3:
                    lon0_raw_deg = float( float( lon0_raw[0]) + (1./60. * ( float(lon0_raw[1]) + (1./60. * float(lon0_raw[2]) ))))
                else:
                    lon0_raw_deg = float(lon0_raw[0])
 
                
            lat0 = lat0_raw_deg
            lon0 = lon0_raw_deg         
            midpoint_coords  = [lat0,lon0]
            
            if int(gs_type_of_section) == 21:

                #type 'circle', so calculate points within radius
                try:
                    circle_radius = float(cfg['gs_radius_of_circle'])
                except:
                    print 'no radius provided - sectioning skipped'
                    list_of_gridpoint_section = lo_gridpoints
                    print 'Grid coordinates ok!\n\n'
                    return 1
                
                for ii in lo_gridpoints:
                    coords2 = coord_array[ii-1]
                    distance,azi,back_azi = distance_azi_backazi(midpoint_coords,coords2)
                    if distance <= circle_radius and (depth_boundaries[0] <= coords2[2] <= depth_boundaries[1] ):
                        list_of_gridpoint_section.append(ii)


            if int(gs_type_of_section) == 22:
                #type 'lat/lon range', so calculate points in given range
                if cfg.has_key('gs_lat_range_in_m'):
                    dist_in_m = float(cfg['gs_lat_range_in_m'])
                    new_lat_n, new_lon = distance2coordinates(midpoint_coords,dist_in_m,0)
                    new_lat_s, new_lon = distance2coordinates(midpoint_coords,dist_in_m,180)
                    new_lat_range      = 0.5*(new_lat_n - new_lat_s)
                    cfg['gs_lat_range']= new_lat_range
                if cfg.has_key('gs_lon_range_in_m'):
                    dist_in_m = float(cfg['gs_lon_range_in_m'])
                    new_lon, new_lon_e = distance2coordinates(midpoint_coords,dist_in_m,90)
                    new_lat, new_lon_w = distance2coordinates(midpoint_coords,dist_in_m,270)
                    new_lon_range      = 0.5*(new_lon_e - new_lon_w)
                    cfg['gs_lon_range']= new_lon_range
                try:
                    lat_range  = float(cfg['gs_lat_range']) 
                    lon_range  = float(cfg['gs_lon_range'])
                except:
                    print 'no lat/lon range provided - sectioning skipped'
                    list_of_gridpoint_section = lo_gridpoints
                    print 'Grid coordinates ok!\n\n'
                    return 1
                
                lat_lower = float(   midpoint_coords[0] - lat_range )
                lat_upper = float(   midpoint_coords[0] + lat_range )
                lon_lower = float(   midpoint_coords[1] - lon_range )
                lon_upper = float(   midpoint_coords[1] + lon_range )

                for ii in lo_gridpoints:
                    if  lat_lower <= coord_array[ii-1,0] <= lat_upper and lon_lower <= coord_array[ii-1,1] <= lon_upper and (depth_boundaries[0] <= coord_array[ii-1,2] <= depth_boundaries[1] ):
                        list_of_gridpoint_section.append(ii)
 
        if not list_of_gridpoint_section:
            print 'no grid points in selected area - take original grid instead'
            list_of_gridpoint_section = lo_gridpoints
        else:
            print 'using only section of source grid for analysis - code %i\n'%(gs_type_of_section)
        
    cfg['list_of_gridpoint_section'] = list_of_gridpoint_section    
    print 'Grid coordinates ok!\n\n'

#     d_break(locals(),'gridpoints')
#     exit('dfsdfsdfsdf')
    if _debug:
        print 'list of %4i selected gridpoints:\n\n'%(len(list_of_gridpoint_section)), list_of_gridpoint_section,'\n'
        #raise SystemExit

    return 1
#---------------------------------------------------------------------
def set_list_of_datafiles(cfg):
    """
    Find all files in file structure that contain data of the given time window. Naming of the files has to be according to mseed standard (including julian day in file name).

    input:
    -- config dictionary

    output:
    -- config dictionary entries
    [list_of_datafiles]

    indirect output:
    -- control parameter

    """

    startdate = time.gmtime(float(cfg['starttime_of_main_window']) )
    enddate = time.gmtime(float(cfg['endtime_of_main_window']) )
    current_year = int(startdate[0])
    parent_data_directory = cfg['parent_data_directory']
    days = []
    days.append( int(startdate[7]) )
    days.append( int(enddate[7]) )

    days = list( set( days)  )

    if _debug:
        print 'days:  ',days
        print 'data in:  ',parent_data_directory
    
    lo_datafiles = []
    mm = re.compile(r'\.(?<!\d)(?P<day>\d{3})(?!\d)')
    for root,dirs,files in os.walk(parent_data_directory):
        for name in files:
            fn = os.path.join(root,name)
            kk = mm.search(name)
            #if not kk:
            #    continue
            
            #kk_dict=kk.groupdict()
            
            if 1:#int(kk_dict['day']) in days:
                lo_datafiles.append(fn)

                
    #    list_of_datafiles =   pymseed.select_files([parent_data_directory],selector=lambda x: int(x.day) in days , regex=r'(?<!\d)(?P<day>\d{3})(?!\d)' )

    cfg['list_of_datafiles'] = lo_datafiles

    print shape(lo_datafiles)
    
    #%d_break(locals(),'sdfffffff')
    #exit()

    return 1   

#---------------------------------------------------------------------

def set_raw_data_for_main_window(cfg):
    """
    Read in MiniSeed data for given time window with tools from PYMSEED. Data has to be stored under path given in config dictionary.

    input:
    -- config dictionary

    indirect output:
    -- config dictionary entries
    [data_pile_for_main_window,list_of_ContributingStations,list_of_receivers,list_of_channels,mseed_data_pile]

    output:
    -- control parameter

    """

    
    
    mseed_data_pile        = pymseed.MSeedPile(cfg['list_of_datafiles'],cachefilename='/tmp/arctic_cache')
    cfg['mseed_data_pile'] = mseed_data_pile
    chopped_data_pile      = []
    starttime              = cfg['starttime_of_main_window']
    endtime                = cfg['endtime_of_main_window']
    lo_stations            = cfg['list_of_stations']

    #d_break(locals(),'chopped data_pile')

    deltat                 = mseed_data_pile.get_deltats()[0]


    #-----------------------------
    #PLANNED: insert padding, allowing a consistent downsampling, needed e.g. for restitution
    #    CHECK!!
    filter_flag = int( float( cfg.get('filter_flag',0) ) )
    bp_hp_freq  = float(cfg.get('bp_lower_corner'))


    pad_length = 1./bp_hp_freq*2
    if filter_flag == 0:
        pad_length = 10

    pad_length =0
    #----------------------------
    
    
   #use PYMSEED chopper to slice out data piles from MiniSeed files in arbitrary file structure
    for ii in mseed_data_pile.chopper(tmin=starttime,tmax=endtime+deltat,tpad=pad_length, want_incomplete=False):
        chopped_data_pile.append(ii)

    #bandpass vielleicht spaeter wg. tapering effekten - GF-fesnster hat andere groesse !!!

    #if int(cfg['filter_flag']) == 1:
        #for jj in chopped_data_pile[0]:
            #jj.bandpass(float(cfg['bp_order']), float(cfg['bp_lower_corner']), float(cfg['bp_upper_corner']) )


    list_of_ContributingStations = []
    list_of_receivers            = []
    list_of_channels             = []
    
    for kk in chopped_data_pile[0]:
        if kk.station in lo_stations:
            list_of_ContributingStations.append(kk.station)
            list_of_channels.append(kk.channel)
            receiver_name = str(kk.station)+'.'+str(kk.channel)
            list_of_receivers.append(receiver_name)
        
    #d_break(locals(),'chopped data_pile')
    #exit()
    
    list_of_ContributingStations = list(set(list_of_ContributingStations ))
    list_of_receivers            = list(set(list_of_receivers))
    list_of_channels             = list(set(list_of_channels))

    cfg['data_pile_for_main_window']      = chopped_data_pile[0]
    cfg['list_of_stations_with_data']     = list(sort(list_of_ContributingStations))
    cfg['list_of_receivers_with_data']    = list(sort(list_of_receivers))
    cfg['list_of_channels_with_data']     = list(sort(list_of_channels))

    if len(list(sort(list_of_receivers))) == 0 :
        print 'No stations with data available for time span\n ', time.gmtime(starttime),time.gmtime(endtime)
        raise

  #   d_break(locals(),'data read in ')
#     exit()

    
    make_dict_ContributingStations_indices(cfg)
    
    make_dict_receivers_indices(cfg)
    #make_dict_channels_indices(cfg)   

    if 0:# _debug:
        print 'current content of cfg after "set_raw_data" :\n'
        for ii in sort(cfg.keys()):
            print ii

    #if receivers without data existing, update inv_A:
    # update A - correct for potentially missing traces or changed weighting factors
    if not correct_inv_A_4_missing_traces_and_weights(cfg):
        print 'ERROR !! Updating inv_A not poosible'
        raise SystemExit


#     print sum(cfg['inv_A'])
#     exit()

    #if not set(list_of_receivers) == set(cfg['list_of_all_receivers']):
    if _debug:
        print 'list of receivers available \n',sort(cfg['list_of_all_receivers'])
        print 'list of receivers with data \n',sort(list_of_receivers)
        

    return 1
    
#---------------------------------------------------------------------

def make_dict_ContributingStations_indices(cfg):
    """  Setup of bijection 'stations whose data is contributing to the inversion <-> index 1...M'.

    input:
    -- config dictionarys
    
    output:
    -- ascii file with calculated ContributingStation index dictionary

    indirect output:
    -- entry within config dictionary
    [ContributingStation_index_dictionary]
    """
    
    
    
    list_of_ContributingStations = cfg['list_of_stations_with_data']

    ContributingStations_index_dict = {}
    counter = 0
    for ii in list_of_ContributingStations:
        ContributingStations_index_dict[str(counter)] = ii
        ContributingStations_index_dict[ii] = counter
        counter+= 1
        
    cfg['ContributingStation_index_dictionary'] = ContributingStations_index_dict
    
    list_of_keys = []
    list_of_values = []
    for ii in ContributingStations_index_dict:
        list_of_keys.append(ii)
        list_of_values.append(ContributingStations_index_dict[ii])      

      
    sort_idx = argsort(list_of_keys)
    sorted_list_of_keys = list(take(list_of_keys,sort_idx))
    sorted_list_of_values = list(take(list_of_values,sort_idx))

    Cont_station_idx_outfile_name  = path.realpath(path.abspath( path.join( cfg['temporary_directory'],'current_ContributingStation_index_dictionary.dat' ) ))
    
    Cont_station_idx_outfile = file(Cont_station_idx_outfile_name,'w')
    for ii in arange(len((sorted_list_of_keys))):
        write_string = '%s    =    %s  \n' %(sorted_list_of_keys[ii],sorted_list_of_values[ii])
        Cont_station_idx_outfile.write(write_string)
    Cont_station_idx_outfile.close()

    
#---------------------------------------------------------------------
def make_dict_channels_indices(cfg):
    """
    Setup of bijection '[N,E,Z] <-> [0,1,2]'.
    
    input:
    -- config dictionary

    indirect output:
    -- entry within config dictionary
    [channel_index_dictionary]
    """


    channel_index_dict = {}
    list_of_channels   = cfg['list_of_all_channels']
    
    if cfg.has_key('provided_list_of_channels'):
        list_of_channels = cfg['provided_list_of_channels']
    if cfg.has_key('channels2skip_list'):
        lo_channnels2skip = cfg['channels2skip_list']
        for channel in list_of_channels:
            if channel in lo_channnels2skip:
                list_of_channels.remove(channel)

    for uu in list_of_channels:
        if uu.endswith('N'):
            channel_index_dict[uu] = 0
        elif uu.endswith('E'):
            channel_index_dict[uu] = 1
        elif uu.endswith('Z'):
            channel_index_dict[uu] = 2
        else:      
            print 'ERROR 234'
            exit()
            
    channel_index_dict['0'] = 'N'
    channel_index_dict['1'] = 'E'
    channel_index_dict['2'] = 'Z'
    channel_index_dict['N'] =  0
    channel_index_dict['E'] =  1    
    channel_index_dict['Z'] =  2
    channel_index_dict['n'] =  0
    channel_index_dict['e'] =  1
    channel_index_dict['z'] =  2
    
    cfg['list_of_channels']         = list_of_channels
    cfg['channel_index_dictionary'] = channel_index_dict

#---------------------------------------------------------------------

def make_dict_stations_indices_complete(cfg):
    """
    Setup of bijection 'all existing stations <-> index 1...N'. Also enumerating stations out of bounds of chosen geometrical section.
    
    input:
    -- config dictionary

    output:
    -- ascii file with calculated station index dictionary

    indirect output:
    -- entry within config dictionary
    [station_index_dictionary]
    """

    list_of_all_stations = cfg['list_of_all_stations']
    station_index_dict = {}
    counter = 0
    for ii in list_of_all_stations:
        station_index_dict[str(counter)] = ii
        station_index_dict[ii] = counter
        counter+= 1
        
    cfg['station_index_dictionary_complete'] = station_index_dict
    
    list_of_keys = []
    list_of_values = []
    for ii in station_index_dict:
        list_of_keys.append(ii)
        list_of_values.append(station_index_dict[ii])      

      
    sort_idx = argsort(list_of_keys)
    sorted_list_of_keys = list(take(list_of_keys,sort_idx))
    sorted_list_of_values = list(take(list_of_values,sort_idx))

    station_idx_outfile_name  = path.realpath(path.abspath( path.join( cfg['temporary_directory'],'current_station_index_dictionary_complete.dat' ) ))
    
    station_idx_outfile = file(station_idx_outfile_name,'w')
    for ii in arange(len((sorted_list_of_keys))):
        write_string = '%s    =    %s  \n' %(sorted_list_of_keys[ii],sorted_list_of_values[ii])
        station_idx_outfile.write(write_string)
    station_idx_outfile.close()

#---------------------------------------------------------------------
def make_dict_receivers_indices(cfg):
    """
    Setup of bijection 'existing receivers <-> index 1...N'.
    
    input:
    -- config dictionary

    output:
    -- ascii file with calculated receiver index dictionary

    indirect output:
    -- 2 entries within config dictionary
    [receiver_index_dictionary,list_of_all_receivers ]

    """

    lo_a_stats    = cfg['list_of_stations']
    lo_a_channels = cfg['list_of_all_channels']
    lo_a_recs     = []

    for stat in lo_a_stats:
        for chan in lo_a_channels:
            receiver = '%s.%s'%(stat,chan)
            lo_a_recs.append(receiver)
    cfg['list_of_all_receivers'] = lo_a_recs

    list_of_receivers = sort(cfg['list_of_receivers_with_data'])
    
    rec_index_dict = {}
    counter = 0
    for ii in list_of_receivers:
        rec_index_dict[str(counter)] = ii
        rec_index_dict[ii] = counter
        counter+= 1
        
    cfg['receiver_index_dictionary'] = rec_index_dict
    
    list_of_keys    = []
    list_of_values  = []
    for ii in rec_index_dict:
        list_of_keys.append(ii)
        list_of_values.append(rec_index_dict[ii])

    sort_idx = argsort(list_of_keys)
    sorted_list_of_keys = list(take(list_of_keys,sort_idx))
    sorted_list_of_values = list(take(list_of_values,sort_idx))

    rec_idx_outfile_name  = path.realpath(path.abspath( path.join( cfg['temporary_directory'],'current_receiver_index_dictionary.dat' ) ))
    
    rec_idx_outfile = file(rec_idx_outfile_name,'w')
    for ii in arange(len((sorted_list_of_keys))):
        write_string = '%s    =    %s  \n' %(sorted_list_of_keys[ii],sorted_list_of_values[ii])
        rec_idx_outfile.write(write_string)
    rec_idx_outfile.close()
    
#---------------------------------------------------------------------

def read_datetime_to_epoch(datetime):
    #TODO
    #replace with python routine from module 'calendar'
    #!!!
    
    if len(datetime.split('.')) == 2:
        milisecs = float('0.'+datetime.split('.')[1])
    else:
        milisecs = 0.            
    date = datetime.split('.')[0]
    print date
    format = '%Y-%m-%dT%H:%M:%S'
    time_tuple = time.strptime(date,format)
    epoch_seconds = calendar.timegm(time_tuple) + milisecs

    return epoch_seconds
#---------------------------------------------------------------------
def distance2coordinates(coords, distance, angle):
    #TODO CHECK !!!!
    
    """
    Calculation of the new coordinates if walking a given distance on Earth's surface from given position. The coordinates and the bearing angle (azimuth) shall be given in degrees. 


    input:
    -- array or tuple with coordinates with 'latitude', 'longitude' in decimal degrees  as first two entries
    -- distance on Earth's surface in metres
    -- azimuth in decimal degrees

    output:
    -- 2 element result 'latitude', 'longitude' of the new point (in decimal degrees)
    """


    lat0      = coords[0]/rad_to_deg
    lon0      = coords[1]/rad_to_deg

    azimuth   = angle/rad_to_deg
    angular_distance = distance/R


    lat2 = arcsin(sin(lat0)*cos(angular_distance) + cos(lat0)*sin(angular_distance)*cos(azimuth))

    lon2 = lon0 + arctan2(  sin(azimuth)*sin(angular_distance)*cos(lat0), cos(angular_distance) - sin(lat0) * sin(lat2)   )  	

    lat_goal = lat2 * rad_to_deg
    len_goal = lon2 * rad_to_deg
    
    return lat_goal, len_goal

#---------------------------------------------------------------------
#---------------------------------------------------------------------
def restitute_trace(data,cfg):
    """
    Restitution of raw data trace.

    TODO    Is to be calculated by the import of Sebastian's tool!!!

    input:
    -- raw data trace
    -- config-dictionary

    output:

    -- restituted data trace
    
    """
    #read n trace, remove mean, normalization, take poles and zeros, calculate transferfunction, fourier trafo, division of transfer, fourier trafo, normalisation, remove mean
    #
    #


    pass



    return data
#---------------------------------------------------------------------
def find_weights_for_stations(cfg,station_distance_dict,lo_dist):
    """
    Weighting factor for data from a station in the given distance.

    Either a file with weights is provided or a linear decrease is
    assumed. In the latter case, the weight of data at source point is
    1, weight at the farthest point is 0.5.

    #To be optimized: future
    weighting by inverting average noise level (standard deviation of
    signal); either by station or by component

    input:
    -- index of current station
    -- config-dictionary

    output:
    -- dictionary with station = weight (Maximum is set to 1, others are scaled respectively

    """

    print 'reading/finding station weights...\n'

    distmax = max(lo_dist)
    distmin = min(lo_dist)
    
    station_weights_dict = {}
    lo_stats    = cfg['list_of_all_stations']   
        
    for stat in lo_stats:
        station_weights_dict[stat] = 1

    weighting_flag = int(cfg.get('weight_stations','0'))


    if not (weighting_flag in[0,1,2,3] ):
        weighting_flag = 0


    if weighting_flag == 2:

        try:
            station_weights_dict2 = station_weights_dict.copy()
            if cfg.has_key('file_with_weighting_factors'):

                fn   = path.join(cfg['base_dir'],cfg['file_with_weighting_factors'] )
                frmt = dtype({'names':['stationname','weight'],'formats':['S6',float]})
                weights_array = loadtxt(fn, dtype=frmt,comments='#')

                tmp_lo_w = []
                print weights_array
                print shape(array(weights_array))

                for st in weights_array:
                    print st
                    if st[0] in lo_stats and 0 <= float(st[1])< 10000 :
                        station_weights_dict2[st[0]] = float(st[1])
                        tmp_lo_w.append(float(st[1]))

                #max_weight = float(max(tmp_lo_w))

                array_w_weights = array(lo_stats)
                                
                #for s_in_dict in sort(station_weights_dict2.keys()):
                #    station_weights_dict2[s_in_dict] = (station_weights_dict2[s_in_dict])#/max_weight

                station_weights_dict = station_weights_dict2.copy()
            else:
                print 'you forgot to provide the name of the file with weights - keyword "file_with_weighting_factors" not found in dictionary - assuming evenly weighted stations instead'
        
            #return station_weights_dict2
        
        except:
            print 'cannot read provided file with weighting informations'
            print 'stations unweighted'


        
    if weighting_flag in [1,3]:
        for station in sort(station_weights_dict.keys()):
            dist      = station_distance_dict[station]
            if weighting_flag == 1:
                #linear weighting - nearest station gets weight 1, farthest gets 0.5
                station_weights_dict[station] = ((0.5*dist + 0.5*distmin - distmax)/(distmin-distmax))
                print 'stations weighted decreased linear by distance'
            if weighting_flag == 3:
                station_weights_dict[station] = sqrt(dist)
                print 'stations weighted increasing in squareroot of distance'

                



    #try to read original dictionary with weights from file in temporary folder 
    try:
        fn =  path.abspath(path.realpath(path.join(cfg['temporary_directory'],'station_weights.txt')))
        FH = file(fn,'r')
        original_weights_dict = cP.load(FH)
        FH.close()

        cfg['original_station_weights_dict'] = original_weights_dict
        
    except:
        print 'no original weights found, set up new file with dictionary containing current weights ("station_weights.txt" in temp-folder)'
    
        fn =  path.abspath(path.realpath(path.join(cfg['temporary_directory'],'station_weights.txt')))
        FH = file(fn,'w')
        cP.dump(station_weights_dict,FH)
        FH.close()

        cfg['original_station_weights_dict'] = station_weights_dict

        

    cfg['station_weights_dict'] = station_weights_dict
    #     print original_weights_dict
    print 'gewichte gesetzt zu:\n',station_weights_dict
    #    exit()
    
    return station_weights_dict

 
#---------------------------------------------------------------------
def estimate_amplitude(distance, depth, cfg):
    """
    Factor for the decrease of amplitude, based on inherent damping. Dependent on distance (source, receiver) and approximate depth of the source.

    Coefficents have to be calculated/estimated for the respective model and must be provided in the config file as 'q_distance' and  'q_depth'. If not existent in the file, both equal '1'.

    input:
    -- distance
    -- depth
    -- config-dictionary

    output:
    -- amplitude damping factor (w.r.t '1')
    """

    pass

    Qfactor_distance = float(cfg.get('q_distance',1))
    Qfactor_depth    = float(cfg.get('q_depth',1))
        
    amplitude_factor  = 1 

    
    return amplitude_factor


#---------------------------------------------------------------------

def correct_inv_A_4_missing_traces_and_weights(cfg):
    """
    If not all receivers ( = station-channel-combinations) contain data, the respective entries in the correlation matrix A have to be deleted by subtraction of their contribution to the sum.

    The same holds for changed weights of the stations participating in the inversion !!!!



    input:
    -- config dictionary

    indirect output:
    -- new entry
    -- binary pickle file in 'temp' directory, if A must be re-read later
    -- ascii file with list of current receivers (if in the next call, the same receiver configuration occurs, A is simply re-read)

    direct output:
    -- control parameter '1'
    """


    lo_recs_w_data        = cfg['list_of_receivers_with_data']
    lo_recs               = cfg['list_of_all_receivers']

    set_of_recs_wo_data   = set(lo_recs)-set(lo_recs_w_data)


    fn_lor                = 'current_list_of_receivers_w_data' 
    fn_lists_of_recs      = path.realpath(path.abspath(path.join(cfg['temporary_directory'],fn_lor)))

    fn_inv_A_red          = 'current_invA_reduced' 
    fn_inv_A_reduced      = path.realpath(path.abspath(path.join(cfg['temporary_directory'],fn_inv_A_red)))

    fn_sta_wghts_tmp      = 'station_weights_temp' 
    fn_station_weights_tmp= path.realpath(path.abspath(path.join(cfg['temporary_directory'],fn_sta_wghts_tmp)))

    remove_traces = 0

    #if the same reduced inv_A is still in memory:
    #
    if cfg.has_key('reduced_inv_A') and cfg.has_key('set_of_recs_wo_data'):
        set_of_recs_wo_data_old = set(cfg['set_of_recs_wo_data'])
        if set_of_recs_wo_data == set_of_recs_wo_data_old:
            new_inv_A               = cfg['reduced_inv_A']
            new_set_of_recs_wo_data = set_of_recs_wo_data_old
            print 'reduced inv_A already available in config dictionary\n'

            return 1


    same_traces_w_data = 1
    same_weights       = 1


    # read file with list of receivers which had contained data in former run
    try:
#         if cfg.has_key('old_so_recs_w_data'):
#             if cfg['old_so_recs_w_data'] == set(lo_recs_w_data):
#                 same_traces_w_data = 1
#         else:
#             same_traces_w_data = 0
#             cfg['old_so_recs_w_data'] = 
        #not possible, if grid has changed
        raise
        # FH = file(fn_lists_of_recs,'r')
#         old_lo_recs_w_data = cP.load(FH)[1]
#         FH.close()
#         print set(old_lo_recs_w_data )
#         print set(lo_recs_w_data)
#         if not ( set(old_lo_recs_w_data ) == set(lo_recs_w_data) ):
        #    same_traces_w_data = 0 
    except:
        same_traces_w_data = 0 

    
    
    # compare dictionaries with weights of stations of the run before with current weights
    weights_dict          = cfg.get('station_weights_dict',0)
    if not weights_dict:
        print 'ERROR - No weights in config dictionary!!\n'
        raise SystemExit


    try:
        FH = file(fn_station_weights_tmp,'r')
        old_weights_dict  = cP.load(FH)
        FH.close()
        for rec_w_data in lo_recs_w_data:
            station = rec_w_data.split('.')[0]
            if old_weights_dict[station] != weights_dict[station]:
                same_weights *= 0                  
    except:
        same_weights = 0

    # write current weights to file for faster corrections in next run (if nothing changes)
    #
    if same_weights == 0:
        FH = file(fn_station_weights_tmp,'w')
        cP.dump(weights_dict,FH)
        FH.close()

    # if the same reduced inv_A is stored as file try to read from this file:
    #
    if same_traces_w_data * same_weights == 1:
        try:
            FH = file(fn_inv_A_reduced,'r')
            reduced_inv_A = cP.load(FH)
            FH.close()
            cfg['reduced_inv_A'] = reduced_inv_A
            cfg['inv_A'] = reduced_inv_A
            print 'nothing changed from last run - reduced inv_A read from file\n'            
            return 1 

        except:
            print 'nothing changed, but cannot read file for existing inv_A\n setting up new one \n'
            pass
        
    #otherwise setting up a new reduced inv A
    #
    #first find weights from the setup of A and inv_A
    try:
        original_weights_dict = cfg.get('original_station_weights_dict',0)
    except:
        fn =  path.abspath(path.realpath(path.join(cfg['temporary_directory'],'station_weights.txt')))
        FH = file(fn,'r')
        original_weights_dict = cP.load(FH)
        FH.close()
        
    if original_weights_dict == 0:
        print 'ERROR - Entry "original_station_weights_dict" in config dict wrong \n'
        raise SystemExit

             
    if not set(lo_recs_w_data) == set(lo_recs):
        print 'remove entries for receivers ...\n',sort(list(set_of_recs_wo_data)),'\n\n'


    lo_recs_weights_to_change = []
    for rec_w_data in sort(lo_recs_w_data):
        station = rec_w_data.split('.')[0]
        if original_weights_dict[station] != weights_dict[station]:
            lo_recs_weights_to_change.append(rec_w_data)
    if lo_recs_weights_to_change:
        print '\nweights of following receivers have changed: \n',lo_recs_weights_to_change,'\n'        


    lo_all_gridpoints        = cfg['list_of_all_gridpoints']
    if cfg.has_key('list_of_gridpoint_section'):
        current_lo_gridpoints    = cfg['list_of_gridpoint_section']
    else:
        current_lo_gridpoints    = lo_all_gridpoints

    n_gridpoints             = len(current_lo_gridpoints)

    original_A               = cfg['A']
    new_inv_A                = zeros((6,6,n_gridpoints),float)

    channel_index_dict        = cfg['channel_index_dictionary']
    stat_idx_dict             = cfg['station_index_dictionary']
    stations_array            = cfg['station_coordinates']
    time_axis                 = cfg['time_axis_of_GF']


    #    print 'A[122] - Summe  ', sum(original_A[:,:,121]),'\n'
    
    if len(lo_recs_weights_to_change) == 0 and len(set_of_recs_wo_data) == 0:
        if cfg.has_key('inv_A'):
            return 1

    for gp in arange(n_gridpoints):

        gp_abs         =   current_lo_gridpoints[gp]
        temp_A         =   original_A[:,:,gp].copy()
        
        temp_red_mat   =   zeros((6,6))
    
        for rec in lo_recs :
            stationname   = rec.split('.')[0]
            channel       = rec.split('.')[1]
            station_idx   = int(stat_idx_dict[stationname])-1
            station_weight= original_weights_dict[stationname]
            chan_idx      = int(channel_index_dict[channel])


            if rec in lo_recs_weights_to_change or rec in set_of_recs_wo_data:
                if rec in set_of_recs_wo_data:
                    temp_red_mat += calc_corr_mat(gp,station_idx,chan_idx,station_weight, time_axis, cfg)
                    print_string  = 'inv_A  for gridpoint %4i updated - removed receiver  %.9s with weight %.2f \r \t' %(gp_abs,rec,station_weight )
                else:
                    station_weight_new =  weights_dict[stationname]
                    station_weight_red =  station_weight - station_weight_new
                    
                    temp_red_mat += calc_corr_mat(gp,station_idx,chan_idx,station_weight_red, time_axis, cfg)
                    print_string  = 'inv_A  for gridpoint %4i updated - \t \t receiver  %.9s  \r \t' %(gp_abs,rec )
                sys.stdout.write(print_string)
            

        reduced_A     = temp_A -  temp_red_mat
            

        
        
        a_tmp2         =  matrix(reduced_A)
        a_tmp3         =  inv(a_tmp2)
        
        new_inv_A[:,:,gp] = a_tmp3

    print '\n'


    new_set_of_recs_wo_data    = set_of_recs_wo_data

   #  #save list configurations and reduced inv_A in file
#     FH_rec_w_data         = file(fn_lists_of_recs,'w')
#     lists_recs            = [lo_recs,lo_recs_w_data]
#     cP.dump(lists_recs,FH_rec_w_data)
#     FH_rec_w_data.close()


    # FH_inv_A_out = file(fn_inv_A_reduced,'w')
#     cP.dump(new_inv_A,FH_inv_A_out)
#     FH_inv_A_out.close()
#     print 'inverse correlation matrix inv_A for reduced number of traces with data is written to file'            
     
    #cfg['set_of_recs_wo_data']  = new_set_of_recs_wo_data

    
    cfg['inv_A']        = new_inv_A

    #print new_inv_A[:,:,0],'\n'
    #     print new_inv_A[:,:,-1],'\n'

    
    return 1
#----------------------------------------------------------------------
def d_break(locs,*message):
    """
    Break for debugging. All local variables available, if called as d_break( locals() ) with optional message.
    """
    
    if _debug:
        print '\n\n'
        print message
        print '\n'
        doc=""" IPython shell for debugging -- all variables available \n"""
        try:
            import IPython
            IPython.Shell.IPShell(user_ns=locs).mainloop(sys_exit=1, banner=doc)
        except:
            import code
            code.interact(local = locs)
    else:
        pass
#----------------------------------------------------------------------
def rescale_M_opt(cfg):
    """
    Re-scaling of best moment tensor M_opt, optimising the total residuum of synthetics and data.
    
    For all data traces, used in the inversion, the respective
    synthetic is calculated by using the moment tensor solution
    M_opt. The latter is being re-scaled to minimise the total misfit
    of all data-synthetics comparisons. Re-scaling is been done by a
    least squares algorithm.
     
    input:
    -- config-dictionary
       
    direct output:
    -- control parameter
       
       indirect output:
    -- config-dictionary entries
    [re-scaled 'M_opt_scaled', 'M_opt_scaling_factor']        
    """

    if _debug:
        print 'start rescaling ofo M by least squares optimisation'

    M                    = cfg['M_array']
    idx_s                = int(cfg['optimal_source_index'] )
    gf                   = cfg['GF']
    best_time_window_idx = int(cfg['optimal_time_window_index'])
    data_array_total     = cfg['data_array']
    best_data_start_idx  = cfg['list_of_moving_window_startidx'][best_time_window_idx]
    #data                 = data_array_total[best_time_window_idx,:,:]
    data                 = data_array_total[:,best_data_start_idx:best_data_start_idx + cfg['data_trace_samples']].copy()
    #print shape(data)
    #exit()

    m         = M
    m_opt     = m[idx_s,:]
     
    lo_CR  = cfg['list_of_receivers_with_data']
    n_traces = len(lo_CR)

    gf_sampling_in_Hz   =   float(cfg['gf_sampling_rate'])
    data_sampling_in_Hz =   float(cfg['data_sampling_rate'])   
    data_sampling_in_s  =   1./data_sampling_in_Hz
     
    dt_s                =   data_sampling_in_s

    if not  gf_sampling_in_Hz == data_sampling_in_Hz:
        print 'Warning!! Sampling of data (%g Hz) and Greens functions (%g Hz) do not coincide'%(data_sampling_in_Hz, gf_sampling_in_Hz)
        print 'Taking sampling of data as true value'
    dt = dt_s

    #build synthetics for best moment tensor
    #set time axis
    
    #todo:check this:
    time_axis        = arange(int( float(cfg['time_window_length']) * gf_sampling_in_Hz ) +1 ) /float(gf_sampling_in_Hz)        
    plottraces_data  = zeros((n_traces,3,len(time_axis)))
    integrals_synth_data   = zeros((n_traces))
    integrals_synth_synth  = zeros((n_traces))
    maxdata_by_max_synth   = zeros((n_traces))

    data_synth_array = zeros(( n_traces, 2,cfg['data_trace_samples']) )    
    
    for idx_rec in arange(n_traces):
        
        current_receiver  = lo_CR[idx_rec]
        current_station   = current_receiver.split('.')[0]
        current_channel        = current_receiver.split('.')[1]
        #set indizes
        station_index_dict     = cfg['station_index_dictionary']
        receiver_index_dict    = cfg['receiver_index_dictionary']
        channel_index_dict     = cfg['channel_index_dictionary']
        receiver_index         =  int(receiver_index_dict[current_receiver])
        station_index          =  int(station_index_dict[current_station])-1
        channel_index          =  int(channel_index_dict[current_channel[-1]])

        n_time         = int( float(cfg['time_window_length']) * data_sampling_in_Hz ) + 1
        tmp_gf         = zeros((6, n_time),float)
        tmp_synth      = zeros((n_time),float)
        gp_index       = idx_s

        for m_component in arange(6):
            tmp_gf[m_component,:]         =  gf[station_index, gp_index, channel_index, m_component, :]
        for t in xrange(n_time):
            tmp_synth[t]                  = dot( tmp_gf[:,t], m_opt )
         
        current_synth_data       = tmp_synth 
        current_real_data        = data[receiver_index,:]

        #print shape(data_synth_array)
        #print len(current_real_data),len(current_synth_data)
        #exit()
        data_synth_array[idx_rec,0,:] = current_real_data
        data_synth_array[idx_rec,1,:] = current_synth_data
         
         
        if _debug:
            print 'set up synthetic and real data for receiver %s' %(current_receiver)         
        
        integrand_1 = current_synth_data *current_synth_data
        integrals_synth_data[idx_rec]  = simps(integrand_1, dx=dt)
        integrand_2 = current_synth_data * current_real_data
        integrals_synth_synth[idx_rec] = simps(integrand_2, dx=dt)

        maxdata_by_max_synth[idx_rec] = max(abs(current_real_data))/max(abs(current_synth_data))

    uppersum = sum(integrals_synth_data[:])
    lowersum = sum(integrals_synth_synth[:])
    
    psi  = uppersum/lowersum

    #DEBUG  --  TEST !!!!:
    #psi = mean(maxdata_by_max_synth[:])

    if _debug:
        print 'calculated scaling for best moment tensor M: \n\n %g \n'%(psi)      

    #TODO check if necessary
    M_scaled = M * psi

    cfg['M_scaled']            = M_scaled
    cfg['M_scaling_factor']    = psi
    #cfg['optimal_M']           = psi*cfg['optimal_M']

    
    #d_break(locals(),'data synth vergleich - array: data_synth_array  ')

    #calculate new residuum array
    # if _debug:
#         print 'calculate new residuals with optimised M'
#     if not  estimate_res(data, M_scaled, cfg):
#         exit( 'ERROR! could not estimate updated RES!')
#     if _debug:
#         print 'calculate new VR for optimised M'      
#     if not estimate_vr(data, cfg):
#         exit( 'ERROR! could not estimate VR!')
     

     
    # control parameter
    return 1   

#----------------------------------------------------------------------
