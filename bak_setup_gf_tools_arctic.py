#!/usr/bin/env python
#coding=utf-8


import os
import sys
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
#import pickle as pp
from scipy.integrate import simps as simps
from subprocess import *

import  Scientific.IO.NetCDF as sioncdf
import time
import calendar as cldr

    
import pymseed
#----------------------------------------------------------------------




#TODO: source indizes bei 1 anfangen lassen !!!!



#----------------------------------------------------------------------

pi  = math.pi

#Earth's radius in metres
R           = 6371000
#Conversion of degrees to/from radians
rad_to_deg  = 180./pi

#accuracy factor - needed in comparison of time axes
accuracy = 0.001
#----------------------------------------------------------------------
_debug = 0

import code
import pdb

console = code.InteractiveConsole()

#----------------------------------------------------------------------

def read_in_config_file(filename):
    """Reads given config-file for setup GF-DB.

    input:
    -- filename (absolute or relative))

    output:
    -- configuration-dictionary

    """

    print 'reading config_file...'

    configfile    = path.abspath(path.realpath(filename))

    #set up configuration-dictionary
    config        = ConfigParser.ConfigParser()

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
            if _debug:
                print '%s \t\t = \t %s'%(str(oi),str(oi_v))

    #if (1000%float(cfg['gf_in_dist_step']) !=0):
    #    print 'inadequate step-size...must be integer part of 1000'
    #    raise SystemExit
        
    #working paths:
    base_dir     = path.abspath(path.realpath(cfg['base_dir']))
    gf_dir       = path.abspath(path.realpath(path.join(cfg['base_dir'],'DB','GF')))
    data_dir     = path.abspath(path.realpath(path.join(base_dir, cfg['data_dir'])))

    #data input location:
    data_dir_in  = path.abspath(path.realpath(path.join(cfg['base_dir'],cfg['data_dir'])))

    # Green's functions input location
    gfdb_base_dir = path.abspath(path.realpath(cfg['gf_db_base_dir'] ))
    gf_dir_input  = path.abspath(path.realpath(path.join(gfdb_base_dir,cfg['gf_dir_input'] )))

    # directory for temporary files:
    temp_dir     = path.abspath(path.realpath(path.join(cfg['base_dir'], 'temp' ) ))
    # directory for temporary collecting suited GFs
    temp_gf_dir  = path.abspath(path.realpath(path.join(temp_dir,'gf_collected')))

    if (not path.exists( base_dir ) ):
        os.makedirs(base_dir)
    if (not path.exists(gf_dir)):
        os.makedirs(gf_dir)
    if (not path.exists(data_dir) ):
        os.makedirs(data_dir)
    if (not path.exists(temp_dir)):
        os.makedirs(temp_dir)            
    if (not path.exists(temp_gf_dir)):
        os.makedirs(temp_gf_dir)            


    cfg['base_dir']            =  base_dir
    cfg['gf_dir']              =  gf_dir
    cfg['gf_dir_input']        =  gf_dir_input
    cfg['gfdb_base_dir']       =  gfdb_base_dir
    cfg['data_dir']            =  data_dir
    cfg['data_dir_in']         =  data_dir_in
    cfg['temporary_directory'] =  temp_dir
    cfg['temp_dir']            =  temp_dir
    cfg['temp_gf_dir']         =  temp_gf_dir

    return cfg

#----------------------------------------------------------------------

def set_GF_parameters(cfg):
    
    print 'setting parameters...'
    
    # event- and window- times:
    eventtime =  read_datetime_to_epoch(cfg['event_id'])
    cfg['event_datetime_in_epoch'] = eventtime
    current_year = time.gmtime(eventtime)[0]
    cfg['current_year'] = current_year
    current_day = time.gmtime(eventtime)[7]
    cfg['current_day'] = current_day

    parent_data_directory = path.realpath(path.abspath( path.join( str(cfg['data_dir']))))

#     if (not path.exists(parent_data_directory)):
#         print 'could not find data directory %s'%(parent_data_directory)
#         os.makedirs(parent_data_directory)
#         print 'dummy directory artificially set up'
    cfg['parent_data_directory'] = parent_data_directory
    


        
    if cfg.has_key('network_names'):
        list_of_all_networks   = cfg['network_names']
    else:
        cfg['network_names']   = ''
        list_of_all_networks   = cfg['network_names']
    nw_list_raw = list_of_all_networks.split(',')
    list_of_all_nw = []
    for ii in nw_list_raw:
        dummy_element = ii.strip()
        dummy_element = dummy_element.upper()
        list_of_all_nw.append(dummy_element)
    cfg['list_of_networks'] = list_of_all_nw
    list_of_all_networks = cfg['network_names']

    if cfg.has_key('location_names'):
        list_of_all_locations   = cfg['location_names']
    else:
        cfg['location_names'] = ''
        list_of_all_locations   = cfg['location_names']
    loc_list_raw = list_of_all_locations.split(',')
    list_of_all_loc = []
    for ii in loc_list_raw:
        dummy_element = ii.strip()
        dummy_element = dummy_element.upper()
        list_of_all_loc.append(dummy_element)
    cfg['list_of_locations'] = list_of_all_loc

    if cfg.has_key('list_of_all_channels'):
        list_of_all_channels_raw_string  = cfg['list_of_all_channels']
    else:
        cfg['list_of_all_channels'] = 'BHN,BHE,BHZ'
        list_of_all_channels_raw_string = cfg['list_of_all_channels']
    channel_list_raw = list_of_all_channels_raw_string.split(',')
    list_of_all_channels = []
    for ii in channel_list_raw:
        dummy_element = ii.strip()
        dummy_element = dummy_element.upper()
        list_of_all_channels.append(dummy_element)
    cfg['list_of_channels'] = list_of_all_channels

    # building config-dict entry 'channel_index_dictionary'
    if not make_dict_channels_indices(cfg):
        print 'ERROR'
        exit()

    # building config-dict entry 'grid_coordinates'
    if not set_sourcepoint_configuration(cfg):
        print 'ERROR in setting of sourcepoints'
        exit()
    
    #building config-dict entries 'station_coordinate_dictionary'(dict), 'station_index_dictionary'(dict),'list_of_stations'(list),'station_coordinates'(array)
    if not set_station_configuration(cfg):
        print 'ERROR in setting of stations '
        exit()

     #building config-dict entries 'receiver_nslc_dictionary', 'list_of_receivers', 'receiver_index_dictionary'
    if not set_receiver_names(cfg):
        print 'ERROR in setting of receiver names'
        exit()       

    return 1

#---------------------------------------------------------------------
def read_datetime_to_epoch(datetime):

    #TODO ersetze timie durch calendar
    import calendar as cc
    if len(datetime.split('.')) == 2:
        milisecs      = float('0.'+datetime.split('.')[1])
    else:
        milisecs      = 0.            

    date          = datetime.split('.')[0]
    #print date,'\n'
    format        = '%Y-%m-%dT%H:%M:%S'
    time_tuple    = time.strptime(date,format)
    epoch_seconds = time.mktime(time_tuple) - time.altzone + milisecs

    return epoch_seconds

#---------------------------------------------------------------------

def make_dict_channels_indices(cfg):

    list_of_channels   = cfg['list_of_channels']
    channel_index_dict ={}

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
    
    cfg['channel_index_dictionary'] = channel_index_dict

    return 1   

#---------------------------------------------------------------------
def set_receiver_names(cfg):

    lo_stat  = cfg['list_of_stations']
    lo_chan  = cfg['list_of_channels']
    lo_loc   = cfg['list_of_locations']   
    lo_nw    = cfg['list_of_networks']


    r_nslc_dict     = {}
    reclist         = []
    rec_index_dict  = {}
    idx_count       = 1
    
    for idx_nw in lo_nw:
        for idx_stat in lo_stat:
            for idx_loc in lo_loc:
                for idx_chan in lo_chan:               
                    temp_dict              = {}
                    temp_dict['network']   = idx_nw
                    temp_dict['station']   = idx_stat
                    temp_dict['location']  = idx_loc
                    temp_dict['channel']   = idx_chan

                    receivername = '%s.%s.%s.%s'%(idx_nw,idx_stat,idx_loc,idx_chan)
                    rec_index_dict[receivername]   = idx_count
                    rec_index_dict[str(idx_count)] = receivername
                    reclist.append(receivername)
                    r_nslc_dict[receivername]      = temp_dict
                    idx_count += 1               
    
    
    
    cfg['receiver_nslc_dictionary']      = r_nslc_dict
    cfg['list_of_receivers']             = reclist
    cfg['receiver_index_dictionary']     = rec_index_dict
    
    return 1   

#---------------------------------------------------------------------

def set_station_configuration(cfg):

    print 'setting station coordinates...'

    station_coords_filename  = path.realpath(path.abspath( path.join(cfg['base_dir'],cfg['station_coords_file']) ))

    if int(cfg['use_station_file']):

        if path.isfile(station_coords_filename):
            print 'by reading from existing file:\n',station_coords_filename
        else:
        
            station_coords_filename_in = path.realpath(path.abspath( path.join(cfg['base_dir'],cfg['station_coords_file'])))

            if path.isfile(station_coords_filename_in):
                print 'by reading from file:\n',station_coords_filename_in
                shutil.copy(station_coords_filename_in,station_coords_filename)
            else:
                exit('station coordinate file %s not found '%(station_coords_filename_in))

        if not read_station_coordinates(cfg):
                exit('ERROR! Could not read station coordinates from file %s !\n'%(station_coords_filename))
        #else:
        #    exit( 'ERROR - station coordinate file not found !\n  Please provide file %s !\n' %(station_coords_filename))

    else:         
        print 'by building artificial station grid (spiral shaped) using specifications in config file\n'

        central_latitude_raw_string     = cfg['central_latitude']
        central_latitude_raw            = central_latitude_raw_string.split(',')
        if len(central_latitude_raw) == 3 or len(central_latitude_raw) == 1 :
            for central_latitude_raw_element in central_latitude_raw:
                dummy5 = central_latitude_raw_element.split('.')
                if not ( len(dummy5) in [1,2]):
                    print 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!'
                    exit()
                for dummy5_element in dummy5:
                    if not ( dummy5_element.isdigit() ):
                        print 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!'
                        exit()
                        
            if len(central_latitude_raw) == 3:
                central_latitude_deg = float( float( central_latitude_raw[0]) + (1./60. * ( float(central_latitude_raw[1]) + (1./60. * float(central_latitude_raw[2]) ))))
            else:
                central_latitude_deg = float(central_latitude_raw[0])
    
    
    
        central_longitude_raw_string    = cfg['central_longitude']
        central_longitude_raw           = central_longitude_raw_string.split(',')
        if len(central_longitude_raw) == 3 or len(central_longitude_raw) == 1 :
            for central_lonitude_raw_element in central_longitude_raw:
                dummy5 = central_lonitude_raw_element.split('.')
                if not ( len(dummy5) in [1,2]):
                    print 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!'
                    exit()
                for dummy5_element in dummy5:
                    if not ( dummy5_element.isdigit() ):
                        print 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!'
                        exit()
                            
            if len(central_longitude_raw) == 3:
                central_longitude_deg = float( float( central_longitude_raw[0]) + (1./60. * ( float(central_longitude_raw[1]) + (1./60. * float(central_longitude_raw[2]) ))))
            else:
                central_longitude_deg = float(central_longitude_raw[0])

    
        lat0        = central_latitude_deg
        lon0        = central_longitude_deg
        distmin     = int(cfg['stat_dist_min'])
        diststep    = int(cfg['stat_radial_dist_step'])

        station_index_dict ={}

        if cfg.has_key('list_of_all_stations'):
            list_of_stations_raw_string = cfg['list_of_all_stations']
            station_list_raw = list_of_stations_raw_string.split(',')
            list_of_stations = []
            count = 0
            for ii in station_list_raw:
                dummy_element = ii.strip()
                dummy_element = dummy_element.upper()
                list_of_stations.append(dummy_element)
                index = count
                station_coordinate_dict[dummy_element] = index
                station_coordinate_dict[str(index)] = dummy_element
                count += 1
                

        else:
            if not cfg.has_key('n_stations'):
                print "ERROR! Either provide number of stations as 'n_stations' or give 'list_of_all_stations'"
                raise SystemExit
        
            else:
                list_of_stations=[]
                number_of_stations = int(cfg['n_stations'])
                chars = 'abcdefghijklmnopqrstuvwxyz'
                for jj in xrange(number_of_stations):
                    charslist = ''.join( [ chars[jj] ] + [ rdm.choice(chars) for i in xrange(4) ])
                    station = charslist.upper()
                    list_of_stations.append(station)
                    station_index = jj
                    station_index_dict[station] = jj
                    station_index_dict[str(jj)] = station

        s_number    = len(list_of_stations)
        stat_coords = zeros((s_number,4),float)
        
        for idx_r in xrange(s_number):            
            azimuth    = (idx_r + 1) * 350./float(s_number)
            dist       = float(distmin + idx_r * diststep)
            dist_north = dist * cos(azimuth / rad_to_deg)
            dist_east  = dist * sin(azimuth / rad_to_deg)
            lat_shift  = dist_north * rad_to_deg /R
            lat_rec    = lat0 + lat_shift 
            lon_shift  = dist_east * rad_to_deg / (R * sin((90.-lat_rec)/rad_to_deg))
            lon_rec    = lon0 + lon_shift 


            stat_coords[idx_r,0] = int(idx_r)
            stat_coords[idx_r,1] = lat_rec
            stat_coords[idx_r,2] = lon_rec
            stat_coords[idx_r,3] = 0.
                
        rec_co_file = file(station_coords_filename,'w')
        stat_co_conf = ConfigParser.ConfigParser()

        station_coordinate_dict = {}
        
        for hh in xrange(s_number):
            temp_dict={}
            stat_idx = int(stat_coords[hh,0])
            stationname = station_index_dict[str(stat_idx)]
            lat = stat_coords[hh,1]
            lon = stat_coords[hh,2]
            stat_co_conf.add_section(stationname)
            stat_co_conf.set(stationname,'lat',str(lat))
            stat_co_conf.set(stationname,'lon',str(lon))
            stat_co_conf.set(stationname,'index',str(stat_idx))
            temp_dict['lat'] = lat
            temp_dict['lon'] = lon
            temp_dict['index'] = stat_idx

            station_coordinate_dict[stationname]  = temp_dict

        stat_co_conf.write(rec_co_file)
        rec_co_file.close()

        cfg['list_of_stations']              = list_of_stations
        cfg['station_coordinates']           = stat_coords
        cfg['station_coordinate_dictionary'] = station_coordinate_dict
        cfg['station_index_dictionary']      = station_index_dict

    print " station <-> coordinates dictionary set up " #in %s "%(cfg['station_coordinate_dictionary'] )

    return 1

#---------------------------------------------------------------------
    
def read_station_coordinates(cfg):

    #station_coordinate_filename  = cfg['station_coords_file']
    station_coordinate_file      = path.realpath(path.abspath(path.join(cfg['base_dir'], cfg['station_coords_file']  )))
    
    if not path.isfile(station_coordinate_file):
        exit( 'file with station coordinates not found !')
        
    
    config_filehandler = ConfigParser.ConfigParser()
    config_filehandler.read(station_coordinate_file)

    list_of_stations         = []
    station_coordinate_dict  = {}
    station_index_dict       = {}
    lo_stat_index            = []
    
    sections = config_filehandler.sections()
    for sec in sections:
        list_of_stations.append(sec)
        temp_lat_lon_dict ={}
        options = config_filehandler.options(sec)
        for opt in options:
            value = config_filehandler.get(sec,opt)
            if opt == 'lat' or opt == 'latitude' or opt == 'lon' or opt == 'longitude':
                if opt == 'latitude':
                    opt = 'lat'
                if opt == 'longitude':
                    opt = 'lon'

                try:
                    lat_lon =  value.split(',')
                except:
                    print 'ERROR!! Wrong coordinate format for ',opt, ' of station ', sec,'!!!'
                    raise SystemExit
                    
                if len(lat_lon) == 3 or len(lat_lon) == 1 :
                    for lat_lon_element in lat_lon:
                        dummy3 = lat_lon_element.split('.')
                        if not ( len(dummy3) in [1,2]):
                            print 'ERROR!! Wrong coordinate format for ',opt, ' of station ', sec,'!!! '
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

            
            if opt == 'index' or opt == 'idx':
                station_index_dict[sec] = int(value)
                station_index_dict[value] = sec
                temp_lat_lon_dict[opt]    = value

        if not len(temp_lat_lon_dict) == 3:
            print 'coordinates of station ',sec, ' are wrong - latitude and/or longitude and/or index missing  !!!!!'
            exit()
            
        station_coordinate_dict[sec] = temp_lat_lon_dict

    #filling array 'stat_coords'
    s_number    = len(list_of_stations)
    stat_coords = zeros((s_number,4),float)
    for hh in xrange(s_number):
        stationname = station_index_dict[str(hh+1)]
        temp_co_dict = station_coordinate_dict[stationname]
        lat = float(temp_co_dict['lat'])
        lon = float(temp_co_dict['lon'])
        idx = int(temp_co_dict['index'])
        stat_coords[hh,0] = int(idx)
        stat_coords[hh,1] = lat
        stat_coords[hh,2] = lon

          
    cfg['station_coordinate_dictionary']   = station_coordinate_dict
    cfg['station_index_dictionary']        = station_index_dict
    cfg['list_of_stations']                = list_of_stations
    cfg['station_coordinates']             = stat_coords
    
    #save array for checking and plotting
    #station_co_fn   = 'station_coordinates.dat'
    #station_co_file = path.abspath(path.realpath(path.join(cfg['base_dir'],station_co_fn)))
    #savetxt(station_co_file,stat_coords,fmt=['%i','%.4f','%.4f','%.2f'])
    
    print 'Station coordinates for %i stations ok!\n\n'%(s_number)

    return 1   

#---------------------------------------------------------------------

def set_sourcepoint_configuration(cfg):
    """
    Setting geographical coordinates of source locations. 

    Indexing is done by looping over all source-points in the order (N,E,Z)


    Input:
    -- Config-dictionary

    indirect Output:
    -- File with coordinate-array (lat,lon,depth) in in base directory 
    -- array with source coordinates in config-dictionary

    direct output:
    -- control parameter
    """

    print 'setting grid_coords...'

    source_coords_filename = path.realpath(path.abspath( path.join(cfg['base_dir'],cfg['source_coords_file']) ))

    northdim    = int(cfg['northdim'])
    eastdim     = int(cfg['eastdim'])
    depthdim    = int(cfg['depthdim'])
    northstep   = int(cfg['northstep'])
    eaststep    = int(cfg['eaststep'])
    depthstep   = int(cfg['depthstep'])

    N_N         = int(2 * northdim + 1)
    N_E         = int(2 * eastdim + 1)
    N_Z         = int(depthdim)
    N_tot       = int(N_N * N_E * N_Z)

    check_flag  = 1

    # if already existing and in right dimensions:
    if path.isfile(source_coords_filename):
        try:
            File7                  = file(source_coords_filename,'r')
            grid_coords_array      = loadtxt(File7, usecols=tuple(range(0,3)))
            File7.close()
        except:
            print 'cannot read existing source coordinate file - generating new one' 
            check_flag = 0

        if len(grid_coords_array) == N_tot:
            print 'by reading from file:\n',source_coords_filename
        else:
            print 'existing source coordinate file has wrong number of entries - generating new file'
            check_flag = 0

    # otherwise set up new file:
    if (not path.isfile(source_coords_filename)) or check_flag == 0 :         
        print 'by building source point grid (rectangular) using specifications in config file\n'

        central_latitude_raw_string   = cfg['central_latitude']
        central_latitude_raw          = central_latitude_raw_string.split(',')
        if len(central_latitude_raw) == 3 or len(central_latitude_raw) == 1 :
            for central_latitude_raw_element in central_latitude_raw:
                dummy5 = central_latitude_raw_element.split('.')
                if not ( len(dummy5) in [1,2]):
                    print 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!'
                    exit()
                for dummy5_element in dummy5:
                    if not ( dummy5_element.isdigit() ):
                        print 'ERROR!! Wrong coordinate format for latitude of sourcepoint grid center!!!'
                        exit()
                        
            if len(central_latitude_raw) == 3:
                central_latitude_deg = float( float( central_latitude_raw[0]) + (1./60. * ( float(central_latitude_raw[1]) + (1./60. * float(central_latitude_raw[2]) ))))
            else:
                central_latitude_deg = float(central_latitude_raw[0])
    
    
    
        central_longitude_raw_string    = cfg['central_longitude']
        central_longitude_raw           = central_longitude_raw_string.split(',')
        if len(central_longitude_raw) == 3 or len(central_longitude_raw) == 1 :
            for central_lonitude_raw_element in central_longitude_raw:
                dummy5 = central_lonitude_raw_element.split('.')
                if not ( len(dummy5) in [1,2]):
                    print 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!'
                    exit()
                for dummy5_element in dummy5:
                    if not ( dummy5_element.isdigit() ):
                        print 'ERROR!! Wrong coordinate format for longitude of sourcepoint grid center!!!'
                        exit()
                            
            if len(central_longitude_raw) == 3:
                central_longitude_deg = float( float( central_longitude_raw[0]) + (1./60. * ( float(central_longitude_raw[1]) + (1./60. * float(central_longitude_raw[2]) ))))
            else:
                central_longitude_deg = float(central_longitude_raw[0])

    
        lat0        = central_latitude_deg
        lon0        = central_longitude_deg
        depth0      = int(cfg['min_depth'])
        

        list_of_source_points = []
        
        coord_array_temp      = zeros((N_N,N_E,N_Z, 3),float64)
        grid_coords_array     = zeros((N_tot, 3),float64)

        if _debug:
            count = 1

        gp_count = 0
        for z1 in (arange(N_Z)):
            depth           = z1 * depthstep + depth0
            current_radius  = float(R) - depth

            for e2 in (arange(N_E) - eastdim):
                for n3 in (arange(N_N) - northdim):
                    #lat = lat0 + ( n3 * northstep * rad_to_deg / current_radius)

                    north_distance =  n3 * northstep
                    east_distance  =  e2 * eaststep
                    

                    #angle in degrees w.r.t. north 
                    angle    = float((90-arctan2(north_distance,east_distance)*float(rad_to_deg)))%360
                    #effective distance on regular rectangular grid
                    distance = sqrt(float(north_distance) **2 + float(east_distance)**2 ) 

                    #needed for input to trigonometric functions
                    azimuth          = float(angle)/rad_to_deg
                    angular_distance = distance/float(current_radius)

                    #calculate coordinates for current grid point
                    lat_rad  = arcsin(sin(lat0/rad_to_deg)*cos(angular_distance) + cos(lat0/rad_to_deg)*sin(angular_distance)*cos(azimuth))
                    lat      = lat_rad*rad_to_deg
                    lon_rad  = lon0/rad_to_deg + arctan2(  sin(azimuth)*sin(angular_distance)*cos(lat0/rad_to_deg), cos(angular_distance) - sin(lat0/rad_to_deg) * sin(lat/rad_to_deg)   )
                    lon      = lon_rad*rad_to_deg
                    #lon = lon0 + ( e2 * eaststep * rad_to_deg / current_radius / sin( (90-lat)/rad_to_deg ) )

                    #put coordinates into array
                    #coord_array_temp[n3+int(northdim),e2+int(eastdim),z1, 0] = lat
                    #coord_array_temp[n3+int(northdim),e2+int(eastdim),z1, 1] = lon
                    #coord_array_temp[n3+int(northdim),e2+int(eastdim),z1, 2] = depth
                    grid_coords_array[gp_count, 0] = lat
                    grid_coords_array[gp_count, 1] = lon
                    grid_coords_array[gp_count, 2] = depth
                    gp_count += 1
                    
                    if int(distance)==0:
                        print gp_count-1,lat,lon,depth

                    #if _debug:
                    #    print count,north_distance,east_distance,angle,distance,'(',lat0,lon0,')',lat,lon
                    #    count +=1
        #d_break(locals(),'grid')
        #exit()
               
        #tot = 0
        #    for dummy1 in arange(N_tot):
        #for z1 in arange(N_Z):
        #    for e2 in arange(N_E):
        #        for n3 in arange(N_N):
        #            grid_coords_array[tot, :] =  coord_array_temp[n3,e2,z1,:]
        #            source_list_entry         = [tot,n3,e2,z1]
        #            list_of_source_points.append(source_list_entry)
        #            tot += 1


        #exit()

        #File7          = file(source_coords_filename,'w')
        savetxt(source_coords_filename,grid_coords_array)
        #File7.close()

        print 'wrote grid coordinates as array to file:\n',source_coords_filename

    cfg['grid_coordinates'] = grid_coords_array

    print 'Grid of %i source points ok!\n' %(len(grid_coords_array))

    d_break(locals(),'SPs')
    
    return 1



#----------------------------------------------------------------------

def rotate_traces(data_x,data_y,angle):
    """Rotating incoming traces (x,y) around the given angle in degrees (!) into (x',y'). 
    
     Input data is given w.r.t. the basis (e_x,e_y), whereas the output data is presented w.r.t. new basis (e_x',e_y'). The rotation angle is positive from e_y' to e_y (counter clockwise) 

    Assuming orthogonal base and canonical naming: e.g. input 'x' (tranversal) ist turned to be 'east' and input 'y' (radial) to 'north' 
    
    Input:
    -- data in positive x-direction
    -- data in positive y-direction
    -- angle in degrees
    
    Output: 
    -- x'-data , y'-data
    
    """
    
    from pylab import plot
    rad_to_deg  = 180./pi
    rad_angle   = angle/rad_to_deg 
    cphi        = math.cos(rad_angle)
    sphi        = math.sin(rad_angle)
              
    if ( len(data_x) != len(data_y) ):
        print 'ERROR! Rotation not possible!  Length of first trace : %i samples -- Length of second trace : %i samples'%(len(data_x),len(data_y))
        print 'ERROR! Invoke function only with data traces of equal length!'
        raise SystemExit

    new_x, new_y  =  (data_x * cphi + data_y * sphi) ,  - data_x * sphi + data_y * cphi
    
    return new_x, new_y



#----------------------------------------------------------------------


def distance_azi_backazi(coord_s,coord_r):
    """
    Calculating distance between two geographical points on the earth's surface (given by coordinate pairs [latitude,longitude]), the azimuth from first to second point (clockwise) and the back_azimuth from second to first point (clockwise). Angles are given in degrees w.r.t north.
    
    Input:
    -- array with source coordinates in (lat,lon [,depth])
    -- array with receiver coordinates in (lat,lon [,depth])

    Output:
    -- distance between surface projections of the source and receiver (in m), azimuth, back_azimuth
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

    back_azimuth_r_s = calc_backazi(lat1,lon1,lat2,lon2,ddd)
    azimuth_s_r      = calc_backazi(lat2,lon2,lat1,lon1,ddd)

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
    

    return  distance_in_m, azimuth_s_r, back_azimuth_r_s

#----------------------------------------------------------------------

# def calc_distance(coord1,coord2,cfg):
#     """ 
#     Calculates the surface-distance between two points on earth.

#     Input:
#     -- array with coordinates of point 1 in (lat,lon)
#     -- array with coordinates of point 2 in (lat,lon)

#     Output:
#     -- Distance on earth-surface between the the points (in m)
#     """


#     lat1  =  coord1[0]/rad_to_deg
#     lon1  =  coord1[1]/rad_to_deg
#     lat2  =  coord2[0]/rad_to_deg
#     lon2  =  coord2[1]/rad_to_deg

#     distance = R * arccos( sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1) )

#     return distance

#----------------------------------------------------------------------

def neighbouring_distances(distance,dist_step):
    """ 
    Calculating the two best fitting distances for setting the respective interval, containing the actual distance.   


    Input:
    -- real distance in metres
    -- step size of distance-grid/mesh in metres

    Output:
    -- array (lower_corner_distance,upper_corner_distance) in m
    -- array (weighting factor for lower corner, weighting factor for upper corner) 
    """


    #calculate the div and modulo values    
    residual  =  distance%dist_step
    lower     =  int(distance/dist_step)*dist_step
    upper     =  lower + dist_step

    w_lower   = float(dist_step-residual)/float(dist_step)
    w_upper   = 1 - w_lower 
    
    distances = array([lower,upper])
    weights   = array([w_lower,w_upper])
    #     print distance,dist_step,anz, steps,km,m ,best_idx,upper_idx,lower_idx,dist_lower,dist_upper, weights
    #     exit()
    return distances, weights

#----------------------------------------------------------------------
#----------------------------------------------------------------------
def setup_db_white_noise(cfg):
    
    source_coords           = cfg['grid_coordinates']
    station_coords          = cfg['station_coordinates']   
    station_coordinate_dict = cfg['station_coordinate_dictionary']
    station_index_dict      = cfg['station_index_dictionary']
    receiver_index_dict     = cfg['receiver_index_dictionary']
     
    
    lo_gridpoints     = list(arange(len(source_coords)))
    lo_stat           = cfg['list_of_stations']
    lo_nw             = cfg['list_of_networks']
    lo_loc            = cfg['list_of_locations']
    lo_chan           = cfg['list_of_channels']
    lo_rec            = cfg['list_of_receivers']


    n_stations   = len(lo_stat)
    n_source     = len(lo_gridpoints)

    gf_dir_out = path.realpath(path.abspath(cfg['gf_dir']))
                        
    length      = int(float(cfg['window_length_in_s']))
    gf_sampling = float(cfg['gf_sampling_rate'])
    gf_version  = int(cfg['gf_version'])

    gf_sampling_in_s = 1./gf_sampling
        
    n_time      = int(float(length) * gf_sampling) + 1
    gf          = zeros((n_stations, n_source,  3,6, n_time),float32)
    

    for station in lo_stat:
        outfilename = path.realpath(path.abspath(path.join( gf_dir_out, 'gf_v%(ver)i_length%(twl)i_sampling%(samp)i_station_%(stat)s.nc' %{'ver':gf_version,'twl':length,'samp':gf_sampling,'stat':station })))

        idx_stat = int(float(station_index_dict[station]))
        t_idx = 0                   
        time_array = arange(n_time)*gf_sampling_in_s
            
        alldataout = []
            
        for idx_s in arange(n_source):
            for idx_comp in arange(3):
                for idx_m_comps in arange(6):
                    dummy_data = numpy.random.randn(n_time)*10.
                    alldataout.append(dummy_data)

        print 'writing to file:  ',outfilename,'\n'
        
        outfile          = sioncdf.NetCDFFile(outfilename, 'w')#, 'Created ' + time.ctime(time.time())) 
        outfile.title    = "GF for station " + station
        outfile.version  = int(cfg['gf_version'])

        outfile.createDimension('GF_idx', len(alldataout)+1)
        outfile.createDimension('t', n_time)

        GF_for_act_rec   = outfile.createVariable('GF_for_cur_stat', 'f', ('t', 'GF_idx'))

        temp = zeros((len(time_array),len(alldataout)+1), dtype=float32)
    
        temp[:,0] = time_array.astype('float32')
        temp[:,1:]  = array(alldataout, dtype=float32).transpose()
            
        GF_for_act_rec[:] = temp
        #GF_for_act_rec[:,0]  = time_array.astype('float32')

 #       for ll in arange(len(alldataout)):
 #           print "xxx"
  #          GF_for_act_rec[:,ll+1]  = array(alldataout)[ll,:].astype('float32')
        
        GF_for_act_rec.units = "'seconds' and 'arbitrary'"
        
        outfile.close()
        
        print 'gf for station  '+ station + ' ...  ok !!\n\n'

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def setup_db_qseis(cfg):
    """
        Building a Green's function database, stored in NetCDF format.

    TODO :blabla
    
    Input:
    -- config-dictionary
       
    indirect output
    -- ...   
        
    """

    #code.interact()
    
    #arrays:
    source_coords           = cfg['grid_coordinates']
    station_coords          = cfg['station_coordinates']   

    #dictionaries:
    station_coordinate_dict = cfg['station_coordinate_dictionary']
    station_index_dict      = cfg['station_index_dictionary']
    receiver_index_dict     = cfg['receiver_index_dictionary']
     
    #lists:
    lo_gridpoints     = list(arange(len(source_coords)))
    lo_stat           = cfg['list_of_stations']
    
    lo_nw             = cfg['list_of_networks']
    lo_loc            = cfg['list_of_locations']
    lo_chan           = cfg['list_of_channels']
    lo_rec            = cfg['list_of_receivers']

    n_stations  = len(lo_stat)
    n_source    = len(lo_gridpoints)

    gf_dir_in   = path.realpath(path.abspath(cfg['gf_dir_input']))    
    gf_dir_out  = path.realpath(path.abspath(cfg['gf_dir']))    
    temp_gf_dir = path.abspath(path.realpath(cfg['temp_gf_dir']))
    db_name    =  cfg['gf_db_chunks_db_name']
    gf_db_chunks_modelname = cfg['gf_db_chunks_modelname']
    
    length      = int(float(cfg['window_length_in_s']))
    gf_sampling = float(cfg['gf_sampling_rate'])
    gf_version  = int(cfg['gf_version'])
    version     = int(cfg['gf_version'] )

    gf_sampling_in_s = 1./gf_sampling
        
    n_time      = int(float(length) * gf_sampling) + 1

    if _debug:
        print 'size of GF: %i stations, %i sources, 3,6, %i time samples\n'%(n_stations,n_source,n_time)
    #gf          = zeros((n_stations, n_source,  3,6,n_time),float32)

    time_array  = arange(n_time)*gf_sampling_in_s

    dist_step   = int(cfg['gf_in_dist_step'])

    # order of coefficents for the 8 components of QSEIS outcome to 18 components needed
    order       = [1,1,3,1,2,2,4,4,1,4,5,5,6,6,8,6,7,7]
    
    stationcounter = 0
    plotdata    = zeros((len(lo_stat),len(time_array) ))
    
    for station in lo_stat:
    
        factor_array = zeros((n_source,18))


        # build GF NetCDF file for current station in 'base_dir'/DB/GF/
        outfilename = path.realpath(path.abspath(path.join( gf_dir_out, 'gf_v%(ver)i_length%(twl)i_sampling%(samp)i_station_%(stat)s.nc'%{'ver':gf_version,'twl':length,'samp':gf_sampling,'stat':station })))


        station_index = int(float(station_index_dict[station]))
        temp_dict     = station_coordinate_dict[station]
        
        station_coordinates = [float(temp_dict['lat']),float(temp_dict['lon'])]

        if _debug:
            idx2 = float(temp_dict['index'])
            print 'check station indizes: %i - %i '%(idx2,station_index)
            
        stationcounter += 1
        t_idx = 0
        files      = []
        count      = 0   

        # no need for double calculations:
        if path.exists(outfilename):
            print outfilename, 'already existing - jump to next station'
            continue

        #for each station every source point needs to be analysed:
        for source in lo_gridpoints:

            #source point coordintaes list with 2 entries [latitude,longitude]:
            source_coordinates = [source_coords[source,0],source_coords[source,1],source_coords[source,2]]

            #depth of source point in metres:
            depth              = int(source_coordinates[2])

            #calculate distance and azimuth between source and station:
            distance, azimuth,  back_azimuth  = distance_azi_backazi(source_coordinates,station_coordinates)

            # azimuth needed in radians:
            phi                = azimuth  / rad_to_deg

            # geometrical weighting factors for the 18 components 
            factors            = [cos(phi)**2,sin(phi)**2,1.,sin(2*phi),cos(phi),sin(phi),-0.5*sin(2*phi),0.5*sin(2*phi),0.,cos(2*phi),-sin(phi),cos(phi),cos(phi)**2,sin(phi)**2,1.,sin(2*phi),cos(phi),sin(phi)]

            factor_array[source,:] = array(factors)
            
            #get nearest fitting GF from QSEIS database, depending on the distance steps in the QSEIS configuration.
            #TODO optimisation: weighted mean of two nearest functions, not only nearest function
            distances,weights            = neighbouring_distances(distance,dist_step)
            nearest_neighbour_distance   = int(distances[weights.argmax()])
            
                
            if nearest_neighbour_distance < int(cfg['nearest_distance_m']):
                nearest_neighbour_distance = int(cfg['nearest_distance_m'])
            if nearest_neighbour_distance > int(cfg['farthest_distance_m']):
                nearest_neighbour_distance = int(cfg['farthest_distance_m'])

            #loop over 8 QSEIS seismograms, indexed 1-8
            for gf_idx in arange(8):
                    
                #build filename for read out respective GF
                # copy file from somewhere (gf_dir_in) into 'base_dir'/temp/temporary_gf/  and append new filename to list of files to handle
                temp_filename_in  = 'gf%(idx1)i.dep.%(idx2)i.dist.%(idx3)i' %{'idx1':gf_idx+1,'idx2':depth,'idx3':nearest_neighbour_distance}
                temp_file_in      = path.abspath(path.realpath(path.join(gf_dir_in,temp_filename_in)))

                # if file with this name is existing in the db, copy it to local GF database in 'base_dir':
                if path.isfile(temp_file_in):
                    shutil.copy(temp_file_in, temp_gf_dir)
                else:
                    if int(float(cfg['gf_db_chunks_flag'])) == 0:
                        print 'File not found : %s'%(temp_file_in)
                        raise SystemExit
                    
                temp_filename = path.abspath(path.realpath(path.join(temp_gf_dir,temp_filename_in)))
                
                files.append(temp_filename)

                # if gf are not ascii but in db-chunks, the latter file has to be created by extraction from the packed db-chunk:
                if int(float(cfg['gf_db_chunks_flag'])) == 1  :
                    #build temporary dummy file for unpacking gfdb chunk

                    dummy7   = path.abspath(path.realpath(path.join(cfg['temporary_directory'],'readout_chunks_data.tmp')))
                    
                    gf_chunks_base_dir = path.abspath(path.realpath(cfg['gf_db_base_dir']))
                    FH = file(dummy7,'w')
                    current_string = "%i   %i   %i   '%s'  \n"%(nearest_neighbour_distance,depth,gf_idx+1,temp_filename)
                    FH.write(current_string)
                    FH.close()
                    basepath = path.realpath(path.abspath(path.join(gf_chunks_base_dir,gf_db_chunks_modelname)))
                    db_in = basepath+'/'+db_name
                    FH_in = file(dummy7,'r')

                    input_cmd  = 'gfdb_extract'
                    input_arg  = db_in
                    s_out      = PIPE
                    s_in       = PIPE

                    subprcs    = Popen([input_cmd, input_arg], stdout = s_out, stdin = s_in)

                    run = subprcs.communicate( FH_in.read() )
    
                    FH_in.close()
                    
                    if run[1]:
                        print 'ERROR on stderr by reading out gfdb chunks'
                        exit(run[1])
                    
                count             += 1
            
                if _debug:
                    #show every single line 
                    info_string =  'reading station %.5s  -  source %4i (depth %5i m  -  distance %7.2f m  - nearest avail. dist.: %5i m ) \n'%(station,source+1,int(depth), distance, int(nearest_neighbour_distance) )
                    
                else:
                    #no newline - just carriage return
                    info_string =  'reading station %.5s  -  source %4i (depth %5i m  -  distance %7.2f m  - nearest avail. dist.: %5i m ) \r'%(station,source+1,int(depth), distance,int(nearest_neighbour_distance) )

                sys.stdout.write(info_string)

        #for current station, now having a file list 'files', in which each a two column array is provided, meaning the respective Greens functions. The list is in order of the source points within the array 'source_coords'
        #Now looping over the list and get (huge) GF file by concatenating the arrays in right order

        #rudimente:
        #time1      = file(files[0],'r').readlines()
        #time_list  = [x.split()[0] for x in time1]

        print '\n setup output array ...'

        alldatain  =[]
        alldataout =[]
        gf_idx     = 0
        source_idx = 0
        

        for filename in files:

            #take the weighting factors for the given source point
            factors = list(factor_array[source_idx,:])
            if _debug:
                print 'reading file no. %3i of %i (source %3i, gf-index %i )'%(source_idx*8+(gf_idx+1),len(files),source_idx+1,gf_idx+1 )

            readfile_raw   = loadtxt(filename)    
                                                          #readfile_raw  = file(filename,'r').readlines()
            data2          = readfile_raw[:,1]            #[float(x.split()[1]) for x in readfile_raw]
            time1          = list(readfile_raw[:,0])      #[float(x.split()[0]) for x in readfile_raw]

            #check to get consistent time series starting all at t0 = 0:
            if _debug:
                print 'check, if trace begins at t=0'
            if min(abs(0-array(time1))) < accuracy*gf_sampling:
                startarg = abs(0-array(time1)).argmin()
            else:
                print 'no beginning of trace found (no t=0 available), station %s not usable !!!'%(station) 
                continue
            try:
                data3 = data2[startarg:startarg+n_time]
            except:
                data3              = zeros((n_time))
                data3[:len(data2)] = data2
            
            #            for ii in time_array:
            #                if min(abs(ii-array(time1))) < accuracy*gf_sampling:
            #                    arg = abs(ii-array(time1)).argmin()
            #                    data3.append(data2[arg])
            #                else:
            #                    print 'no suiting time sample found for time step %f.3 in %s - "0" added'%(ii,filename)
            #                    data3.append(0)
                    
            #simply listing 8 data sets in the right order.
            alldatain.append(list(data3.transpose()))
            gf_idx += 1

            #after 8 files, the data for one source is complete
            if (gf_idx == 8):
                if _debug:
                    print 'data for source no. %i complete - doing re-arrangement '%(source_idx+1)
               #now build the 18 (3x6) components of the tensor G:
                for tmp_idx in xrange(18):
                    # set gf time series for component:
                    data_tmp = array(alldatain[order[tmp_idx]-1]) *array(factors[tmp_idx])

                    #debug:
                    #print shape(array(alldatain[order[tmp_idx]-1]))

                    #-------------------------------------------------------------------------------------
                    #Error in QSEIS - sometimes one sample is missing - can be corrected by this:
                    if not len(data_tmp) == n_time:
                        dummy8 = zeros((len(data_tmp)+1))
                        dummy8[1:] = data_tmp
                        data_tmp = dummy8
                        del dummy8
                    #-------------------------------------------------------------------------------------

                    #listing all GF successively
                    alldataout.append(data_tmp)

                #clear indizes and temp list for new source point:
                gf_idx = 0
                source_idx += 1                  
                alldatain=[]


        if _debug:
            print 'preparing NetCDF output parametres '
        
        #prepare NetCDF output
        # set file name:
        outfile          = sioncdf.NetCDFFile(outfilename, 'w')#, 'Created ' + time.ctime(time.time())) 

        #mandatory arguments for NetCDF
        outfile.title    = "GF for station "+station
        outfile.version  = int(cfg['gf_version'])

        # decription of axes
        outfile.createDimension('GF_idx', len(alldataout)+1)
        outfile.createDimension('t', len(time_array))

        #only important step - build a variable to be stored
        GF_for_act_rec   = outfile.createVariable('GF_for_current_station', 'f', ('t', 'GF_idx'))

        # set array for timie axis and GF
        #print len(alldataout)
        temp = zeros((len(time_array),len(alldataout)+1), dtype=float32)

        # fill array with time axis
        temp[:,0] = time_array.astype('float32')
        
        #fill array with GF
        #print shape(temp[:,1:]),shape(array(alldataout, dtype=float32).transpose())

        temp[:,1:]  = array(alldataout, dtype=float32).transpose()

        if _debug:
            print shape(temp)

        #up to here, data is given in the components 'radial R', 'tangential phi', down Z' within the right handed coordinate system (R, phi, Zdown)
        # rotate data into left handed coordinate system (N,E,Zup)
        print 'rotating elementary seismograms to NEU ...'
        for source in arange(n_source):

            source_coordinates   = [source_coords[source,0],source_coords[source,1],source_coords[source,2]]

            dist, azi, back_azi  =  distance_azi_backazi(source_coordinates,station_coordinates)
            rot_angle            = (back_azimuth-180.)%360
             
            start_index_of_first_horizontal_component_R   = int(1 + 18*source)
            start_index_of_second_horizontal_component_T  = start_index_of_first_horizontal_component_R + 6
            start_index_of_vertical_component_Zdown       = start_index_of_first_horizontal_component_R + 12

            for lid in arange(6):
                elemental_trace_R     =  temp[: , start_index_of_first_horizontal_component_R + lid]
                elemental_trace_phi   =  temp[: , start_index_of_second_horizontal_component_T + lid]
                elemental_trace_Zdown =  temp[: , start_index_of_vertical_component_Zdown + lid]

                elemental_trace_E, elemental_trace_N  = rotate_traces(elemental_trace_phi,elemental_trace_R ,rot_angle)
                
                # GF array is set in NORTH, EAST, UP !!!!!!!!!!!!!!
                temp[: , start_index_of_first_horizontal_component_R + lid]  =  elemental_trace_N
                temp[: , start_index_of_second_horizontal_component_T + lid] =  elemental_trace_E
                temp[: , start_index_of_vertical_component_Zdown + lid]      = -elemental_trace_Zdown

            
        # fill variable with values from GF array:
        GF_for_act_rec[:] = temp

        #mandatory argument for NetCDF
        GF_for_act_rec.units = "'seconds' and 'arbitrary'"

        print 'writing to file:  ',outfilename,'\n'
        
        outfile.close()
        
        print 'gf for station  '+station+ ' ...  ok !! (%3i of %3i)\n' %(stationcounter,n_stations)

        del temp
        del alldatain
        del alldataout
        del factor_array
        del temp_dict
        del readfile_raw
        del data2
        del data3
        del data_tmp
        del elemental_trace_Zdown
        del elemental_trace_E
        del elemental_trace_N
        del elemental_trace_R
        del elemental_trace_phi
        
                
        #delete temporary filelist in temp directory
        #if int(float(cfg['gf_db_chunks_flag'])) == 1  :
        #    loc_filelist = os.listdir(gf_dir_in)
        #    for ff in loc_filelist:
        #        realname = path.abspath(path.realpath(path.join(gf_dir_in,ff)))
        #        os.remove(realname)

    print 'Green`s functions - database for current source-station-combinations ready !'
            
#----------------------------------------------------------------------

def setup_synth_data(cfg):

    """ Beispieldaten fuer die gegebenen receiver fuer ein event an gegebener Quelle ('source'-Index)
    
    """

    import pickle as pp
    import numpy.random as Rand  
    import time


    base_dir        = path.realpath(path.abspath(cfg['base_dir']))   
    gf_dir          = path.realpath(path.abspath(path.join(base_dir,'DB','GF')))
    parent_data_dir = path.realpath(path.abspath(path.join(base_dir,cfg['data_dir'])))

    cfg['parent_data_directory'] = parent_data_dir
    temp_dir        = path.realpath(path.abspath(path.join(base_dir,'temp')))

    if not path.exists(temp_dir):
        os.makedirs(temp_dir)

    if not path.exists(parent_data_dir):
        os.makedirs(parent_data_dir)
        
    print '\nsynthetics in folder',parent_data_dir,'\n'

    grid_coords     = cfg['grid_coordinates']
    stat_coords     = cfg['station_coordinates']
    lo_rec          = cfg['list_of_receivers']
    n_receiver      = len(lo_rec)
    rec_index_dict  = cfg['receiver_index_dictionary']
    rec_nslc_dict   = cfg['receiver_nslc_dictionary']
    stat_index_dict = cfg['station_index_dictionary']
    stat_coords_dict= cfg['station_coordinate_dictionary']
    lo_stats        = cfg['list_of_stations']
    lo_chans        = cfg['list_of_channels']
    chan_index_dict = cfg['channel_index_dictionary']


    try:
        s_idx         = int(cfg.get('source_index') )-1
    except:
        try:
            source_depth  = float(cfg.get('source_depth'))
            s_idx         = find_idx_from_coordinates([float(cfg['central_latitude']),float(cfg['central_longitude']),float(source_depth)],grid_coords)
            print 'source index: %i - source depth: %f'%(s_idx+1,source_depth)
        except:
            print "ERROR! please specify source in config file as 'source_index' or 'source_coordinates'"
            raise SystemExit()
    
    source_coordinates = grid_coords[s_idx]

    filename1a = path.abspath(path.realpath(path.join(temp_dir,'current_GF_for_synth_data.dat')))
    filename2a = path.abspath(path.realpath(path.join(temp_dir,'current_time_for_synth_data.dat')))

    lp_flag    = int(cfg['lp_flag'])

        
    print 'synthetic data for source no. '+ str(s_idx+1) + ' at '+ str(source_coordinates )+'  is being built ... \n'

    try:
        noise      = int(round(100.*float(cfg.get('noiselevel',0))))

        if (int(lp_flag) == 1): 
            data_dir = path.abspath(path.realpath(path.join(parent_data_dir, 'lp-data_wn_%i'% noise )))
        else:
            data_dir = path.abspath(path.realpath(path.join(parent_data_dir, 'data_wn_%i'% noise )))
        if noise == 0:
            data_dir = path.abspath(path.realpath(path.join(parent_data_dir, 'wo_noise' )))
     
    except:
        data_dir = parent_data_dir


    if path.exists(data_dir):
        shutil.rmtree(data_dir)
    makedirs(data_dir)
     

     
    gf_sampling  = int(float(cfg['gf_sampling_rate']))
    gf_version   = int(float(cfg['gf_version']))
    version      = int(float(cfg['data_version']) )
    length       = int(float(cfg['window_length_in_s']))
    
    windowtracelength   = length*gf_sampling + 1
    eventtime           = cfg['event_datetime_in_epoch']
    ev_time_tuple       = time.gmtime(eventtime)

    tmin_day = cldr.timegm((ev_time_tuple[0], ev_time_tuple[1], ev_time_tuple[2],0 ,0 ,0 ,ev_time_tuple[6] , ev_time_tuple[7],0 ))


    current_day       = ev_time_tuple[7]
    current_year      = ev_time_tuple[0]   
    tmax_day          = tmin_day + 86400 - gf_sampling

    totaltracelength    = 86400 * gf_sampling 

    chng_length_of_data  = cfg.get('reduce_synth_data_length',0)
    try: 
        chng_length_of_data = int(chng_length_of_data) 
        if (not chng_length_of_data in [0,1]):
            chng_length_of_data = 0
    except:
        chng_length_of_data = 0

    if chng_length_of_data:
        try:
            length_of_data = cfg.get('data_trace_length',0)
            length_of_data = float(length_of_data)
            if 86400 > length_of_data > 10/gf_sampling:
                tmin_day = eventtime - length_of_data
                tmax_day = eventtime + 2*length_of_data
                totaltracelength = round(tmax_day - tmin_day) * gf_sampling
                print 'length of data traces adjusted to %.1f seconds (%i samples)\n'%(totaltracelength/gf_sampling,totaltracelength )
        except:   
            pass

        
    GF_column_min = (s_idx)*18+1
    GF_column_max = (s_idx+1)*18+1

    print 'Spalten: ',GF_column_min, ' -- ', GF_column_max
    
    #M = array([1,-2,4,6,0,-1]) #bsp aus jost und hermann mit dominantem vertical strike-slip
    M = array([0,0,0,0,1,1])

    data_temp       = zeros((totaltracelength,4))
    time_axis       = (arange(totaltracelength)/gf_sampling) + tmin_day
    data_temp[:,0]  = time_axis
    dummy4          = abs(time_axis - eventtime)
    startindex_of_window = dummy4.argmin()


    GF_total_dict = {}

    if not ( path.exists(filename1a) and path.exists(filename2a) ) :

        for station_index in arange(len(lo_stats)):

            station       = lo_stats[station_index]
         
            current_stat_coo_dict = stat_coords_dict[station]
            lat_stat = float(current_stat_coo_dict['lat'])
            lon_stat = float(current_stat_coo_dict['lon'])
            station_coordinates = [lat_stat,lon_stat]
            
            print 'and station '+ station + ' ... \n'
            
            t_infilename  = path.realpath(path.abspath(path.join( gf_dir, 'gf_v%(ver)i_length%(twl)i_sampling%(samp)i_station_%(stat)s.nc' %{'ver':gf_version,'twl':length,'samp':gf_sampling,'stat':station })))
            
            tmp_ncfile_gf  = sioncdf.NetCDFFile(t_infilename,'r')
            contained_vars_dict = tmp_ncfile_gf.variables
            contained_vars = contained_vars_dict.keys()
            gf_var         = contained_vars[0]
            tmp_data       = tmp_ncfile_gf.variables[gf_var]
            gf_raw         = array(tmp_data).astype('float64')
            tmp_ncfile_gf.close()


            count = 0


            print 'lese zeitachse von ', t_infilename
            time        = gf_raw[:,0]
            print '... done\n'
            print 'lese GF-Matrix von ', t_infilename
            tmp_gf1     = gf_raw[:,GF_column_min:GF_column_max]
            print '... done\n'
            print 'transponiere GF-Matrix\n'
            GF          =  transpose(matrix(tmp_gf1))
            print '... done\n'

            GF_total_dict[str(station_index)] = GF
            
            print 'Groesse GF: ', [size(GF,i) for i in range(2)]


            
        GF_total = zeros((len(lo_stats), size(GF,0), size(GF,1)  ))
        print shape(GF_total), shape(GF)

        
        for stat_idx in arange(len(lo_stats)):
           station      = lo_stats[stat_idx]
           GF           = GF_total_dict[str(stat_idx)]
           GF_total[stat_idx,:,:] = GF 

        File1a = file(filename1a,'w')
        File2a = file(filename2a,'w')
        pp.dump(GF_total,File1a)
        pp.dump(time,File2a)
        File1a.close()
        File2a.close()

    else:
        print 'Einlesen von gespeicherten GF und time...\n aus ',filename1a,'\n',filename2a

        File1a    = file(filename1a,'r')
        GF_total  = pp.load(File1a)
        File1a.close()
        
        File2a = file(filename2a,'r')
        time   = pp.load(File2a)
        File2a.close()

    for station_index in arange(len(lo_stats)):
        station    = lo_stats[station_index]
        data_temp_1 = zeros((totaltracelength))
        data_temp_2 = zeros((totaltracelength))
        data_temp_3 = zeros((totaltracelength))

            
        lo_nw     = cfg['list_of_networks']
        network   = lo_nw[0]
        lo_loc    = cfg['list_of_locations']
        location  = lo_loc[0]
   
        current_stat_coo_dict = stat_coords_dict[station]
        lat_stat = float(current_stat_coo_dict['lat'])
        lon_stat = float(current_stat_coo_dict['lon'])
        station_coordinates = [lat_stat,lon_stat]
    

        GF = GF_total[station_index,:,:]

        G1 = (GF[0:6])
        G2 = (GF[6:12])
        G3 = (GF[12:18])

        
        tmpD1 = array(transpose(matrix(dot (M, G1))))
        tmpD1 = array([float(tmpD1[i]) for i in range(len(tmpD1))] )

        tmpD2 = array(transpose(matrix(dot (M, G2))))
        tmpD2 = array([float(tmpD2[i]) for i in range(len(tmpD2))] )

        tmpD3 = array(transpose(matrix(dot (M, G3))))
        tmpD3 = array([float(tmpD3[i]) for i in range(len(tmpD3))] )

        data_temp_1[startindex_of_window : startindex_of_window+windowtracelength] =  tmpD1[:]
        data_temp_2[startindex_of_window : startindex_of_window+windowtracelength] =  tmpD2[:]
        data_temp_3[startindex_of_window : startindex_of_window+windowtracelength] =  tmpD3[:]


#         figure(2)
#         plot(tmpD2)
#         figure(3)
#         plot(tmpD3)
                
 

        if (not noise == 0 ):

            print 'addiere noise %i \n' %(noise)
 
            data_temp_1 += float(cfg['noiselevel'])*max(abs(data_temp_1))*Rand.randn(len(data_temp_1))
            data_temp_2 += float(cfg['noiselevel'])*max(abs(data_temp_2))*Rand.randn(len(data_temp_2))
            data_temp_3 += float(cfg['noiselevel'])*max(abs(data_temp_3))*Rand.randn(len(data_temp_3))
            
        
        data_N = data_temp_1
        data_E = data_temp_2
        data_Z = data_temp_3

        data_temp[:,1] = data_N
        data_temp[:,2] = data_E
        data_temp[:,3] = data_Z


        for channel in lo_chans:
            channel_index = int(chan_index_dict[channel])
            data_trace = data_temp[:,channel_index+1]
            filename = '%s.%s.%s.%s.D.%s.%s' %(network,station,location,channel,current_year,current_day)
            #filename = '%s.%s'%(station,channel)
            trtup = (network,station,location,channel,int(tmin_day*pymseed.HPTMODULUS), int(tmax_day*pymseed.HPTMODULUS), gf_sampling, data_trace)

            if 0:#channel_index ==2 :
                figure(10 * (station_index+1) +channel_index )
                
                pd = data_temp[startindex_of_window : startindex_of_window+windowtracelength,channel_index+1]
                plot( pd)

            fn = path.abspath(path.realpath(path.join( data_dir,filename )))
#             print [trtup],'\n',fn
#             print 'plotted in window',10 * (station_index+1) +channel_index,'\n'
            
            pymseed.store_traces([trtup], fn)
 
        print 'daten fuer station '+station+' geschrieben \n'

    print '....set !!!\n'

#----------------------------------------------------------------------
def find_idx_from_coordinates(input_coords,coord_array):

    #both in N,E,Z
    array_lat =  coord_array[:,0]
    array_lon =  coord_array[:,1]
    array_dep =  coord_array[:,2]


    s_depth     = input_coords[2]
    mindistdep  = min(abs(array_dep - s_depth))
    s_depth_idx_list =[]
    for ad_idx in arange(len(array_dep)):
        if abs(array_dep[ad_idx] - s_depth) == mindistdep:
            s_depth_idx_list.append(ad_idx)

    s_lat = input_coords[0]
    s_lon = input_coords[1]

    hor_dist_list=[]
    for s_d_idx in s_depth_idx_list:
        dist = (array_lon[s_d_idx] - s_lon )**2 + (array_lat[s_d_idx] - s_lat)**2
        hor_dist_list.append(dist)
    
    index = s_depth_idx_list[ array(hor_dist_list).argmin()]

    #print hor_dist_list,array(hor_dist_list).argmin() 

    return index


#---------------------------------------------------------------------
def distance2coordinates(coords, distance, angle):
    
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
    len0      = coords[1]/rad_to_deg

    azimuth   = angle/rad_to_deg
    angular_distance = distance/R


    lat2 = arcsin(sin(lat0)*cos(angular_distance) + cos(lat0)*sin(angular_distance)*cos(azimuth))

    lon2 = lon0 + arctan2(  sin(azimuth)*sin(angular_distance)*cos(lat0), cos(angular_distance) - sin(lat0) * sin(lat2)   )  	

    lat_goal = lat2 * rad_to_deg
    lon_goal = lon2 * rad_to_deg
    
    return lat_goal, lon_goal
#---------------------------------------------------------------------
def d_break(locs,*message):
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
