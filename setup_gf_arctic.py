#!/usr/bin/env python
#coding=utf-8
#----------------------------------------------------------------------
#  version 1.03    -   24.02.10   -   LK   -
#----------------------------------------------------------------------
#
#
#  Tool for setting up a database of Green's functions for a
#   - circular shaped configuration of receivers
#   - rectangular setup of source-grid
#  around a given center (point, given in (lat,lon)-coordinates
#
#  Function for setting up synthetic data-set for arbitrarily chosen source-point from source-grid
#
#  USES FUNCTIONS FROM "setup_gf_functions.py" !!!!!
#
#----------------------------------------------------------------------
import sys
import os
import os.path as op

sys.path.append('/net/scratch2/gp/u25400/lasse/arctic/')
import setup_gf_tools_arctic
reload(setup_gf_tools_arctic)
from   setup_gf_tools_arctic import *

#----------------------------------------------------------------------

_debug = 1
#----------------------------------------------------------------------



def setup_gf(configfilename,qs=True, data=False):

    if not _debug:
        saveout = sys.stdout
        fsock = file('log_setup_arctic_current_run.log', 'w')
        sys.stdout = fsock 
 


    print '\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n'
    
    cfg          = read_in_config_file(configfilename)
    print '...done!\n'   
   
    cfg['datetime'] = cfg['event_id']
    if not set_GF_parameters(cfg):
        print 'ERROR in setting the Greens functions database parameters'
        raise SystemExit           
    print '...done!\n'   
    #if _debug:
    #    for i in cfg:
    #        print i,cfg[i]
    if qs==False and data==False:
        sys.exit('\n  NOTHING DONE !! \n')            


    #read in QSEIS database and prepare for given grid-receiver-configuration. create NetCDF GF database, ordered by station

    #-------------------------
    if qs:
        setup_db_qseis(cfg)

    #-------------------------
    
    # if no real model available, take random GF made from white noise:
    #setup_db_white_noise(cfg)

    #-------------------------
    if data:
        #set artificial data
        setup_synth_data(cfg)

    #-------------------------

    print '\n\n database ready !!!!!!!\n\n'
    if cfg['number_of_gf'] == 8:
        print '\n\n !!!!!!!!!!!     ATTENTION!! Far field approximation applied  !!!!!!!\n\n'
    print '\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n'

    if not _debug:
        sys.stdout = saveout        
        fsock.close()                  



#----------------------------------------------------------------------

if not len(sys.argv)> 1:
    sys.exit('\n  NO input - provide config file !! \n')    


cfg_file = op.realpath(op.abspath(op.join(os.curdir,sys.argv[1])))

qs_key=False
data_key=False

if len(sys.argv)> 2:
    for aa in sys.argv[2:]:
        if 'qs' in aa.lower():
            qs_key = True
        if 'data' in  aa.lower():
            data_key = True


setup_gf(cfg_file, qs=qs_key, data=data_key)

