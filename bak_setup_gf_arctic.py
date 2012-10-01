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

import setup_gf_tools_ruhr
reload(setup_gf_tools_ruhr)
from   setup_gf_tools_ruhr import *

#----------------------------------------------------------------------

_debug = 1
#----------------------------------------------------------------------



def setup_gf(configfilename):

    if not _debug:
        saveout = sys.stdout
        fsock = file('../inversion/log_setup_selby.log', 'w')
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
        
    #read in QSEIS database and prepare for given grid-receiver-configuration. create NetCDF GF database, ordered by station


    #setup_db_qseis(cfg)
    
    # if no real model available, just take random GF:
    #setup_db_white_noise(cfg)

    #set artificial data
    setup_synth_data(cfg)

    print '\n\n database ready !!!!!!!\n\n'
    print '\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n'

    if not _debug:
        sys.stdout = saveout        
        fsock.close()                  



#----------------------------------------------------------------------

setup_gf('setup_gf_config_ruhr.cfg')

