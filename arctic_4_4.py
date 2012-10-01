#!/usr/bin/env python
#coding:utf-8
#----------------------------------------------------------------------
#   artmotiv
#  37th version
#  4.4.0
#----------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
##
#     using arctic tools v.14
#
#     (last change : 14.02.11)
#
#     LK
#----------------------------------------------------------------------
#----------------------------------------------------------------------



import os
import os.path as op


import sys

sys.path.append('/net/scratch2/gp/u25400/lasse/arctic/')
#sys.path.reverse()

import arctic_tools_0_14
reload(arctic_tools_0_14)
from   arctic_tools_0_14 import *

#---------------------------
_debug = 1

import code
#---------------------------


def run_arctic(configfile, event_type_code, datetime_code):
    """Automatic  Real-time Cmt Time domain Inversion Code.

    Version 4.4.0

    The one and only wrapping for the single steps of the inversion...!!!!
    """

    
    #write output to log file instead of screen 
    if not _debug:
        saveout = sys.stdout
        fsock = file('arctic_output.log', 'w')
        sys.stdout = fsock 
             
    print '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n \n'
             
    print 'start config_file parsing...'
    cfg = read_in_config_file(configfile)
    print '...done \n '

    cfg['event_type'] = event_type_code
    cfg['datetime'] = datetime_code 
    cfg['event_ID'] = datetime_code 


    print 'setting parameters (time, files, coordinates, etc.)...'
    if not set_general_parameters(cfg):
        print 'ERROR! could not set up parameters'
        exit()
    print '...done \n '
    #cfg['m_cmp'] = array([-13,73,-60,10,46,56])*1e24
    cfg['m_cmp'] = array([1,2,3,-4,-5,0])*2e12

    #d_break(locals(),'nMain\n')
    
    print 'setting GF... \n'
    if not   read_in_GF(cfg):
        print 'ERROR! could not set up GF array' 
        exit()
    print '...GF ready \n'


    
    print 'setting A and A^(-1)... \n'
    if not setup_A(cfg):
        print 'ERROR! could not set up correlation matrix A'
        exit()

    if not calc_inv_A(cfg):
        print 'ERROR! could not set up inverted matrix inv_A'
        exit()
    print '...done \n'
    
    print 'theoretically begin time-loop here...later ...\n'

    print 'setting list of datafiles'
    if not set_list_of_datafiles(cfg):
        print 'ERROR! could not set up list of datafiles'
    print '...done \n '

    print 'setting raw data pile'
    if not set_raw_data_for_main_window(cfg):
        print 'ERROR! could not set up raw_data_pile'
    print '...done \n '

    print 'setting data traces for moving time windows'
    if not read_in_data(cfg):
        print 'ERROR! could not read in data array'
    print '...done '


    # print 'calc correlation vector(s) b'
    # if not calc_corr_vec(cfg):
    #     print 'ERROR! could not set up correlation vector b'
    # print '...done \n'

    print 'finding optimal source point'
    if not find_source(cfg):
        print 'ERROR! could not find optimal source'
    print '...done!\n'      
    print '...fertig gesucht und gefunden!!! \n'

    #d_break(locals(),'main')
    
    print 'plot'
    if not plot_result(cfg):
        print 'ERROR! could not plot'
    print '...done!\n'      
    
    if not present_results(cfg):
        print 'ERROR! could not display results'
    

    #print 'collect solution parameters in "solution_dictionary"'
    if 0:#not make_solution_dict(cfg):
        print 'ERROR! could not set up solution dictionary'
    if 0:#not  make_parameter_dict(cfg):
        print 'ERROR! could not set up parameter dictionary'
    #print '...done!\n'      

    
    print 'build xml file'
    if not build_xml_hardcode(cfg):
        print 'ERROR! could not build xml file'
    print '...done!\n'      
    #print 'submit xml file to database'
    #if not put_xml_to_database(cfg):
    #    print 'ERROR! could not send xml file '
    #print '...done!\n'      
  
    
    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'

    #set standard out back to screen 
    if not _debug:
        sys.stdout = saveout        
        fsock.close()                  

    return 1

#-----------------------------------------------------------------

if __name__=='__main__':

    cfg_file = op.realpath(op.abspath(sys.argv[1]))

#synthetisch    
#    run_arctic(cfg_file, '2', '1983-05-18T12:00:00.0') 

#synth - 1 seconds
#    run_arctic(cfg_file, '2', '1983-05-18T11:59:59.0') 

#synth - 5 seconds
#    run_arctic(cfg_file, '2', '1983-05-18T11:59:55.0') 

#synth -10 seconds
#    run_arctic(cfg_file, '2', '1983-05-18T11:59:50.0') 

#synth -20 seconds
    run_arctic(cfg_file, '2', '1983-05-18T11:59:40.0') 

#synth rotenburg continuous
#    run_arctic(cfg_file, '2', '2004-10-20T06:59:14.4') 


#rotenburg 
#    run_arctic(cfg_file, '2', '2004-10-20T06:59:15.1')

#rotenburg best
#    run_arctic(cfg_file, '2', '2004-10-20T06:59:22.1')

#rotenburg conti best sol 
#    run_arctic(cfg_file, '2', '2004-10-20T06:59:18.2')

#rotenburg -10 seconds
#    run_arctic(cfg_file, '2', '2004-10-20T06:59:05.1')

#rotenburg -20 seconds
#    run_arctic(cfg_file, '2', '2004-10-20T06:58:55.1')

#ekofisk  
#    run_arctic(cfg_file, '2', '2001-05-07T09:43:34.0')

#ekofisk -10 seconds
#    run_arctic(cfg_file, '2', '2001-05-07T09:43:24.0')

#erebus -3 s    
#    run_arctic(cfg_file, '2', '2005-12-31T10:53:43.0')

#erebus -10 s    
#    run_arctic(cfg_file, '2', '2005-12-31T10:53:36.0')

#erebus -20 s    
#    run_arctic(cfg_file, '2', '2005-12-31T10:53:26.0')

#erebus 
#    run_arctic(cfg_file, '2', '2005-12-31T10:53:46.0') 

#ekofisk -1 seconds
#    run_arctic(cfg_file, '2', '2001-05-07T09:43:32.5')
