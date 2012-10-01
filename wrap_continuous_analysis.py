#!/usr/bin/env python
#coding:utf-8
#----------------------------------------------------------------------
#   arctic_continuous
#  1st version
#  4.4.0
#----------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------
##
#     using arctic tools v.14
#
#     (last change : 24.02.11)
#
#     LK
#----------------------------------------------------------------------
#----------------------------------------------------------------------


import os
import os.path as op
from numpy import *


import sys
import pickle 

sys.path.reverse()
sys.path.append('/net/scratch2/gp/u25400/lasse/arctic/')
sys.path.reverse()

import arctic_tools_0_14 as AT
reload(AT)


import time
from   pyrocko import pile, util
import pyrocko.trace as pt

#==================================================================



def d_break(locs,*message):
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


#==================================================================
def restitute_t(tr,cfg,bp_freq):
    
    
    if cfg.has_key('restitute') and cfg['restitute'].upper() in ['TRUE', '1']:
        tr.highpass(4,bp_freq/2.)
        dat_in     = tr.ydata-mean(tr.ydata)
        dat_out    = cumsum(dat_in)
        dat_out    *= tr.deltat


    desired_deltat = 1./float(cfg['gf_sampling_rate'])
    tr.downsample_to(desired_deltat)
    #d_break(locals())
    #exit()
    try:
        tr.chop(tr.wmin, tr.wmax,include_last=True, inplace=True)
    except pt.NoData:
        pass
        #continue

    

#==================================================================

def analysis_4_given_window(traces,cfg):

    log_file = cfg['log_file']
    logg = file(log_file,'a')
    old_stderr = sys.stderr
    sys.stderr = logg
    cfg_pickle_file = cfg['cfg_pickle_file'] 
    solution_pickle_file  =   cfg['solution_pickle_file']
    time_stamp =  cfg['current_window_starttime']

    try:
        VR  = AT.read_in_data_and_find_source_continuous(traces,cfg)
    except:
        sys.stderr.write( '\nERROR! could not read data or find source\n')
        raise

    print '\n',VR,time_stamp,'\n'

    #write VR value  into file to save time history
    f_handle = file(cfg['vr_history_file'], 'a')
    f_handle.write('%s \t %s \n'%(str(VR),time_stamp))
    f_handle.close()

    VR_threshold = float(cfg['vr_threshold_value'])
    
    if VR >=  VR_threshold:
        print '\nEVENT detected !! \n VR = %.2f  \n %s'%(VR,util.time_to_str(time_stamp))
        
        print 'building xml file...'
        event_count          = int(float(cfg['event_count'])) + 1
        cfg['event_count']   = event_count
        event_id             = '%i-%s'%(event_count,util.time_to_str(time_stamp))
        cfg['event_ID']      = event_id
        
        cfg['xml_file_name'] = 'event_nr_%i.%s'%(event_count,cfg['xml_file_name_basis'])
        
        if not AT.build_xml_hardcode(cfg):
             sys.stderr.write( '\nERROR! could not build xml file\n')
             raise
        print '...done!\n'      
        print 'submitting xml file to database...'
        if 0: #not  AT.put_xml_to_database(cfg):
            sys.stderr.write( '\nERROR! could not send xml file \n')
            raise
        print '...done!\n'      

        print 'plot'
        if not AT.plot_result(cfg):
            print 'ERROR! could not plot'
        print '...done!\n'      
    
        if 0: #not AT.present_results(cfg):
            print 'ERROR! could not display results'

        #exit()

    #for manual checking:
    # cfg_pickle_fh = file(cfg_pickle_file,'w')
#     pickle.dump(cfg,cfg_pickle_fh)
#     cfg_pickle_fh.close()

#    print 'written temporary config to file'
    
        solution_pickle_fh = file(solution_pickle_file,'w')
        pickle.dump(cfg['temp_solution_dictionary'],solution_pickle_fh )
        solution_pickle_fh.close()

    print 'written temporary solution to file %s'%(solution_pickle_file) 
    
    #set output back to standard:
    sys.stderr = old_stderr
    logg.close()
#==================================================================

def run_cont_inversion(cfg_file,tmin_data):


    #read in config file
    #including setup of path structure 
    cfg       = AT.read_in_config_file(cfg_file) 
    cfg['datetime'] = util.time_to_str(tmin_data,format='%Y-%m-%dT%H:%M:%S')

    event_count = 0
    cfg['event_count'] = event_count

    filter_flag = int( float( cfg.get('filter_flag',0) ) )
    bp_hp_freq  = float(cfg.get('bp_lower_corner'))

    pad_length = 1./bp_hp_freq*2
    if filter_flag == 0:
        pad_length = 10


    #------------------------------------------
    #   set files
    #
    cont_temp_dir =op.realpath(op.abspath(op.join(cfg['temporary_directory'],'cont_run'))) 
    if not op.isdir(cont_temp_dir):
        os.makedirs(cont_temp_dir)
    cfg['temporary_directory']  = cont_temp_dir 

    # set file for 'time - VR' history
    VR_history_file = op.realpath(op.abspath(op.join(cfg['temporary_directory'],'VR_over_time.txt')))
    open( VR_history_file,'w').close()
    cfg['vr_history_file'] = VR_history_file

    # set file for logging
    log_file = op.realpath(op.abspath(op.join(cfg['temporary_directory'],'arctic_log_file.log')))
    open(log_file,'w').close()
    cfg['log_file'] = log_file
 
    # set file for storing last config dictionary for debugging 
    cfg_pickle_file =  op.realpath(op.abspath(op.join(cfg['temporary_directory'],'cfg_pickle.pp')))
    open(cfg_pickle_file,'w').close()
    cfg['cfg_pickle_file'] = cfg_pickle_file

    # set file for storing last solution dictionary for debugging 
    solution_pickle_file =  op.realpath(op.abspath(op.join(cfg['temporary_directory'],'cfg_pickle.pp')))
    open(solution_pickle_file,'w').close()
    cfg['solution_pickle_file'] = solution_pickle_file
    #------------------------------------------    # set time parameters
    #
    #safety time delay in seconds
    #data are analysed maximal to the current time minus the latency
    # allows small buffer for incoming data in real data test case
    latency = 5.

    # length of analysis window
    wlen = float(cfg['time_window_length'])

    #minimum window step size, avoiding too high cpu load
    minstep = float(cfg['window_steps_in_s'])

    # inintialise window end time
    tlast = None


    #------------------------------------------
    # set other parameters
    #
    print 'setting parameters (time, files, coordinates, etc.)...'
    if not AT.set_general_parameters(cfg):
        print 'ERROR! could not set up parameters'
        raise 
    print '...done \n '

    print 'setting GF... \n'
    if not   AT.read_in_GF(cfg):
        print 'ERROR! could not set up GF array' 
        exit()
    print '...GF ready \n'

   
    if not AT.setup_A(cfg):
        print 'ERROR! could not set up correlation matrix A'
        raise
    if not AT.calc_inv_A(cfg):
        print 'ERROR! could not set up inverted matrix inv_A'
        raise
    print '...done \n'
   

    #------------------------------------------
    #
    #define data pile from given file location
    p = pile.make_pile(cfg['data_dir'])
    
    # read in GFs
    if not AT.read_in_GF(cfg):
        exit('ERROR in reading GFs')

    #------------------------------------------
        
    #d_break(locals(), 'last exit before while loop')

    #------------------------------------------

    #set current real time as basis time 
    tmin_real = time.time()

    #------------------------------------------
    #start infinite run
    while True:        
        # check, if enough time has pased since last loop run
        while True:
            # define internal time
            tfake = ( time.time() - (tmin_real-tmin_data))
            # either beginning of analysis or minimal time for window movement has passed
            if tlast is None or tfake - tlast > minstep:
                break
            
            #wait time, needed to fill minumum window step length
            time.sleep(minstep-(tfake-tlast))

        print 'real_time: %s -- data_time: %s'%(util.time_to_str(time.time()),util.time_to_str(tfake) )

        #set temporary analysis window
        tmin = tfake - latency - wlen
        tmax = tfake - latency + p.get_deltats()[0]
        cfg['current_window_starttime'] = tmin
 
        print 'analysing from %s  --  %s'%(util.time_to_str(tmin),util.time_to_str(tmax)) 
        
        #read in all data traces, available for this window 
        try:
            traces = p.all( tmin=tmin, tmax=tmax, want_incomplete=False ,tpad = pad_length)
            
            for t in traces:
                restitute_t(t,cfg,bp_hp_freq)
        except Exception, e:
            sys.stderr.write( '%s' % e )
            pass

        #d_break(locals(),'%s'%(util.time_to_str(tfake)))
        # analyse the data
        #try:
        analysis_4_given_window(traces,cfg)
        #except Exception, e:
        #    sys.stderr.write( '%s' % e )

        #save current time stamp for next loop run
        tlast = tfake

        #break

#-----------------------------------------------------------------
#==================================================================

if __name__=='__main__':


    #read given start date
    try:
        tmin_data = AT.read_datetime_to_epoch(sys.argv[2])
        #util.str_to_time( sys.argv[2], format='%Y-%m-%dT%H:%M:%S.OPTFRAC')
    except:
        print '\n ERROR - provide date as 2. argument in format < year-month-dayThour:minute:seconds.milliseconds > \n'
        exit()

    try:
        cfg_file    = op.realpath(op.abspath(op.join('.',sys.argv[1])))
        if not os.path.isfile(cfg_file):
            raise
    except:
        print '\n ERROR - provide config file position  as 1. argument \n'
        exit()
 
    
    run_cont_inversion(cfg_file,tmin_data)
