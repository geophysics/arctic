#!/usr/bin/env python

import numpy as N

from pyrocko import io

import sys, os
import os.path as op
import glob

station_amplification_file = sys.argv[1]
stat_amp_fn = op.abspath(op.realpath(op.join('.',station_amplification_file)))

lo_amps = N.loadtxt(stat_amp_fn, dtype = 'string')

#make amplification_dict
so_stats = set()
ampl_dict = {}

for i in lo_amps:
    stat = i[0].upper()
    so_stats.add(stat)
    ampl_dict[stat] = {}

for i in lo_amps:
    stat = i[0].upper()
    channel = i[1].upper()
    ampl = float(i[2])
    loc_dict = ampl_dict[stat]
    loc_dict[channel] = ampl
    #print stat, channel, ampl



#--------------------------------

mseed_path_raw = sys.argv[2]
mseed_path = op.abspath(op.realpath(op.join('.',mseed_path_raw)))

WD = op.abspath(op.realpath(os.curdir))

os.chdir(mseed_path)

lo_mseeds  = glob.glob('*')

os.chdir(WD)

outpathname = 'corrected_traces'
outpath     = op.abspath(op.realpath(op.join('.',outpathname)))

if not op.exists(outpath):
    os.makedirs(outpath)

for F in lo_mseeds:
    in_fn  = op.abspath(op.realpath(op.join(mseed_path,F)))
    out_fn = op.abspath(op.realpath(op.join(outpath,F)))
    
    traces = io.load(in_fn)

    for tr in traces:
        stat = tr.station
        channel = tr.channel.upper()[-1]
        fac = ampl_dict[stat][channel] 
        tr.ydata = fac * tr.get_ydata() /10**9

        print F, stat, channel, fac
  
    io.save(traces,out_fn )
