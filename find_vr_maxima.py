#!/usr/bin/env python

from numpy import *
import cPickle as cp
import time

from pylab import *
import sys

time_step = 20
North = 11
East  = 11
Down  = 3


#=======================
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

#=======================



FH = file(sys.argv[1],'r')
full_dict = cp.load(FH)
FH.close()

lo_results = []
lo_vrs     = []
maxvr = 0
mvg   = zeros((North, East))
update = Down + 1 

for time_step in arange(len(full_dict.keys())):
    

    sol_dict = full_dict[str(time_step)]
    lo_results.append(sol_dict)
    VR_total       = sol_dict['VR_array']

    VR_grid = zeros((North, East, Down))
    #d_break(locals(),'')

    for idx,vr in enumerate(VR_total):
        d        = int(idx/(North*East))
        horz_idx = idx%(North*East)
        e        = int(horz_idx/North)
        n        = horz_idx%North
        VR_grid[n,e,d]  = vr
        if vr > maxvr:
            maxvr = vr
            update = d
            
    if update == 1:
        mvg = VR_grid[:,:,d]
    lo_vrs.append(VR_grid)
    
#d_break(locals(),'')

ion()
aa = contourf(mvg)
aa.set_clim(vmin=0,vmax=maxvr)

cb = colorbar(aa)
cb.set_clim(vmin=0, vmax=maxvr)

hold(False)

dinx = int(sys.argv[2])-1

for idx, vr in enumerate(lo_vrs):
    aa = contourf(vr[:,:,dinx])
    #lo_vrs[idx][:,:,1] 
    clim(0,maxvr)
    #draw()
    print idx
    time.sleep(0.5)
    #raw_input()
   
