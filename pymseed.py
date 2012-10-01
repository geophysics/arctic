'''PyMSEED -- libmseed wrapper for Python and seismic waveform processing platform.'''


# Copyright (c) 2009, Sebastian Heimann <sebastian.heimann@zmaw.de>
#
# This file is part of pymseed. For licensing information please see the file
# COPYING which is included with pymseed.


from  pymseed_ext import *
import sys, os, logging, time, math, copy, re, calendar
import cPickle as pickle
import numpy as num
import scipy.signal as signal
import progressbar
from os.path import join as pjoin
import evalresp

def gmctime(t):
    return time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(t))

def gmctime_fn(t):
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(t))


class X:
    pass

reuse_store = dict()
def reuse(x):
    if not x in reuse_store:
        reuse_store[x] = x
    return reuse_store[x]

config = X()
config.show_progress = True

def plural_s(n):
    if n == 1:
        return ''
    else:
        return 's' 

def progress_beg(label):
    if config.show_progress:
        sys.stderr.write(label)
        sys.stderr.flush()

def progress_end(label=''):
    if config.show_progress:
        sys.stderr.write(' done. %s\n' % label)
        sys.stderr.flush()
        
def t2ind(t,tdelta):
    return int(round(t/tdelta))

def minmax(traces, key=lambda tr: (tr.network, tr.station, tr.location, tr.channel), mode='minmax'):
    
    '''Get data range given traces grouped by selected pattern.
    
    A dict with the combined data ranges is returned. By default, the keys of
    the output dict are tuples formed from the selected keys out of network,
    station, location, and channel, in that particular order.
    '''
    
    ranges = {}
    for trace in traces:
        if mode == 'minmax':
            mi, ma = trace.ydata.min(), trace.ydata.max()
        else:
            mean = trace.ydata.mean()
            std = trace.ydata.std()
            mi, ma = mean-std*mode, mean+std*mode
            
        k = key(trace)
        if k not in ranges:
            ranges[k] = mi, ma
        else:
            tmi, tma = ranges[k]
            ranges[k] = min(tmi,mi), max(tma,ma)
    
    return ranges
        
def minmaxtime(traces, key=lambda tr: (tr.network, tr.station, tr.location, tr.channel)):
    
    '''Get time range given traces grouped by selected pattern.
    
    A dict with the combined time ranges is returned. By default, the keys of
    the output dict are tuples formed from the selected keys out of network,
    station, location, and channel, in that particular order.
    '''
    
    ranges = {}
    for trace in traces:
        mi, ma = trace.tmin, trace.tmax
        k = key(trace)
        if k not in ranges:
            ranges[k] = mi, ma
        else:
            tmi, tma = ranges[k]
            ranges[k] = min(tmi,mi), max(tma,ma)
    
    return ranges
    
def degapper(in_traces, maxgap=5, fillmethod='interpolate'):
    
    '''Try to connect traces and remove gaps.
    
    This method will combine adjacent traces, which match in their network, 
    station, location and channel attributes. Overlapping parts will be removed.
    
    Arguments:
    
       in_traces:   input traces, must be sorted by their full_id attribute.
       maxgap:      maximum number of samples to interpolate.
       fillmethod:  what to put into the gaps: 'interpolate' or 'zeros'.
       
    '''
    
    out_traces = []
    if not in_traces: return out_traces
    out_traces.append(in_traces.pop(0))
    while in_traces:
        
        a = out_traces[-1]
        b = in_traces.pop(0)

        if (a.nslc_id == b.nslc_id and a.deltat == b.deltat and 
            len(a.ydata) >= 1 and len(b.ydata) >= 1 and a.ydata.dtype == b.ydata.dtype):
            
            dist = (b.tmin-(a.tmin+(len(a.ydata)-1)*a.deltat))/a.deltat
            idist = int(round(dist))
            if abs(dist - idist) > 0.05:
                logging.warn('cannot degap traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
            else:
                idist = int(round(dist))
                if 1 < idist <= maxgap:
                    if fillmethod == 'interpolate':
                        filler = a.ydata[-1] + (((1.+num.arange(idist-1,dtype=num.float))/idist)*(b.ydata[0]-a.ydata[-1])).astype(a.ydata.dtype)
                    elif fillmethod == 'zeros':
                        filler = num.zeros(idist-1,dtype=a.ydist.dtype)
                    a.ydata = num.concatenate((a.ydata,filler,b.ydata))
                    a.tmax = b.tmax
                    if a.mtime and b.mtime:
                        a.mtime = max(a.mtime, b.mtime)
                    continue

                elif idist == 1:
                    a.ydata = num.concatenate((a.ydata,b.ydata))
                    a.tmax = b.tmax
                    if a.mtime and b.mtime:
                        a.mtime = max(a.mtime, b.mtime)
                    continue
                    
                elif idist <= 0:
                    if b.tmax > a.tmax:
                        a.ydata = num.concatenate((a.ydata[:idist-1], b.ydata))
                        a.tmax = b.tmax
                        if a.mtime and b.mtime:
                            a.mtime = max(a.mtime, b.mtime)
                        continue
                    
        if len(b.ydata) >= 1:
            out_traces.append(b)
        
    return out_traces

def decimate(x, q, n=None, ftype='iir', axis=-1):
    """downsample the signal x by an integer factor q, using an order n filter
    
    By default, an order 8 Chebyshev type I filter is used or a 30 point FIR 
    filter with hamming window if ftype is 'fir'.

    (port to python of the GNU Octave function decimate.)

    Inputs:
        x -- the signal to be downsampled (N-dimensional array)
        q -- the downsampling factor
        n -- order of the filter (1 less than the length of the filter for a
             'fir' filter)
        ftype -- type of the filter; can be 'iir' or 'fir'
        axis -- the axis along which the filter should be applied
    
    Outputs:
        y -- the downsampled signal

    """

    if type(q) != type(1):
        raise Error, "q should be an integer"

    if n is None:
        if ftype == 'fir':
            n = 30
        else:
            n = 8
    if ftype == 'fir':
        b = signal.firwin(n+1, 1./q, window='hamming')
        y = signal.lfilter(b, 1., x, axis=axis)
    else:
        (b, a) = signal.cheby1(n, 0.05, 0.8/q)
        y = signal.lfilter(b, a, x, axis=axis)

    return y.swapaxes(0,axis)[n/2::q].swapaxes(0,axis)

class UnavailableDecimation(Exception):
    pass
    
class Glob:
    decitab_nmax = 0
    decitab = {}

def mk_decitab(nmax=100):
    tab = Glob.decitab
    for i in range(1,10):
        for j in range(1,i+1):
            for k in range(1,j+1):
                for l in range(1,k+1):
                    for m in range(1,l+1):
                        p = i*j*k*l*m
                        if p > nmax: break
                        if p not in tab:
                            tab[p] = (i,j,k,l,m)
                    if i*j*k*l > nmax: break
                if i*j*k > nmax: break
            if i*j > nmax: break
        if i > nmax: break
    
def decitab(n):
    if n > Glob.decitab_nmax:
        mk_decitab(n*2)
    if n not in Glob.decitab: raise UnavailableDecimation('ratio = %g' % ratio)
    return Glob.decitab[n]

def moving_avg(x,n):
    n = int(n)
    cx = x.cumsum()
    nn = len(x)
    y = num.zeros(nn)
    y[n/2:n/2+(nn-n)] = (cx[n:]-cx[:-n])/n
    y[:n/2] = y[n/2]
    y[n/2+(nn-n):] = y[n/2+(nn-n)-1]
    return y

def nextpow2(i):
    return 2**int(math.ceil(math.log(i)/math.log(2.)))
    
def snapper(nmax, delta, snapfun=math.ceil):
    def snap(x):
        return max(0,min(snapfun(x/delta),nmax))
    return snap

def costaper(a,b,c,d, nfreqs, deltaf):
    hi = snapper(nfreqs, deltaf)
    tap = num.zeros(nfreqs)
    tap[hi(a):hi(b)] = 0.5 - 0.5*num.cos((deltaf*num.arange(hi(a),hi(b))-a)/(b-a)*num.pi)
    tap[hi(b):hi(c)] = 1.
    tap[hi(c):hi(d)] = 0.5 + 0.5*num.cos((deltaf*num.arange(hi(c),hi(d))-c)/(d-c)*num.pi)
    
    return tap

class TraceTooShort(Exception):
    pass    

class FrequencyResponse(object):
    '''Evaluates frequency response at given frequencies.'''
    
    def evaluate(self, freqs):
        coefs = num.ones(freqs.size, dtype=num.complex)
        return coefs
   
class InverseEvalresp(FrequencyResponse):
    '''Calls evalresp and generates values of the inverse instrument response for 
       deconvolution of instrument response.'''
    
    def __init__(self, respfile, trace, target='dis'):
        self.respfile = respfile
        self.nslc_id = trace.nslc_id
        self.instant = trace.tmin
        self.target = target
        
    def evaluate(self, freqs):
        network, station, location, channel = self.nslc_id
        x = evalresp.evalresp(sta_list=station,
                          cha_list=channel,
                          net_code=network,
                          locid=location,
                          instant=self.instant,
                          freqs=freqs,
                          units=self.target.upper(),
                          file=self.respfile,
                          rtype='CS')
        
        transfer = x[0][4]
        return 1./transfer

class PoleZeroResponse(FrequencyResponse):
    
    def __init__(self, poles, zeros):
        self.poles = poles
        self.zeros = zeros
        
    def evaluate(self, freqs):
        pass
        
        

class MSeedGroup(object):
    def __init__(self):
        self.empty()
    
    def empty(self):
        self.networks, self.stations, self.locations, self.channels, self.nslc_ids = [ set() for x in range(5) ]
        self.tmin, self.tmax = num.inf, -num.inf
    
    def update_from_contents(self, contents):
        self.empty()
        for c in contents:
            self.networks.update( c.networks )
            self.stations.update( c.stations )
            self.locations.update( c.locations )
            self.channels.update( c.channels )
            self.nslc_ids.update( c.nslc_ids )
            self.tmin = min(self.tmin, c.tmin)
            self.tmax = max(self.tmax, c.tmax)
        
        if len(self.networks) < 32:
            self.networks = reuse(tuple(self.networks))
        if len(self.stations) < 32:
            self.stations = reuse(tuple(self.stations))
        if len(self.locations) < 32:
            self.locations = reuse(tuple(self.locations))
        if len(self.channels) < 32:
            self.channels = reuse(tuple(self.channels))
        if len(self.nslc_ids) < 32:
            self.nslc_ids = reuse(tuple(self.nslc_ids))
      
    def overlaps(self, tmin,tmax):
        return not (tmax < self.tmin or self.tmax < tmin)
    
    def is_relevant(self, tmin, tmax, selector=None):
        return  not (tmax <= self.tmin or self.tmax < tmin) and (selector is None or selector(self))
    
class MSeedTrace(MSeedGroup):
    def __init__(self, trace, substitutions=None):
        self.network, self.station, self.location, self.channel = [reuse(x) for x in trace[1:5]]
        self.tmin = float(trace[5])/float(HPTMODULUS)
        self.tmax = float(trace[6])/float(HPTMODULUS)
        self.deltat = reuse(float(1.0)/float(trace[7]))
        self.ydata = None
        if trace[8] is not None:
            ydata = trace[8]
            self.ydata = ydata
            
        if substitutions:
            for k,v in substitutions.iteritems():
                if k in self.__dict__:
                    self.__dict__[k] = v
            
        self.update_ids()

    def update_ids(self):
        self.full_id = (self.network,self.station,self.location,self.channel,self.tmin)
        self.nslc_id = reuse((self.network,self.station,self.location,self.channel))
        # for MSeedGroup interface
        self.networks, self.stations, self.locations, self.channels = [ reuse((x,)) for x in self.nslc_id ]
        self.nslc_ids = reuse((self.nslc_id,))
        
    def as_tuple(self):
        itmin = int(round(self.tmin*HPTMODULUS))
        itmax = int(round(self.tmax*HPTMODULUS))
        srate = num.float64(1.0)/num.float64(self.deltat)
        return (self.network, self.station, self.location, self.channel, 
                itmin, itmax, srate, self.ydata)

    def make_xdata(self):
        return self.tmin + num.arange(len(self.ydata), dtype=num.float64) * self.deltat

    def drop_data(self):
        self.ydata = None
    
    def chop(self, tmin, tmax, selector=None):
        
        if not self.is_relevant(tmin,tmax,selector): return None
        ibeg = max(0, t2ind(tmin-self.tmin,self.deltat))
        iend = min(len(self.ydata), t2ind(tmax-self.tmin,self.deltat))
        #if ibeg == iend: return None
        tracecopy = copy.copy(self)
        tracecopy.ydata = self.ydata[ibeg:iend].copy()
        tracecopy.tmin = self.tmin+ibeg*self.deltat
        tracecopy.tmax = tracecopy.tmin+(len(tracecopy.ydata)-1)*tracecopy.deltat
        return tracecopy
        
    def copy(self):
        tracecopy = copy.copy(self)
        tracecopy.ydata = self.ydata.copy()
        return tracecopy
        
    def downsample(self, ndecimate):
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = decimate(data, ndecimate, ftype='fir')
      #  self.tmin = self.tmin + ndecimate*self.deltat/2.
        self.deltat = reuse(self.deltat*ndecimate)
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        
    def downsample_to(self, deltat):
        ratio = deltat/self.deltat
        rratio = round(ratio)
        if abs(rratio - ratio) > 0.0001: raise UnavailableDecimation('ratio = %g' % ratio)
        deci_seq = decitab(int(rratio))
        for ndecimate in deci_seq:
             if ndecimate != 1:
                self.downsample(ndecimate)
            
    def lowpass(self, order, corner):
        (b,a) = signal.butter(order, corner*2.0*self.deltat, btype='low')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = signal.lfilter(b,a, data)
        
    def highpass(self, order, corner):
        (b,a) = signal.butter(order, corner*2.0*self.deltat, btype='high')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = signal.lfilter(b,a, data)
        
    def bandpass(self, order, corner_hp, corner_lp):
        (b,a) = signal.butter(order, [corner*2.0*self.deltat for corner in (corner_hp, corner_lp)], btype='band')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = signal.lfilter(b,a, data)
        
    def bandpass_fft(self, corner_hp, corner_lp):
        data = self.ydata.astype(num.float64)
        n = len(data)
        fdata = num.fft.rfft(data)
        nf = len(fdata)
        df = 1./(n*self.deltat)
        freqs = num.arange(nf)*df
        fdata *= num.logical_and(corner_hp < freqs, freqs < corner_lp)
        data = num.fft.irfft(fdata,n)
        assert len(data) == n
        self.ydata = data
        
    def shift(self, tshift):
        self.tmin += tshift
        self.tmax += tshift
        
    def sta_lta_centered(self, tshort, tlong, quad=True):
    
        nshort = tshort/self.deltat
        nlong = tlong/self.deltat
    
        if quad:
            sqrdata = self.ydata**2
        else:
            sqrdata = self.ydata
    
        mavg_short = moving_avg(sqrdata,nshort)
        mavg_long = moving_avg(sqrdata,nlong)
    
        self.ydata = num.maximum((mavg_short/mavg_long - 1.) * float(nshort)/float(nlong), 0.0)
        
    def peaks(self, threshold, tsearch):
        y = self.ydata
        above =  num.where(y > threshold, 1, 0)
        itrig_positions = num.nonzero((above[1:]-above[:-1])>0)[0]
        tpeaks = []
        apeaks = []
        for itrig_pos in itrig_positions:
            ibeg = max(0,itrig_pos - 0.5*tsearch/self.deltat)
            iend = min(len(self.ydata)-1, itrig_pos + 0.5*tsearch/self.deltat)
            ipeak = num.argmax(y[ibeg:iend])
            tpeak = self.tmin + (ipeak+ibeg)*self.deltat
            apeak = y[ibeg+ipeak]
            tpeaks.append(tpeak)
            apeaks.append(apeak)
            
        return tpeaks, apeaks
        
    def transfer(self, tfade, freqlimits, transfer_function=None, cut_off_fading=True):
        '''Return new trace with transfer function applied.
        
        tfade -- rise/fall time in seconds of taper applied in timedomain at both ends of trace.
        freqlimits -- 4-tuple with corner frequencies in Hz.
        transfer_function -- FrequencyResponse object; must provide a method 'evaluate(freqs)', which returns the
                             transfer function coefficients at the frequencies 'freqs'.
        cut_off_fading -- cut off rise/fall interval in output trace.
        '''
    
        if transfer_function is None:
            transfer_function = FrequencyResponse()
    
        if self.tmax - self.tmin <= tfade*2.:
            raise TraceTooShort('trace too short for fading length setting. trace length = %g, fading length = %g' % (self.tmax-self.tmin, tfade))

        ndata = self.ydata.size
        ntrans = nextpow2(ndata*1.2)
        coefs = self._get_tapered_coefs(ntrans, freqlimits, transfer_function)
        
        data = self.ydata
        data_pad = num.zeros(ntrans, dtype=num.float)
        data_pad[:ndata]  = data - data.mean()
        data_pad[:ndata] *= costaper(0.,tfade, self.deltat*(ndata-1)-tfade, self.deltat*ndata, ndata, self.deltat)
        fdata = num.fft.rfft(data_pad)
        fdata *= coefs
        ddata = num.fft.irfft(fdata)
        output = self.copy()
        output.ydata = ddata[:ndata]
        if cut_off_fading:
            output = output.chop(output.tmin+tfade, output.tmax-tfade)
        else:
            output.ydata = output.ydata.copy()
        return output
        
    def _get_tapered_coefs(self, ntrans, freqlimits, transfer_function):
    
        deltaf = 1./(self.deltat*ntrans)
        nfreqs = ntrans/2 + 1
        transfer = num.ones(nfreqs, dtype=num.complex)
        hi = snapper(nfreqs, deltaf)
        a,b,c,d = freqlimits
        freqs = num.arange(hi(d)-hi(a), dtype=num.float)*deltaf + hi(a)*deltaf
        transfer[hi(a):hi(d)] = transfer_function.evaluate(freqs)
        
        tapered_transfer = costaper(a,b,c,d, nfreqs, deltaf)*transfer
        return tapered_transfer
        
    def fill_template(self, template):
        params = dict(zip( ('network', 'station', 'location', 'channel'), self.nslc_id))
        params['tmin'] = gmctime_fn(self.tmin)
        params['tmax'] = gmctime_fn(self.tmax)
        return template % params
        
    def __str__(self):
        s = 'MSeedTrace (%s, %s, %s, %s)\n' % self.nslc_id
        s += '  timerange: %s - %s\n' % (gmctime(self.tmin), gmctime(self.tmax))
        s += '  delta t: %g\n' % self.deltat
        return s
    
    
class MSeedFile(MSeedGroup):
    def __init__(self, abspath, mtime, substitutions=None):
        self.abspath = abspath
        self.mtime = mtime
        self.traces = []
        self.data_loaded = False
        self.substitutions = substitutions
        self.load_headers()
        
    def load_headers(self):
        
        for tr in get_traces( self.abspath, False ):
            trace = MSeedTrace(tr, self.substitutions)
            self.traces.append(trace)
        self.data_loaded = False
        self.update_from_contents(self.traces)
        
    
    def load_data(self):
        logging.info('loading data from file: %s' % self.abspath)
        self.traces = []
        for tr in get_traces( self.abspath, True ):
            trace = MSeedTrace(tr, self.substitutions)
            self.traces.append(trace)
        self.data_loaded = True
        self.update_from_contents(self.traces)
        
    def drop_data(self):
        logging.info('forgetting data of file: %s' % self.abspath)
        for tr in self.traces:
            tr.drop_data()
        self.data_loaded = False
            
    def chop(self,tmin,tmax,selector):
        
        chopped = []
        for trace in self.traces:
            chopped_trace = trace.chop(tmin,tmax,selector)
            if chopped_trace is not None:
                chopped_trace.mtime = self.mtime
                chopped.append(chopped_trace)
            
        return chopped
        
    def get_deltats(self):
        deltats = set()
        for trace in self.traces:
            deltats.add(trace.deltat)
            
        return deltats
    
    def iter_traces(self):
        for trace in self.traces:
            yield trace
    
    def gather_keys(self, gather):
        keys = set()
        for trace in self.traces:
            keys.add(gather(trace))
            
        return keys
    
    def __str__(self):
        
        def sl(s):
            return sorted(list(s))
        
        s = 'MSeedFile\n'
        s += 'abspath: %s\n' % self.abspath
        s += 'file mtime: %s\n' % gmctime(self.mtime)
        s += 'number of traces: %i\n' % len(self.traces)
        s += 'timerange: %s - %s\n' % (gmctime(self.tmin), gmctime(self.tmax))
        s += 'networks: %s\n' % ', '.join(sl(self.networks))
        s += 'stations: %s\n' % ', '.join(sl(self.stations))
        s += 'locations: %s\n' % ', '.join(sl(self.locations))
        s += 'channels: %s\n' % ', '.join(sl(self.channels))
        return s
    
def load_cache(cachefilename):
        
    if os.path.isfile(cachefilename):
        progress_beg('reading cache...')
        f = open(cachefilename,'r')
        cache = pickle.load(f)
        f.close()
        progress_end()
    else:
        cache = {}
        
    # weed out files which no longer exist
    progress_beg('weeding cache...')
    for fn in cache.keys():
        if not os.path.isfile(fn):
            del cache[fn]
    
    progress_end()
            
    return cache

def dump_cache(cache, cachefilename):
    
    progress_beg('writing cache...')
    f = open(cachefilename+'.tmp','w')
    pickle.dump(cache, f)
    f.close()
    os.rename(cachefilename+'.tmp', cachefilename)
    progress_end()
    
class FilenameAttributeError(Exception):
    pass

class MSeedPile(MSeedGroup):
    def __init__(self, filenames, cachefilename=None, filename_attributes=None):
        msfiles = []
        
        if filenames:
            # should lock cache here...
            if cachefilename:
                cache = load_cache(cachefilename)
            else:
                cache = {}
                
            if config.show_progress:
                widgets = ['Scanning files', ' ',
                        progressbar.Bar(marker='-',left='[',right=']'), ' ',
                        progressbar.Percentage(), ' ',]
                
                pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(filenames)).start()
            
            regex = None
            if filename_attributes:
               regex = re.compile(filename_attributes)
            
            failures = []
            cache_modified = False
            for ifile, filename in enumerate(filenames):
                try:
                    abspath = os.path.abspath(filename)
                    
                    substitutions = None
                    if regex:
                        m = regex.search(filename)
                        if not m: raise FilenameAttributeError(
                            "Cannot get attributes with pattern '%s' from path '%s'" 
                                % (filename_attributes, filename))
                        substitutions = m.groupdict()
                        
                    mtime = os.stat(filename)[8]
                    if abspath not in cache or cache[abspath].mtime != mtime or substitutions:
                        cache[abspath] = MSeedFile(abspath, mtime, substitutions)
                        if not substitutions:
                            cache_modified = True
                        
                except (MSEEDERROR, OSError, FilenameAttributeError), xerror:
                    failures.append(abspath)
                    logging.warn(xerror)
                else:
                    msfiles.append(cache[abspath])
               
                if config.show_progress: pbar.update(ifile+1)
            
            if config.show_progress: pbar.finish()
            if failures:
                logging.warn('The following file%s caused problems and will be ignored:\n' % plural_s(len(failures)) + '\n'.join(failures))
            
            if cachefilename and cache_modified: dump_cache(cache, cachefilename)
        
            # should unlock cache here...
        
        self.msfiles = msfiles
        self.update_from_contents(self.msfiles)
        self.open_files = set()
        
    def chopper(self, tmin=None, tmax=None, tinc=None, tpad=0., selector=None, 
                      want_incomplete=True, degap=True, keep_current_files_open=False,
                      ndecimate=None):
        
        if tmin is None:
            tmin = self.tmin+tpad
                
        if tmax is None:
            tmax = self.tmax-tpad
            
        if tinc is None:
            tinc = tmax-tmin
        
        if not self.is_relevant(tmin,tmax,selector): return
        
        files_match_full = [ f for f in self.msfiles if f.is_relevant( tmin-tpad, tmax+tpad, selector ) ]
        
        if not files_match_full: return
        
        ftmin = num.inf
        ftmax = -num.inf
        for f in files_match_full:
            ftmin = min(ftmin,f.tmin)
            ftmax = max(ftmax,f.tmax)
        
        iwin = max(0, int(((ftmin-tpad)-tmin)/tinc-2))
        files_match_partial = files_match_full
        
        partial_every = 50
        
        while True:
            chopped = []
            wmin, wmax = tmin+iwin*tinc, tmin+(iwin+1)*tinc
            if wmin >= ftmax or wmin >= tmax: break
                        
            if iwin%partial_every == 0:  # optimization
                swmin, swmax = tmin+iwin*tinc, tmin+(iwin+partial_every)*tinc
                files_match_partial = [ f for f in files_match_full if f.is_relevant( swmin-tpad, swmax+tpad, selector ) ]
                
            files_match_win = [ f for f in files_match_partial if f.is_relevant( wmin-tpad, wmax+tpad, selector ) ]
            
            if files_match_win:
                used_files = set()
                for file in files_match_win:
                    used_files.add(file)
                    if not file.data_loaded:
                        self.open_files.add(file)
                        file.load_data()
                    chopped.extend( file.chop(wmin-tpad, wmax+tpad, selector) )
                                
                chopped.sort(lambda a,b: cmp(a.full_id, b.full_id))
                if degap:
                    chopped = degapper(chopped)
                    
                if not want_incomplete:
                    wlen = (wmax+tpad)-(wmin-tpad)
                    chopped_weeded = []
                    for trace in chopped:
                        if abs(wlen - round(wlen/trace.deltat)*trace.deltat) > 0.001:
                            logging.warn('Selected window length (%g) not nicely divideable by sampling interval (%g).' % (wlen, trace.deltat) )
                        if len(trace.ydata) == t2ind((wmax+tpad)-(wmin-tpad), trace.deltat):
                            chopped_weeded.append(trace)
                    chopped = chopped_weeded
                
                if ndecimate is not None:
                    for trace in chopped:
                        trace.downsample(ndecimate)
                    
                yield chopped
                
                unused_files = self.open_files - used_files
                for file in unused_files:
                    file.drop_data()
                    self.open_files.remove(file)
                
            
            iwin += 1
        
        if not keep_current_files_open:
            while self.open_files:
                file = self.open_files.pop()
                file.drop_data()
            
        
    def all(self, *args, **kwargs):
        alltraces = []
        for traces in self.chopper( *args, **kwargs ):
            alltraces.extend( traces )
            
        return alltraces
        
    def iter_all(self, *args, **kwargs):
        for traces in self.chopper( *args, **kwargs):
            for trace in traces:
                yield trace
            
    def gather_keys(self, gather):
        keys = set()
        for file in self.msfiles:
            keys |= file.gather_keys(gather)
            
        return sorted(keys)
    
    def get_deltats(self):
        deltats = set()
        for file in self.msfiles:
            deltats.update(file.get_deltats())
        return sorted(list(deltats))
    
    def iter_traces(self, load_data=False):
        for file in self.msfiles:
            
            must_close = False
            if load_data and not file.data_loaded:
                file.load_data()
                must_close = True
            
            for trace in file.iter_traces():
                yield trace
            
            if must_close:
                file.drop_data()
               
    def __str__(self):
        
        def sl(s):
            return sorted([ x for x in s ])
        
        s = 'MSeedPile\n'
        s += 'number of files: %i\n' % len(self.msfiles)
        s += 'timerange: %s - %s\n' % (gmctime(self.tmin), gmctime(self.tmax))
        s += 'networks: %s\n' % ', '.join(sl(self.networks))
        s += 'stations: %s\n' % ', '.join(sl(self.stations))
        s += 'locations: %s\n' % ', '.join(sl(self.locations))
        s += 'channels: %s\n' % ', '.join(sl(self.channels))
        return s
    
    
    
class Anon:
    def __init__(self,dict):
        for k in dict:
            self.__dict__[k] = dict[k]


def select_files( paths, selector=None,  regex=None ):

    progress_beg('selecting files...')
    if logging.getLogger().isEnabledFor(logging.DEBUG): sys.stderr.write('\n')

    good = []
    if regex: rselector = re.compile(regex)

    def addfile(path):
        
        
        if regex:
            logging.debug("looking at filename: '%s'" % path) 
            m = rselector.search(path)
            if m:
                infos = Anon(m.groupdict())
                logging.debug( "   regex '%s' matches." % regex)
                for k,v in m.groupdict().iteritems():
                    logging.debug( "      attribute '%s' has value '%s'" % (k,v) )
                if selector is None or selector(infos):
                    good.append(os.path.abspath(path))
                
            else:
                logging.debug("   regex '%s' does not match." % regex)
        else:
            good.append(os.path.abspath(path))
        
        
    for path in paths:
        if os.path.isdir(path):
            for (dirpath, dirnames, filenames) in os.walk(path):
                for filename in filenames:
                    addfile(pjoin(dirpath,filename))
        else:
            addfile(path)
        
    progress_end('%i file%s selected.' % (len( good), plural_s(len(good))))
    
    return good


def save(all_traces, filename_template):
    
    fn_tr = {}
    for trace in all_traces:
        fn = trace.fill_template(filename_template)
        if fn not in fn_tr:
            fn_tr[fn] = []
        
        fn_tr[fn].append(trace)
        
    for fn, traces in fn_tr.items():
        trtups = []
        traces.sort(lambda a,b: cmp(a.full_id, b.full_id))
        for trace in traces:
            trtups.append(trace.as_tuple())
            
        store_traces(trtups, fn)
        
    return fn_tr.keys()

import unittest
class MSeedTestCase( unittest.TestCase ):
    
    def testWriteRead(self):
        import tempfile
        n = 1000
        deltat = 0.1
        tmin = calendar.timegm( (2008,2,2,0,0,0) )
        itmin = int(round(tmin*HPTMODULUS))
        tmax = tmin + (n-1)*deltat
        itmax = int(round(tmax*HPTMODULUS))
        freq = 1./deltat
        data = num.arange(n)
        tr = ('NETWORK', 'STATION', 'LOCATION', 'CHANNEL', itmin, itmax, freq, data)
        tempfn = tempfile.mkstemp()[1]
        store_traces([tr], tempfn)
        tr2 = get_traces(tempfn, True)[0]
        os.unlink(tempfn)
        assert tr[:-1] == tr2[:-1], 'MSeed trace headers not identical.'
        assert num.all(tr[-1] == tr[-1]), 'MSeed trace data not identical.'
    
    def testPileTraversal(self):
        import tempfile, shutil
        config.show_progress = False
        nfiles = 200
        nsamples = 100000
        datadir = self.makeManyFiles(nfiles=nfiles, nsamples=nsamples)
        filenames = select_files([datadir])
        cachefilename = pjoin(datadir,'_cache_')
        pile = MSeedPile(filenames, cachefilename)
        s = 0
        for traces in pile.chopper(tmin=None, tmax=None, tinc=1234.): #tpad=10.):
            for trace in traces:
                s += num.sum(trace.ydata)
                
        os.unlink(cachefilename)
        shutil.rmtree(datadir)
        assert s == nfiles*nsamples
    
    def makeManyFiles(self, nfiles=200, nsamples=100000):
        import tempfile
        import random
        from random import choice as rc

        abc = 'abcdefghijklmnopqrstuvwxyz' 
        
        def rn(n):
            return ''.join( [ random.choice(abc) for i in xrange(n) ] )
        
        stations = [ rn(4) for i in xrange(10) ]
        components = [ rn(3) for i in xrange(3) ]
        networks = [ 'xx' ]
        
        datadir = tempfile.mkdtemp()
        for i in xrange(nfiles):
            tbeg = 1234567890+i*60*60*24*10 # random.randint(1,int(time.time()))
            srate = 1.0
            tend = tbeg + (1.0/srate)*(nsamples-1)
            data = num.ones(nsamples)
            trtup = (rc(networks),rc(stations),'',rc(components), 
                     int(tbeg*HPTMODULUS), int(tend*HPTMODULUS), srate, data)
            fn = pjoin( datadir, '%s_%s_%s_%s_%s.mseed' % (trtup[:4]+(rn(5),)))
            store_traces([trtup], fn)
        
        return datadir

    
if __name__ == '__main__':
    unittest.main()
    
