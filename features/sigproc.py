##@package sigproc 
#contains the signal processing functionality

#The MIT License (MIT)
#
#Copyright (c) 2013 James Lyons
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
#the Software, and to permit persons to whom the Software is furnished to do so,
#subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012

import numpy
import math

##Frame a signal into overlapping frames.
#
#@param sig the audio signal to frame.
#@param frame_len length of each frame measured in samples.
#@param frame_step number of samples after the start of the previous frame that the next frame should begin.
#@param winfunc the analysis window to apply to each frame. By default no window is applied.    
#
#@return an array of frames. Size is NUMFRAMES by frame_len.
def framesig(sig,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):
    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len: 
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))
        
    padlen = int((numframes-1)*frame_step + frame_len)
    
    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig,zeros))
    
    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len),(numframes,1))
    return frames*win
    
##Does overlap-add procedure to undo the action of framesig. 
#
#@param frames the array of frames.
#@param siglen the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.    
#@param frame_len length of each frame measured in samples.
#@param frame_step number of samples after the start of the previous frame that the next frame should begin.
#@param winfunc the analysis window to apply to each frame. By default no window is applied. 
#   
#@return a 1-D signal.
def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
 
    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    padlen = (numframes-1)*frame_step + frame_len   
    
    if siglen <= 0: siglen = padlen
    
    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)
    
    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]
        
    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]
    
##Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 
#
#@param frames the array of frames. Each row is a frame.
#@param NFFT the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
#
#@return If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
def magspec(frames,NFFT):
    complex_spec = numpy.fft.rfft(frames,NFFT)
    return numpy.absolute(complex_spec)
          
##Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 
#
#@param frames the array of frames. Each row is a frame.
#@param NFFT the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
#
#@return If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
def powspec(frames,NFFT):   
    return 1.0/NFFT * numpy.square(magspec(frames,NFFT))
    
##Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 
#
#@param frames the array of frames. Each row is a frame.
#@param NFFT the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
#@param norm If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
#
#@return If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.  
def logpowspec(frames,NFFT,norm=1): 
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps
    
##perform preemphasis on the input signal.
#    
#@param signal The signal to filter.
#@param coeff The preemphasis coefficient. 0 is no filter, default is 0.95.
#
#@return the filtered signal.    
def preemphasis(signal,coeff=0.95):
    return numpy.append(signal[0],signal[1:]-coeff*signal[:-1])



