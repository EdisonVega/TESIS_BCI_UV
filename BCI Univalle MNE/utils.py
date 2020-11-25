import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne

from scipy.signal import butter, sosfilt, lfilter, welch
mne.set_log_level('WARNING')


#Funcion filtro pasabanda 4to orden
def butter_bandpass(lowcut, highcut, fs, order=4): 

    nyq = 0.5 * fs 
    low = lowcut/nyq 
    high = highcut/nyq 
    b, a = butter(order, [low, high], btype='band') 

    return b, a 

def butter_bandpass_filter(data, banda, Fs, order=4): 

    b, a = butter_bandpass(banda[0],banda[1], fs=Fs, order=order) 
    y = lfilter(b, a, data) 
  
    return y


# Filtro FIR pasabanda Butter
def bandpass_filter_butter_IIR(X, Fs, banda, order=4): 

    low  = banda[0]
    high = banda[1]
        
    sos = butter(order,[low, high], btype='band', output='sos', fs=Fs)         
    X   = sosfilt(sos, X) 
  
    return X

# Filtro IIR padabanda con MNE por defecto
def bandpass_filter_mne(X,Fs,banda,phase='zero'):
    
    X = mne.filter.filter_data(X,Fs,banda[0],banda[1], phase=phase, verbose=0)
    
    return X

# Aplica filtro butter 4 orden por columnas en un DataFrame
def filtrar_dataframes(data=None, fs=250, bandas=None, canales=None, metodo='butter'):
        
    X=[]  
    for i in range(len(data)):
        
        df = pd.DataFrame()  
        
        for canal in canales:
            for banda in bandas:                        
                
                if metodo=='butter':
                    df['{}{}'.format(canal,banda)] = bandpass_filter_butter_IIR(data[i][canal].copy().values, fs, banda, order=4)
                elif metodo=='mne':
                    df['{}{}'.format(canal,banda)] = bandpass_filter_mne(data[i][canal].copy().values, fs, banda)
                elif metodo=='butter_ab':
                    df['{}{}'.format(canal,banda)] = butter_bandpass_filter(data[i][canal].copy().values, banda, fs, order=4)
                    
        X.append(df)
    
    return X

# Cortar DF segun la ventana requerida
def recortar_dataframes(X, Fs, ventana):
    
    t_min = ventana[0]  
    t_max = ventana[1]
    
    for i in range(len(X)):
        X[i] = X[i].iloc[int((t_min+3)*Fs):int((t_max+3)*Fs)]
    return X

# FFT de un solo lado
def FFT_onesided(X,Fs,nfft=None,in_Hz=False,plot=False):
    
    Y = np.fft.fft(X,n=nfft)/len(X)

    Y=np.abs(Y)
    Y=np.fft.fftshift(Y)
    Y=Y[len(Y)//2:len(Y)]*2

    w=np.fft.fftfreq(len(Y)*2)
    w=np.fft.fftshift(w)
    w=w[len(w)//2:len(w)]

    if in_Hz==True:
        w=w*Fs
    
    if plot==True:
        plt.plot(w, Y)
        plt.show()
        
    return w,Y

def PSD_def(X, Fs, in_Hz=False, get_w=False, **kwds):
    
    w,Y = FFT_onesided(X,Fs,in_Hz=in_Hz) 
    Y = Y/np.sum(Y)

    if get_w==False:
        return Y
    else:
        return w,Y

def PSD_multitapper(X, Fs, fmin=0, fmax=40, bandwidth=None, get_w=False, **kwds):
    
    Y,w = mne.time_frequency.psd_array_multitaper(X,sfreq=Fs,fmin=fmin,fmax=fmax,bandwidth=bandwidth,normalization='full', verbose=False)
    
    if get_w==False:
        return Y
    else:
        return w,Y

def PSD_welch_mne(X, Fs, fmin=0, fmax=np.inf, nfft=256, nperseg=256, noverlap=128, average='median', get_w=False, **kwds):
    
    X = X.values
    
    Y,w = mne.time_frequency.psd_array_welch(X,sfreq=Fs,fmin=fmin,fmax=fmax, n_fft=nfft,n_per_seg=nperseg,n_overlap=noverlap, average=average)
    
    if get_w==False:
        return Y
    else:
        return w,Y

def PSD_welch_scipy(X, Fs, nfft=256, nperseg=256, noverlap=128, average='median', get_w=False, **kwds):
    
    w,Y = welch(X, Fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, average=average)
    
    if get_w==False:
        return Y
    else:
        return w,Y

def PSD_welch_old(X, get_w=False, **kwds):
    
    w,Y = welch(X)
    Y = Y/np.sum(Y)
    
    if get_w==False:
        return Y
    else:
        return w,Y

def entropia_(PSD):
    
    H = -np.sum(PSD*np.log(PSD))
    
    return H
    
# Caracteristicas temporales ======================================================
def root_mean_square(X):    
    
    Y = np.sqrt(np.mean(X**2))

    return Y

def valor_abs_medio(X):
    
    Y = np.mean(np.abs(X))
    
    return Y

def integral(X):
    
    Y = np.sum(X)
    
    return Y

def integral_abs(X):
    
    Y = np.sum(np.abs(X))
    
    return Y

def integral_cuadrada(X):
    
    Y = np.sum(X**2)

    return Y

def cambio_amplitud_prom(X):
    
    Xn = np.pad(X,(1,0),constant_values=0)[1:-1]
    Xn1= np.pad(X,(0,1),constant_values=0)[1:-1]
    Y = np.mean(np.abs(Xn1-Xn))
    
    return Y

def varianza(X):
    
    Y = np.var(X)
    
    return Y

def var_log(X):
    
    Y = np.log(np.var(X))
    
    return Y

def media(X):
    
    Y = np.mean(X)
    
    return Y

def desviacion_std(X):
    
    Y = np.std(X)
    
    return Y

def var_coef(X):
    
    Y = np.std(X)/np.abs(np.mean(X))
    
    return Y

def energia_(X):
    
    Y = np.sum(X**2)

    return Y

