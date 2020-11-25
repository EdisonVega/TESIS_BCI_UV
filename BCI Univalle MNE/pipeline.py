import numpy as np
import time
import mne
import pandas as pd
import pickle
import os

from utils import *

mne.set_log_level('ERROR')

class modelCreator():

    def __init__(self, fs = 128, dataFrom = "emotiv"):

        print("Inicializando ModelCreator")
        self.fs     = fs
        self.epochs = None
        self.cars   = None
        self.carsNF = []
        self.model  = []
        self.X_train= None
        self.y_train= None
        self.X_test = None
        self.y_Test = None
        self.dataFrom = dataFrom
        
        if(self.dataFrom == "emotiv"):
            self.Columns_Nuevas   = ['Eventos', 'Tiempo', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        elif(self.dataFrom != "BCICompIV2b"):
            raise TypeError("el argumento dataFrom debe ser 'emotiv' o 'BCICompIV2b'")
            
    def epoching(self, data = None, bandpass_filter = True, banda=(0.5,40),baseline_remove=None, parametros = None):
        
        tmin         = parametros['tmin']         
        tmax         = parametros['tmax']   
        secondsPreCue= parametros['secondsPreCue']      
        channels     = parametros['channels']     
        picked       = parametros['picked']        
        montage      = parametros['montage']      
        descripcion  = parametros['descripcion']  
        fs           = self.fs

        if(self.dataFrom == "emotiv"):
            Columns_Original = data.columns.tolist()
            Columns_Nuevas   = self.Columns_Nuevas

            ColRenameDict    = {} 
            for key in Columns_Original: 
                for value in Columns_Nuevas: 
                    ColRenameDict[key] = value 
                    Columns_Nuevas.remove(value) 
                    break 

            data = data.rename(columns=ColRenameDict)

            print(data.columns)
            
            if bandpass_filter == True:
                for channel in channels:
                    x = data[channel].values.tolist()
                    y = bandpass_filter_butter_IIR(x, fs, banda, order=4)
                    #y = butter_bandpass_filter(x, fs, banda, order=4)
                    data[channel] = y
            else:
                for channel in channels:  
                    x = data[channel].values.tolist()
                    y = x
                    data[channel] = y
            
            # Creando MNE Info
            numChannels   = len(channels)
            channel_types = ['eeg']*numChannels
            
            info          = mne.create_info(channels, fs, channel_types)
            info.set_montage(montage)
            info['description'] = descripcion
                
            #Creando los indices de los eventos  # Se considera el inicio del try en el momento -3 segundos
            eventos       = data['Eventos'].values
            idxEvents     = (np.where(eventos[:-1] != eventos[1:])[0]+1).tolist()
            idxLFCueEvent = [[i,0,1] for i in idxEvents if eventos[i] == 1]
            idxRFCueEvent = [[i,0,2] for i in idxEvents if eventos[i] == 2]
            events        = np.vstack([idxLFCueEvent,idxRFCueEvent])
            events_raw    = events[np.argsort(events[:, 0])]
            event_ids     = dict(izquierda=1, derecha=2)
                    
            # Creando datos RAW del DF original - "data debe tener shape: (n_channels, n_samples)"
            dataChannels = np.transpose(data[channels].values)
            raw          = mne.io.RawArray(dataChannels, info)

            # Canales a escoger
            picks = mne.pick_channels(raw.info['ch_names'], picked)
            
            if baseline_remove is None:
                baseline = None
            elif baseline_remove is tuple:
                baseline = (baseline_remove[0],baseline_remove[1])
            else:
                baseline = None
                
            epochs = mne.Epochs(raw, events_raw, event_ids, tmin, tmax - 1/fs, picks=picks, baseline=baseline)
            
            print(events_raw.shape)
            print(dataChannels.shape)
            print(epochs.get_data().shape)

            # Guardando los eventos
            columns = picked
            
            left_data  = epochs['izquierda'].get_data()
            rigth_data = epochs['derecha'].get_data()
            epoch_data = np.vstack((left_data,rigth_data))
            lista_epocas = []
            for i in range(len(epoch_data)):

                df = pd.DataFrame(data = epoch_data[i].T, columns = columns, index = None)
                df['Evento'] = events_raw[i,2]-1
                lista_epocas.append(df)

            self.epocas = lista_epocas
            return self.epocas

        else: 

            if bandpass_filter == True:
                for channel in channels:
                    x = data[channel].values.tolist()
                    y = bandpass_filter_butter_IIR(x, fs, banda, order=4)
                    #y = butter_bandpass_filter(x, fs, banda, order=4)
                    data[channel] = y
            else:
                for channel in channels:  
                    x = data[channel].values.tolist()
                    y = x
                    data[channel] = y
            
            # Creando MNE Info
            numChannels   = len(channels)
            channel_types = ['eeg']*numChannels
            
            info          = mne.create_info(channels, fs, channel_types)
            info.set_montage(montage)
            info['description'] = descripcion
                
            #Creando los indices de los eventos  
            
            # Se considera inicio del Try desde el segundo 0 del paradigma, se desplaza +secondsPrecue 
            idxLFCue      = [data[(data["Events"]==1)].index + (fs * secondsPreCue)][0].tolist()
            idxRFCue      = [data[(data["Events"]==2)].index + (fs * secondsPreCue)][0].tolist()
            idxLFCueEvent = [[i,0,1] for i in idxLFCue]
            idxRFCueEvent = [[i,0,2] for i in idxRFCue]
            events        = np.vstack([idxLFCueEvent,idxRFCueEvent])
            events_raw    = events.copy()
            event_ids     = dict(izquierda=1, derecha=2)
                    
            # Creando datos RAW del DF original - "data debe tener shape: (n_channels, n_samples)"
            dataChannels = np.transpose(data[channels].values)
            raw          = mne.io.RawArray(dataChannels, info)
                        
            # Canales a escoger
            picks = mne.pick_channels(raw.info['ch_names'], picked)
            
            if baseline_remove is None:
                baseline = None
            elif baseline_remove is tuple:
                baseline = (baseline_remove[0],baseline_remove[1])
            else:
                baseline = None
                
            epochs = mne.Epochs(raw, events_raw, event_ids, tmin, tmax - 1/fs, picks=picks, baseline=baseline)
            
            # Guardando los eventos
            columns = picked
            
            left_data  = epochs['izquierda'].get_data()
            rigth_data = epochs['derecha'].get_data()
            epoch_data = np.vstack((left_data,rigth_data))

            lista_epocas = []
            for i in range(len(epoch_data)):

                df = pd.DataFrame(data = epoch_data[i].T, columns = columns, index = None)
                df['Evento'] = events_raw[i,2]-1
                lista_epocas.append(df)

            self.epochs = lista_epocas
            return self.epochs

    def extraccion(self,epocas=None, ventana=None, bandas=None, labelsCars = None, canales= None):

        epocasFilt = filtrar_dataframes(epocas.copy(), self.fs, bandas=bandas, canales=canales, metodo='butter')
        epocasCut  = recortar_dataframes(epocasFilt.copy(), self.fs, ventana)

        rms        = []
        mav        = []
        ieeg       = []
        ieegabs    = []
        aac        = []
        var        = []
        logvar     = []
        mean       = []
        std        = []
        cvar       = []
        energia    = []
        maxpsd     = []
        varpsd     = []
        entropia   = []
        maxpsdW    = []
        varpsdW    = []
        entropiaW  = []
        maxpsdMT   = []
        varpsdMT   = []
        entropiaMT = []     
        
        for epoca in epocasCut:

            rms.append(epoca.apply(root_mean_square,axis=0))
            mav.append(epoca.apply(valor_abs_medio, axis=0))
            ieeg.append(epoca.apply(integral, axis=0))
            ieegabs.append(epoca.apply(integral_abs, axis=0))
            aac.append(epoca.apply(cambio_amplitud_prom, axis=0))
            var.append(epoca.apply(varianza, axis=0))
            logvar.append(epoca.apply(var_log, axis=0))
            mean.append(epoca.apply(media, axis=0))
            std.append(epoca.apply(desviacion_std, axis=0))
            cvar.append(epoca.apply(var_coef, axis=0))
            energia.append(epoca.apply(energia_, axis=0))                    

            if ('maxpsd' and 'varpsd' and 'entropia' in labelsCars):
                # PSD por definicion
                kwds = {'Fs':self.fs, 'in_Hz':False, 'get_w':False}
                psd_def = epoca.apply(PSD_def, axis=0, **kwds)

                maxpsd.append(psd_def.apply(np.max, axis=0))
                varpsd.append(psd_def.apply(varianza, axis=0))
                entropia.append(psd_def.apply(entropia_, axis=0))

            if ('maxpsdW' and 'varpsdW' and 'entropiaW' in labelsCars):
                # PSD Welch MNE
                kwds = {'Fs':self.fs, 'fmin':0, 'fmax':40, 'nfft':1024, 'nperseg':256, 'noverlap':128, 'average':'median', 'get_w':False}
                psd_W = epoca.apply(PSD_welch_mne, axis=0, **kwds)    

                maxpsdW.append(psd_W.apply(np.max, axis=0))
                varpsdW.append(psd_W.apply(varianza, axis=0))
                entropiaW.append(psd_W.apply(entropia_, axis=0))    

            if ('maxpsdMT' and'varpsdMT' and 'entropiaMT' in labelsCars):
                # PSD con Multitappers
                kwds = {'Fs':self.fs, 'fmin':0, 'fmax':40, 'bandwidth':None, 'get_w':False}
                psd_MT = epoca.apply(PSD_multitapper, axis=0, **kwds)

                maxpsdMT.append(psd_MT.apply(np.max, axis=0))
                varpsdMT.append(psd_MT.apply(varianza, axis=0))
                entropiaMT.append(psd_MT.apply(entropia_, axis=0))     


        rms        = pd.concat(rms, axis=1).transpose()
        mav        = pd.concat(mav, axis=1).transpose()
        ieeg       = pd.concat(ieeg, axis=1).transpose()
        ieegabs    = pd.concat(ieegabs, axis=1).transpose()
        aac        = pd.concat(aac, axis=1).transpose()
        var        = pd.concat(var, axis=1).transpose()
        logvar     = pd.concat(logvar, axis=1).transpose()
        mean       = pd.concat(mean, axis=1).transpose()
        std        = pd.concat(std, axis=1).transpose()
        cvar       = pd.concat(cvar, axis=1).transpose()
        energia    = pd.concat(energia, axis=1).transpose()    

        if ('maxpsd' and 'varpsd' and 'entropia' in labelsCars):
            maxpsd     = pd.concat(maxpsd, axis=1).transpose()
            varpsd     = pd.concat(varpsd, axis=1).transpose()
            entropia    = pd.concat(entropia, axis=1).transpose()
            
        if ('maxpsdW' and 'varpsdW' and 'entropiaW' in labelsCars):
            maxpsdW    = pd.concat(maxpsdW, axis=1).transpose()
            varpsdW    = pd.concat(varpsdW, axis=1).transpose()
            entropiaW   = pd.concat(entropiaW, axis=1).transpose()

        if ('maxpsdMT' and'varpsdMT' and 'entropiaMT' in labelsCars):
            maxpsdMT   = pd.concat(maxpsdMT, axis=1).transpose()
            varpsdMT   = pd.concat(varpsdMT, axis=1).transpose()
            entropiaMT  = pd.concat(entropiaMT, axis=1).transpose()

        cars = pd.concat([ rms      ,
                            mav      , 
                            ieeg     , 
                            ieegabs  , 
                            aac      , 
                            var      , 
                            logvar   , 
                            mean     , 
                            std      , 
                            cvar     , 
                            energia  ,], axis=1)  

        if ('maxpsd' and 'varpsd' and 'entropia' in labelsCars):
            cars=pd.concat([cars,maxpsd,varpsd,entropia], axis=1)
            
        if ('maxpsdW' and 'varpsdW' and 'entropiaW' in labelsCars):
            cars=pd.concat([cars,maxpsdW,varpsdW,entropiaW], axis=1)                

        if ('maxpsdMT' and'varpsdMT' and 'entropiaMT' in labelsCars):
            cars=pd.concat([cars,maxpsdMT,varpsdMT,entropiaMT], axis=1)   

        cars.columns = ['{}{}FB{}'.format(car,i,j) for car in labelsCars for i in canales for j in range(1,len(bandas)+1)]

        evento = []
        for i in range(0,len(epocas)):
            evento.append(epocas[i].at[epocas[i].index[0],'Evento'])
        cars['Evento']=evento

        self.cars = cars
        return self.cars



if __name__ == "__main__":

    # Dataset Selection
    dataFrom = "BCICompIV2b"         # "emotiv" o "BCICompIV2b"

    # Seleccion del sujeto
    sujeto  = "1"     # 1, 2, 3, ..., 9 para BCIComp, Nombre del sujeto para Emotiv
    sesion  = "Sesion 1"        # no usado para BCIComp
    setType = "T"               # T:Train , E:Evaluacion
    corridas= [1,2,3,4,5,6]     # no usado para BCIComp 

    #Save Options
    saveEpochs = True
    saveCars   = True

    ############################################## Pipeline para el Emotiv ##############################################
    if dataFrom=="emotiv":

        # Configuracion de directorios
        dataDir         = '.\\dataset\\Inputs\\Propio\\{}\\{}'.format(sujeto,sesion)
        outputsDir      = '.\\dataset\\Outputs\\Propio\\{}'.format(sujeto)
        os.makedirs(outputsDir, exist_ok=True)

        # Cargando la data
        listaCorridas  = []
        for i in corridas:
                inputFile = '.\\Corrida '+str(i)+'.csv'
                listaCorridas.append(pd.read_csv(dataDir+inputFile).loc[:,:])
        data = pd.concat(listaCorridas, ignore_index=True)

        # Pipeline BCI para entenamiento con Emotiv
        fs = 128 # Hz
        modelCreatorPipeline = modelCreator(fs=fs,dataFrom=dataFrom)

        # --------------------------------- Parametros del epoching ---------------------------------
        secondsPreCue = 2   # no usado con el Emotiv, poner cualquier entero
        tmin, tmax    = -2,6 
        channels      = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        pickedChannels= ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        montage       = 'biosemi64'
        descripcion   = "Dataset de train BCI Univalle"

        parametros    = {   'sampleFreq'    : fs,
                            'secondsPreCue' : secondsPreCue,
                            'tmin'     : tmin,
                            'tmax'     : tmax,
                            'channels' : channels,
                            'picked'   : pickedChannels,
                            'montage'  : montage,
                            'descripcion'  : descripcion}
    
        # Nombre Archivos de salida para las Epocas
        outputFileEpocas= outputsDir+'\\Epocas_{}_{}_({},{}s)_corridas_{}.bin'.format(sujeto,setType,tmin,tmax,corridas)
        
        # Epoching
        listaEpocas = modelCreatorPipeline.epoching(data,bandpass_filter=True,banda=(0.5,40),baseline_remove=None,parametros=parametros)        
        if saveEpochs:
            with open(outputFileEpocas, "wb") as fp:   #Pickling
                pickle.dump(listaEpocas, fp)

        # --------------------------------- Parametros de la extraccion ---------------------------------
        ventana = (1,6)  #Enventanado segundos segun el paradigma de adquisicion
        bandas  = [(8,16),(16,23),(23,31)]
        labels  = ['rms','mav','ieeg','ieegabs','aac','var','logvar','mean','std','cvar','energia','maxpsd','varpsd',
                   'entropia','maxpsdW','varpsdW','entropiaW','maxpsdMT','varpsdMT','entropiaMT']
        # Seleccion de canales - para seleccionar todos poner channelSel = channels
        channelSel  = ['O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] 

        # Nombre Archivos de salida para las Caracteristicas
        outputFileCars  = outputsDir+'\\Cars_{}_{}_ventana{}s_corridas_{}.csv'.format(sujeto,setType,ventana,corridas)  
        
        # Extraccion
        dfCars    = modelCreatorPipeline.extraccion(listaEpocas,ventana,bandas,labels,channelSel)
        if saveCars:
            dfCars.to_csv(outputFileCars,index=None)
    
    ########################################### Pipeline para BCICompetition ###########################################
    elif dataFrom=="BCICompIV2b":

        dataDir         = '.\\dataset\\Inputs\\BCICompetitonIV2b\\{}'.format(setType)
        outputsDir      = '.\\dataset\\Outputs\\BCICompetitonIV2b\\Sujeto_{}'.format(sujeto)
        os.makedirs(outputsDir, exist_ok=True)

        #Cargando la data    -- Pendiente cambiar 
        if setType == "T":
            inputFile = '\\B0'+str(sujeto)+'T.csv'
        elif setType == "E":
            inputFile = '\\B0'+str(sujeto)+'E.csv'
        data = pd.read_csv(dataDir+inputFile)
        

        # Pipeline BCI para entenamiento con BCICompetition
        fs            = 250 # Hz
        modelCreatorPipeline  = modelCreator(fs=fs,dataFrom=dataFrom)

        # # ----------------------------------- Parametros del epoching -----------------------------------
        secondsPreCue = 3   # no usado con el Emotiv, poner cualquier entero
        tmin, tmax    = -3,6 
        channels      = ['C3', 'Cz', 'C4']
        pickedChannels= ['C3', 'Cz', 'C4']
        montage       = 'biosemi64'
        descripcion   = "Dataset de train BCI competition IV 2B"

        parametros    = {   'sampleFreq'    : fs,
                            'secondsPreCue' : secondsPreCue,
                            'tmin'     : tmin,
                            'tmax'     : tmax,
                            'channels' : channels,
                            'picked'   : pickedChannels,
                            'montage'  : montage,
                            'descripcion'  : descripcion}

        # Nombre Archivos de salida para las Epocas
        outputFileEpocas= outputsDir+'\\Epocas_sujeto_{}_{}_({},{}s).bin'.format(sujeto,setType,tmin,tmax)

        # Epoching
        listaEpocas = modelCreatorPipeline.epoching(data,bandpass_filter=True,banda=(0.5,40),baseline_remove=None,parametros=parametros)        
        if saveEpochs:
            with open(outputFileEpocas, "wb") as fp:   #Pickling
                pickle.dump(listaEpocas, fp)

        # --------------------------------- Parametros de la extraccion ---------------------------------
        ventana = (1,6)  #Enventanado segundos segun el paradigma de adquisicion
        bandas  = [(8,16),(16,23),(23,31)]
        labels  = ['rms','mav','ieeg','ieegabs','aac','var','logvar','mean','std','cvar','energia','maxpsd','varpsd',
                   'entropia','maxpsdW','varpsdW','entropiaW','maxpsdMT','varpsdMT','entropiaMT']
        # Seleccion de canales - para seleccionar todos poner channelSel = channels
        channelSel  = ['C3', 'C4']
        
        # Nombre Archivos de salida para las Caracteristicas
        outputFileCars  = outputsDir+'\\Cars_sujeto{}_{}_ventana{}s.csv'.format(sujeto,setType,ventana)  

        # Extraccion
        dfCars    = modelCreatorPipeline.extraccion(listaEpocas,ventana,bandas,labels,channelSel)
        if saveCars:
            dfCars.to_csv(outputFileCars,index=None)



    