#Miscellaneous Imports
import time
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sn
import argparse
import random
import math
import threading

#OSC Imports
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

#lsl custom Imports with resolve stream timeout
from pylsl_custom import StreamInlet, resolve_stream, local_clock

#PyQt5 Imports
from PyQt5 import QtGui, QtWidgets, QtCore, uic
from PyQt5 import QtCore, QtWidgets
#from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QPushButton, QMessageBox

#Matplotlib Imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

#Sklearn Imports
import sklearn.utils._cython_blas
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression

#Scipy Imports
from scipy import signal
from scipy.signal import butter, sosfilt


#GUI Object Creation
qtCreatorFile = "MainWindow.ui" # Nombre del archivo aquí.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        #Variables del python OSC
        self.unityStartFlag  = False

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--serverip", default="127.0.0.1", help="The ip to listen on")
        self.parser.add_argument("--serverport", type=int, default=6161, help="The port the OSC Server is listening on")
        self.parser.add_argument("--clientip", default="127.0.0.1", help="The ip of the OSC server")
        self.parser.add_argument("--clientport", type=int, default=6969, help="The port the OSC Client is listening on")
        self.args   = self.parser.parse_args()

        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/unityStartFlag", self.setUnityStartFlag)

        # Declaracion e inicializacion de servidor
        print("Starting Server")
        self.server = osc_server.ThreadingOSCUDPServer((self.args.serverip, self.args.serverport), self.dispatcher)
        print("Serving on {}".format(self.server.server_address))
        self.OSCStartServer()

        # Declaracion de cliente
        print("Starting Client")
        self.client = udp_client.SimpleUDPClient(self.args.clientip, self.args.clientport)
        print("Sending on {}".format(self.client._address))

        #Variables de los modelos
        self.listaNomModelos = []    
        self.listaDirModelos = []  
        self.modelPath       = None
        self.model           = None

        #Variables de la captura    
        self.tInicio    = None
        self.tCue       = None
        self.tIM        = None
        self.tFinal     = None
        
        self.totalIter  = None  
        self.exitCaptura= False
        self.bufferData = None
        self.cueList    = []
        self.totSamples = None
        self.nSamples   = None
        self.nIter      = None

        #Variables LSL
        self.strams = None
        self.inlet  = None
        self.streamInfo = None

        #Variables de la señal adquirida
        self.eventos    = []       # Organizacion del vector de eventos 0,1/2,-1,0  (2,1,4,2)

        #Variables del epoching
        self.fs = 128
        self.secondsPreCue = 2
        self.tmin, self.tmax    = -2,6 
        self.channels      = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.pickedChannels= ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.montage       = 'biosemi64'
        self.descripcion   = "Dataset de train BCI Univalle"

        self.parametros    = {  'sampleFreq'    : self.fs,
                                'secondsPreCue' : self.secondsPreCue,
                                'tmin'     : self.tmin,
                                'tmax'     : self.tmax,
                                'channels' : self.channels,
                                'picked'   : self.pickedChannels,
                                'montage'  : self.montage,
                                'descripcion'  : self.descripcion}
    
        #Variables de la extraccion

        #Variables de la clasificacion

        #Variables de la interfaz con unity


        #Inicializacion UI
        self.onlyInt = QtGui.QRegExpValidator(QtCore.QRegExp("[0-9]{0,2}"))
        self.lineEditIteraciones.setValidator(self.onlyInt)
        self.MplWidget.canvas.axes.set_visible(False)
        self.listModels()

        #Inicializacion de botones
        self.pushButtonStart.clicked.connect(self.iniciarCapturaControl)
        self.pushButtonInfo.clicked.connect(self.mostrarInfo)
        self.pushButtonClear.clicked.connect(self.limpiarInterfaz)
        self.pushButtonLoadModel.clicked.connect(self.loadModel)

    # funciones de OSC
    def OSCStartServer(self):

        thread = threading.Thread(target=self.server.serve_forever)
        thread.start()

    def setUnityStartFlag(self, address, args):
        
        if(args==1):
            self.unityStartFlag = True
        else:
            self.unityStartFlag = False

    def listModels(self):

        self.listaNomModelos = [""]    
        self.listaDirModelos = [""]    
        for file_name in glob.iglob('.\models\*.sav', recursive=True):
            self.listaNomModelos.append(file_name[9:-4])
            self.listaDirModelos.append(file_name)

        self.comboBoxModelos.clear() 
        self.comboBoxModelos.insertItems(0,self.listaNomModelos)

    def loadModel(self):

        currentIndex = self.comboBoxModelos.currentIndex()
        if(currentIndex==0):
            self.showWarning("Atención","Seleccione un modelo de la lista.")
        else:           
            self.modelPath  = self.listaDirModelos[currentIndex]
        print(self.modelPath)
    
    # Control del hilo de captura
    def iniciarCapturaControl(self):

        if(not self.unityStartFlag):

            self.pushButtonStart.setStyleSheet(("QPushButton{background-color: rgb(255, 255, 255);color: #fff;image: url(:/iconos/iconos/STOP.svg); border: 0px solid #555;  border-radius: 20px; border-style: inset;padding: 5px;}QPushButton:pressed{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #dadbde, stop: 1 #f6f7fa);}"))


            self.tInicio    = self.timeEditInicio.time().second()
            self.tCue       = 1 + self.tInicio
            self.tIM        = self.timeEditDuracion.time().second() + self.tCue
            self.tFinal     = self.timeEditFin.time().second() + self.tIM 

            print(self.tInicio)
            print(self.tCue)
            print(self.tIM)
            print(self.tFinal)

            
            self.totalIter  = int(self.lineEditIteraciones.text())

            try: 
                thread = threading.Thread(target=self.capturaExperimento)
                thread.start()
            except:
                print("No se pudo iniciar la captura")
            
            self.startTime=time.time()
        else:
            self.exitCaptura = True

    def capturaExperimento(self):
        
        # Generacion de los Cue
        self.cueList = np.array([1] * self.totalIter + [2] * self.totalIter)
        np.random.shuffle(self.cueList)
        # print(self.cueList)

        # Start Unity Capture
        self.client.send_message("/Protocol", 0)
        self.unityStartFlag = True
        time.sleep(1)

        self.bufferData = []
        self.eventos    = []
        self.nSamples   = 0
        self.nIter      = 0

        print("looking for an EEG stream...")
        self.streams = resolve_stream('type', 'EEG',1,5)
        if not self.streams:
            print("No se encontró el emotiv")
        else:
            self.inlet   = StreamInlet(self.streams[0])
            
            timeOut = 0.5 #Segundos
            stamp = local_clock()
            
            while(not self.exitCaptura and self.unityStartFlag and self.nIter < self.totalIter*2):

                sample, timestamp = self.inlet.pull_sample(timeout=timeOut)
                self.bufferData.append(np.hstack((timestamp, sample)))

                # Verificacion del timeOut
                if(local_clock()-stamp >= timeOut):
                    print("Emotiv Desconectado")
                    break
                stamp = local_clock()
                
                # Si no ha time out, ejecute el protocolo segun los tiempos,
                # cantidad de muestras e iteraciones indicadas.
                if(self.nSamples == 0 or self.nSamples == self.tIM*self.fs):
                    evento = 0
                elif(self.nSamples == self.tInicio*self.fs):
                    evento = int(self.cueList[self.nIter])
                    self.client.send_message("/Protocol", evento)
                elif(self.nSamples == self.tCue*self.fs):
                    evento = -1
                    self.client.send_message("/Protocol", 3)
                elif(self.nSamples == self.tFinal*self.fs-1):
                    self.nSamples = -1
                    self.nIter   +=  1

                self.eventos.append(evento)
                self.nSamples +=1 
            
            # Verificando la integridad de los datos capturados
            if(self.nIter == self.totalIter*2):
                self.bufferData = np.asarray(self.bufferData)
                self.bufferData[:,0] = self.bufferData[:,0]-self.bufferData[0,0]    
                print(self.bufferData.shape)

            elif(self.nIter < self.totalIter*2 and self.nIter >= 1):
                self.bufferData = np.asarray(self.bufferData)
                self.bufferData[:,0] = self.bufferData[:,0]-self.bufferData[0,0]   
                self.bufferData      = self.bufferData[:,:(self.nIter-1)*self.fs*self.tFinal-1] 
                self.eventos         = self.eventos[:(self.nIter-1)*self.fs*self.tFinal-1]
                print(self.bufferData.shape)

            elif(self.nIter<1):
                self.bufferData = None
                self.evento     = None
                print("Fallo la captura, 0 iteraciones capturadas")

        #Terminando captura de Unity
        self.client.send_message("/Protocol", 4)

        #Resetea las variables de exit del protocolo
        self.unityStartFlag = False
        self.exitCaptura    = False

        #Resetea el boton de play
        print("Terminó la captura")
        self.pushButtonStart.setStyleSheet(("QPushButton{background-color: rgb(255, 255, 255);color: #fff;image: url(:/iconos/iconos/PLAY.svg); border: 0px solid #555;  border-radius: 20px; border-style: inset;padding: 5px;}QPushButton:pressed{background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #dadbde, stop: 1 #f6f7fa);}"))
        

    # Matriz de confusion
    def graficaMatriz(self, y_test=None, y_predict=None, labels=None, titulo='Matriz de confusión'):
        
        if y_test is None and y_predict is None and labels is None:
            y_test      = [1,1,0,0,0]
            y_predict   = [1,1,0,0,0]
            labels      = ["Izquierda","Derecha"]

        cm      = confusion_matrix(y_test, y_predict)
        cm_sum  = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100

        annot = np.empty_like(cm).astype(np.dtype('U30'))
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.2f%%\n%d/%d\n' % (p, c, s)
                else:
                    annot[i, j] = '%.2f%%\n%d/%d\n' % (p, c, s)
                
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name   = 'Verdaderos'
        cm.columns.name = 'Estimados'

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.set_visible(True)
        self.MplWidget.canvas.axes.set_title(titulo,fontdict={'weight': 'normal','size':10})
        self.MplWidget.canvas.axes.axhline(y=0, color='Black',linewidth=0.2)
        self.MplWidget.canvas.axes.axhline(y=2, color='Black',linewidth=0.2)
        self.MplWidget.canvas.axes.axvline(x=0, color='Black',linewidth=0.2)
        self.MplWidget.canvas.axes.axvline(x=2, color='Black',linewidth=0.2)

        sn.heatmap(cm, cmap= "Blues", annot=annot, fmt='', cbar=None, ax=self.MplWidget.canvas.axes)
        
        self.MplWidget.canvas.draw()
        
    def mostrarInfo(self):
        #Code for the second screen
        #from InfoWindow import Ui_Form
        #self.Form = QtWidgets.QWidget()
        #self.ui = Ui_Form()
        #self.ui.setupUi(self.Form)
        #self.Form.show()    
        self.graficaMatriz()

    def limpiarInterfaz(self):

        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Reiniciar Interfaz")
        msg.setText("<strong>¿Desea reiniciar la interfaz?</strong>")
        msg.setInformativeText("Al reiniciar la interfaz se perderá lo realizado en la sesión.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        msg.setIcon(QtWidgets.QMessageBox.Warning) 
        msg.setWindowIcon(QtGui.QIcon('.\imagenes\IM.svg')) 
        ret = msg.exec_()

        if ret == QMessageBox.Ok:
            print("hola")
        elif ret == QMessageBox.Cancel:
            print("adios")

    ## Show message of warning
    def showWarning(self, title, message):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QtWidgets.QMessageBox.Warning) 
        msg.setWindowIcon(QtGui.QIcon('.\imagenes\IM.svg')) 
        msg.exec_()

    def closeEvent(self, event):
        
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Cerrar")
        msg.setText("¿Desea cerrar la interfaz?")
        msg.setIcon(QtWidgets.QMessageBox.Question) 
        msg.setWindowIcon(QtGui.QIcon('.\imagenes\IM.svg')) 
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        ret = msg.exec_()

        if ret == QMessageBox.Yes:
            self.exitCaptura = True
            self.server.shutdown()
            event.accept()
        else:
            event.ignore()


    def emotivRandomData():

        import time
        from random import random as rand

        from pylsl import StreamInfo, StreamOutlet, local_clock

        # first create a new stream info (here we set the name to BioSemi,
        # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
        # last value would be the serial number of the device or some other more or
        # less locally unique identifier for the stream as far as available (you
        # could also omit it but interrupted connections wouldn't auto-recover).
        info = StreamInfo('BioSemi', 'EEG', 14, 128, 'float32', 'myuid2424')

        # append some meta-data
        info.desc().append_child_value("manufacturer", "BioSemi")
        channels = info.desc().append_child("channels")
        for c in ["C3", "C4", "Cz", "FPz", "POz", "CPz", "O1", "O2"]:
            channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "microvolts") \
                .append_child_value("type", "EEG")

        # next make an outlet; we set the transmission chunk size to 32 samples and
        # the outgoing buffer size to 1000 seconds (max.)
        outlet = StreamOutlet(info, 32, 1000)

        print("now sending data...")
        stamp = local_clock()
        tInicial = stamp
        while True:
            if(local_clock()-stamp >= (1./128)):
                stamp = local_clock()
                # make a new random 8-channel sample; this is converted into a
                # pylsl.vectorf (the data type that is expected by push_sample)
                mysample = [rand(), rand(), rand(), rand(), rand(), rand(), 
                            rand(), rand(), rand(), rand(), rand(), rand(), 
                            rand(), rand()]
                # get a time stamp in seconds (we pretend that our samples are actually
                # 125ms old, e.g., as if coming from some external hardware)
                # now send it and wait for a bit
                outlet.push_sample(mysample, stamp-tInicial)


if __name__ == "__main__":

    app =  QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
