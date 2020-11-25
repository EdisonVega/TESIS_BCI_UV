@ECHO OFF
pyuic5 MainWindow.ui -o MainWindow.py
pyuic5 InfoWindow.ui -o InfoWindow.py
pyrcc5 resources.qrc -o resources_rc.py
PAUSE