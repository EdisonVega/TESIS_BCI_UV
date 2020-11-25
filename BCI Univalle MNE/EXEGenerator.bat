@ECHO OFF
pyinstaller --name "Ensayo de Impacto y Penetracion" --icon=.\iconos\installer_icono.ico --add-data ".\configuracion_ajuste.txt;." --add-data ".\MainWindow.ui;." --add-data ".\InfoWindow.ui;." main.py
PAUSE