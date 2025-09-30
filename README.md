
# Ejercicio de Suficiencia de Capital

En este repositorio se encuentran los codigo usados para el Ejercicio de Suficiencia de Capital (ESC).


## Instalacion de Librerias

### Python
Para instalar las librerias necesarias para los codigos en Python, correr el siguiente comando:

    pip install -r requirements.txt
## Cascaron Neural Prophet

### Ejectubable
Para generar el ejectubale del cascaron del modelo Neural Prophet, se debe primero instalar la libreria **PyInstaller**, y despues correr el siguiente comando en la ruta del proyecto:

    pyinstaller --additional-hooks-dir=./hooks --onefile --name <nombre de archivo generado> main.py   
Este en automatico tomara los dos codigos en la carpeta **hooks** para tomar en cuenta los archivos **version.info** de dos librerias: pytorch_lightning y ightning_fabric.

#### Mac
Para poder compartir el ejectuable para Mac, primero se debe poner en misma carpeta **dist** el archivo **quarantine-fix.sh** que se encuentra en la carpeta **utils**, y despues es necesario correr el comando 
    tar -czf <nombre del archivo comprimido>.tar.gz dist

Ese archivo luego puede ser descargado y primero se debera correr el archivo **quarantine-fix.sh** una unica vez, y ya se podra correr el ejectuable.