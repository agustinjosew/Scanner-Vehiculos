import cv2 as escaner
import numpy as np


buscador = escaner.CascadeClassifier('preEntrenadoCascClasif.xml')

def reconocer_vehiculos(recuadro):

    autos = buscador.detectMultiScale(recuadro, 1.15, 4)
    for (x, y, w, h) in autos:
        escaner.rectangle(recuadro, (x, y), (x+w, y+h),
                          color=(128, 255, 0), thickness=2)
    return recuadro

def simulador():
    fuenteDeVideo = escaner.VideoCapture('videos/AccidenteTransito.mp4')
    while fuenteDeVideo.isOpened():
            ret, recuadro = fuenteDeVideo.read()
            controlTecla = escaner.waitKey(1)

            if ret:
                autoRecuadro = reconocer_vehiculos(recuadro)
                escaner.imshow('Ventana de Analisis Video', autoRecuadro)

            else:
                break
            if controlTecla == ord('s'):
                break

    fuenteDeVideo.release()
    escaner.destroyAllWindows()



if __name__ == '__main__':
    simulador()
