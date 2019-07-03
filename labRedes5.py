import matplotlib.pyplot as plt
import numpy as np
PI = np.pi

print('hola')
#Modulacion
signal = [0, 1, 0, 0, 0, 1, 0]

f1 = 100
f2 = 200
fs = 30*f2
tb = 0.1

tiempo = np.linspace(0, tb, num=int(fs*tb))

print(len(tiempo))
cero_signal = np.cos(2*PI*f1*tiempo)
one_signal = np.cos(2*PI*f2*tiempo)

plt.figure(1)
plt.plot(tiempo, cero_signal)

plt.figure(2)
plt.plot(tiempo, one_signal)

salida = []

for bit in signal:
    if (bit == 0):
        salida.extend(cero_signal)
    else:
        salida.extend(one_signal)

print(len(salida))

tiempoSignal = np.linspace(0, tb*len(signal), num=len(salida))
print(len(tiempoSignal))

plt.figure(3)
plt.plot(tiempoSignal, salida)
#plt.show()

#Demodular





