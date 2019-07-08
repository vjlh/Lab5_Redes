import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter
import numpy as np
import sys
PI = np.pi

def main(argv):
    # Modulacion
    sig_len = 1e5
    #signal = np.random.randint(2, size=int(sig_len)) No usar a menos que quieras un freeze
    signal = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1]

    bitrate = 150
    if len(argv) > 1:
        bitrate = int(argv[1])
        tb = 1 / bitrate  # Tiempo bit = 1 / bitrate.
    else:
        tb = 1 / bitrate

    # La frecuencia de la señal se adapta al bitrate para completar al menos 1 ciclo
    # De lo contrario no es posible enviar altos bitrate
    f1 = bitrate
    f2 = 2*f1
    print("Frecuencia bit 0: ", f1)
    print("Frecuencia bit 1: ", f2)

    # Por teorema Nquist, se toma fs > 2*fm
    dt_bit = 1 / (15 * f2)  # Se elige el con mayor frecuencia

    len_bit = int(tb / dt_bit)  # Numero de muestras para 1 bit con frecuencia f2 (mayor)

    tiempo = np.linspace(0, tb, num=len_bit)
    print("Num muestras en 1 bit: ", len_bit)
    cero_signal = np.cos(2 * PI * f1 * tiempo)
    one_signal = np.cos(2 * PI * f2 * tiempo)

    plt.figure(1)
    plt.plot(tiempo, cero_signal)
    plt.title("Frecuencia f1: bit 0")

    plt.figure(2)
    plt.plot(tiempo, one_signal)
    plt.title("Frecuencia f2: bit 1")
    plt.show()

    salida = []
    for bit in signal:
        if bit == 0:
            salida.extend(cero_signal)
        else:
            salida.extend(one_signal)

    # Modulacion FSK

    fc = 1e4
    dt_carrier = 1 / (4 * fc) # Diferencial de tiempo muestreo carrier
    t_carrier = len(signal) * tb  # Tiempo total que se demora en enviar todos los bits
    print("Tiempo total toda la señal: ", t_carrier)
    n_carrier = int(t_carrier / dt_carrier)
    tiempo_signal = np.linspace(0, t_carrier, len(salida))
    tiempo_carrier = np.linspace(0, t_carrier, n_carrier)
    signal_interp = np.interp(tiempo_carrier, tiempo_signal, salida)
    print("Num muestras para toda la señal interpolada: ", len(signal_interp))
    wc = 2*PI*fc
    signal_mod = np.cos(wc*tiempo_carrier + np.cumsum(signal_interp))

    plt.figure(4)
    plt.plot(tiempo_carrier, signal_mod)
    plt.title("Señal modulada")
    plt.show()

    # FFT para ver

    fft_mod = np.abs(fft(signal_mod))
    fft_freq = fftfreq(len(tiempo_carrier), dt_carrier)

    plt.figure(5)
    plt.plot(fft_freq, fft_mod, 'blue', linewidth=1)
    plt.title("Transformada Fourier FSK modulada")
    plt.show()

    # Demodular FSK
    cosf1 = np.cos(2 * PI * f1 * tiempo_carrier)
    cosf2 = np.cos(2 * PI * f2 * tiempo_carrier)

    cos_carrier = np.cos(2 * PI * fc * tiempo_carrier)

    zero_demod = signal_interp*cosf1
    one_demod = signal_interp*cosf2
    #one_demod = signal_mod * cos_carrier # no funciona

    plt.figure(6)
    plt.plot(tiempo_carrier, one_demod, 'blue', linewidth=1)
    plt.title("Demodulacion 1")
    plt.show()

    plt.figure(7)
    plt.plot(tiempo_carrier, zero_demod, 'blue', linewidth=1)
    plt.title("Demodulacion 0")
    plt.show()

    len_bit = int(tb / dt_carrier)
    print("numero muestras en carrier x bit: " + str(len_bit))
    print("numero de bits calculado: " + str(len(one_demod) / len_bit))
    symbol_signal = []  # Señal binaria continua (para mostrar en plot)
    bits_decoded = [] # Señal binaria discreta (bits demodulados)
    for k in range(1, int(len(one_demod) / len_bit) + 1):
        # Se toma una porción de la señal (muestras) equivalentes a 1 bit.
        voltage = one_demod[((k - 1) * len_bit): k * len_bit - 1]
        # Se calcula la media del voltaje (centro de la oscilación)
        # Usando la señal demodulada con f1 (one_demod):
        # Cuando ocurre un 1, se oscila entre 0 y 1, con una media de 0.5
        # Cuando ocurre un 0, se oscila entre -1 y 1, con una media 0.
        voltage_mean = np.mean(voltage)
        symbol_signal.extend([voltage_mean] for i in range(len_bit)) # Se
        # El voltaje máximo es 0.5, por lo tanto se puede tomar la mitad hacia arriba como 1
        if voltage_mean > 0.25:
            bits_decoded.append(1)
        else:
            bits_decoded.append(0)

    offset = len(tiempo_carrier) - len(symbol_signal) # Muestras faltantes por aproximación
    symbol_signal = np.pad(symbol_signal, int(offset/2), 'edge')
    plt.figure(8)
    plt.plot(tiempo_carrier, symbol_signal, 'orange', linewidth=1)
    plt.title("Con detector de envoltura")
    plt.show()

    print("Bits finales: " + str(bits_decoded))
    error_count = 0
    for i in range(len(signal)):
        if signal[i] != bits_decoded[i]:
            error_count += 1
    print("Número de errores en la transmisión: {}".format(error_count))
    err_rate = (error_count/len(signal)) * 100
    print("Error rate: {}%".format(err_rate))


if __name__ == "__main__":
    main(sys.argv)





