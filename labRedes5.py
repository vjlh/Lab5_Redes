import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter
import numpy as np
import sys
PI = np.pi


def noise_channel(signal, snr):
    np.random.seed(127369)
    noise_sig = np.random.normal(0, 1, len(signal))
    energia_s = np.sum(np.abs(signal) * np.abs(signal))
    energia_n = np.sum(np.abs(noise_sig) * np.abs(noise_sig))
    snr_lineal = np.exp(snr/10)
    delta = np.sqrt(energia_s / (energia_n * snr_lineal))
    # print('E signal: ' + str(energia_s))
    # print('E noise: ' + str(energia_n))
    # print('SNR lineal: ' + str(snr_lineal))
    print('Desviación ruido: ' + str(delta))
    noise_sig = delta*noise_sig
    signal_awgn = signal + noise_sig
    return signal_awgn

def ber_vs_snr( bitrate ):
    sig_len = 1e4 # Numero de bits para prueba
    bits = np.random.randint(2, size=int(sig_len))
    colores = ['-b', '-g', '-r']
    plt.figure(10)
    for i in range(0, 3):
        snr_x = []
        ber_y = []
        bitrate = bitrate + i*1000 # Para probar 3 bitrates distintos
        tiempo, signal, len_bit = mod_FSK(bits, bitrate, False)
        for snr in range(-2, 12, 2):
            print("**** Prueba SNR = {}[dB] para bitrate = {}[bits/s] ****".format(snr, bitrate))
            signal_awgn = noise_channel(signal, snr)
            bits_demod = demod_FSK(tiempo, signal_awgn, len_bit, bitrate, False)
            ber = signal_ber(bits, bits_demod)
            snr_x.append(snr)
            ber_y.append(ber)
            lab = str(bitrate) + ' [bps]'
            print("**** Fin prueba ****\n")
        plt.plot(snr_x, ber_y, colores[i], label=lab, marker="o")
        #legend.append(str(bitrate) + ' [bps]')

    plt.grid(True)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (%)')
    plt.title('Rendimiento SNR vs Bitrate')
    plt.legend()
    plt.show()


def mod_FSK( bits, bitrate, plot_results):
    # La frecuencia de la señal se adapta al bitrate para completar al menos 1 ciclo
    # De lo contrario no es posible enviar altos bitrate
    tb = 1 / bitrate
    f1 = bitrate
    f2 = 2 * f1
    print("Frecuencia bit 0: ", f1)
    print("Frecuencia bit 1: ", f2)

    # Por teorema Nquist, se toma fs > 2*fm
    dt_bit = 1 / (10 * f2)  # Se elige el con mayor frecuencia

    len_bit = int(tb / dt_bit)  # Numero de muestras para 1 bit con frecuencia f2 (mayor)
    tiempo = np.linspace(0, tb, num=len_bit)
    print("Num muestras en 1 bit: ", len_bit)
    cero_signal = np.cos(2 * PI * f1 * tiempo)
    one_signal = np.cos(2 * PI * f2 * tiempo)

    if plot_results:
        plt.figure(1)
        plt.plot(tiempo, cero_signal, "k", tiempo, one_signal, "r", linewidth=1.5)
        plt.title("Señal del bit")
        plt.legend(('Bit 0', 'Bit 1'))
        plt.show()

    salida = []
    for bit in bits:
        if bit == 0:
            salida.extend(cero_signal)
        else:
            salida.extend(one_signal)

    t_total = len(bits) * tb  # Tiempo total que se demora en enviar todos los bits
    print("Tiempo total toda la señal: ", t_total)
    tiempo_signal = np.linspace(0, t_total, len(salida))
    print("Num muestras para toda la señal: ", len(salida))

    return tiempo_signal, salida, len_bit


def demod_FSK(tiempo, signal, samp_bit, bitrate, plot_results):
    f1 = bitrate
    f2 = 2 * f1
    cosf1 = np.cos(2 * PI * f1 * tiempo)
    cosf2 = np.cos(2 * PI * f2 * tiempo)
    one_demod = signal * cosf2

    if plot_results:
        plt.figure(6)
        plt.plot(tiempo, one_demod, 'blue', linewidth=1)
        plt.title("Demodulacion 1")

        zero_demod = signal * cosf1  # Basta con calcular one_demod o zero_demod, pero para pruebas se usa ambos
        plt.figure(7)
        plt.plot(tiempo, zero_demod, 'blue', linewidth=1)
        plt.title("Demodulacion 0")
        plt.show()

    # tb = 1 / bitrate
    # len_bit = int(tb / samp_bit) se usa directamente samp_bit en argumento en vez de calcularlo
    n_bits = int(len(one_demod) / samp_bit)
    symbol_signal = []  # Señal binaria continua (para mostrar en plot)
    bits_decoded = []  # Señal binaria discreta (bits demodulados)
    for k in range(1, n_bits + 1):
        # Se toma una porción de la señal (muestras) equivalentes a 1 bit.
        voltage = one_demod[((k - 1) * samp_bit): k * samp_bit - 1]
        # Se calcula la media del voltaje (centro de la oscilación)
        # Usando la señal demodulada con f1 (one_demod):
        # Cuando ocurre un 1, se oscila entre 0 y 1, con una media de 0.5
        # Cuando ocurre un 0, se oscila entre -1 y 1, con una media 0.
        voltage_mean = np.mean(voltage)
        symbol_signal.extend([voltage_mean] for i in range(samp_bit))  # Se
        # El voltaje máximo es 0.5, por lo tanto se puede tomar la mitad hacia arriba como 1
        if voltage_mean > 0.25:
            bits_decoded.append(1)
        else:
            bits_decoded.append(0)

    offset = len(tiempo) - len(symbol_signal)  # Muestras faltantes por aproximación
    symbol_signal = np.pad(symbol_signal, int(offset / 2), 'edge') # Se rellenan los bordes hasta el largo

    if plot_results:
        plt.figure(8)
        plt.plot(tiempo, symbol_signal, 'orange', linewidth=1)
        plt.title("Detector de envoltura")
        plt.show()

    return bits_decoded


def main(argv):

    bits = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    bitrate = 1000
    if len(argv) > 1:
        bitrate = int(argv[1])

    time, signal, len_bit = mod_FSK(bits, bitrate, 1)

    bits_decoded = demod_FSK(time, signal, len_bit, bitrate, True)
    print("Bits finales: " + str(bits_decoded))

    ber_vs_snr(bitrate)

    # Demodulacion FSK

    '''
    plt.figure(4)
    plt.plot(tiempo_signal, salida)
    plt.title("Señal modulada")
    plt.show()
    '''
    # FFT para ver
    '''
    fft_mod = np.abs(fft(salida))
    fft_freq = fftfreq(len(tiempo_signal), dt_bit)

    plt.figure(5)
    plt.plot(fft_freq, fft_mod, 'blue', linewidth=1)
    plt.title("Transformada Fourier FSK modulada")
    plt.show()
    '''
    return 0


def signal_ber(bits_decoded, bits):
    error_count = 0
    for i in range(len(bits)):
        if bits[i] != bits_decoded[i]:
            error_count += 1
    print("Número de errores en la transmisión: {}".format(error_count))
    bit_err_rate = (error_count / len(bits)) * 100
    print("Error rate: {}%".format(bit_err_rate))
    return bit_err_rate


if __name__ == "__main__":
    main(sys.argv)





