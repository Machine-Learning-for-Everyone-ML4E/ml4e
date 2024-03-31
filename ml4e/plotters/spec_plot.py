import librosa
import matplotlib.pyplot as plt
import numpy as np

def SpectrogramPlot(path):
    y, sr = librosa.load(path)

    # Compute spectrogram
    D = librosa.stft(y)

    # Plot spectrogram
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(librosa.amplitude_to_db(abs(D), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()
