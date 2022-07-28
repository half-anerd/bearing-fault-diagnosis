import pickle
import pandas as pd
import numpy as np

real_data = pd.read_csv('accelerometer_readings.csv') #assumed file with accelerometer readings in required format

Fs = 48000
n = 3600
k = np.arange(n)
Ts = n/Fs
frq = k/Ts 
frq = frq[range(int(n))]

model = pickle.load(open('finalized_model.sav', 'rb'))

for i in range(len(real_data)):
    peak_f = []
    peak_A = []
    sum_f = []

    X = np.array(real_data[i:i+1])
    X = np.fft.fft(X)
    peak_A.append(np.max(np.abs(X)))
    peak_f.append(frq[np.argmax(np.abs(X))])
    S = np.abs(X**2)/n
    sum_f.append(np.sum(S))

    peakFrequency = np.array(peak_f)
    peakAmplitude = np.array(peak_A)
    powerSum = np.array(sum_f)
    features = pd.DataFrame({'peakFrequency': peakFrequency, 'peakAmplitude': peakAmplitude, 'powerSum': powerSum}, columns=['peakFrequency', 'peakAmplitude', 'powerSum'])

    prediction = model.predict(features)
    print(prediction)