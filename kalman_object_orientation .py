#object orientation from 6DOF imu using kalman filter
# Import the required libraries
import pandas as pd
from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumtrapz
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy.signal import butter, filtfilt

#do arc tan squared


def integrate(subject_sway):
    subject_sway_df = pd.read_csv(subject_sway)
    # integrate the X, Y, and Z Gyro columns
    gyro_x_integrated = cumtrapz(subject_sway_df['Gyro X'], subject_sway_df['Frame'], initial=0)
    gyro_y_integrated = cumtrapz(subject_sway_df['Gyro Y'], subject_sway_df['Frame'], initial=0)
    gyro_z_integrated = cumtrapz(subject_sway_df['Gyro Z'], subject_sway_df['Frame'], initial=0)

    # add the integrated columns to the dataframe
    subject_sway_df['Gyro X Integrated'] = gyro_x_integrated
    subject_sway_df['Gyro Y Integrated'] = gyro_y_integrated
    subject_sway_df['Gyro Z Integrated'] = gyro_z_integrated


    #low pass the accel data
    Fs = 60 # Sampling frequency (Hz)
    Fc = 10 # Cutoff frequency (Hz)
    Order = 5 # Filter order

    # Create the filter coefficients using a Butterworth filter
    Nyq = 0.5 * Fs
    Fc_norm = Fc / Nyq

    bb, aa = butter(Order, Fc_norm, btype='low', analog=False)

    subject_sway_df['low_pass_accel_x'] = filtfilt(bb, aa, subject_sway_df['linAccel X'])
    subject_sway_df['low_pass_accel_y'] = filtfilt(bb, aa, subject_sway_df['linAccel Y'])
    subject_sway_df['low_pass_accel_z'] = filtfilt(bb, aa, subject_sway_df['linAccel Z'])

    ax = subject_sway_df['low_pass_accel_x']
    ay = subject_sway_df['low_pass_accel_y']
    az = subject_sway_df['low_pass_accel_z']

    subject_sway_df['phi_values'] = np.arctan2(ay, az) 
    subject_sway_df['theta_values']= np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    #accel_x_arctan = np.arctan2(subject_sway_df['linAccel X'],subject_sway_df['Frame'])
    #accel_y_arctan = np.arctan2(subject_sway_df['linAccel Y'], subject_sway_df['Frame'])
    #accel_z_arctan= np.arctan2(subject_sway_df['linAccel Z'], subject_sway_df['Frame'])

    #subject_sway_df['Accel X arctan'] = accel_x_arctan[1:]
    #subject_sway_df['Accel Y arctan'] = accel_y_arctan[1:]
    #subject_sway_df['Accel Z arctan'] = accel_z_arctan[1:]


    subject_sway_df.to_excel("subject_sway_df.xlsx")
    return subject_sway_df

def apply_high_pass_filter(subject_sway_df):
    # Define the filter parameters
    fs = 60 # Sampling frequency (Hz)
    fc = 15 # Cutoff frequency (Hz)
    order = 5 # Filter order

    # Create the filter coefficients using a Butterworth filter
    nyq = 0.5 * fs
    fc_norm = fc / nyq
    b, a = butter(order, fc_norm, btype='high', analog=False)

    # Apply the filter to the desired columns of the DataFrame
    subject_sway_df['high_pass_gyro_x'] = filtfilt(b, a, subject_sway_df['Gyro X Integrated'])
    subject_sway_df['high_pass_gyro_y'] = filtfilt(b, a, subject_sway_df['Gyro Y Integrated'])
    subject_sway_df['high_pass_gyro_z'] = filtfilt(b, a, subject_sway_df['Gyro Z Integrated'])


    return subject_sway_df

def plotting(subject_sway_df):
    # Calculate the time interval between measurements
    time_interval = 1/60

        # Create a time array based on the number of frames and time interval
    time_array = np.arange(subject_sway_df.shape[0]) * time_interval

    # Create a figure and axis objects
    fig = plt.figure(figsize=(15,10))

    # Create the main axis for the combined plot
    ax_main = fig.add_subplot(4, 2, 1)
    ax_main.plot(time_array, subject_sway_df['high_pass_gyro_x'], label='Gyro X')
    ax_main.plot(time_array, subject_sway_df['high_pass_gyro_y'], label='Gyro Y')
    ax_main.plot(time_array, subject_sway_df['high_pass_gyro_z'], label='Gyro Z')
    ax_main.set_xlabel('Time (s)')
    ax_main.set_ylabel('Integrated Gyro Reading (radians)')
    ax_main.set_title('Combined Gyro Readings')
    ax_main.legend()

    # Create the axis for the Gyro X plot
    ax_gyro_x = fig.add_subplot(4, 2, 2)

    ax_gyro_x.plot(time_array, subject_sway_df['high_pass_gyro_x'], label='accel X')

    ax_gyro_x.plot(time_array, subject_sway_df['high_pass_gyro_x'])
    ax_gyro_x.set_xlabel('Time (s)')
    #ax_gyro_x.set_ylabel('Accel Readings (m/s squared)')
    ax_gyro_x.set_ylabel('Integrated Gyro X Reading (radians)')
    ax_gyro_x.set_title('Gyro X Reading')

    # Create the axis for the Gyro Y plot
    ax_gyro_y = fig.add_subplot(4, 2, 3)
    ax_gyro_y.plot(time_array, subject_sway_df['high_pass_gyro_y'], color='orange')
    ax_gyro_y.set_xlabel('Time (s)')
    ax_gyro_y.set_ylabel('Integrated Gyro Y Reading (radians)')
    ax_gyro_y.set_title('Gyro Y Reading')

    # Create the axis for the Gyro Z plot
    ax_gyro_z = fig.add_subplot(4, 2, 4)
    ax_gyro_z.plot(time_array, subject_sway_df['high_pass_gyro_z'], color='green')
    ax_gyro_z.set_xlabel('Time (s)')
    ax_gyro_z.set_ylabel('Integrated Gyro Z Reading (radians)')
    ax_gyro_z.set_title('Gyro Z Reading')

    # Create the axis for the Accel plot
    ax_accel_x = fig.add_subplot(4, 2, 5)
    ax_accel_x.plot(time_array, subject_sway_df['theta_values'], color='red')
    ax_accel_x.set_xlabel('Time (s)')
    ax_accel_x.set_ylabel('arctan sguared acccel x Readings (rad)')
    ax_accel_x.set_title('Theta Reading (roll)')

    # Create the axis for the Accel plot
    ax_accel_y = fig.add_subplot(4, 2, 6)
    ax_accel_y.plot(time_array, subject_sway_df['phi_values'], color='red')
    ax_accel_y.set_xlabel('Time (s)')
    ax_accel_y.set_ylabel('arctan sguared acccel y Readings (rad)')
    ax_accel_y.set_title('Thi Reading (pitch)')


    # Adjust the layout of the subplots
    plt.tight_layout()
    # Save the figure as a file
    fig.savefig('gyro_readings_5.png')
    # Show the plot
    plt.show()

def find_common_frequencies(subject_sway_df):
    # Take the Fourier Transform of each axis in the DataFrame
    freq_data = np.abs(fft(subject_sway_df.values, axis=0))
    
    # Calculate the mean frequency spectrum for each axis
    mean_freq_spectrum = np.mean(freq_data, axis=0)
    
    # Find the frequencies with the highest amplitude for each axis
    #find top 75% frequencies
    top_frequencies = []
    for col in range(subject_sway_df.shape[1]):
        sorted_freq_indices = np.argsort(mean_freq_spectrum[:, col])[::-1]
        top_frequencies.append(sorted_freq_indices[:10])
    
    return top_frequencies



def apply_bandpass_filter(subject_sway_df, top_frequencies, fs=100, order=5, lowcut=0.5, highcut=5):
    # Create a Butterworth filter
    nyquist_freq = fs/2
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to each column in the DataFrame
    for col in subject_sway_df.columns:
        data = subject_sway_df[col].values
        freq_data = fft(data)
        freqs = fftfreq(data.size, 1/fs)
        freqs_shifted = fftshift(freqs)
        
        # Filter out frequencies outside the pass band
        mask = np.zeros_like(freq_data)
        for freq_idx in top_frequencies:
            mask[(freqs_shifted >= freqs_shifted[freq_idx] - 1) & (freqs_shifted <= freqs_shifted[freq_idx] + 1)] = 1
            
        freq_filtered_data = freq_data * mask
        filtered_data = np.real(ifft(freq_filtered_data))
        subject_sway_df[col] = filtfilt(b, a, filtered_data)
        
    return subject_sway_df

if __name__ == '__main__':

    subject_sway = '/Users/felixbrener/Library/CloudStorage/OneDrive-Personal/Lazarus/Sway/Felix-1/3-21-2023-14-02/3-21-2023-14-02_Felix-1_SingleLimbLeftEyesOpen_1.csv'
    cutoff_freq =  0.2
    sampling_freq = 1/60

subject_sway_df = integrate(subject_sway)
subject_sway_df = apply_high_pass_filter(subject_sway_df)
plotting(subject_sway_df)
#top_frequencies = find_common_frequencies(subject_sway_df)
#filtered_subject_sway_df = apply_bandpass_filter(subject_sway_df, top_frequencies)
  