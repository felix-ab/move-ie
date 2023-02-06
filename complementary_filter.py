import numpy as np
from scipy.signal import filtfilt, butter
from pyquaternion import Quaternion
from quaternion import quaternion, from_rotation_vector, rotate_vectors
import csv


def comp_filter():
    with open("/Users/felixbrener/Documents/ARC/sway-18-2/1-18-2023-15-31_TEST-2_SingleLimbRIGHTEyesOpen_1.csv", 'r') as file:
        # Read the first line of the file (the header row)
        fields = file.readline().strip().split(',')
        # Delete the “Frame” field because we do not need to plot the frames
        #del fields[0]
        # Create an empty dictionary to store the data
        my_dict = {}
        # Iterate over the fields and add an empty list to the dictionary for each field
        for field in fields:
            my_dict[field] = []
        # Iterate over the remaining lines in the file
        for line in file:
            values = line.strip().split(',')
            # Delete the “Frame” value because we do not need to plot the frames
            #del values[0]
            # Zip the fields and values for parallel iteration. Add the values to the appropriate lists in the  dictionary
            for field, value in zip(fields, values):
                my_dict[field].append(float(value))

    a = np.transpose([my_dict['linAccel X'], my_dict['linAccel Y'], my_dict['linAccel Z']])
    w = np.transpose([my_dict['Gyro X'], my_dict['Gyro Y'], my_dict['Gyro Z']])
    frame = np.array(my_dict['Frame'])
    t = frame/60

    def estimate_orientation(a, w, t, alpha=0.9, g_ref=(0., 0., 1.),
                            theta_min=1e-6, highpass=.01, lowpass=.05):
        """ Estimate orientation with a complementary filter.

        Fuse linear acceleration and angular velocity measurements to obtain an
        estimate of orientation using a complementary filter as described in
        `Wetzstein 2017: 3-DOF Orientation Tracking with IMUs`_

        .. _Wetzstein 2017: 3-DOF Orientation Tracking with IMUs:
        https://pdfs.semanticscholar.org/5568/e2100cab0b573599accd2c77debd05ccf3b1.pdf

        Parameters
        ----------
        a : array-like, shape (N, 3)
            Acceleration measurements (in arbitrary units).

        w : array-like, shape (N, 3)
            Angular velocity measurements (in rad/s).

        t : array-like, shape (N,)
            Timestamps of the measurements (in s).

        alpha : float, default 0.9
            Weight of the angular velocity measurements in the estimate.

        g_ref : tuple, len 3, default (0., 0., 1.)
            Unit vector denoting direction of gravity.

        theta_min : float, default 1e-6
            Minimal angular velocity after filtering. Values smaller than this
            will be considered noise and are not used for the estimate.

        highpass : float, default .01
            Cutoff frequency of the high-pass filter for the angular velocity as
            fraction of Nyquist frequency.

        lowpass : float, default .05
            Cutoff frequency of the low-pass filter for the linear acceleration as
            fraction of Nyquist frequency.

        Returns
        -------
        q : array of quaternions, shape (N,)
            The estimated orientation for each measurement.
        """

        # initialize some things
        N = len(t)
        dt = np.diff(t)
        g_ref = np.array(g_ref)
        q = np.ones(N, dtype=quaternion)

        # get high-passed angular velocity
        w = filtfilt(*butter(5, highpass, btype='high'), w, axis=0)
        w[np.linalg.norm(w, axis=1) < theta_min] = 0
        q_delta = from_rotation_vector(w[1:] * dt[:, None])

        # get low-passed linear acceleration
        a = filtfilt(*butter(5, lowpass, btype='low'), a, axis=0)
        print(q_delta)

        for i in range(1, N):

            # get rotation estimate from gyroscope
            q_w = q[i - 1] * q_delta[i - 1]

            # get rotation estimate from accelerometer
            v_world = rotate_vectors(q_w, a[i])
            n = np.cross(v_world, g_ref)
            phi = np.arccos(np.dot(v_world / np.linalg.norm(v_world), g_ref))
            q_a = from_rotation_vector(
                (1 - alpha) * phi * n[None, :] / np.linalg.norm(n))[0]

            # fuse both estimates
            q[i] = q_a * q_w

        #create csv file
        with open('quaternions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(q)
        #Test
        my_quaternion = Quaternion(axis=[1, 0, 0], angle=3.14159265)
        np.set_printoptions(suppress=True) # Suppress insignificant values for clarity
        v = np.array([0., 0., 1.]) # Unit vector in the +z direction
        v_prime = my_quaternion.rotate(v)
        #v_prime.array([ 0., 0., -1.])
        print(str(v_prime))

        return q
