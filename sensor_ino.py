import time
import math
import pandas as pd
#from Adafruit_BNO055 import BNO055

#BNO055_DELAY_MS = 0.1
df = pd.read_csv('/Users/felixbrener/Documents/ARC/OpenPose_Aidan/Jillian_csv/8-23-2021-16-24_00119jkt_eyes-open_5.csv')
#sensor = BNO055.BNO055()
#The sensor data in this code is three-dimensional, representing the values of various parameters measured by the accelerometer and gyroscope of the BNO055 sensor
#The three dimensions are:
#X-axis: Represented by variables such as acc.x(), gyr.x(), and phiM, which represent the values measured along the X-axis of the sensor.
#Y-axis: Represented by variables such as acc.y(), gyr.y(), and thetaM, which represent the values measured along the Y-axis of the sensor.
#Z-axis: Represented by variables such as acc.z(), gyr.z(), and phiM, which represent the values measured along the Z-axis of the sensor.

filterPOld, filterPNew = 0, 0 # Precision variables, Derived in setup function

thetaM = 0 # Measured inclination based on accelerometer X axis.
thetaFOld = 0 # Filtered Old Value - Initial value is 0
thetaFNew = 0 # Filtered New Value, will be derived and will replace the old value for next iteration. 
#having an old value is important because it is used to estimate the initial value for the next integration

phiM = 0
phiFOld = 0 # Filtered Old Value - Initial value is 0
phiFNew = 0 # Filtered New Value, will be derived and will replace the old value for next iteration

timePrevious = 0 # Variable to store the old time in milliseconds.
timeDiff = 0 # variable to capture the time difference
thetaG = 0 # Angular distance in terms of angle on x axis
phiG = 0 # Angular distance in terms of angle on y axis

roll = 0 # Complementary filter variable
pitch = 0 # Complementary filter variable
trust = 0.9 # Trust percentage for complementary filter on gyroscope

# Setup function
def setup():
    global filterPOld, filterPNew, timePrevious
    sensor.begin()
    time.sleep(1) #?
    temp = sensor.getTemp() #? why doe sit need to know the tempurature
    sensor.setExtCrystalUse(True) #used for timekeeping for live recordings
    # Derive the filter precision for the old and new value
    filterPOld = 0.95 # FILTER_PRECISION in percentage / 100;
    filterPNew = 1 - filterPOld
    timePrevious = time.time() # Initializing the time in seconds
#instead of defining a loop, would do for i in range len acceleration, loop through the acceleration list of lists and gyro and 
#do the exact same thing but instead of acc
# Loop function
#while dataframe is open() or for I in range(len(dataframe)):
for index, row in df.iterrows():
    #global thetaM, thetaFOld, thetaFNew, phiM, phiFOld, phiFNew, timePrevious, timeDiff, thetaG, phiG, roll, pitch
    # Extract the "Accel X", "Accel Y", and "Accel Z" values from the row
    accel_x = row["Accel X"]
    accel_y = row["Accel Y"]
    accel_z = row["Accel Z"]
    
    # Create a tuple from the extracted values and append it to the list
    accel_tuple = (accel_x, accel_y, accel_z)
        # Extract the "Gyro X", "Gyro Y", and "Gyro Z" values from the row
    gyro_x = row["Gyro X"]
    gyro_y = row["Gyro Y"]
    gyro_z = row["Gyro Z"]
    
    # Create a tuple from the extracted values and append it to the list
    gyro_tuple = (gyro_x, gyro_y, gyro_z)
    # Reading accelerometer data
    #acc = sensor.getVector(BNO055.VECTOR_ACCELEROMETER)
    # Reading Gyroscope data
    #gyr = sensor.getVector(BNO055.VECTOR_GYROSCOPE)
    #system, gyro, accel, mg = 0, 0, 0, 0 we do not need this because it's for the bno550
    #sensor.getCalibration(system, gyro, accel, mg)
    # Calculating the inclination based on accelerometer data x axis
    # the index after acceleration is the 1d list in the array that we need for the respective coordinate direction, first is x, second is z
    thetaM = math.atan2(accel_tuple[0]/9.8, accel_tuple[2]/9.8) * 180 / 3.14159265359
    # Calculate the New theta value based on the measured theta value and the old value
    thetaFNew = filterPOld * thetaFOld + filterPNew * thetaM
    # Calculating the inclination based on accelerometer data y axis
    #first 1d array is y, second is z
    phiM = math.atan2(accel_tuple[1]/9.8, accel_tuple[2]/9.8) * 180 / 3.14159265359
    # Calculate the New phi value based on the measured phi value and the old value
    phiFNew = filterPOld * phiFOld + filterPNew * phiM

    # Evaluate Angular acceleration based on gyro data
    # Time difference calculation (in sec)
    #timeDiff = (time.time() - timePrevious)/1000.0
    timePrevious = time.time() # Resetting the time with current time
    timeDiff = 1/60
    #timePrevious = previous frame number / 60
    thetaG = thetaG - gyro_tuple[1] * timeDiff
    phiG = phiG + gyro_tuple[0] * timeDiff
    #timediff will always be 60 fps
    # Complementary filter implementation roll estimation
    roll = (roll + gyro_tuple[0] * timeDiff) * trust + phiM * (1 - trust)
    # Complementary filter implementation pitch estimation
    pitch = (pitch - gyro_tuple[1] * timeDiff) * trust + thetaM * (1 - trust)

    print(accel_tuple[0]/9.8, ",", accel_tuple[1]/9.8, ",", accel_tuple[2]/9.8, ",", system, ",", accel, ",", gyro, ",", mg, ",", thetaM, ",", phiM, ",", thetaFNew, ",", phiFNew, ",", gyro_tuple[0], ",", gyro_tuple[1], ",", gyro_tuple[2], ",", thetaG, ",", phiG, ",", roll, ",", pitch)

    thetaFOld = thetaFNew
    phiFOld = phiFNew

    #time.sleep(BNO055_DELAY_MS/1000.0)
