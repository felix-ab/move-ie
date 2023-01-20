import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
from scipy.integrate import trapezoid
from scipy.special import erf
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math
import csv

# Open the CSV file
#with open("/Users/felixbrener/Documents/ARC/OpenPose_Aidan/vs_code/csv1.csv", 'r') as file:
with open("/Users/felixbrener/Documents/ARC/sway-18-2/1-18-2023-15-31_TEST-2_SingleLimbRIGHTEyesOpen_1.csv", 'r') as file:
    # Read the first line of the file (the header row)
    fields = file.readline().strip().split(',')
    # Delete the “Frame” field because we do not need to plot the frames
    del fields[0]
    # Create an empty dictionary to store the data
    my_dict = {}
    # Iterate over the fields and add an empty list to the dictionary for each field
    for field in fields:
        my_dict[field] = []
    # Iterate over the remaining lines in the file
    for line in file:
        values = line.strip().split(',')
        # Delete the “Frame” value because we do not need to plot the frames
        del values[0]
        # Zip the fields and values for parallel iteration. Add the values to the appropriate lists in the  dictionary
        for field, value in zip(fields, values):
            my_dict[field].append(float(value))

    #function to find the average jerk and velocity for every field and plot all variables over time
    def over_time(my_dict:dict):
        #Creating an empty list of dictionaries
        list_of_dicts = []
        linAccels = ['linAccel X','linAccel Y','linAccel Z']
        #Iterate through every key, value pair in the data dictionary
        for key_, val_list in my_dict.items():
            if key_ in linAccels:
                #calculate jerk by finding the first derivative for acceleration
                jerk = np.diff(val_list)
                abs_jerk = np.abs(jerk)
                abs_average_jerk = np.mean(abs_jerk)
                print("Average absolute value of jerk in the",key_[-1], "direction =", abs_average_jerk)
                accel = np.array(val_list[:-1])
                #Creating a list of every frame
                frames = [*range(len(accel))]
                #Trapezoidal integration to find velocity
                velocity_cumulative = integrate.cumulative_trapezoid(val_list, x=None, dx=1.0, axis=-1)
                '''
                #simpsons integration
                velocity2 = []
                for t in range(0,(len(val_list)//3)*3,3):
                    velocity_2 = integrate.romb(val_list[t:t+3],[t,t+1,t+2])
                    velocity2.append(velocity_2)
                lindis2 = []
                for t in range(0,(len(velocity2)//3)*3,3):
                    lindis_2 = integrate.simpson(velocity2[t:t+3],[t,t+1,t+2])
                    lindis2.append(lindis_2)
                ''' 
                #velocity from gyroscope data
                the_right_gyro = "Gyro " + key_[-1]
                gyro1 = my_dict[the_right_gyro]   
                #un-cumulating the cumulative integration
                velocity_c1 = np.insert(velocity_cumulative, 0, 0)
                velocity_c2 = np.insert(velocity_cumulative, -1, 0)
                velocity_from_integral = np.subtract(velocity_c2,velocity_c1)
                velocity_from_integral = velocity_from_integral[:-1]
                velocity_from_integral[-1] = velocity_from_integral[-2]
                velocity = list(np.round(velocity_from_integral, 4))

                #double integrating linAccel
                linDisplacement_cumulative = integrate.cumulative_trapezoid(velocity, x=None, dx=1.0, axis=-1)
                #un-cumulating the cumulative integration
                
                linDisplacement_c1 = np.insert(linDisplacement_cumulative, 0, 0)
                linDisplacement_c2 = np.insert(linDisplacement_cumulative, -1, 0)
                linDisplacement_from_integral = np.subtract(linDisplacement_c2, linDisplacement_c1)
                linDisplacement_from_integral = linDisplacement_from_integral[:-1]
                linDisplacement_from_integral[-1] = linDisplacement_from_integral[-2]
                linDisplacement = list(np.round(linDisplacement_from_integral, 4))
                
                #Finding displacement from v = rw equation
                #Creating a dictionary to store the data for each field and adding it to the list of dicionaries
                dict_ = {'linDisplacement':linDisplacement,'rad_velocity':gyro1,'Velocity': velocity, 'Acceleration': accel, 'Jerk': jerk, 'Title':key_}
                #dict_ = {'rad_velocity':gyro1,'linDisplacement':lindis2, 'Velocity': velocity2, 'Acceleration': accel, 'Jerk': jerk, 'Title':key_}
                list_of_dicts.append(dict_)

        #creating the figure
        fig = plt.figure(figsize=(14, 7),layout="constrained")
        ax1 = fig.add_subplot(121,projection='3d')
        ax2 = fig.add_subplot(122)
    
        # Position Arrays for 3d animation
        x_animate = list_of_dicts[0]['linDisplacement']
        y_animate = list_of_dicts[1]['linDisplacement']
        z_animate = list_of_dicts[2]['linDisplacement']
        #position arrays for 2d animation

        #position arrays for 2d animation
        speed = []
        for v in range(len(list_of_dicts[0]['rad_velocity'])):
            speed.append(math.sqrt((list_of_dicts[0]['rad_velocity'][v]**2) + (list_of_dicts[1]['rad_velocity'][v]**2) + (list_of_dicts[2]['rad_velocity'][v]**2)))
        window_size = 5
        smooth_speed = signal.convolve(speed, np.ones(window_size)/window_size, mode='same')
    
        magnitute_displacement = []
        for v in range(len(list_of_dicts[0]['linDisplacement'])):
            magnitute_displacement.append(math.sqrt((list_of_dicts[0]['linDisplacement'][v]**2) + (list_of_dicts[1]['linDisplacement'][v]**2) + (list_of_dicts[2]['linDisplacement'][v]**2)))

        # Setting up Data Set for 3d Animation
        dataSet = np.array([x_animate, z_animate, y_animate,])  # Combining our position coordinates
        numDataPoints = len(x_animate)
        t = np.linspace(0,1,numDataPoints)

        def animate_3D(num):
            ax1.clear()  # Clears the figure to update the line, point,   
                        # title, and axes
            # Updating Trajectory Line (num+1 due to Python indexing)
            ax1.plot3D(dataSet[0, :num+1], dataSet[1, :num+1], dataSet[2, :num+1],linewidth=0.75, c='#0093FF')
        
            xx = np.linspace(-1, 1, 10)
            yy = np.linspace(-1, 1, 10)
            X, Y = np.meshgrid(xx, yy)
            Z = np.zeros_like(X)

            # Plot the surface
            ax1.plot_surface(X, Y, Z, color = '#000000', alpha = 0.08)
            ax1.plot_surface(Z, X, Y, color = '#000000', alpha = 0.08)
            ax1.plot_surface(Y, Z, X, color = '#000000', alpha = 0.08)

            # Updating Point Location 
            ax1.scatter(dataSet[0, num], dataSet[1, num], dataSet[2, num], c='blue', marker='o')
            # Adding Constant Origin
            ax1.plot3D(dataSet[0, 0], dataSet[1, 0], dataSet[2, 0], c='#DC0000', marker='o')
            # Setting Axes Limits
            ax1.set_xlim3d([-1, 1])
            ax1.set_ylim3d([-1, 1])
            ax1.set_zlim3d([-1, 1])

            # Adding Figure Labels
            ax1.set_title('Linear Displacement \nTime = ' + str(np.round(t[num],    
                        decimals=2)) + ' sec') #change this for the time to be right
            ax1.set_xlabel('x (meters)')
            ax1.set_ylabel('z(meters)')
            ax1.set_zlabel('y(meters)')
        
        # Setting up Data Set for 2d Animation
        dataSet_1 = np.array([frames[:-1], smooth_speed[:-2],magnitute_displacement])  # Combining our position coordinates
        numDataPoints_1 = len(frames)
        t_1 = np.linspace(0,1,numDataPoints_1)
        def animate_2D(num):
            ax2.clear()  # Clears the figure to update the line, point,   
            # title, and axes

            # Updating Trajectory Line (num+1 due to Python indexing)
            #ax2.plot(dataSet_1[0, :num+1], dataSet_1[1, :num+1], c='#0093FF')
            ax2.plot(dataSet_1[0, :], dataSet_1[1, :], alpha = 0.5, linewidth=0.75, c='#0093FF', label='Speed(rad/s)')
            ax2.plot(dataSet_1[0, :], dataSet_1[2, :], alpha = 0.5, linewidth=0.75, c='#FF0D47', label='Displacement(m)')
            #adding lines for wobble
            window_size = 5
            smooth_jerk_x = signal.convolve(list_of_dicts[0]['Jerk'], np.ones(window_size)/window_size, mode='same')
            smooth_jerk_z = signal.convolve(list_of_dicts[2]['Jerk'], np.ones(window_size)/window_size, mode='same')
            #finding peaks
            peaks_x, _ = find_peaks(smooth_jerk_x, height=np.percentile(smooth_jerk_x, 99))
            peaks_z, _ = find_peaks(smooth_jerk_z, height=np.percentile(smooth_jerk_z, 99))
            peaks_height = np.ones_like(peaks_x)
            peaks_height2 = np.ones_like(peaks_z)
            plt.plot(peaks_x, (peaks_height+1),'^',color='#FFC90F',label='Jerk Side to Side')
            plt.plot(peaks_z, (peaks_height2+1),'^',color='#FF892C',label='Jerk Front to Back')
            ax2.plot([0,0], [5, 5], linewidth=0.7, color='#FFC90F', label='Wobble Side to Side')
            ax2.plot([0,0], [5, 5], linewidth=0.7, color='#FF892C', label='Wobble Front to Back')

            for starts_wobbling in range(len(peaks_x)):
                for i in range(peaks_x[starts_wobbling], len(list_of_dicts[0]['linDisplacement'])):
                    #Finding out when subject retured to center after starting to wobble
                    if -0.001 <= list_of_dicts[0]['linDisplacement'][i]:
                        centered = i 
                        break
                    else: centered = len(list_of_dicts[0]['linDisplacement'])
                ax2.plot([peaks_x[starts_wobbling],centered], [5, 5], linewidth=1, color='#FFC90F')
            for starts_wobbling in range(len(peaks_z)):
                for i in range(peaks_z[starts_wobbling], len(list_of_dicts[2]['linDisplacement'])):
                    #Finding out when subject retured to center after starting to wobble
                    if -0.001 <= list_of_dicts[2]['linDisplacement'][i] <= 0.001:
                        centered = i 
                        break
                    else: centered = len(list_of_dicts[2]['linDisplacement'])
                ax2.plot([peaks_z[starts_wobbling],centered], [5, 5], linewidth=1, color='#FF892C')
            
            ax2.legend()
            # Updating Point Location 
            ax2.scatter(dataSet_1[0, num], dataSet_1[1, num], c='blue', marker='o')
            # Setting Axes Limits
            ax2.set_xlim([0, numDataPoints_1])
            ax2.set_ylim([0, max(smooth_speed)+1])

            # Adding Figure Labels
            ax2.set_title('Radial Velocity \nTime = ' + str(np.round(t_1[num], decimals=2)) + ' sec')
            ax2.set_xlabel('Frames')
            ax2.set_ylabel('Speed')

        # Plotting the Animation
        line_ani = animation.FuncAnimation(fig, animate_3D, interval=50/3, frames=numDataPoints)
        line_ani2 = animation.FuncAnimation(fig, animate_2D, interval=50/3, frames=numDataPoints_1)
        #fig.tight_layout()

        plt.show()
over_time(my_dict)
