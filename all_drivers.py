import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
# Python file for mapping gps data and color information without down sampling for precise results)
num_drivers = 10
matlab_var = sio.loadmat('Driver.mat')
# Function for matching the lengths
def mapping(inp, _ratio, length1, length2):
    for j in range(2):
        for _i in range(length1):
            if int(_i * _ratio) <= length2:
                path_new[int(_i * _ratio), j] = inp[_i, j]
    for i in range(length2):
        if path_new[i, 0] == 0.0 or path_new[i,1] == 0.0:
            if path_new[i, 0] == 0.0:
                path_new[i, 0] = path_new[i-1,0]
            if path_new[i, 1] == 0.0:
                path_new[i, 1] = path_new[i-1,1]
    return path_new


# Normalizing function
def norm(inp):
    inp = np.array(inp)
    flag = (inp < 1).all() and (inp > 0).all()
    if flag==False:
        inp = (inp - inp.min(0)) / (inp.max(0) - inp.min(0))
    return inp

def sub2mat(driver_id):
    x=int(driver_id/4)
    y=driver_id%4
    return x,y

def read_path(driver_id,lng = [],lat = []):
    data = matlab_var['Driver']
    path = data[0, driver_id]['path']
    for ii in range(3):
        gps_data = path[0, ii]['gps_data']
        for jj in range(len(gps_data)):
            lng.append(gps_data[jj, 1])
            lat.append(gps_data[jj, 2])

    return np.array([lat, lng]).transpose()

# -------------------Main code-------------------

# Give the input color map you wish to plot on the GPS data
f, axarr = plt.subplots(3,4)
for driver_id in range(num_drivers):
    color_map =[]
    color_map = np.loadtxt('tf_out_3/outtest_%d.txt'%driver_id, delimiter=',')
    dim_cmap = np.shape(color_map)
    # Normalize the input
    color_map = norm(color_map)
    # Getting the ratio between color_map and gps data length
    path =read_path(driver_id);
    ratio = dim_cmap[0]/len(path)
    path_new = np.zeros(dim_cmap)
    # Changing the sampling rate for gps data to fit the color map
    path_new = mapping(path, ratio, len(path), dim_cmap[0])
    # Updating the gps data
    dim_new=np.shape(path_new)
    lat = path_new[:, 0]
    lon = path_new[:, 1]
    axarr[sub2mat(driver_id)].scatter(lat, lon, facecolor=color_map)
plt.show()
