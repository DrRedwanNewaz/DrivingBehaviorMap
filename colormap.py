import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
from sklearn.preprocessing import normalize


def mapping(inp, _ratio, length1, length2):
    path_new = np.zeros((length2,2))
    for j in range(2):
        for _i in range(length1):
            if int(_i * _ratio) <= length2:
                path_new[int(_i * _ratio), j] = inp[_i, j]
    for i in range(length2):
        if path_new[i, 0] == 0.0 or path_new[i, 1] == 0.0:
            if path_new[i, 0] == 0.0:
                path_new[i, 0] = path_new[i - 1, 0]
            if path_new[i, 1] == 0.0:
                path_new[i, 1] = path_new[i - 1, 1]
    return path_new


class colormap(object):
    def __init__(self,driver_id):
        self.driver_id = driver_id
        self.matlab_var = sio.loadmat('Driver.mat')

    def read_path(self,driver_id, lng=[], lat=[]):
        data = self.matlab_var['Driver']
        path = data[0, driver_id]['path']
        for ii in range(3):
            gps_data = path[0, ii]['gps_data']
            for jj in range(len(gps_data)):
                lng.append(gps_data[jj, 1])
                lat.append(gps_data[jj, 2])
        return np.array([lat, lng]).transpose()



    def show(self):
        color_map = np.loadtxt('output2/driver_%d_map.txt' % (self.driver_id), delimiter=',')
        color_map = normalize(color_map,axis=0)
        np.savetxt('color_map.txt',color_map,delimiter=',')
        path = self.read_path(self.driver_id)
        min_size = min(len(path),len(color_map))
        max_size = max(len(path), len(color_map))
        ratio = (max_size/min_size)
        print("ratio ",ratio)
        path_new = mapping(path,ratio, len(path), len(color_map))
        lat = path_new[:,0]
        lng = path_new[:,1]


        plt.scatter(lat, lng, facecolor=color_map)
        plt.show()



class rgb_converter(object):
    """docstring for rgb_converter"""

    def __init__(self, driver_id):
        self.path = 'driver_%d_map.txt'%driver_id
        self.R_values = []
        self.G_values = []
        self.B_values = []

    def load_rgb_data(self):
        # read the output of DSAE
        matrix = np.loadtxt(self.path, delimiter=',')
        return matrix

    def map(self, _values):
        #  map data to 1 to 255
        _min = np.amin(_values)
        _max = np.amax(_values)
        _values = np.subtract(_values, _min)
        _values = (np.divide(_values, _max - _min) * 255)
        _values = _values.astype(int)
        # Running mean for smoothening the characteristics
        _norm = np.zeros((len(_values),))
        N = int(len(_values) / 2)
        for ctr in range(len(_values), ):
            _norm[ctr] = int(np.sum(_values[ctr:(ctr + N)]) / N)
        _norm = _norm.astype(int);

        return _norm

    def get_spectreum(self):
        matrix = self.load_rgb_data();
        self.R_values = self.map(matrix[:, 0])
        self.G_values = self.map(matrix[:, 1])
        self.B_values = self.map(matrix[:, 2])
        spectrum_array = np.array([self.R_values, self.G_values, self.B_values])
        return spectrum_array

    def draw_img(self):
        spectrum_array = self.get_spectreum()
        dim = np.shape(spectrum_array)
        img = Image.new("RGB", (dim[0], 50))
        pix = img.load()
        for i in range(dim[0]):
            for j in range(50):
                pix[i, j] = (self.R_values[i], self.G_values[i], self.B_values[i])
        img.save("driving_behavior.png", "PNG")

        print("image width ", img.width, "image height ", img.height)
        new_width = 1080
        new_height = img.height
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        return img

    def write_csv(self, filename):
        spectrum_array = self.get_spectreum()
        np.savetxt(filename, spectrum_array, delimiter=',')

    def show(self):
        img = self.draw_img()
        plt.imshow(img)
        plt.show()






