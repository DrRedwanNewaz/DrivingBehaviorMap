import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
from sklearn.preprocessing import normalize

# Python file for mapping gps data and color information without down sampling for precise results)


class CMap(object):
    def __init__(self, input_file_path, driver_id):
        self.input_file_path = input_file_path
        self.driver_id = driver_id
        self.lat = []
        self.lng = []
        self.path =[]

    def read_path(self):
        matlab_var = sio.loadmat('Driver.mat')
        data = matlab_var['Driver']
        path = data[0, self.driver_id]['path']
        for ii in range(3):
            gps_data = path[0, ii]['gps_data']
            for jj in range(len(gps_data)):
                self.lng.append(gps_data[jj, 1])
                self.lat.append(gps_data[jj, 2])

        return np.array([self.lat, self.lng]).transpose()
    def mapping(self, inp, _ratio, length1, length2):
        path_new = np.matrix(np.zeros((length2,2)))
        inp = np.matrix(inp)
        # print(np.shape(inp))
        for j in range(2):
            for _i in range(length1):
                if int(_i * _ratio) <= length2:
                    path_new[int(_i * _ratio), j] = inp[_i, j]
        for i in range(length2):
            if path_new[i, 0] == 0.0 or path_new[i, 1] == 0.0:
                if path_new[i, 0] == 0.0:
                    path_new[i,0] = path_new[i - 1, 0]
                if path_new[i,1] == 0.0:
                    path_new[i,1] = path_new[i - 1, 1]
        return path_new

    def norm(self,inp):
        inp = np.array(inp)
        flag = (inp < 1).all() and (inp > 0).all()
        if flag==False:
            inp = (inp - inp.min(0)) / (inp.max(0) - inp.min(0))
        return inp

    def load_cmap(self):
        cmap = np.loadtxt(self.input_file_path, delimiter=',')

        dim_cmap = np.shape(cmap)
        path = self.read_path()
        dim_path = len(self.lat)
        ratio = dim_cmap[0]/dim_path
        new_path = self.mapping(path, ratio, dim_path, dim_cmap[0])
        self.lat = new_path[:, 0]
        self.lng = new_path[:, 1]
        cmap = self.norm(cmap)
        plt.scatter(self.lat, self.lng, facecolor=cmap)
        plt.show()







class DataBase(object):
    def __init__(self,data, driver_id):
        self.data = data
        self.driver_id = driver_id
        self.lat = []
        self.lng = []
        self.path =[]

    def read_path(self):
        matlab_var = sio.loadmat('matlabFile/Driver.mat')
        data = matlab_var['Driver']
        path = data[0, self.driver_id]['path']
        for ii in range(3):
            gps_data = path[0, ii]['gps_data']
            for jj in range(len(gps_data)):
                self.lng.append(gps_data[jj, 1])
                self.lat.append(gps_data[jj, 2])

        return np.array([self.lat, self.lng]).transpose()
    def mapping(self, inp, _ratio, length1, length2):
        path_new = np.matrix(np.zeros((length2,2)))
        inp = np.matrix(inp)
        # print(np.shape(inp))
        for j in range(2):
            for _i in range(length1):
                if int(_i * _ratio) <= length2:
                    path_new[int(_i * _ratio), j] = inp[_i, j]
        for i in range(length2):
            if path_new[i, 0] == 0.0 or path_new[i, 1] == 0.0:
                if path_new[i, 0] == 0.0:
                    path_new[i,0] = path_new[i - 1, 0]
                if path_new[i,1] == 0.0:
                    path_new[i,1] = path_new[i - 1, 1]
        return path_new

    def norm(self,inp):
        inp = np.array(inp)
        flag = (inp < 1).all() and (inp > 0).all()
        if flag==False:
            inp = (inp - inp.min(0)) / (inp.max(0) - inp.min(0))
        return inp

    def show(self):
        cmap = self.data
        dim_cmap = np.shape(cmap)
        path = self.read_path()
        dim_path = len(self.lat)
        ratio = dim_cmap[0] / dim_path
        # path =np.vstack((self.lat, self.lng))
        # path = np.array(path)
        #
        # path = np.shape(path)
        # print(np.shape(self.path))
        # print(dim_path, dim_cmap)
        new_path = self.mapping(path, ratio, dim_path, dim_cmap[0])
        self.lat = new_path[:, 0]
        self.lng = new_path[:, 1]
        cmap = normalize(cmap)
        plt.figure()
        plt.scatter(self.lat, self.lng, facecolor=(cmap))

    def view(self):
        plt.show()


class rgb_converter(object):
    """docstring for rgb_converter"""

    def __init__(self, data):
        self.data = data
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
        matrix = self.data
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


