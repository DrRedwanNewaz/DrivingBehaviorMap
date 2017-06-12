import numpy as np
from keras.models import load_model
from dataloader import data_base
from update_cmap import DataBase,rgb_converter

# loading h5 file
generator = load_model('model/generator-2.h5')
encoder = load_model('model/encoder_driver_2.h5')

for id in range(9):
    id = id+1
    driver_id = [id]
    file_reader = data_base(driver_id)
    test = file_reader.read_files()
    zsamples = np.random.normal(size=(len(test), 3))
    gen = np.array(generator.predict(zsamples))
    gen = test.reshape((test.shape[0], -1), order='F')
    colormap = encoder.predict(gen)
    obj = DataBase(data=colormap, driver_id=id)
    obj.show()

obj.view()



#
# from update_cmap import CMap
#
# id = 1
# path = 'driver_%d_map.txt' % id
# obj = CMap(input_file_path=path, driver_id=id)
# obj.load_cmap()