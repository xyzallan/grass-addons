#!/usr/bin/env python3
# -- coding: utf-8 --
#
############################################################################
#
# MODULE:        r.keras.class
#
# AUTHOR:        Allan Sims
#
# PURPOSE:       Supervised classification and regression of GRASS rasters
#                using the python keras.io package
#
# COPYRIGHT: (c) 2020 Allan Sims, and the GRASS Development Team
#                This program is free software under the GNU General Public
#                for details.
############################################################################

#%module
#% description: Selects values from raster above value of mean plus standard deviation
#% keyword: raster
#% keyword: select
#% keyword: standard deviation
#%end
#%option G_OPT_R_INPUT
#% key: classes
#%end
#%option G_OPT_R_INPUT
#% key: input
#% label: level 1 data
#% multiple: yes
#%end
#%option G_OPT_R_INPUT
#% key: subinput
#% label: level 2 data
#% multiple: yes
#% required: no
#%end
#%option
#% key: subdims
#% type: string
#% description: Submatrix dimension
#%end
#%option
#% key: modelfile
#% type: string
#% description: model file name
#%end
#%option string
#% key: activation
#% label: Activation
#% description: Activation function
#% answer: tanh
#% options: relu,sigmoid,softmax,softplus,softsigh,tanh,selu,elu
#% guisection: Classifier settings
#% required: no
#%end
#%option string
#% key: optimizer
#% label: Optimizer
#% description: Supervised learning model to use
#% answer: adam
#% options: adam,adamax,SGD
#% guisection: Classifier settings
#% required: no
#%end


import sys
import numpy as np

import grass.script as gscript
from grass.exceptions import CalledModuleError

from grass.pygrass.gis.region import Region
from grass.pygrass.raster import RasterRow
from grass.pygrass.raster import numpy2raster


############################################################################
# Reading data from layer
############################################################################
def getData(dataLayer):
    print("Reading data: ", dataLayer)
    if RasterRow(dataLayer).exist() is True:
        dataFile = RasterRow(dataLayer)
        dataFile.open('r')
        data = np.array(dataFile)
        data[np.isnan(data)] = 0
        return data
    else:
        print("Is missing: ", dataLayer)


############################################################################
#
############################################################################
def fitModel(y_train, x_train_sat, x_train_ldr):
    options, flags = gscript.parser()
    from keras.models import Sequential, Model
    from keras.layers import Dense, Conv1D, Conv2D, Flatten, Input, concatenate
    
    optim_func = options['optimizer']
    loss_func = 'sparse_categorical_crossentropy'
    activ_func = options['activation']
    epoch_count = 5
    num_of_cl = (np.amax(y_train) + 1)
    
    satInput = Input(shape=(x_train_sat.shape[1], x_train_sat.shape[2], 1))
    #satX = Conv2D(256, (x_train_sat.shape[1], x_train_sat.shape[2]), activation=activ_func)(satInput)
    satX = Conv2D(256, (x_train_sat.shape[1], x_train_sat.shape[2]), activation=activ_func)(satInput)
    satX = Flatten()(satX)
    satX = Dense(pow(num_of_cl, 3), activation = activ_func)(satX)
    satX = Dense(pow(num_of_cl, 2), activation = activ_func)(satX)
    satX = Model(inputs = satInput, outputs = satX)
    #print(satX.summary())

    ldrInput = Input(shape=(x_train_ldr.shape[1],))
    ldrX = Dense(pow(num_of_cl, 3), activation = activ_func)(ldrInput)
    ldrX = Dense(pow(num_of_cl, 2), activation = activ_func)(ldrX)
    ldrX = Model(inputs = ldrInput, outputs = ldrX)
    #print(ldrX.summary())

    combined = concatenate([satX.output, ldrX.output])

    finX = Dense(pow(num_of_cl, 2), activation = activ_func)(combined)
    finX = Dense(num_of_cl, activation='softmax')(finX)

    model = Model([satX.input, ldrX.input], outputs = finX)
    print(model.summary())
    model.compile(optimizer=optim_func, loss=loss_func, metrics=["accuracy"])
    history = model.fit([x_train_sat, x_train_ldr], y_train, epochs = epoch_count)
    model.save("{}.h5".format(options['modelfile']))

############################################################################
#
############################################################################
def main():
    options, flags = gscript.parser()
    class_raster = getData(options['classes'])
    print(class_raster.shape)
    y_train = class_raster[class_raster > 0]
    print(y_train.shape)
    input_files = (options['input']).split(',')
    x_train_input = np.zeros((y_train.shape[0], len(input_files)))
    print(x_train_input.shape)
    for i in range(len(input_files)):
        x_train_input[:,i] = getData(input_files[i])[class_raster > 0]
    subDims = tuple(np.array(options['subdims'].split(",")).astype(np.int))
    print(subDims)

    subinput_files = (options['subinput']).split(',')
    x_train_subinput = np.zeros((y_train.shape[0], len(subinput_files)))
    for i in range(len(subinput_files)):
        x_train_subinput[:,i] = getData(subinput_files[i])[class_raster > 0]

    x_train_subinput = x_train_subinput.reshape((x_train_subinput.shape[0], subDims[0], subDims[1]))
    print(x_train_subinput.shape)
    fitModel(y_train, x_train_subinput, x_train_input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
