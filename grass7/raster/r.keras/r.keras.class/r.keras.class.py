#!/usr/bin/env python3
# -- coding: utf-8 --
#
############################################################################
#
# MODULE:        r.keras.class
#
# AUTHOR:        Allan Sims
#
# PURPOSE:       Supervised classification of GRASS rasters 
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
#% description: Training classes
#%end
#%option G_OPT_I_INPUT
#% key: input
#% label: Level data
#% description: Input data
#%end
#%option G_OPT_I_INPUT
#% key: subinput
#% label: Sublevel data
#% description: Input data
#% required: no
#%end
#%option
#% key: modelfile
#% type: string
#% description: model output file name
#%end
#%option
#% key: epochs
#% type: integer
#% answer:10
#% description: epoch count
#%end
#%option string
#% key: activation
#% label: Activation
#% description: Activation function used for fitting
#% answer: tanh
#% options: relu,sigmoid,softmax,softplus,softsigh,tanh,selu,elu
#% guisection: Classifier settings
#% required: no
#%end
#%option string
#% key: optimizer
#% label: Optimizer
#% description: Optimizer used for fitting
#% answer: adam
#% options: adam,adamax,SGD
#% guisection: Classifier settings
#% required: no
#%end

import sys, os

import grass.script as gscript
from grass.exceptions import CalledModuleError

from grass.pygrass.gis.region import Region
from grass.pygrass.raster import RasterRow
from grass.pygrass.raster import numpy2raster

#from keras.models import Sequential, Model
#from keras.layers import Dense, Conv1D, Conv2D, Flatten, Input, concatenate

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import psutil



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
        exit()


############################################################################
# 
############################################################################
def calcKappa(classes, predictions):
    import sklearn.metrics as sm

    resu = np.argmax(predictions, axis=1)
    clas = np.argmax(classes, axis=1)
    print("= Cohen's Kappa score: ===========================================")
    print("{:.3f}".format(sm.cohen_kappa_score(clas, resu)))
    print("= Confusion matrix: ==============================================")
    print(sm.confusion_matrix(clas, resu))
    print("= Classification report: =========================================")
    print(sm.classification_report(clas, resu))

    with open('result_data.npy', 'wb') as f:
        np.save(f, classes)
        np.save(f, predictions)
    return 0
        

############################################################################
#
############################################################################
def fitModelSingleLevel(y_tr, x_tr_input):
    options, flags = gscript.parser()
    batch_size = 512
    epochs = int(options['epochs'])
    
    optim_func = options['optimizer']
    loss_func = 'sparse_categorical_crossentropy'
    activ_func = options['activation']

    y_tr_c = keras.utils.to_categorical(y_tr)
    num_classes = y_tr_c.shape[1]
    
    input_model = keras.Input(shape=x_tr_input.shape[1:])
    input_sequence = layers.Dense(pow(num_classes, 3), activation=activ_func)(input_model)
    input_sequence = layers.Dense(pow(num_classes, 2), activation=activ_func)(input_sequence)
    input_sequence = layers.Dense(num_classes, activation="softmax")(input_sequence)
    model_to_fit = models.Model(inputs = input_model, outputs = input_sequence)
    print(model_to_fit.summary())
    model_to_fit.compile(loss="categorical_crossentropy", optimizer=optim_func, metrics=["binary_accuracy"])
    model_to_fit.fit(x_tr_input, y_tr_c, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    model_to_fit.save(options['modelfile'])
    num_cpus = psutil.cpu_count(logical=True)
    resu_cl = model_to_fit.predict(x_tr_input, batch_size=128, use_multiprocessing=True, verbose=1, workers=num_cpus)
    calcKappa(y_tr_c, resu_cl)
    return 0
    


############################################################################
#
############################################################################
def fitModelWithSub(y_tr, x_tr_subinput, x_tr_input):
    options, flags = gscript.parser()
    batch_size = 512
    epochs = int(options['epochs'])
    
    optim_func = options['optimizer']
    loss_func = 'sparse_categorical_crossentropy'
    activ_func = options['activation']

    y_tr_c = keras.utils.to_categorical(y_tr)
    num_classes = y_tr_c.shape[1]
    x_tr_subinput = np.expand_dims(x_tr_subinput, -1)

    print(x_tr_subinput.shape)

    input_1 = keras.Input(shape=x_tr_subinput.shape[1:])
    seq_1 = layers.Conv1D(256, kernel_size=x_tr_subinput.shape[1], activation=activ_func)(input_1)
    seq_1 = layers.Dense(pow(num_classes, 3), activation=activ_func)(seq_1)
    seq_1 = layers.Dense(pow(num_classes, 2), activation=activ_func)(seq_1)
    seq_1 = layers.Flatten()(seq_1)
    mudel1 = models.Model(inputs = input_1, outputs = seq_1)
    print(mudel1.summary())

    input_2 = keras.Input(shape=x_tr_input.shape[1:])
    seq_2 = layers.Dense(pow(num_classes, 3), activation=activ_func)(input_2)
    seq_2 = layers.Dense(pow(num_classes, 2), activation=activ_func)(seq_2)
    mudel2 = models.Model(inputs = input_2, outputs = seq_2)
    print(mudel2.summary())

    combined = keras.layers.concatenate([mudel1.output, mudel2.output])

    mudelLopp = layers.Dense(64, activation=activ_func)(combined)
    mudelLopp = layers.Dense(num_classes, activation="softmax")(mudelLopp)

    model_to_fit = models.Model(inputs=[mudel1.input, mudel2.input], outputs=mudelLopp)
    print(model_to_fit.summary())

    model_to_fit.compile(loss="categorical_crossentropy", optimizer=optim_func, metrics=["binary_accuracy"])
    model_to_fit.fit([x_tr_subinput, x_tr_input], y_tr_c, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    model_to_fit.save(options['modelfile'])
    num_cpus = psutil.cpu_count(logical=True)
    resu_cl = model_to_fit.predict([x_tr_subinput, x_tr_input], batch_size=128, use_multiprocessing=True, verbose=1, workers=num_cpus)
    calcKappa(y_tr_c, resu_cl)
    #resu = np.argmax(resu_cl, axis=1)

    return 0
    
    

############################################################################
#
############################################################################
def main():
    options, flags = gscript.parser()
    print(options)
    if not options['optimizer'] in ['adam', 'adamax', 'SGD']:
        print("Wrong optimizer")
        exit()
    if not options['activation'] in ['relu', 'sigmoid', 'softmax', 'softplus', 'softsigh', 'tanh', 'selu', 'elu']:
        print("Wrong activation")
        exit()

    class_raster = getData(options['classes'])
    print(class_raster.shape)
    y_train = class_raster[class_raster > 0]
    print(y_train.shape)
    
    input_files = gscript.read_command("i.group", group=options['input'], flags="g").split(os.linesep)[:-1]
    print(input_files)
    x_train_input = np.zeros((y_train.shape[0], len(input_files)), dtype=np.float16)
    print(x_train_input.shape)
    for i in range(len(input_files)):
        x_train_input[:,i] = getData(input_files[i])[class_raster > 0]

    if len(options['subinput']) > 0:
        subinput_files = gscript.read_command("i.group", group=options['subinput'], flags="g").split(os.linesep)[:-1]
        print(subinput_files)

        x_train_subinput = np.zeros((y_train.shape[0], len(subinput_files)), dtype=np.float16)
        for i in range(len(subinput_files)):
            x_train_subinput[:,i] = getData(subinput_files[i])[class_raster > 0]

        print(x_train_subinput.shape)
        with open('training_data.npy', 'wb') as f:
            np.save(f, y_train)
            np.save(f, x_train_input)
            np.save(f, x_train_subinput)
        
        fitModelWithSub(y_train, x_train_subinput, x_train_input)
    else:
        print("Only single level data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
