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
#% description: Supervised classification and regression of GRASS rasters using the python tensorflow and keras packages
#% keyword: raster
#% keyword: classification
#% keyword: machine learning
#% keyword: tensorflow
#% keyword: keras
#%end

#%option G_OPT_I_GROUP
#% key: group
#% label: Group of raster layers to be classified
#% description: GRASS imagery group of raster maps representing feature variables to be used in the machine learning model
#% required: yes
#% multiple: no
#%end

#%option G_OPT_R_INPUT
#% key: trmap
#% label: Labelled pixels
#% description: Raster map with labelled pixels for training
#% required: no
#% guisection: Required
#%end

#%option G_OPT_R_OUTPUT
#% key: output
#% label: Labelled pixels
#% description: Raster map with labelled pixels for training
#% required: no
#% guisection: Required
#%end


import atexit
import os, glob
import grass.script as gs
from grass.pygrass.modules.shortcuts import raster as r
from grass.pygrass.gis.region import Region
from grass.pygrass.raster import RasterRow
from grass.pygrass.raster import numpy2raster


############################################################################
# Checking and Loading required packages
############################################################################
def loadPackages():
    try:
        globals()["np"] = __import__("numpy")
    except:
        gs.fatal("Package numpy is not installed")
        exit()
    try:
        globals()["Model"] = __import__("keras").models.Model
        globals()["Sequential"] = __import__("keras").models.Sequential
        globals()["Dense"] = __import__("keras").layers.Dense
        globals()["Conv2D"] = __import__("keras").layers.Conv2D
        globals()["Flatten"] = __import__("keras").layers.Flatten
        globals()["Input"] = __import__("keras").layers.Input
        #globals()["concatenate"] = __import__("keras").layers.concatenate
    except:
        gs.fatal("Packages keras and tensorflow are not installed")
        exit()
    try:
        globals()["train_test_split"] = __import__("sklearn.model_selection", fromlist = ['train_test_split']).train_test_split
        globals()["LabelEncoder"] = __import__("sklearn.preprocessing", fromlist = ['LabelEncoder']).LabelEncoder
        #from sklearn.preprocessing import LabelEncoder
    except:
        gs.fatal("Package sklearn is not installed")
        exit()
    
############################################################################
#
############################################################################
def cleanup():
    for rast in tmp_rast:
        gs.run_command(
            "g.remove", name=rast, type='raster', flags='f', quiet=True)

############################################################################
#
############################################################################
def getData(dataLayer):
    if RasterRow(dataLayer).exist() is True:
        dataFile = RasterRow(dataLayer)
        dataFile.open('r')
        data = np.array(dataFile)
        return data
    else:
        print("Missing", dataLayer)
    
############################################################################
#
############################################################################
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc    

############################################################################
#
############################################################################
def main():
    print("Starting r.keras.class")
    loadPackages()

    # required gui section
    trainingmap = options['trmap']
    response_np = getData(trainingmap)
    print(response_np.shape)
    trData = response_np[response_np > 0]
    group = options['group']
    maplist = gs.read_command("i.group", group=group, flags="g").split(os.linesep)[:-1]
    inputData = np.zeros((trData.shape[0], len(maplist)))
    inputAllData = np.zeros((response_np.shape[0],response_np.shape[1], len(maplist)))
    #allData = np.zeros((trData.shape[0], trData.shape[1], len(maplist)))
    numberOfInputs = len(maplist)
    for i in range(numberOfInputs):
        tmp = getData(maplist[i])
        print(maplist[i], tmp.shape)
        inputData[:,i] = tmp[response_np > 0]
        inputAllData[:,:,i] = tmp

    print(maplist)
    print(inputData.shape)
    
    print(trData.shape)
    u = sorted(np.unique(trData))
    print(u)
    numberOfClasses = len(u)
    trData2 = np.zeros((trData.shape[0]), dtype=np.int)
    print(trData2.shape)
    for i in range(len(u)):
        print(u[i], "=>", np.sum(np.where(trData == u[i], 1, 0)))
        trData2[trData == u[i]] = i

    print(trData2.shape)
    print(inputData.shape)
    inpShape = (1, inputData.shape[1])
    print(inpShape)
    inputData2 = inputData.reshape((inputData.shape[0], 1, inputData.shape[1]))

    predModel = Sequential([
        Flatten(input_shape=inpShape),
        Dense(pow(numberOfClasses, 3), activation="relu"),
        Dense(pow(numberOfClasses, 2), activation="relu"),
        Dense(numberOfClasses, activation="softmax")
    ])
    
    predModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(predModel.summary())
    inputData2.tofile("test.txt", sep="", format="%s")
    predModel.fit(inputData2, trData2, epochs=10)
    predModel.save("path_to_my_model.h5")
    #prediction = 
    #numpy2raster(array=prediction, mtype='CELL', rastname=options['output'], overwrite=True)
                


tmp_rast = []
if __name__ == "__main__":
    options, flags = gs.parser()
    atexit.register(cleanup)
    main()
