#!/usr/bin/env python3
# -- coding: utf-8 --
#
############################################################################
#
# MODULE:        r.keras.predict
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
#% key: mask
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

#%option G_OPT_R_OUTPUT
#% key: output
#% label: Output Map
#% description: Raster layer name to store result from classification or regression model. The name will also used as a perfix if class probabilities or intermediate of cross-validation results are ordered as maps.
#% guisection: Required
#% required: no
#%end

import sys, os
import numpy as np

import grass.script as gscript
from grass.exceptions import CalledModuleError

from grass.pygrass.gis.region import Region
from grass.pygrass.raster import RasterRow
from grass.pygrass.raster import numpy2raster
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

############################################################################
#
############################################################################
def main():
    options, flags = gscript.parser()
    import keras
    from keras.models import load_model
    #from keras.layers import Dense, Conv2D, Flatten, Input, concatenate
    input_files = gscript.read_command("i.group", group=options['input'], flags="g").split(os.linesep)[:-1]
    print(input_files)
    subinput_files = gscript.read_command("i.group", group=options['subinput'], flags="g").split(os.linesep)[:-1]
    print(subinput_files)
    class_raster = getData(options['mask'])
        
    print(class_raster.shape)
    y_train = class_raster[class_raster > 0]
    print(y_train.shape)
    x_train_input = np.zeros((y_train.shape[0], len(input_files)), dtype=np.float16)
    print(x_train_input.shape)
    for i in range(len(input_files)):
        x_train_input[:,i] = getData(input_files[i])[class_raster > 0]

    x_train_subinput = np.zeros((y_train.shape[0], len(subinput_files)), dtype=np.float16)
    for i in range(len(subinput_files)):
        x_train_subinput[:,i] = getData(subinput_files[i])[class_raster > 0]

    print(x_train_subinput.shape)
    model = load_model(options['modelfile'])
    num_cpus = psutil.cpu_count(logical=True)
    resu_cl = model.predict([x_train_subinput, x_train_input], batch_size=128, use_multiprocessing=True, verbose=1, workers=num_cpus)
    resu = np.argmax(resu_cl, axis=1)
    predictions = np.zeros((class_raster.shape))
    predictions[class_raster > 0] = resu
    numpy2raster(array=predictions, mtype='CELL', rastname=options['output'], overwrite=True)
    for i in range(1, resu_cl.shape[1]):
        predictions = np.zeros((class_raster.shape))
        predictions[class_raster > 0] = resu_cl[:,i]
        numpy2raster(array=predictions, mtype='FCELL', rastname="probs_pred_{}".format(i), overwrite=True)

    
    return 0

if __name__ == "__main__":
    sys.exit(main())
