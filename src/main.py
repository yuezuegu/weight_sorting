'''
Created on May 29, 2018

@author: acy
'''

import numpy as np
import helpers
import time
import sys

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import backend as K

model = VGG16(weights='imagenet')

im_dir = sys.argv[1:2][0]
im_name = sys.argv[2:3][0]

#img_ids, im_list = helpers.choose_subset(im_dir, no_imgs)

logfile_nv = open("log_noverbose.txt","a") 

perc_procastinate = 0.2

#for i, im_name in enumerate(im_list):
print("Image name: ", im_name)
img = image.load_img(im_dir+im_name, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

pred = model.predict(x)

start_time = time.time()

top1correct, top5correct = helpers.check_accuracy(pred[0], im_name)

total_mac_cnt = 0
total_ws_cnt = 0
for idx, layer in enumerate(model.layers):
    print(layer.name)
    print("Layer: ", layer.name)

    #f = K.function([model.input], [layer.input])
    #layer_input = f([x])[0]
    
    #f = K.function([model.input], [layer.output])
    #layer_output = f([x])[0]
    
    if layer.name.__eq__("block1_conv1"):
        #next_layer_input = layer_output
        [weights, biases] = layer.get_weights()
        ofmap = helpers.conv3d(x[0,:,:,:], weights, biases)
        next_layer_input = np.expand_dims(ofmap, axis=0)
        
    elif layer.name.__contains__("conv"):
        [weights, biases] = layer.get_weights()
        
        ofmap, mac_cnt, ws_cnt, mp_cnt = helpers.conv3dsorted(next_layer_input[0,:,:,:], weights, biases, layer.name, perc_procastinate)
        total_mac_cnt = total_mac_cnt + mac_cnt
        total_ws_cnt = total_ws_cnt + ws_cnt
        
        #print("Max error: ", np.max(np.abs(layer_output-ofmap)))
        print("# of mac: ", mac_cnt, " # of ws: ", ws_cnt, " , % of skipped: ", 1 - float(ws_cnt)/mac_cnt )
        logfile_nv.write(str(1 - float(ws_cnt)/mac_cnt) + " ")
        
        if model.layers[idx+1].name.__contains__("pool"):
            print("% of skipped with MP:", 1 - float(mp_cnt)/mac_cnt)        

        next_layer_input = np.expand_dims(ofmap, axis=0)

    elif layer.name.__contains__("pool"):
        max_out = helpers.max_pool(next_layer_input[0,:,:,:])
        next_layer_input = np.expand_dims(max_out, axis=0)

    elif layer.name.__eq__("flatten"):
        next_layer_input = helpers.flatten(next_layer_input)
        
    elif layer.name.__contains__("fc"):
        [weights, biases] = layer.get_weights()
        
        ofmap, mac_cnt, ws_cnt = helpers.fcSorted(next_layer_input[0,:], weights, biases, layer.name)
        
        total_mac_cnt = total_mac_cnt + mac_cnt
        total_ws_cnt = total_ws_cnt + ws_cnt
        
        #print("Max error: ", np.max(np.abs(layer_output-ofmap)))
        print("# of mac: ", mac_cnt, " # of ws: ", ws_cnt, " , % of skipped: ", 1 - float(ws_cnt)/mac_cnt )
        logfile_nv.write(str(1 - float(ws_cnt)/mac_cnt) + " ")
        
        next_layer_input = np.expand_dims(ofmap, axis=0)
        
    elif layer.name.__eq__("predictions"):    
        [weights, biases] = layer.get_weights()
        pred_ws = helpers.fcOutput(next_layer_input[0,:], weights, biases)
        
print("Max error: ", np.max(np.abs(pred-pred_ws)))   
print("--- Total time: %s seconds ---" % (time.time() - start_time))

logfile_nv.write(str(1 - float(total_ws_cnt)/total_mac_cnt) + " ")

top1correct, top5correct = helpers.check_accuracy(pred[0], im_name)
logfile_nv.write(str(top1correct) + " " + str(top5correct) + " ")

top1correct, top5correct = helpers.check_accuracy(pred_ws[0], im_name)
logfile_nv.write(str(top1correct) + " " + str(top5correct) + "\n")

logfile_nv.flush()
