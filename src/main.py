'''
Created on May 29, 2018

@author: acy
'''

import numpy as np
import helpers
import time

import sys
import gc

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras import backend as K

model = VGG16(weights='imagenet')

im_dir = sys.argv[1:2][0]
no_imgs = int(sys.argv[2:3][0])

img_ids, im_list = helpers.choose_subset(im_dir, no_imgs)

logfile = open("log.txt","w") 
logfile_nv = open("log_noverbose.txt","w") 

perc_procastinate = 0

org_top1_cnt = 0
org_top5_cnt = 0
ws_top1_cnt = 0
ws_top5_cnt = 0

for i, im_name in enumerate(im_list):
    logfile.write("Image name: " + im_name + "\n")
    img = image.load_img(im_name, target_size=(224, 224))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    pred = model.predict(x)
    
    start_time = time.time()
    
    top1correct, top5correct = helpers.check_accuracy(pred[0], i)
    
    total_mac_cnt = 0
    total_ws_cnt = 0
    for idx, layer in enumerate(model.layers):
        print(layer.name)
        logfile.write("Layer: " + layer.name+ "\n")
        logfile.flush()
        
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
            #logfile.write("Max error: " + str(np.max(np.abs(layer_output-ofmap)))+ "\n")
            logfile.write("# of mac: " + str(mac_cnt) + " # of ws: " + str( ws_cnt) + " , % of skipped: " + str(1 - float(ws_cnt)/mac_cnt) + "\n")
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
            #logfile.write("Max error: " + str(np.max(np.abs(layer_output-ofmap))) + "\n")
            logfile.write("# of mac: " + str(mac_cnt) + " # of ws: " + str( ws_cnt) + " , % of skipped: " + str(1 - float(ws_cnt)/mac_cnt) + "\n")            
            logfile_nv.write(str(1 - float(ws_cnt)/mac_cnt) + " ")
            
            next_layer_input = np.expand_dims(ofmap, axis=0)
            
        elif layer.name.__eq__("predictions"):    
            [weights, biases] = layer.get_weights()
            pred_ws = helpers.fcOutput(next_layer_input[0,:], weights, biases)
            
    print("Max error: ", np.max(np.abs(pred-pred_ws)))   
#     np.savetxt('pred.txt', pred, delimiter=' ') 
#     np.savetxt('pred_ws.txt', pred_ws, delimiter=' ')
    
    print("--- Total time: %s seconds ---" % (time.time() - start_time))
    logfile.write("Final max error: " +  str(np.max(np.abs(pred-pred_ws))) + "\n")
    logfile.write("Total # of mac: " + str(total_mac_cnt) + " total # of ws: " + str(total_ws_cnt) + " , total % of skipped: " +  str(1 - float(total_ws_cnt)/total_mac_cnt) + "\n")
    logfile.write("Total elapsed time: " + str(time.time() - start_time) + "\n")
    
    logfile_nv.write(str(1 - float(total_ws_cnt)/total_mac_cnt) + "\n")
    
    top1correct, top5correct = helpers.check_accuracy(pred[0], i)
    org_top1_cnt = org_top1_cnt + top1correct
    org_top5_cnt = org_top5_cnt + top5correct
    
    top1correct, top5correct = helpers.check_accuracy(pred_ws[0], i)
    ws_top1_cnt = ws_top1_cnt + top1correct
    ws_top5_cnt = ws_top5_cnt + top5correct

    logfile.write("Top1: " + str(top1correct) + " Top5: " + str(top5correct)+ "\n")

    del x
    del ofmap 
    gc.collect()


print('Original top1 accuracy: ', float(org_top1_cnt) / no_imgs)
print('Original top5 accuracy: ', float(org_top5_cnt) / no_imgs)
print('WS top1 accuracy: ', float(ws_top1_cnt) / no_imgs)
print('WS top5 accuracy: ', float(ws_top5_cnt) / no_imgs)

logfile.write("Original top1 accuracy: " +  str(float(org_top1_cnt) / no_imgs) + "\n")
logfile.write("Original top5 accuracy: " +  str(float(org_top5_cnt) / no_imgs) + "\n")
logfile.write("WS top1 accuracy: " +  str(float(ws_top1_cnt) / no_imgs) + "\n")
logfile.write("WS top5 accuracy: " +  str(float(ws_top5_cnt) / no_imgs) + "\n")
