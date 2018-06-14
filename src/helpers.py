'''
Created on May 29, 2018

@author: acy
'''

import numpy as np
import time
import json 
import os.path

import multiprocessing

def choose_subset(im_dir, no_imgs):
    np.random.seed(0)
    
    img_list = sorted(os.listdir(im_dir))
    
    idx = np.random.permutation(len(img_list))[0:no_imgs]
    #idx = range(no_imgs)
    
    sublist = []
    for i in idx:
        sublist.append( img_list[i] )
    
    return idx, [im_dir+l for l in sublist]


def check_accuracy(prob, img_name):
    synset = [l.strip() for l in open('../data/synset.txt').readlines()]

    gt = [i for i in open('../data/ground_truth.txt').readlines()]
    
    img_names = [i for i in open('../data/image_names.txt').readlines()]
    
    idx = img_names.index(img_name+'\n')
    
    # print prob
    pred = np.argsort(prob)[::-1]
    
    if int(gt[idx]) == pred[0]:
        top1correct = True
        print('Top 1 correct')
    else:
        top1correct = False
        print('Top 1 NOT correct')
    if int(gt[idx]) in pred[0:5]:
        top5correct = True
        print('Top 5 correct')
    else:
        top5correct = False
        print('Top 5 NOT correct')
         
#     top1 = synset[pred[0]]
#     print(("Top1: ", top1, prob[pred[0]]))
#     top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
#     print(("Top5: ", top5))
    
    return top1correct, top5correct


def mult3dsorted(im_sorted, filt_sorted, b):
    mult_out = np.multiply(im_sorted, filt_sorted)
    
    cumsum = np.cumsum(mult_out) + b
    max_ind = np.argmax(cumsum)
    max_val = cumsum[max_ind]
    
    cutoff_point = np.argmax((cumsum[max_ind:]<=0))
    
    if cutoff_point == 0:
        ws_cnt = mult_out.shape[0]
        midsum = max(cumsum[-1], 0)
    else:
        ws_cnt = cutoff_point + max_ind
        midsum = 0
    
    
    

    
    return midsum, ws_cnt, max_ind, max_val

def conv3d(im, filt, b):
    lx = im.shape[0]
    ly = im.shape[1]
    lz = filt.shape[3]
    
    pad_size = int((int(filt.shape[0])-1)/2)
    im_padded = np.pad(im, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=(0, 0))

    ofmap = np.zeros((lx,ly,lz))
    
    for f in range(lz):
        for i in range(lx):
            for j in range(ly):
                ofmap[i,j,f] = np.sum(np.multiply(im_padded[i:i+filt.shape[0], j:j+filt.shape[1], :], filt[:,:,:,f])) + b[f]
    
    ofmap[ofmap<0] = 0
    
    return ofmap
                
def conv3dkernel(args):
    f, filt, filt_ind, b, lx, ly = args
    global im_padded
    
    filt_ind_array = np.array(filt_ind)
    
    filt_sorted = filt[filt_ind_array[:,0], filt_ind_array[:,1], filt_ind_array[:,2]]
    
    ofmap_slice = np.zeros((lx,ly))
    
    ws_cnt = 0
    for i in range(lx):
        for j in range(ly):
            im_sorted = im_padded[filt_ind_array[:,0]+i, filt_ind_array[:,1]+j, filt_ind_array[:,2]]
            
            ofmap_slice[i,j], ws_cnt_, _,_ = mult3dsorted(im_sorted, filt_sorted, b)
            ws_cnt = ws_cnt + ws_cnt_
            
    return ofmap_slice, ws_cnt
    
def conv3dsorted(im, filt, b, layer_name, perc_procastinate):
    start_time = time.time()
    
    lx = im.shape[0]
    ly = im.shape[1]
    lz = filt.shape[3]
    
    pad_size = int((int(filt.shape[0])-1)/2)
    global im_padded
    im_padded = np.pad(im, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=(0, 0))
    
    ofmap = np.zeros((lx,ly,lz))
    
    loadSortedIndex = False
    indexFileName = '../sorted_indexes/filt_ind_' + layer_name + '.json'
    if loadSortedIndex and os.path.exists(indexFileName):
        print('Loading json file')
        with open(indexFileName, 'r') as infile:
            filt_ind = json.load(infile)
    else:
        filt_ind = sortConvFilters(filt, perc_procastinate)
        print('Writing json file')
        with open(indexFileName, 'w') as outfile:
            json.dump(filt_ind, outfile)
    
    ws_cnt = 0
    mac_cnt = filt.shape[0]*filt.shape[1]*filt.shape[2]*im.shape[0]*im.shape[1]*filt.shape[3]
    
    maxVals = [np.zeros((lx, ly)), np.zeros((lx, ly)), np.zeros((lx, ly))]
    mp_cnt = 0
    
    arg_list = []
    for f in range(lz):
        arg_list.append([f, filt[:,:,:,f], filt_ind[f], b[f], lx, ly])
        
    pool = multiprocessing.Pool(processes=lz)
    out_list = pool.map(conv3dkernel, arg_list)        
    
    for f in range(lz):    
        ws_cnt = ws_cnt + out_list[f][1]
        ofmap[:,:,f] = out_list[f][0]

    pool.close()
    pool.join()

    print("--- %s seconds ---" % (time.time() - start_time))
    return ofmap, mac_cnt, ws_cnt, mp_cnt

def sortConvFilters(filt, perc_procrastinate):
    dx = filt.shape[0]
    dy = filt.shape[1]
    dz = filt.shape[2]
    df = filt.shape[3]

    filt_ind = []
    
    for f in range(df):
        filt_ind.append([])
        posW = []
        negW = []
        for i in range(dx):
            for j in range(dy):
                for k in range(dz):
                    if filt[i,j,k,f] > 0:
                        posW.append( (filt[i,j,k,f], (i,j,k)))
                    elif filt[i,j,k,f] < 0:
                        negW.append( (filt[i,j,k,f], (i,j,k)))
        
        posW_sorted = sorted(posW, reverse=True)            
        negW_sorted = sorted(negW)

        s = int(len(posW_sorted) * (1-perc_procrastinate))
        for i in posW_sorted[0:s]:              
            filt_ind[f].append(i[1])

        for i in negW_sorted:              
            filt_ind[f].append(i[1])
    
        for i in posW_sorted[s:]:              
            filt_ind[f].append(i[1])
                
    return filt_ind

def max_pool(im):
    start_time = time.time()
        
    lx = im.shape[0]
    ly = im.shape[1]
    lz = im.shape[2]
    
    out = np.zeros((lx/2,ly/2,lz))
    
    for z in range(lz):
        for y in range(ly/2):
            for x in range(lx/2):
                p1 = im[2*x+0,2*y+0,z]
                p2 = im[2*x+0,2*y+1,z]
                p3 = im[2*x+1,2*y+0,z]
                p4 = im[2*x+1,2*y+1,z]
                
                maxp = np.max([p1,p2,p3,p4])
                out[x,y,z] = maxp
                
    print("--- %s seconds ---" % (time.time() - start_time))
    return out
    
def fc(im,w,b):
    out = np.matmul(im,  w) + b
    out[out<0] = 0
    return out
    
def fcOutput(im,w,b):
    out = np.matmul(im,  w) + b
    out = softmax(out)
    return [out]
    

def fcSorted(im, w, b, layer_name):
    start_time = time.time()

    loadSortedIndex = True
    indexFileName = '../sorted_indexes/filt_ind_' + layer_name + '.json'
    if loadSortedIndex and os.path.exists(indexFileName):
        print('Loading json file')
        with open(indexFileName, 'r') as infile:
            filt_ind = json.load(infile)
    else:
        filt_ind = sortFCFilters(w, 0)
        print('Writing json file')
        with open(indexFileName, 'w') as outfile:
            json.dump(filt_ind, outfile)
    
    print('filter sorted.')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    out = np.zeros((w.shape[1],))
    
    mac_cnt = w.shape[0] * w.shape[1]
    
    ws_cnt = 0
    for i in range(w.shape[1]):
        filt_ind_array = np.array(filt_ind[i])
        w_sorted = w[filt_ind_array, i]
        im_sorted = im[filt_ind_array]
        
        out[i], ws_cnt_ = vectormultSorted(im_sorted, w_sorted, b[i])
        ws_cnt = ws_cnt + ws_cnt_
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return out, mac_cnt, ws_cnt
    

def vectormultSorted(im_sorted, w_sorted, b):
    mult = np.multiply(im_sorted, w_sorted)
    
    cumsum = np.cumsum(mult) + b
    max_ind = np.argmax(cumsum)
    
    cutoff_point = np.argmax((cumsum[max_ind:]<=0))
    
    if cutoff_point == 0:
        ws_cnt = mult.shape[0]
        midsum = max(cumsum[-1], 0)
    else:
        ws_cnt = cutoff_point + max_ind
        midsum = 0

    return midsum, ws_cnt
    
def sortFCFilters(w, perc_procrastinate):
    df = w.shape[1]
    
    filt_ind = []
    
    for f in range(df):  
        filt_ind.append([])
        
        posInd = np.where(w[:,f]>0)[0]
        posW = w[posInd,f].tolist()
        posW_sorted = sorted(zip(posW,posInd), reverse=True)
        
        negInd = np.where(w[:,f]<0)[0]
        negW = w[negInd,f].tolist()
        negW_sorted = sorted(zip(negW,negInd))
        
        s = int(len(posW_sorted) * (1-perc_procrastinate))
        if s < 1.0:
            filt_ind[f] = zip(*posW_sorted[0:s+1])[1] + zip(*negW_sorted)[1] + zip(*posW_sorted[s+1:])[1]
        else:
            filt_ind[f] = zip(*posW_sorted)[1] + zip(*negW_sorted)[1]
        
    return filt_ind
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
    
def flatten(im):
    shape = im.shape
    dim = 1
    for d in shape[0:]:
        dim *= d
    im = np.reshape(im, [-1, dim])
    
    return im
    
    
    
    