'''
Created on Jun 17, 2018

@author: acy
'''
import random

len_sublist = 200

img_names = [i for i in open('../data/image_names.txt').readlines()]
rand_smpl = [ img_names[i][0:28] for i in sorted(random.sample(xrange(len(img_names)), len_sublist)) ]

f = open('sublist.txt', 'w')
f.writelines(["%s\n" % item  for item in rand_smpl])
f.close()