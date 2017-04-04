#!/usr/bin/env python

import numpy as np
import subprocess
import os
#import struct
from preprocess_image import read_preprocess_image

# TODO mozna by vubec nebylo potreba pouzit imagemagick, protoze stejne operace bych zvladl s PIL
# img.resize
# diff:
# http://stackoverflow.com/questions/16720594/comparing-two-images-pixel-wise-with-pil-python-imaging-library

FUZZ = 12
image_div = 2
cluster_size = 10

image_size_x = 1920 / image_div
image_size_y = 1080 / image_div

data_dir = "/home/pepa/projects/camera_filter/learning/"
picture_dir = data_dir + "pictures"
tf_dirs = [ name for name in os.listdir(picture_dir) if os.path.isdir(os.path.join(picture_dir, name)) ]

for d in sorted(tf_dirs):
    out_dir = data_dir + "diff-f%s-%s/%s" % (FUZZ, image_size_x, d)
    cmd = "mkdir -p %s" % (out_dir)
    p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (out, err) = p.communicate()

    in_dir = picture_dir + "/" + d
    files = [ name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name)) ]
    it = iter(sorted(files))
    for f1, f2 in zip(it, it):
        print "Processing files %s %s" % (f1, f2)
        df1 = in_dir + "/" + f1
        df2 = in_dir + "/" + f2
        diff_file = out_dir + "/" + ("%s-diff.png" % f2[:-4])
        if os.path.exists(diff_file):
            print diff_file + " already exists"
        else:
            cmd = "convert %s -resize %sx%s res1.jpg" % (df1, image_size_x, image_size_y)
            p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (out, err) = p.communicate()
            cmd = "convert %s -resize %sx%s res2.jpg" % (df2, image_size_x, image_size_y)
            p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (out, err) = p.communicate()
            cmd = "compare -metric AE -fuzz %s%% res1.jpg res2.jpg -compose src -highlight-color White -lowlight-color Black %s" % (FUZZ, diff_file)
            print cmd
            p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (out, err) = p.communicate()

        pp_file = out_dir + "/" + ("%s-c%s.pp" % (f2[:-4], cluster_size))
        if os.path.exists(pp_file):
            print pp_file + " already exists"
        else:
            npi = read_preprocess_image(diff_file, cluster_size)
            #print npi
            f = open(pp_file, "w")
            npi.astype(np.int16).tofile(f)
            #f.write(struct.pack('h', npi))
            f.close()
            print "%s written" % pp_file

