#!/usr/bin/env python

import numpy as np
import subprocess
import os
from lib.preprocess_image import difference_image, read_preprocess_image
from PIL import Image

#TODO promenne z yml configu

image_div = 2
cluster_size = 30

image_size_x = 1920 / image_div
image_size_y = 1080 / image_div

#TODO do konfigu
data_dir = "/home/pepa/projects/camera_filter/learning/"
picture_dir = data_dir + "pictures"
tf_dirs = [ name for name in os.listdir(picture_dir) if os.path.isdir(os.path.join(picture_dir, name)) ]


for d in sorted(tf_dirs):
    out_dir = data_dir + "diff-%s/%s" % (image_size_x, d)
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

        # cernobily 1 kanalovy diff
        diff_file = out_dir + "/" + ("%s-diff2.png" % f2[:-4])
        if os.path.exists(diff_file):
            print diff_file + " already exists"
        else:
            img1 = Image.open(df1)
            img2 = Image.open(df2)
            img1 = img1.resize((image_size_x, image_size_y), Image.ANTIALIAS)
            img2 = img2.resize((image_size_x, image_size_y), Image.ANTIALIAS)
            img1 = img1.convert('L')
            img2 = img2.convert('L')
            np_img1 = np.array(img1.getdata(),dtype=np.uint8).reshape((image_size_y, image_size_x))
            np_img2 = np.array(img2.getdata(),dtype=np.uint8).reshape((image_size_y, image_size_x))
            np_img_diff = difference_image(np_img1, np_img2)
            img_diff = Image.fromarray(np_img_diff, mode='L')
            img_diff.save(diff_file)
            print "%s written" % diff_file

#       # barevny 3 kanalovy diff
#       diff_file_3 = out_dir + "/" + ("%s-diffc.png" % f2[:-4])
#       if os.path.exists(diff_file_3):
#           print diff_file_3 + " already exists"
#       else:
#           img1 = Image.open(df1)
#           img2 = Image.open(df2)
#           img1 = img1.resize((image_size_x, image_size_y), Image.ANTIALIAS)
#           img2 = img2.resize((image_size_x, image_size_y), Image.ANTIALIAS)
#           np_img1 = np.array(img1.getdata(),dtype=np.uint8).reshape((image_size_y, image_size_x, 3))
#           np_img2 = np.array(img2.getdata(),dtype=np.uint8).reshape((image_size_y, image_size_x, 3))
#           np_img_diff = difference_image(np_img1, np_img2)
#           img_diff = Image.fromarray(np_img_diff)
#           img_diff.save(diff_file_3)
#           print "%s written" % diff_file_3


        pp_file = out_dir + "/" + ("%s-c%s.pp" % (f2[:-4], cluster_size))
        if os.path.exists(pp_file):
            print pp_file + " already exists"
        else:
            npi = read_preprocess_image(diff_file, cluster_size)
            f = open(pp_file, "w")
            npi.astype(np.uint16).tofile(f)
            f.close()
            print "%s written" % pp_file

            # kontrolni zobrazeni preprocesovaneho souboru
            # normalizaci co rozsahu 0-255 a vykreslenim jako png
            if False:
                npi_max = np.amax(npi)
                npi_png = npi.astype(np.float64) / npi_max
                npi_png = npi_png * 255
                npi_png = npi_png.astype(np.uint8)
                png = Image.fromarray(npi_png, mode='L')
                png.save(pp_file + ".png")



