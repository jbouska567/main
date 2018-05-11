#!/usr/bin/env python

from datetime import datetime
import numpy as np
import subprocess
import os
from lib.config import OptionParser, Configuration
from lib.preprocess_image import difference_image, read_preprocess_image
import logging
from PIL import Image
from time import sleep
import sys

#TODO promenne z yml configu

image_div = 2
cluster_size = 30

image_size_x = 1920 / image_div
image_size_y = 1080 / image_div

def main(argv):
    parser = OptionParser()
    options, args = parser.parse_args_dict()

    cfg = Configuration(options)

    logging.basicConfig(
        filename=(cfg.yaml['main']['log_dir']+"/preprocessor-"+datetime.now().strftime('%Y%m%d')+".log"),
        level=cfg.log_level,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Started')

    data_dir = cfg.yaml['main']['learn_dir']

    picture_dir = data_dir + "/pictures"
    tf_dirs = [ name for name in os.listdir(picture_dir) if os.path.isdir(os.path.join(picture_dir, name)) ]

    total_pp_files = 0
    new_pp_files = 0

    for d in sorted(tf_dirs):
        out_dir = data_dir + "/diff-%s/%s" % (image_size_x, d)
        cmd = "mkdir -p %s" % (out_dir)
        p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (out, err) = p.communicate()

        in_dir = picture_dir + "/" + d
        files = [ name for name in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, name)) ]
        it = iter(sorted(files))
        for f1, f2 in zip(it, it):
            logging.debug("Processing files %s %s", f1, f2)
            df1 = in_dir + "/" + f1
            df2 = in_dir + "/" + f2

            # cernobily 1 kanalovy diff
            diff_file = out_dir + "/" + ("%s-diff2.png" % f2[:-4])
            if os.path.exists(diff_file):
                logging.debug("%s already exists", diff_file)
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
                logging.info("%s written", diff_file)
                #sleep(0.5)

#       # barevny 3 kanalovy diff
#       diff_file_3 = out_dir + "/" + ("%s-diffc.png" % f2[:-4])
#       if os.path.exists(diff_file_3):
#           logging.debug("%s already exists", diff_file_3)
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
#           logging.info("%s written", diff_file_3)


            pp_file = out_dir + "/" + ("%s-c%s.pp" % (f2[:-4], cluster_size))
            if os.path.exists(pp_file):
                logging.debug("%s already exists", pp_file)
            else:
                npi = read_preprocess_image(diff_file, cluster_size)
                f = open(pp_file, "w")
                npi.astype(np.uint16).tofile(f)
                f.close()
                logging.info("%s written", pp_file)
                new_pp_files += 1

                # kontrolni zobrazeni preprocesovaneho souboru
                # normalizaci co rozsahu 0-255 a vykreslenim jako png
                if False:
                    npi_max = np.amax(npi)
                    npi_png = npi.astype(np.float64) / npi_max
                    npi_png = npi_png * 255
                    npi_png = npi_png.astype(np.uint8)
                    png = Image.fromarray(npi_png, mode='L')
                    png.save(pp_file + ".png")

                #sleep(0.5)
            total_pp_files += 1

    logging.info("Found %s new diff files", new_pp_files)
    logging.info("Found %s total diff files", total_pp_files)

    if not new_pp_files:
        logging.info("No now diff files found")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
