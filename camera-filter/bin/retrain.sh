#!/bin/bash

/www/camera-filter/bin/preprocessor.py -f /www/camera-filter/conf/camera-filter.yml

# retrain if new pictures exists, or force parametr is set
if [ $? -eq 0 ] || [ $# -eq 1 -a "$1" = "-f" ] ; then
    /www/camera-filter/bin/classifier.py -f /www/camera-filter/conf/camera-filter.yml >> /www/camera-filter/log/classifier-"$(date +\%Y\%m\%d)".log && /usr/sbin/service camera-filter restart
fi
