#!/bin/bash

INSIDE=`owread /28.624183080000/temperature`
OUTSIDE=`owread /28.BACE83080000/temperature`
WATER=`owread /28.FF2FCB801402/temperature`
PI=`vcgencmd measure_temp | cut -d"=" -f2 | cut -d"'" -f1`

echo "`date +\%Y-\%m-\%d\ \%H:\%M:\%S`  $INSIDE $OUTSIDE $WATER $PI" >> /www/temprecorder/records/rec_"$(date +\%Y\%m\%d)"
