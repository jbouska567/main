#!/bin/bash

INSIDE=`owread /28.624183080000/temperature`
OUTSIDE=`owread /28.BACE83080000/temperature`

echo "`date +\%Y-\%m-\%d\ \%H:\%M:\%S`  $INSIDE $OUTSIDE" >> /www/temprecorder/records/rec_"$(date +\%Y\%m\%d)"
