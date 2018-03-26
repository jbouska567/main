*/5 * * * * root /www/camera-filter/bin/mailer.py -f /www/camera-filter/conf/camera-filter.yml >> /www/camera-filter/log/mailer-"$(date +\%Y\%m\%d)".log

0 20 * * * root /www/camera-filter/bin/preprocessor.py -f /www/camera-filter/conf/camera-filter.yml && /www/camera-filter/bin/classifier.py -f /www/camera-filter/conf/camera-filter.yml >> /www/camera-filter/log/classifier-"$(date +\%Y\%m\%d)".log && /usr/sbin/service camera-filter restart

2 0 * * * root find /www/camera-filter/trash/ -mtime +7 -delete; find /www/camera-filter/alarm/ -mtime +7 -delete; find /www/camera-filter/input/ -mtime +1 -delete; find /www/camera-filter/log/ -mtime +14 -delete;
