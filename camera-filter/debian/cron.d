*/5 * * * * root /www/camera-filter/bin/mailer.py

1 0 * * * root find /www/camera-filter/trash/ -mtime +7 -delete; find /www/camera-filter/alarm/ -mtime +7 -delete; find /www/camera-filter/input/ -mtime +1 -delete;
