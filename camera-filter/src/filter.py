#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from lib.multilayer_perceptron import MultilayerPerceptron
from PIL import Image
from lib.preprocess_image import difference_image, read_preprocess_image

import subprocess
import traceback
from sys import exit, argv, stdin, stdout
from time import localtime, strftime, sleep
from ftplib import FTP
from optparse import OptionParser as BaseParser
import yaml

import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

# TODO doplnit debian zavislosti

# TODO udelat z toho service
# TODO pozor na posilani chyb az to bude sluzba, mohlo by se stat, ze pri nejake nezotavitelne chybe me to pekne vyspamuje!

# TODO zlepsit nacitani z FTP

#TODO ovladani mailem zasalnym zpet na adresu odesilatele (napr. vypinani, zapinani upozorneni, poslani statistik, atd.)


def build_dict(seq, key):
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))

class OptionParser(BaseParser):
    def __init__(self, **kwargs):
        BaseParser.__init__(self, **kwargs)

        self.add_option('-f', '--config', help='the config file')
        self.add_option('-v', '--verbose', action='store_true')

    def parse_args_dict(self):
        options, args = self.parse_args()
        return vars(options), args

class Configuration:
    def __init__(self):
        parser = OptionParser()
        parser.add_option('--once', action='store_true', help='start once, only to process batch')
        parser.add_option('--noftp', action='store_true', help='prevent loading files from FTP server (overrides config)')
        parser.add_option('--nomail', action='store_true', help='prevent sending mail (overrides config)')
        options, args = parser.parse_args_dict()
        config_file = options['config']
        if not config_file:
            parser.print_help()
            raise Exception("Config file argument is requeired")

        stream = open(config_file)
        self.yaml = yaml.load(stream)
        self.once_opt = ('once' in options) and (options['once'] is True)
        self.ftp_opt = self.yaml['main']['ftp_opt']
        if 'noftp' in options and options['noftp']:
            self.ftp_opt = False
        self.mail_opt = self.yaml['main']['mail_opt']
        if 'nomail' in options and options['nomail']:
            self.mail_opt = False

        self.input_batch_size = self.yaml['main']['input_batch_size']
        self.image_size_x = self.yaml['classifier']['image_size_x'] / self.yaml['classifier']['image_div']
        self.image_size_y = self.yaml['classifier']['image_size_y'] / self.yaml['classifier']['image_div']
        self.cluster_size = self.yaml['classifier']['cluster_size']
        self.n_input = (self.image_size_x / self.cluster_size) * (self.image_size_y / self.cluster_size) * self.yaml['classifier']['channels']


#TODO nejake standardni logovani
class Logger:
    def __init__(self, cfg_yaml):
        self.log_dir = cfg_yaml['main']['log_dir']
        self.log_expire_days = cfg_yaml['expiration']['log_expire_days']
        self.log_file = open("%s/log" % self.log_dir, 'a')

    def rotate(self):
        self.log_file.close()
        print "mv %s/log %s/log-%s" % (self.log_dir, self.log_dir,
            strftime("%Y-%m-%d", localtime()))
        subprocess.call("mv %s/log %s/log-%s" % (self.log_dir, self.log_dir,
            strftime("%Y-%m-%d", localtime())), shell=True)
        self.log_file = open('%s/log' % self.log_dir, 'a')
        # expirace
        subprocess.call("find %s/* -mtime +%s -exec rm {} \;" %
            (self.log_dir, self.log_expire_days), shell=True)

    def log(self, text, level="INFO"):
        line = strftime("%Y-%m-%d %H:%M:%S ", localtime()) + level + ": " + text
        print line
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def get_hour():
    return int(strftime("%H", localtime()))

def connect_ftp(cfg_yaml):
    ftp = FTP(cfg_yaml['ftp']['server'])
    ftp.login(cfg_yaml['ftp']['user'], cfg_yaml['ftp']['passwd'])
    ftp.cwd(cfg_yaml['ftp']['dir'])
    return ftp

#TODO toto je potreba zlepsit, protoze se stava, ze se nacte seznam souboru, kdyz tam jeste nejsou
# vsechny 3. To dela problem hlavne, kdyz se nekdy predtim nacetly jen 2 fotky celkem, coz se obcas
# taky stava (nevim proc), ale pak se to spatne sparuje a hned potom se donacte ten zbytek.
# -> melo by se to pokouset nekolirat nacitat znovu dokavad nejsou 3, a pokud ne a jsou aspon 2
# tak vratit pocet. Ten seznam ftp.nlst() by to chtelo taky seradit, protoze jich muze byt i vic nez 3.
# TODO lepe ostrit chyby spojeni
# zatim zname chyby
#2016-11-06 13:49:00: ftp connection lost: 421 No Transfer Timeout (300 seconds): closing control connection
#2016-11-06 13:57:15: ftp connection lost: [Errno 32] Broken pipe

def fetch_files(ftp, cfg_yaml, logger):
    # zkusime zda je navazane spojeni s FTP serverem
    try:
        ftp.voidcmd("NOOP")
    except Exception as ex:
        # s nejvetsi pravdepodobnosti doslo k timeoutu, nebo jinemu duvodu odpojeni
        logger.log("ftp connection lost: %s" %ex, level="WARN")
        # quit nejspis neni mozny, protoze spojeni uz v tu chvili neexistuje
        #ftp.quit()
        # zkusime se znovu spojit
        ftp = connect_ftp(cfg_yaml)
        # pokud stale nic, koncime s chybou
        ftp.voidcmd("NOOP")

    ftp_list = [x for x in ftp.nlst() if x[:3]=="ARC"]
    if len(ftp_list) < 3:
        # kamera jeste neuploadnula vsechny 3 fotky, pockame
        # TODO obcas se ale stalo, ze 3. uz neprisla - co s tim? Pak se treba nahraje prvni
        # z dalsi serie 3 fotek a je problem.
        return
    # kratka pauza, jelikoz bez ni muze dojit k tomu, ze nejaky soubor z precteneho seznamu
    # jeste nemusi byt kompletne nahran na FTP ze ktereho cteme
    sleep(1)
    for file in ftp_list:
        ftp.retrbinary("RETR %s" % file, open("%s/%s" % (cfg_yaml['main']['input_dir'], file), "wb").write)
        logger.log("retrieved %s file from ftp" % file)
        ftp.delete(file)
    sleep(1)


def send_mail(logger, cfg_yaml, subj, text, files=None, notify=False):

    msg = MIMEMultipart()
    msg['From'] = cfg_yaml['mail']['from']
    msg['To'] = cfg_yaml['mail']['to']
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subj

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)

    smtp = smtplib.SMTP()
    tries = 5
    while tries:
        try:
            smtp.connect(cfg_yaml['mail']['server'], 587) #port 587 TLS, 25 nezabezpecene spojeni
            smtp.ehlo()     #TLS only
            smtp.starttls() #TLS only
            smtp.login(cfg_yaml['mail']['from'], cfg_yaml['mail']['from_passwd'])
            tries = 0
        except Exception as ex:
            logger.log("%s" %ex, level="ERROR")
            sleep(5 - tries)
            tries = tries - 1
            if not tries:
                raise ex

    smtp.sendmail(cfg_yaml['mail']['from'], cfg_yaml['mail']['to'], msg.as_string())

    if notify:
        msg = MIMEMultipart()
        msg['From'] = cfg_yaml['mail']['from']
        msg['To'] = cfg_yaml['mail']['notify_to']
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subj

        smtp.sendmail(cfg_yaml['mail']['from'], cfg_yaml['mail']['notify_to'], msg.as_string())

    smtp.close()

def get_np_img(file_name, image_size_x, image_size_y):
    img = Image.open(file_name)
    img = img.resize((image_size_x, image_size_y), Image.ANTIALIAS)
    img = img.convert('L')
    np_img = np.array(img.getdata(),dtype=np.uint8).reshape((image_size_y, image_size_x))
    return np_img

def process_mp(cfg, logger, sess, model, files):

    diff_file = files[-1][:-4] + "-diff.jpg"
    time_str = strftime("%Y-%m-%d %H:%M:%S", localtime())

    try:
        np_img1 = get_np_img(files[0], cfg.image_size_x, cfg.image_size_y)
        np_img2 = get_np_img(files[-1], cfg.image_size_x, cfg.image_size_y)
        np_img_diff = difference_image(np_img1, np_img2)
        img_diff = Image.fromarray(np_img_diff, mode='L')
        img_diff.save(diff_file)
        print "%s written" % diff_file
    except Exception as ex:
        if cfg.mail_opt:
            send_mail(logger, cfg.yaml, 'Camera ALARM Error ' + time_str,
                "Error while preprocessing images", files=[files[0], files[-1]], notify=True)
        subprocess.call("mv %s %s %s/" % (files[0], files[-1],
            cfg.yaml['main']['error_dir']), shell=True)
        raise ex

    try:
        npi = read_preprocess_image(diff_file, cfg.cluster_size)
        npi = npi.reshape(cfg.n_input)
        cl = sess.run(tf.argmax(model.out_layer, 1), feed_dict={model.input_ph: [npi]})
    except Exception as ex:
        if cfg.mail_opt:
            send_mail(logger, cfg.yaml, 'Camera ALARM Error ' + time_str,
                "Error while evaluating images", files=[files[0], files[-1], diff_file], notify=True)
        subprocess.call("mv %s %s %s %s/" % (files[0], files[-1], diff_file,
            cfg.yaml['main']['error_dir']), shell=True)
        raise ex

    if cl:
        logger.log("! ALARM")
        if cfg.mail_opt:
            send_mail(logger, cfg.yaml, 'Camera ALARM ' + time_str, '',
                files=[files[0], files[-1], diff_file], notify=True)
        subprocess.call("mv %s %s %s %s/" % (files[0], files[-1], diff_file,
            cfg.yaml['main']['alarm_dir']), shell=True)
    else:
        logger.log("False alarm")
        subprocess.call("mv %s %s %s %s/" % (files[0], files[-1], diff_file,
            cfg.yaml['main']['trash_dir']), shell=True)

    if len(files) > 2:
        subprocess.call("rm %s" % (' '.join(files[1:len(files)-1]), ), shell=True)

    logger.log("----------")
    return cl


def main():
    cfg = Configuration()
    logger = Logger(cfg.yaml)

    # Construct model
    model_path = cfg.yaml['main']['model_dir'] + '/' + cfg.yaml['classifier']['model_name']
    model = MultilayerPerceptron(
        cfg.n_input,
        cfg.yaml['classifier']['n_hidden_1'],
        cfg.yaml['classifier']['n_hidden_2'],
        cfg.yaml['classifier']['n_classes'])
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()
    print "opening model %s" % model_path
    saver.restore(sess, model_path)

    prev_hour = get_hour()
    stats_true = 0
    stats_false = 0

    if cfg.ftp_opt:
        ftp = connect_ftp(cfg.yaml)

    err_count = 0

    while(err_count < 2):
        try:
            hour = get_hour()
            if prev_hour != hour:
                # logrotate a expirace starych souboru o pulnoci
                if hour == 0:
                    logger.rotate()

                    logger.log("Deleting files from alarm and trash older than %s days" % cfg.yaml['expiration']['expire_days'])
                    subprocess.call("find trash/* -mtime +%s -delete" % cfg.yaml['expiration']['expire_days'], shell=True)
                    subprocess.call("find alarm/* -mtime +%s -delete" % cfg.yaml['expiration']['expire_days'], shell=True)

                # monitoring + hodinove statistiky
                p = subprocess.Popen(["du -sh /www/camera-filter/", ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                (out, err) = p.communicate()
                status = p.wait()
                text = ("Hourly statistics:\nALARMS count: %s\nFalse alarms count: %s\n\nError count: %s\n\nTotal size of dir: %s" %
                    (stats_true, stats_false, err_count, out))
                logger.log(text)
                if cfg.mail_opt:
                    send_mail(logger, cfg.yaml, "Camera stats", text)

                stats_true = 0
                stats_false = 0
                err_count = 0
                prev_hour = hour

            if cfg.ftp_opt:
                fetch_files(ftp, cfg.yaml, logger)

            # TODO lepsi prace se soubory, pres pythoni libky
            # TODO lepe zpracovat davku (nenacitat znovu porad dokola)
            # TODO co se souborama s nulovym timestampem? (rozhodi nam parovani fotek)
            p = subprocess.Popen("find %s -maxdepth 1 -type f | grep ARC | grep -v diff | sort -r | head -n%s" % (cfg.yaml["main"]["input_dir"], cfg.input_batch_size),
                stdout=subprocess.PIPE, shell=True)
            (output, err) = p.communicate()
            p_status = p.wait()

            files = output.split()
            if not files:
                if cfg.once_opt:
                    # pokud nechcem sluzbu a mame zpracovano, koncime
                    break
                logger.log("no files to compare, sleeping for %s seconds.." % cfg.yaml['main']['sleep_sec'])
                sleep(cfg.yaml['main']['sleep_sec'])
                continue
            if len(files) < cfg.input_batch_size:
                logger.log("less than %s files to compare, trying again.." % cfg.input_batch_size)
                sleep(1)
                continue

            logger.log("%s files to compare: %s" % (len(files), files))

            if process_mp(cfg, logger, sess, model, files):
                stats_true += 1
            else:
                stats_false += 1

            sleep(1)

        except Exception as ex:
            print ex
            traceback.print_exc(file=stdout)
            logger.log("%s" %ex, level="ERROR")
            traceback.print_exc(file=logger.log_file)
            if cfg.mail_opt:
                try:
                    send_mail(logger, cfg.yaml, 'Camera ALARM Error', "%s" % ex, notify=True)
                except Exception as ex:
                    logger.log("sending error mail failed");
                    logger.log("%s" %ex, level="ERROR")
            err_count += 1
            logger.log("error counter %s" % err_count);
            sleep(err_count)

    logger.log("Too many errors in one hour, giving up...");

    if cfg.ftp_opt:
        ftp.quit()

    logger.close()

if __name__ == "__main__":
    exit(main())
