#!/usr/bin/python

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

# TODO spusteni po startu systemu

# TODO doplnit debian zavislosti

# TODO udelat z toho service
# TODO pozor na posilani chyb az to bude sluzba, mohlo by se stat, ze pri nejake nezotavitelne chybe me to pekne vyspamuje!

# TODO trida Configuration, ktera by ulozila nastaveni do clenskych promennych

# TODO zlepsit nacitani z FTP

#TODO ovladani mailem zasalnym zpet na adresu odesilatele (napr. vypinani, zapinani upozorneni, poslani statistik, atd.)


config = None
logger = None

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

#TODO nejake standardni logovani
class Logger:
    def __init__(self, config):
        self.log_dir = config['main']['log_dir']
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
            (self.log_dir, config['expiration']['log_expire_days']), shell=True)

    def log(self, text, level="INFO"):
        line = strftime("%Y-%m-%d %H:%M:%S ", localtime()) + level + ": " + text
        print line
        self.log_file.write(line + "\n")

    def close(self):
        self.log_file.close()

def get_hour():
    return int(strftime("%H", localtime()))

def connect_ftp(server, login, passwd):
    ftp = FTP(server)
    ftp.login(login, passwd)
    ftp.cwd(config['ftp']['ftp_dir'])
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

def fetch_files(ftp):
    # zkusime zda je navazane spojeni s FTP serverem
    try:
        ftp.voidcmd("NOOP")
    except Exception as ex:
        # s nejvetsi pravdepodobnosti doslo k timeoutu, nebo jinemu duvodu odpojeni
        logger.log("ftp connection lost: %s" %ex, level="WARN")
        ftp.quit()
        # zkusime se znovu spojit
        ftp = connect_ftp(config['ftp']['ftp_server'],
            config['ftp']['ftp_user'], config['ftp']['ftp_passwd'])
        # pokud stale nic, koncime s chybou
        ftp.voidcmd("NOOP")

    ftp_list = [x for x in ftp.nlst() if x[:3]=="ARC"]
    # kratka pauza, jelikoz bez ni muze dojit k tomu, ze nejaky soubor z precteneho seznamu
    # jeste nemusi byt kompletne nahran na FTP ze ktereho cteme
    sleep(1)
    for file in ftp_list:
        ftp.retrbinary("RETR %s" % file, open("%s/%s" % (config['main']['input_dir'], file), "wb").write)
        logger.log("retrieved %s file from ftp" % file)
        ftp.delete(file)
    sleep(1)


def send_mail(subj, text, files=None, notify=False):

    msg = MIMEMultipart()
    msg['From'] = config['mail']['mail_from']
    msg['To'] = config['mail']['mail_to']
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

    smtp.connect(config['mail']['mail_server'], 587) #port 587 TLS, 25 nezabezpecene spojeni
    smtp.ehlo()     #TLS only
    smtp.starttls() #TLS only
    smtp.login(config['mail']['mail_from'], config['mail']['mail_from_passwd'])

    smtp.sendmail(config['mail']['mail_from'], config['mail']['mail_to'], msg.as_string())

    if notify:
        msg = MIMEMultipart()
        msg['From'] = config['mail']['mail_from']
        msg['To'] = config['mail']['notify_to']
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subj

        smtp.sendmail(config['mail']['mail_from'], config['mail']['notify_to'], msg.as_string())

    smtp.close()

def process(files, mail_opt):
    # TODO toto by se melo provest jednou pri startu v nejake tride s nastavenim
    input_dir = config['main']['input_dir']
    alarm_dir = config['main']['alarm_dir']
    trash_dir = config['main']['trash_dir']
    error_dir = config['main']['error_dir']
    fuzz = config['detect_compare']['fuzz']
    zones = config['detect_compare']['zones']
    contra_zones = config['detect_compare']['contra_zones']

    pos = files[2].find(".jpg")
    if pos > 0:
        diff_file = files[2][:-4] + "-diff.jpg"

    # porovnani fotek
    try:
        # pripravit a poslat prikaz na porovnani celeho obrazku
        cmd = "compare -metric AE -fuzz %s%% %s %s %s" % (fuzz, files[0], files[2], diff_file)
        logger.log(cmd)
        p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (out, err) = p.communicate()

        # pripravit a poslat prikazy na porovnani zon
        diff_files = ""
        for z in zones + contra_zones:
            # souradnice vyrezu jsou velikost XxY a levy horni roh +X+Y
            cmd = "compare -metric AE -fuzz %s%% -extract %sx%s+%s+%s %s %s diff%s.jpg" % (
                fuzz, z["extract"][0], z["extract"][1], z["extract"][2], z["extract"][3], files[0], files[2], z["id"])
            z["p"] = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (z["out"], z["err"]) = z["p"].communicate()
            diff_files += "diff%s.jpg " % (z["id"], )
            logger.log("zone %s: %s" % (z["id"], cmd))

        # pockat na vysledek porovnani obrazku
        status = p.wait()
        diff = float(err)

        # pockat na vysledky a zpracovat porovnani zon
        diff_zones_sum = 0.0
        diff_zones = ""
        alarm_zones = ""
        alarm_zones_count = 0
        for z in zones + contra_zones:
            z["alarm"] = False
            z["status"] = z["p"].wait()
            z["diff"] = float(z["err"])
            z["area"] = z["extract"][0] * z["extract"][1]
            z["pct_diff"] = round((z["diff"] / z["area"]) * 100, 1)
            if z["pct_diff"] >= z["pct_thr"]:
                z["alarm"] = True
                alarm_zones_count += 1
            diff_zones_sum += z["diff"]
            diff_zones += "%s%s: %s%% (%s), " % (("! " if z["alarm"] else ""), z["id"], z["pct_diff"], z["diff"])
    except Exception as ex:
        send_mail("Camera ALARM: Error Pictures", "error while comparing pictures",
            files=[files[0], files[2], ], notify=True)
        subprocess.call("mv %s %s %s %s %s/" % (files[0], files[1], files[2], diff_file, error_dir), shell=True)
        subprocess.call("rm *.jpg", shell=True)
        raise ex

    # oramovat a otextovat zony
    rectangles = ""
    texts = ""
    for z in zones + contra_zones:
        # souradnice obdelniku jsou odkud X,Y kam X,Y
        rectangles += "-draw \"rectangle %s,%s %s,%s\" " % (
            z["extract"][2], z["extract"][3], z["extract"][2]+z["extract"][0], z["extract"][3]+z["extract"][1])
        texts += "-draw \"text %s,%s \'%s%s: %s%% (%s)\'\" " % (
            z["extract"][2] + 5, z["extract"][3] + 30, ("! " if z["alarm"] else ""), z["id"], z["pct_diff"], z["diff"])
    try:
        cmd = "convert %s -fill none -stroke black -strokewidth 2 %s %s" % (diff_file, rectangles, diff_file)
        p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (out, err) = p.communicate()
        status = p.wait()
        cmd = "convert %s -stroke black -pointsize 30 %s %s" % (diff_file, texts, diff_file)
        p = subprocess.Popen([cmd, ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (out, err) = p.communicate()
        status = p.wait()
    except Exception as ex:
        send_mail("Camera ALARM: Error Pictures", "error while converting diff file",
            files=[files[0], files[2], diff_file, ], notify=True)
        subprocess.call("mv %s %s %s %s %s/" % (files[0], files[1], files[2], diff_file, error_dir), shell=True)
        subprocess.call("rm *.jpg", shell=True)
        raise ex

    # vyruseni falesnych alarmu kontra zonami
    zone_by_id = build_dict(zones, key="id")
    for zc in contra_zones:
        if zc["alarm"]:
            for z_id in zc["contra"]:
                if (zone_by_id[z_id]["pct_diff"] - zc["pct_diff"] * zc["mult"]) < zone_by_id[z_id]["pct_thr"]:
                    zone_by_id[z_id]["alarm"] = False

    result = False
    for key, z in zone_by_id.iteritems():
        if z["alarm"]:
            result = True
            alarm_zones += "%s: %s%%, " % (z["id"], z["pct_diff"])

    # vic jak 2 (nebo 3?) zony najednou je s velkou pravdepodobnosti plany poplach zpusobeny
    # plosnou zmenu jasu/barev ve scene
    if result and alarm_zones_count < 4:
        subj = 'Camera ALARM ' + strftime("%Y-%m-%d %H:%M:%S", localtime())
        text = "in zones: %s\nimage diff: %s, zones diff sum: %s\nzones diffs: %s" % (alarm_zones, diff, diff_zones_sum, diff_zones)
        logger.log("! ALARM " + text)
        if mail_opt:
            send_mail(subj, text, files=[files[0], files[2], diff_file, ], notify=True)
        subprocess.call("mv %s %s %s %s %s/" % (files[0], files[1], files[2], diff_file, alarm_dir), shell=True)
        subprocess.call("rm %s" % (diff_files, ), shell=True)
    else:
        logger.log("False alarm, diff: %s, zones diff sum: %s\nzones diffs: %s" % (diff, diff_zones_sum, diff_zones))
        subprocess.call("mv %s %s %s %s %s/" % (files[0], files[1], files[2], diff_file, trash_dir), shell=True)
        subprocess.call("rm %s" % (diff_files, ), shell=True)
        result = False

    logger.log("----------")
    return result

def main():
    ftp_opt = False
    mail_opt = False

    parser = OptionParser()
    parser.add_option('--once', action='store_true', help='start once, only to process batch')
    parser.add_option('--noftp', action='store_true', help='prevent loading files from FTP server (overrides config)')
    parser.add_option('--nomail', action='store_true', help='prevent sending mail (overrides config)')
    options, args = parser.parse_args_dict()
    config_file = options['config']
    if not config_file:
        parser.print_help()
        print "Config file argument is requeired"
        return

    global config
    with open(config_file) as stream:
        config = yaml.load(stream)

    global logger
    logger = Logger(config)

    once_opt = ('once' in options) and (options['once'] is True)
    ftp_opt = config['main']['ftp_opt']
    if 'noftp' in options and options['noftp']:
        ftp_opt = False
    mail_opt = config['main']['mail_opt']
    if 'nomail' in options and options['nomail']:
        mail_opt = False

    prev_hour = get_hour()
    stats_true = 0
    stats_false = 0

    if ftp_opt:
        ftp = connect_ftp(config['ftp']['ftp_server'],
            config['ftp']['ftp_user'], config['ftp']['ftp_passwd'])

    err_count = 0

    while(err_count < 5):
        try:
            hour = get_hour()
            if prev_hour != hour:
                # logrotate a expirace starych souboru o pulnoci
                if hour == 0:
                    logger.rotate()

                    logger.log("Deleting files from alarm and trash older than %s days" % config['expiration']['expire_days'])
                    subprocess.call("find trash/* -mtime +%s -exec rm {} \;" % config['expiration']['expire_days'], shell=True)
                    subprocess.call("find alarm/* -mtime +%s -exec rm {} \;" % config['expiration']['expire_days'], shell=True)

                # monitoring + hodinove statistiky
                p = subprocess.Popen(["du -sh", ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                (out, err) = p.communicate()
                status = p.wait()
                text = ("Hourly statistics:\nALARMS count: %s\nFalse alarms count: %s\n\nError count: %s\n\nTotal size of dir: %s" %
                    (stats_true, stats_false, err_count, out))
                logger.log(text)
                if mail_opt:
                    send_mail("Camera stats", text)

                stats_true = 0
                stats_false = 0
                err_count = 0
                prev_hour = hour

            if ftp_opt:
                fetch_files(ftp)

            # TODO lepsi prace se soubory, pres pythoni libky
            p = subprocess.Popen(["find %s -maxdepth 1 -type f | grep ARC | grep -v diff | sort -r" % config['main']['input_dir']],
                stdout=subprocess.PIPE, shell=True)
            (output, err) = p.communicate()
            p_status = p.wait()

            files = output.split()
            if not files:
                if once_opt:
                    # pokud nechcem sluzbu a mame zpracovano, koncime
                    break
                logger.log("no files to compare, sleeping for %s seconds.." % config['main']['sleep_sec'])
                sleep(config['main']['sleep_sec'])
                continue
            if len(files) < 3:
                logger.log("less than 3 files to compare, trying again..")
                sleep(1)
                continue
            #TODO osetrit stav, kdy je mensi pocet souboru < 3, napr. se podari nahrat jen jeden,
            # pak tri nejedou, takze se porovnaji k sobe nepatrici obrazky?

            logger.log("%s files to compare" % (len(files), ))
            if process(files, mail_opt):
                stats_true += 1
            else:
                stats_false += 1

            sleep(1)

        except Exception as ex:
            print ex
            traceback.print_exc(file=stdout)
            logger.log("%s" %ex, level="ERROR")
            traceback.print_exc(file=logger.log_file)
            if mail_opt:
                try:
                    send_mail('Camera ALARM: Error', "%s" % ex, notify=True)
                except Exception as ex:
                    logger.log("sending error mail failed");
                    logger.log("%s" %ex, level="ERROR")
            err_count += 1
            logger.log("error counter %s" % err_count);
            sleep(err_count)

    logger.log("Too many errors in one hour, giving up...");

    if ftp_opt:
        ftp.quit()

    logger.close()

if __name__ == "__main__":
    exit(main())
