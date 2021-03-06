#!/usr/bin/env python

from lib.config import OptionParser, Configuration
import poplib
import email
import email.parser
import os
import sys
import re
import html2text
import logging
from datetime import datetime

# TODO cesty nacitat z konfigurace
# TODO lepsi pojmenovani trid
# TODO automaticky preucovat sit na novych datech

class email_attachment:
    def __init__(self, attachmentnum, filename, contents):
        '''
        arguments:

        attachmentnum - attachment number for this attachment
        filename - filename for this attachment
        contents - attachment's contents
        '''
        self.attachmentnum=attachmentnum
        self.filename=filename
        self.contents=contents
        return

    def save(self, savepath, savefilename=None):
        '''
        Method to save the contents of an attachment to a file
        arguments:

        savepath - path where file is to be saved
        safefilename - optional name (if None will use filename of attachment
        '''

        savefilename=savefilename or self.filename
        f=open(os.path.join(savepath, savefilename),"wb")
        f.write(self.contents)
        f.close()
        return

class email_msg:
    def __init__(self, messagenum, contents):
        self.messagenum=messagenum
        self.contents=contents
        self.attachments_index=0  # Index of attachments for next method
        self.ATTACHMENTS=[]       # List of attachment objects

        self.msglines='\n'.join(contents[1])
        #
        # See if I can parse the message lines with email.Parser
        #
        self.msg=email.parser.Parser().parsestr(self.msglines)
        self.mail_from=self.msg.get('From')
        self.mail_to=self.msg.get('To')
        self.subject=self.msg.get('Subject')

    def read_body(self):
        if self.msg.is_multipart():
            attachmentnum=0
            for part in self.msg.walk():
                # multipart/* are just containers
                mptype=part.get_content_maintype()
                filename = part.get_filename()
                if mptype == "multipart": continue
                if filename: # Attached object with filename
                    attachmentnum+=1
                    self.ATTACHMENTS.append(email_attachment(attachmentnum,filename,part.get_payload(decode=1)))
                    logging.debug("Attachment filename=%s", filename)

                else: # Must be body portion of multipart
                    self.body=part.get_payload(decode=True)

        else: # Not multipart, only body portion exists
            self.body=self.msg.get_payload(decode=True)

        return


    def get(self, key):
        try: return self.msg.get(key)
        except:
            emsg="email_msg-Unable to get email key=%s information" % key
            logging.error(emsg)
            sys.exit(emsg)

    def has_attachments(self):
        return (len(self.ATTACHMENTS) > 0)

    def __iter__(self):
        return self

    def next(self):
        #
        # Try to get the next attachment
        #
        try: ATTACHMENT=self.ATTACHMENTS[self.attachments_index]
        except:
            self.attachments_index=0
            raise StopIteration
        #
        # Increment the index pointer for the next call
        #
        self.attachments_index+=1
        return ATTACHMENT

class pop3_inbox:
    def __init__(self, server, userid, password):
        logging.debug("pop3_inbox.__init__-Entering")
        self.result=0             # Result of server communication
        self.messages_index=0     # Index of message for next method
        #
        # See if I can connect using information provided
        #
        try:
            logging.debug("pop3_inbox.__init__-Calling poplib.POP3(server)")
            self.connection=poplib.POP3_SSL(server, 995)
            logging.debug("pop3_inbox.__init__-Calling connection.user(userid)")
            self.connection.user(userid)
            logging.debug("pop3_inbox.__init__-Calling connection.pass_(password)")
            self.connection.pass_(password)

        except:
            logging.debug("pop3_inbox.__init__-Login failure, closing connection")
            self.result=1
            self.connection.quit()

        #
        # Get count of messages and size of mailbox
        #
        logging.debug("pop3_inbox.__init__-Calling connection.stat()")
        self.msgcount, self.size=self.connection.stat()
        logging.debug("pop3_inbox.__init__- stat msgcount = %s, size = %s", self.msgcount, self.size)
        #
        # Loop over all the messages processing each one in turn
        #

        logging.debug("pop3_inbox.__init__-Leaving")
        return

    def close(self):
        self.connection.quit()
        return

    def remove(self, msgnumorlist):
        if isinstance(msgnumorlist, int): self.connection.dele(msgnumorlist)
        elif isinstance(msgnumorlist, (list, tuple)):
            map(self.connection.dele, msgnumorlist)
        else:
            emsg="pop3_inbox.remove-msgnumorlist must be type int, list, or tuple, not %s" % type(msgnumorlist)
            logging.error(emsg)
            sys.exit(emsg)

        return

    def __iter__(self):
        return self

    def next(self):
        #
        # for next file
        #
        try: MESSAGE=self.MESSAGES[self.messages_index]
        except:
            self.messages_index=0
            raise StopIteration
        #
        # incr. pointer index for next..
        #
        self.messages_index+=1
        return MESSAGE

def getMail(cfg):
    data_dir = cfg.yaml['main']['learn_dir'] + "/pictures"

    inbox=pop3_inbox(cfg.yaml['mail']['pop_server'], cfg.yaml['mail']['login'], cfg.yaml['mail']['password'])
    if inbox.result:
        emsg="Connection error with pop3.."
        logging.error(emsg)
        sys.exit(emsg)
    if not inbox.msgcount:
        logging.info("no messages..")
        return []

    cmds = []

    for msgnum in range(1, inbox.msgcount+1):
        m = email_msg(msgnum,inbox.connection.retr(msgnum))

        logging.info("Processing message with subject: '%s'", m.subject)
        l_subj = m.subject.strip().lower()
        if l_subj.startswith("cmd reboot"):
            cmds.append('reboot')
        elif l_subj.startswith("cmd restart"):
            cmds.append('/usr/sbin/service camera-filter restart')
        elif l_subj.startswith("cmd retrain"):
            cmds.append('/www/camera-filter/bin/retrain.sh -f')
        elif l_subj.startswith("re:") and "arc" in l_subj:
            files = re.findall("ARC[0-9]*\.jpg", m.subject)
            if len(files) != 2:
                logging.warning("Wrong count of files in subject, need exactly 2 files")
                continue
            m.read_body()
            body_text = html2text.html2text(unicode(m.body, 'utf-8')).strip()
            tf = body_text[0].lower() if body_text else ""
            if tf not in ["t", "f"]:
                logging.warning("Neither True nor False in mail body")
                continue
            dest = "true" if tf == "t" else "false"
            dest_dir = "%s/%s" % (data_dir, dest)
            # searching not only in alarm dir, because we need to support change location between false or true
            for source_dir in [cfg.yaml['main']['alarm_dir'], data_dir + "/true", data_dir + "/false"]:
                all_files = True
                for f in files:
                    # some files are broken from camera, because they have not time in filename
                    # (000000 at the end of filename). These files we wan't.
                    if not os.path.isfile("%s/%s" % (source_dir, f)) or "000000." in f:
                        all_files = False
                if all_files:
                    break
            if not all_files:
                logging.warning("Some file is missing")
                continue
            for f in files:
                os.rename("%s/%s" % (source_dir, f), "%s/%s" % (dest_dir, f))
        elif l_subj.startswith("model"):
            #TODO nahrat model a restartnou sluzbu
            logging.debug("attachments %s", len(m.ATTACHMENTS))
            #if m.has_attachments():
            #    acounter=0
            #    for a in m:
            #        acounter+=1
            #        logging.debug("%s: %s", acounter, a.filename)
            #        if not "diff" in a.filename:
            #            a.save(r"/home/pi/learning/%s" % tf) # select save path in ur computer

        else:
            continue

        # TODO proc to nefunguje?
        logging.info("deleting message")
        inbox.remove(msgnum)

    inbox.close()

    return cmds

def main(argv):
    parser = OptionParser()
    options, args = parser.parse_args_dict()
    cfg = Configuration(options)

    logging.basicConfig(
        filename=(cfg.yaml['main']['log_dir']+"/mailer-"+datetime.now().strftime('%Y%m%d')+".log"),
        level=cfg.log_level,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Started')

    cmds = getMail(cfg)

    for cmd in cmds:
        os.system(cmd)

    logging.info('Successfully finished')

if __name__ == "__main__":
    main(sys.argv[1:])
