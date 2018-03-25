from optparse import OptionParser as BaseParser
import yaml

class OptionParser(BaseParser):
    def __init__(self, **kwargs):
        BaseParser.__init__(self, **kwargs)

        self.add_option('-f', '--config', help='the config file')
        self.add_option('-v', '--verbose', action='store_true')

    def parse_args_dict(self):
        options, args = self.parse_args()
        return vars(options), args

class Configuration:
    def __init__(self, options):
        config_file = options['config']
        if not config_file:
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

