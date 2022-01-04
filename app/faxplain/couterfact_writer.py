import os
from random import random
import csv

class Faxplain_writer():
    def __init__(self) -> None:
        self.session_id = hex(int(random() * 1e13))[2:]
        self.base_dir = './'
        
        self.res_folder = self.base_dir + 'res/'
        Faxplain_writer.maybe_create_res_folder()
        
        self.init_res_fnames(self.session_id)
        Faxplain_writer.maybe_create_res_file(self.ugc_data_fname)
        Faxplain_writer.maybe_create_res_file(self.mgc_data_fname)
                
    @classmethod
    def maybe_create_res_file(fname):
        if not os.path.isfile(fname):
            with open(fname, 'w+', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerow('query url evidence label'.split())
    
    @classmethod
    def maybe_create_res_folder(dirname):
        os.makedirs(dirname, exist_ok=True)
        
    def init_res_fnames(self):
        self.ugc_data_fname = self.res_folder + f'ugc_{self.session_id}.csv'  # user generated content
        self.mgc_data_fname = self.res_folder + f'mgc_{self.session_id}.csv'  # machine genarated content
        self.temp_data_fname = self.res_folder + f'data/temp_{self.session_id}.pkl'
