import os
import datetime
import zipfile
import urllib

import json

import numpy

from utils import find_unique_classes
from MDBN import train_MDBN

# batch_size changed from 1 as in M.Liang to 20

def prepare_OV_TCGA_datafiles(config, datadir='data'):
    base_url = 'http://nar.oxfordjournals.org/content/suppl/2012/07/25/gks725.DC1/'
    archive = 'nar-00961-n-2012-File005.zip'

    datafiles = dict()
    for key in config["pathways"]:
        datafiles[key] = config[key]["datafile"]

    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    root_dir = os.getcwd()
    os.chdir(datadir)
    for name, datafile in datafiles.iteritems():
        if not os.path.isfile(datafile):
            if not os.path.isfile(archive):
                print('Downloading TCGA_Data from ' + base_url)
                testfile = urllib.URLopener()
                testfile.retrieve(base_url + archive, archive)
            zipfile.ZipFile(archive, 'r').extract(datafile)
    os.chdir(root_dir)
    return datafiles

if __name__ == '__main__':
    config_dir = 'config/'

    with open(config_dir + 'ov_config.json') as config_file:
        config = json.load(config_file)

    datafiles = prepare_OV_TCGA_datafiles(config)

    output_dir = 'MDBN_run'
    run_start_date = datetime.datetime.now()
    run_start_date_str = run_start_date.strftime("%Y-%m-%d_%H%M")
    results = []
    for i in range(1):
        dbn_output = train_MDBN(datafiles,
                                config,
                                output_folder=output_dir,
                                output_file='Exp_%s_run_%d.npz' %
                                            (run_start_date_str, i),
                                holdout=0.0, repeats=1)
        results.append(find_unique_classes((dbn_output > 0.5) * numpy.ones_like(dbn_output)))

    current_date_time = datetime.datetime.now()
    print('*** Run started at %s' % run_start_date.strftime("%H:%M:%S on %B %d, %Y"))
    print('*** Run completed at %s' % current_date_time.strftime("%H:%M:%S on %B %d, %Y"))

    root_dir = os.getcwd()
    os.chdir(output_dir)
    numpy.savez('Results_%s.npz' % run_start_date_str,
                results=results)
    os.chdir(root_dir)