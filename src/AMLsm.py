import os
import sys
import datetime
import json
import numpy
import getopt

from MDBN import train_MDBN

from utils import find_unique_classes
from utils import usage

def prepare_AML_TCGA_datafiles(config):

    datafiles = dict()
    for key in config["pathways"]:
        datafiles[key] = config[key]["datafile"]

    return datafiles

def main(argv):
    config_dir = 'config/'
    verbose = False
    config_filename = 'aml_config.json'

    try:
        opts, args = getopt.getopt(argv, "hc:v", ["help", "config=", "verbose"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-c", "--config"):
            config_filename = arg

    with open(config_dir + config_filename) as config_file:
        config = json.load(config_file)

    datafiles = prepare_AML_TCGA_datafiles(config)

    numpy_rng = numpy.random.RandomState(config["seed"])

    results = []
    batch_start_date = datetime.datetime.now()
    batch_start_date_str = batch_start_date.strftime("%Y-%m-%d_%H%M")

    output_dir = 'MDBN_run/AML_Batch_%s' % batch_start_date_str
    os.mkdir(output_dir)

    for i in range(config["runs"]):
        run_start_date = datetime.datetime.now()
        print('*** Run %i started at %s' % (i, run_start_date.strftime("%H:%M:%S on %B %d, %Y")))
        dbn_output = train_MDBN(datafiles,
                                config,
                                output_folder=output_dir,
                                output_file='Exp_%s_run_%d.npz' %
                                            (batch_start_date_str, i),
                                holdout=0.0, repeats=1,
                                run=i,
                                verbose=verbose,
                                rng=numpy_rng)
        current_date_time = datetime.datetime.now()
        classes = find_unique_classes((dbn_output > 0.5) * numpy.ones_like(dbn_output))
        print('*** Run %i identified %d classes' % (i,numpy.max(classes[0])))
        results.append(classes[0])
        print('*** Run %i completed at %s' % (i, current_date_time.strftime("%H:%M:%S on %B %d, %Y")))

    root_dir = os.getcwd()
    os.chdir(output_dir)
    numpy.savez('Results_%s.npz' % batch_start_date_str,
                results=results)
    os.chdir(root_dir)

if __name__ == '__main__':
    main(sys.argv[1:])