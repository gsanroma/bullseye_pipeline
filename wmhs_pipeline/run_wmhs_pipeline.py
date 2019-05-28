#!/usr/bin/env python

from __future__ import print_function

# from .wmhs_pipeline import wmhs_pipeline_bullseye
from wmhs_pipeline import wmhs_pipeline_bullseye

from nipype import config, logging

import os, sys,glob
import argparse
from itertools import chain

def wmhs_bullseye_wf(scans_dir, wmh_dir, work_dir, outputdir, subject_ids, wfname='wmhs_bullseye'):
    wf = wmhs_pipeline_bullseye(scans_dir, wmh_dir, work_dir, outputdir, subject_ids, wfname)
    wf.inputs.inputnode.subject_ids = subject_ids
    return wf


def main():
    """
    Command line wrapper for preprocessing data
    """
    parser = argparse.ArgumentParser(description='Computes regional WMHs load using bullseye representation',
                                     epilog='Example-1: {prog} -s ~/data/scans -x ~/data/wmh_dir -w ~/data/work_dir -p 2 --subjects subj1 subj2 \n\n'
                                     .format(prog=os.path.basename(sys.argv[0])), formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-s', '--scansdir', help='Scans directory with the data for each subject', required=True)
    parser.add_argument('-x', '--wmhdir', help='Directory with the WMH segmentation masks (same as \'scansdir\' if not provided)')
    parser.add_argument('-w', '--workdir', help='Work directory where data will be processed', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory where results will be stored', required=True)
    parser.add_argument('--subjects', help='One or more subject IDs (space separated)', default=None, required=False, nargs='+', action='append')
    parser.add_argument('-b', '--debug', help='debug mode', action='store_true')
    parser.add_argument('-p', '--processes', help='overall number of parallel processes', default=1, type=int)
    parser.add_argument('-n', '--name', help='Pipeline workflow name', default='wmh_bullseye_pipeline')
    
    args = parser.parse_args('-s /home/sanromag/DATA/WMH/data_nodenoise/scans '
                             '-x /home/sanromag/DATA/WMH/data_nodenoise/wmh_pipeline/out2 '
                             '-w /home/sanromag/DATA/WMH/data_nodenoise/bullseye_pipeline/work '
                             '-o /home/sanromag/DATA/WMH/data_nodenoise/bullseye_pipeline/out '
                             '--subjects 0825d8e6-db27-4802-b848-3e408cbf38ba '
                             ''.split())

    # args = parser.parse_args()
    
    scans_dir = os.path.abspath(os.path.expandvars(args.scansdir))
    if not os.path.exists(scans_dir):
        raise IOError("Scans directory does not exist.")
        
    
    subject_ids = []
    
    if args.subjects:
        subject_ids = list(chain.from_iterable(args.subjects))
    else:
        subject_ids = glob.glob(scans_dir.rstrip('/') +'/*')
        subject_ids = [os.path.basename(s.rstrip('/')) for s in subject_ids]


    print ("Creating wmhs pipeline workflow...")
    wmh_dir = scans_dir
    if args.wmhdir is not None:
        wmh_dir = os.path.abspath(os.path.expandvars(args.wmhdir))
    work_dir = os.path.abspath(os.path.expandvars(args.workdir))
    output_dir = os.path.abspath(os.path.expandvars(args.outputdir))
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    config.update_config({
        'logging': {'log_directory': args.workdir, 'log_to_file': True},
        'execution': {'job_finished_timeout' : 60,
                      'poll_sleep_duration' : 20,
                      'hash_method' : 'content',
                      'local_hash_check' : False,
                      'stop_on_first_crash':False,
                      'crashdump_dir': args.workdir,
                      'crashfile_format': 'txt'
                       },
                       })

    #config.enable_debug_mode()
    logging.update_logging(config)
    

    wmhs_bullseye_pipeline = wmhs_bullseye_wf(scans_dir, wmh_dir, work_dir, output_dir, subject_ids, wfname='wmhs_lobes')

    # Visualize workflow
    if args.debug:
        wmhs_bullseye_pipeline.write_graph(graph2use='colored', simple_form=True)

    wmhs_bullseye_pipeline.run(plugin='MultiProc', plugin_args={'n_procs' : args.processes})


    print('Done WMHs bullseye pipeline!!!')

    
if __name__ == '__main__':
    sys.exit(main())
