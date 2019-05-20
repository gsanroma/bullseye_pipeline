#!/usr/bin/env python

from __future__ import print_function

# from .wmhs_pipeline import wmhs_pipeline
# import sys,os
# sys.path.insert(0, os.path.join(os.environ['HOME'], 'tmp', 'pycharm_deploy_base', 'rs_wmhs_pipeline'))
from wmhs_pipeline import wmhs_pipeline

from nipype import config, logging

import os, sys,glob
import argparse
from itertools import chain

def setenviron():

    import subprocess

    os.environ['PATH'] = '/groups/mri-rhinelandstudy/software/bin:' + os.environ['PATH']
    # os.environ['LD_LIBRARY_PATH'] = '/groups/mri-rhinelandstudy/software/curl/lib:' + os.environ['LD_LIBRARY_PATH']
    # os.environ[''] = ''
    os.environ['ANTSPATH'] = '/groups/mri-rhinelandstudy/software/ants2.3'
    os.environ['PATH'] = os.path.join(os.environ['ANTSPATH'], 'bin') + ':' + os.environ['PATH']
    os.environ['FSLDIR'] = '/groups/mri-rhinelandstudy/software/fsl/fsl6.0.0'
    os.environ['PATH'] = os.path.join(os.environ['FSLDIR'], 'bin') + ':' + os.environ['PATH']
    # subprocess.call(os.path.join(os.environ['FSLDIR'], 'etc', 'fslconf', 'fsl.sh'))
    os.environ['FREESURFER_HOME'] = '/groups/mri-rhinelandstudy/software/freesurfer/freesurfer6.0.0'
    os.environ['PATH'] = os.path.join(os.environ['FREESURFER_HOME'], 'bin') + ':' + os.environ['PATH']
    # subprocess.call(os.path.join(os.environ['FREESURFER_HOME'], 'SetUpFreeSurfer.sh'))
    os.environ['PATH'] = '/groups/mri-rhinelandstudy/software/c3d-1.1.0-Linux-gcc64/bin:' + os.environ['PATH']
    os.environ['PATH'] = os.path.join(os.environ['HOME'], 'CODE', 'external', 'deepmedic') + ':' + os.environ['PATH']

def wmhs_preproc_wf(scans_dir, work_dir, outputdir,subject_ids, threads, device, opp=False, oseg=False,  wfname='wmhs_preproc'):
    wf = wmhs_pipeline(scans_dir, work_dir, outputdir, subject_ids, threads,device, opp, oseg, wfname)
    wf.inputs.inputnode.subject_ids = subject_ids
    return wf
    
    
def main():
    """
    Command line wrapper for preprocessing data
    """
    parser = argparse.ArgumentParser(description='Run WMHS pipelines for FLAIR '\
                                     'imaging data.',
                                     epilog='Example-1: {prog} -s '\
                                     '~/data/scans -w '\
                                     '~/data/work_dir -p 2 -t 2 '\
                                     '--subjects subj1 subj2 '\
                                     '\nExample-2: {prog} -s ~/data/scans -w ~/data -o'\
                                     ' ~/output_dir -p 2 -t 2 '\
                                     '\n\n'
                                     .format(prog=os.path.basename\
                                             (sys.argv[0])),\
                                     formatter_class=argparse.\
                                     RawTextHelpFormatter)

    parser.add_argument('-s', '--scansdir', help='Scans directory where data' \
                        ' is already downloaded for each subject.', required=True)
    
    parser.add_argument('-w', '--workdir', help='Work directory where data' \
                        ' is processed for each subject.', required=True)

    parser.add_argument('-o', '--outputdir', help='Output directory where ' \
                        'results will be stored.', required=True)

    parser.add_argument('--subjects', help='One or more subject IDs'\
                        '(space separated).', \
                        default=None, required=False, nargs='+', action='append')
    
    parser.add_argument('-r', '--only_preproc', help="Only perform pre-processing",
                        required=False, action='store_true')
    parser.add_argument('-e', '--only_segment', help="Only perform till segmentation",
                        required=False, action='store_true')

    parser.add_argument('-b', '--debug', help='debug mode', action='store_true')
    
    parser.add_argument('-p', '--processes', help='overall number of parallel processes', \
                        default=1, type=int)
    parser.add_argument('-g', '--ngpus', help='number of gpus to use (emb-) parallel', \
                        default=1, type=int)
    parser.add_argument('-gp', '--ngpuproc', help='number of processes per gpu', \
                        default=1, type=int)
    parser.add_argument('-d', '--device', help='deepmedic -dev flag', \
                        default='cpu', type=str)
    
    parser.add_argument('-t', '--threads', help='ITK threads', default=1,\
                        type=int)
    
    parser.add_argument('-n', '--name', help='Pipeline workflow name', 
                        default='wmhs_pipeline')
    
    args = parser.parse_args('-s /home/sanromag/DATA/WMH/preproc/scans '
                             '-w /home/sanromag/DATA/WMH/preproc/work '
                             '-o /home/sanromag/DATA/WMH/preproc/out '
                             '--subjects fff5fd4e-94dc-4f66-9ac7-950b5c4e28b5 '
                             '-r '
                             '-d cpu '.split())
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
    work_dir = os.path.abspath(os.path.expandvars(args.workdir))
    outputdir = os.path.abspath(os.path.expandvars(args.outputdir))
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)        
    
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
    

    wmhs_pipeline = wmhs_preproc_wf(scans_dir, work_dir, outputdir, subject_ids,
                                   args.threads,args.device, opp=args.only_preproc,
                                    oseg=args.only_segment, wfname='wmhs_preproc')

    # Visualize workflow
    if args.debug:
        wmhs_pipeline.write_graph(graph2use='colored', simple_form=True)



    wmhs_pipeline.run(
                            plugin='MultiProc',
                            plugin_args={'n_procs' : args.processes,'n_gpus': args.ngpus, 'ngpuproc': args.ngpuproc}
                           )


    print('Done WMHS pipeline!!!')

    
if __name__ == '__main__':
    setenviron()
    sys.exit(main())
