from __future__ import division

import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util

from nipype import IdentityInterface, DataSink

# from .utils import *
from utils import *

import os

def create_bullseye_pipeline(scans_dir, work_dir, outputdir, subject_ids, name='bullseye_pipeline'):

    # set freesurfer subjects_dir to scans_dir
    os.environ['SUBJECTS_DIR'] = scans_dir

    bullwf = pe.Workflow(name=name)
    bullwf.base_dir = work_dir

    inputnode = pe.Node(interface=IdentityInterface(fields=['subject_ids', 'outputdir']), name='inputnode')
    inputnode.iterables = [('subject_ids', subject_ids)]
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.outputdir = outputdir

    #template for input files
    template = {"ASEG": "{subject_id}/mri/aseg*gz",
                "RIBBON": "{subject_id}/mri/ribbon.mgz",
                "ANNOT_LH": "{subject_id}/label/lh.aparc.annot",
                "ANNOT_RH": "{subject_id}/label/rh.aparc.annot",
                "WHITE_LH": "{subject_id}/surf/lh.white",
                "WHITE_RH": "{subject_id}/surf/rh.white",
                "PIAL_LH": "{subject_id}/surf/lh.pial",
                "PIAL_RH": "{subject_id}/surf/rh.pial",
                "subject_id": "{subject_id}"}

    fileselector = pe.Node(SelectFiles(template), name='fileselect')
    fileselector.inputs.base_directory = scans_dir

    # lobar parcellation
    annot2label_lh = pe.Node(interface=Annot2Label(), name='annot2label_lh')
    annot2label_lh.inputs.hemi = 'lh'
    annot2label_lh.inputs.lobes = 'lobes'

    annot2label_rh = pe.Node(interface=Annot2Label(), name='annot2label_rh')
    annot2label_rh.inputs.hemi = 'rh'
    annot2label_rh.inputs.lobes = 'lobes'

    # aparc2aseg to map lobes into white matter volume
    aparc2aseg = pe.Node(interface=Aparc2Aseg(), name='aparc2aseg')
    aparc2aseg.inputs.annot = 'lobes'
    aparc2aseg.inputs.labelwm = True
    aparc2aseg.dmax = 1000
    aparc2aseg.inputs.rip = True
    aparc2aseg.inputs.hypo = True
    aparc2aseg.inputs.out_file = 'lobes+aseg.nii.gz'

    # group some lobes and discard others
    filter_lobes = pe.Node(interface=util.Function(input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'], output_names=['out_file'],
                                                   function=filter_labels), name='filter_lobes')
    # Here we include insula (3007, 4007) with frontal (3001, 4001)
    # We exclude the structure in the superior part spanning from anterior to posterior (3003, 4003)
    filter_lobes.inputs.include_superlist = [[3001, 3007], [4001, 4007], [3004], [4004], [3005], [4005], [3006], [4006]]  # lobar labels in WM
    filter_lobes.inputs.fixed_id = None
    # we give some informative label-ids
    filter_lobes.inputs.map_pairs_list = [[3001, 11], [4001, 21], [3004, 12], [4004, 22], [3005, 13], [4005, 23], [3006, 14], [4006, 24]]

    # create ventricles and cortex masks
    ventricles_mask = pe.Node(interface=util.Function(input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'], output_names=['out_file'],
                                                function=filter_labels), name='ventricles_mask')
    ventricles_mask.inputs.include_superlist = [[43, 4]]
    ventricles_mask.inputs.fixed_id = [1]
    ventricles_mask.inputs.map_pairs_list = None

    cortex_mask = pe.Node(interface=util.Function(input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'], output_names=['out_file'],
                                                  function=filter_labels), name='cortex_mask')
    cortex_mask.inputs.include_superlist = [[1001, 2001, 1004, 2004, 1005, 2005, 1006, 2006]]  # lobar labels in cortex
    cortex_mask.inputs.fixed_id = [1]
    cortex_mask.inputs.map_pairs_list = None

    # create mask with basal ganglia + thalamus
    bgt_mask = pe.Node(interface=util.Function(input_names=['in_file', 'include_superlist', 'fixed_id', 'map_pairs_list'], output_names=['out_file'],
                                               function=filter_labels), name='bgt_mask')
    bgt_mask.inputs.include_superlist = [[10, 49, 11, 12, 50, 51, 26, 58, 13, 52]]  # basal ganglia + thalamus
    bgt_mask.inputs.fixed_id = [5]
    bgt_mask.inputs.map_pairs_list = None

    # create normalized distance map
    ndist_map = pe.Node(interface=util.Function(input_names=['orig_file', 'dest_file'], output_names=['out_file'],
                                                function=norm_dist_map), name='ndist_map')

    # generate WM parcellations by filling the discarded lobes (3003, 4003) and unsegmented white matter (5001, 5002)
    gen_wmparc = pe.Node(interface=util.Function(input_names=['incl_file', 'ndist_file', 'label_file', 'incl_labels', 'verbose'], output_names=['out_file'],
                                                 function=generate_wmparc), name='gen_wmparc')
    gen_wmparc.inputs.incl_labels = [3003, 4003, 5001, 5002] # the labels that need to be 'filled'
    gen_wmparc.inputs.verbose = False

    # include bgt into wmparc to create the final lobar wmparc
    lobe_wmparc = pe.Node(interface=util.Function(input_names=['in1_file', 'in2_file', 'out_file', 'intersect'], output_names=['out_file'],
                                                  function=merge_labels), name='lobe_wmparc')
    lobe_wmparc.inputs.intersect = False
    lobe_wmparc.inputs.out_file = 'lobes_wmparc.nii.gz'

    # create depth shells using normalized distance maps
    depth_wmparc = pe.Node(interface=util.Function(input_names=['ndist_file', 'n_shells', 'out_file', 'mask_file'], output_names=['out_file'],
                                                   function=create_shells), name='depth_wmparc')
    depth_wmparc.inputs.n_shells = 4
    depth_wmparc.inputs.out_file = 'shells_wmparc.nii.gz'

    # final bullseye parcellation by intersecting depth and lobar parcellations
    bullseye_wmparc = pe.Node(interface=util.Function(input_names=['in1_file', 'in2_file', 'out_file', 'intersect'], output_names=['out_file'],
                                                      function=merge_labels), name='bullseye_wmparc')
    bullseye_wmparc.inputs.intersect = True
    bullseye_wmparc.inputs.out_file = 'bullseye_wmparc.nii.gz'

    # collect outputs
    datasinkout = pe.Node(interface=DataSink(), name='datasinkout')
    datasinkout.inputs.parameterization=False

    ##### CONNECTIONS #####

    bullwf.connect(inputnode        , 'subject_ids',      fileselector,'subject_id')

    bullwf.connect(fileselector     , 'subject_id',       annot2label_lh, 'subject')
    bullwf.connect(fileselector     , 'ANNOT_LH',         annot2label_lh, 'in_annot')
    bullwf.connect(fileselector     , 'subject_id',       annot2label_rh, 'subject')
    bullwf.connect(fileselector     , 'ANNOT_LH',         annot2label_rh, 'in_annot')

    bullwf.connect(annot2label_rh     , 'out_annot_file',         aparc2aseg, 'in_lobes_rh')
    bullwf.connect(annot2label_lh     , 'out_annot_file',         aparc2aseg, 'in_lobes_lh')
    bullwf.connect(fileselector     , 'subject_id',         aparc2aseg, 'subject')
    bullwf.connect(aparc2aseg     , 'out_file',         filter_lobes, 'in_file')

    bullwf.connect(aparc2aseg     , 'out_file',         ventricles_mask, 'in_file')
    bullwf.connect(aparc2aseg     , 'out_file',         cortex_mask, 'in_file')
    bullwf.connect(aparc2aseg     , 'out_file',         bgt_mask, 'in_file')

    bullwf.connect(ventricles_mask     , 'out_file',         ndist_map, 'orig_file')
    bullwf.connect(cortex_mask     , 'out_file',         ndist_map, 'dest_file')

    bullwf.connect(aparc2aseg     , 'out_file',         gen_wmparc, 'incl_file')
    bullwf.connect(ndist_map     , 'out_file',         gen_wmparc, 'ndist_file')
    bullwf.connect(filter_lobes     , 'out_file',         gen_wmparc, 'label_file')

    bullwf.connect(gen_wmparc     , 'out_file',         lobe_wmparc, 'in1_file')
    bullwf.connect(bgt_mask     , 'out_file',         lobe_wmparc, 'in2_file')

    bullwf.connect(ndist_map     , 'out_file',         depth_wmparc, 'ndist_file')
    bullwf.connect(lobe_wmparc     , 'out_file',         depth_wmparc, 'mask_file')

    bullwf.connect(lobe_wmparc     , 'out_file',         bullseye_wmparc, 'in1_file')
    bullwf.connect(depth_wmparc     , 'out_file',         bullseye_wmparc, 'in2_file')

    # outputs
    bullwf.connect(inputnode               , 'subject_ids',       datasinkout, 'container')
    bullwf.connect(inputnode               , 'outputdir',         datasinkout, 'base_directory')

    bullwf.connect(lobe_wmparc   , 'out_file',   datasinkout, '@lobes_wmparc')
    bullwf.connect(depth_wmparc   , 'out_file',   datasinkout, '@shells_wmparc')
    bullwf.connect(bullseye_wmparc   , 'out_file',   datasinkout, '@bullseye_wmparc')

    return(bullwf)

