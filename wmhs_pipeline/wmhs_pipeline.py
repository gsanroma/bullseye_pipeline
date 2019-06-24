from __future__ import division

import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util

from nipype.interfaces import fsl
from nipype.interfaces.fsl.utils import Reorient2Std


from nipype import IdentityInterface, DataSink

#from .utils import *
from utils import *

import os

def wmhs_pipeline_bullseye(scans_dir, wmh_dir, work_dir, outputdir, subject_ids, name='wmhs_bullseye'):

    # set freesurfer subjects_dir to scans_dir
    os.environ['SUBJECTS_DIR'] = scans_dir


    wmhsbullwf = pe.Workflow(name=name)
    wmhsbullwf.base_dir = work_dir

    inputnode = pe.Node(interface=IdentityInterface(fields=['subject_ids', 'outputdir']), name='inputnode')
    inputnode.iterables = [('subject_ids', subject_ids)]
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.outputdir = outputdir

    #template for input files
    template_scans = {"FLAIR": "{subject_id}/*FLAIR*gz",
                      "T1FS": "{subject_id}/mri/orig*gz",
                      "ASEG": "{subject_id}/mri/aseg*gz",
                      "RIBBON": "{subject_id}/mri/ribbon.mgz",
                      "ANNOT_LH": "{subject_id}/label/lh.aparc.annot",
                      "ANNOT_RH": "{subject_id}/label/rh.aparc.annot",
                      "WHITE_LH": "{subject_id}/surf/lh.white",
                      "WHITE_RH": "{subject_id}/surf/rh.white",
                      "PIAL_LH": "{subject_id}/surf/lh.pial",
                      "PIAL_RH": "{subject_id}/surf/rh.pial",
                      "subject_id": "{subject_id}"}

    template_wmh = {"WMH": "{subject_id}/*Segm.nii.gz"}

    fileselector_scans = pe.Node(SelectFiles(template_scans), name='fileselect_scans')
    fileselector_scans.inputs.base_directory = scans_dir

    fileselector_wmh = pe.Node(SelectFiles(template_wmh), name='fileselect_wmh')
    fileselector_wmh.inputs.base_directory = wmh_dir

    # # convert aseg mgz to aseg.nii.gz if in mgz format
    # convert_aseg_mgz = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
    #                                                    function=convert_mgz), name='convert_aseg_mgz')

    # # reorient 2 std
    # reorient2std_aseg = pe.Node(interface=Reorient2Std(), name= 'reorient2std_aseg')

    # lobar parcellation
    annot2label_lh = pe.Node(interface=Annot2Label(), name='annot2label_lh')
    annot2label_lh.inputs.hemi = 'lh'
    annot2label_lh.inputs.lobes = 'lobes'

    annot2label_rh = pe.Node(interface=Annot2Label(), name='annot2label_rh')
    annot2label_rh.inputs.hemi = 'rh'
    annot2label_rh.inputs.lobes = 'lobes'

    # apar2aseg
    aparc2aseg = pe.Node(interface=Aparc2Aseg(), name='aparc2aseg')
    aparc2aseg.inputs.annot = 'lobes'
    aparc2aseg.inputs.labelwm = True
    aparc2aseg.dmax = 1000
    aparc2aseg.inputs.rip = True
    aparc2aseg.inputs.hypo = True
    aparc2aseg.inputs.out_file = 'lobes+aseg.nii.gz'

    filter_lobes = pe.Node(interface=util.Function(input_names=['in_file', 'include_superlist', 'fixed_id'], output_names=['out_file'],
                                                   function=filter_labels), name='filter_lobes')
    filter_lobes.inputs.include_superlist = [[3001, 3007], [4001, 4007], [3004], [4004], [3005], [4005], [3006], [4006]]  # lobar labels in WM
    filter_lobes.inputs.fixed_id = None

    ventricles_mask = pe.Node(interface=util.Function(input_names=['in_file', 'include_superlist', 'fixed_id'], output_names=['out_file'],
                                                function=filter_labels), name='ventricles_mask')
    ventricles_mask.inputs.include_superlist = [[43, 4]]
    ventricles_mask.inputs.fixed_id = [1]

    cortex_mask = pe.Node(interface=util.Function(input_names=['in_file', 'include_superlist', 'fixed_id'], output_names=['out_file'],
                                                  function=filter_labels), name='cortex_mask')
    cortex_mask.inputs.include_superlist = [[1001, 2001, 1004, 2004, 1005, 2005, 1006, 2006]]  # lobar labels in cortex
    cortex_mask.inputs.fixed_id = [1]

    ndist_map = pe.Node(interface=util.Function(input_names=['orig_file', 'dest_file'], output_names=['out_file'],
                                                function=norm_dist_map), name='ndist_map')

    gen_wmparc = pe.Node(interface=util.Function(input_names=['incl_file', 'ndist_file', 'label_file', 'incl_labels', 'verbose'], output_names=['out_file'],
                                                 function=generate_wmparc), name='gen_wmparc')
    gen_wmparc.inputs.incl_labels = [3003, 4003, 5001, 5002] # the labels that need to be 'filled'
    gen_wmparc.inputs.verbose = False

    # collect outputs
    datasinkout = pe.Node(interface=DataSink(), name='datasinkout')
    datasinkout.inputs.parameterization=False

    # connections
    wmhsbullwf.connect(inputnode        , 'subject_ids',      fileselector_scans,'subject_id')

    wmhsbullwf.connect(fileselector_scans     , 'subject_id',       annot2label_lh, 'subject')
    wmhsbullwf.connect(fileselector_scans     , 'ANNOT_LH',         annot2label_lh, 'in_annot')
    wmhsbullwf.connect(fileselector_scans     , 'subject_id',       annot2label_rh, 'subject')
    wmhsbullwf.connect(fileselector_scans     , 'ANNOT_LH',         annot2label_rh, 'in_annot')

    wmhsbullwf.connect(annot2label_rh     , 'out_annot_file',         aparc2aseg, 'in_lobes_rh')
    wmhsbullwf.connect(annot2label_lh     , 'out_annot_file',         aparc2aseg, 'in_lobes_lh')
    wmhsbullwf.connect(fileselector_scans     , 'subject_id',         aparc2aseg, 'subject')
    wmhsbullwf.connect(aparc2aseg     , 'out_file',         filter_lobes, 'in_file')

    # wmhsbullwf.connect(fileselector_scans     , 'ASEG',         convert_aseg_mgz, 'in_file')
    wmhsbullwf.connect(aparc2aseg     , 'out_file',         ventricles_mask, 'in_file')
    wmhsbullwf.connect(aparc2aseg     , 'out_file',         cortex_mask, 'in_file')

    wmhsbullwf.connect(ventricles_mask     , 'out_file',         ndist_map, 'orig_file')
    wmhsbullwf.connect(cortex_mask     , 'out_file',         ndist_map, 'dest_file')

    wmhsbullwf.connect(aparc2aseg     , 'out_file',         gen_wmparc, 'incl_file')
    wmhsbullwf.connect(ndist_map     , 'out_file',         gen_wmparc, 'ndist_file')
    wmhsbullwf.connect(filter_lobes     , 'out_file',         gen_wmparc, 'label_file')

    wmhsbullwf.connect(gen_wmparc   , 'out_file',   datasinkout, '@wmparc')

    return(wmhsbullwf)



# def wmhs_pipeline_preproc(scans_dir, work_dir, outputdir, subject_ids, num_threads, device, opp=False, name='wmhs_preproc'):
#
#     wmhsppwf = pe.Workflow(name=name)
#     wmhsppwf.base_dir = work_dir
#
#     inputnode = pe.Node(interface=IdentityInterface(fields=['subject_ids', 'outputdir']), name='inputnode')
#     inputnode.iterables = [('subject_ids', subject_ids)]
#     inputnode.inputs.subject_ids = subject_ids
#     inputnode.inputs.outputdir = outputdir
#
#     #template for input files
#     templates = {"FLAIR": "{subject_id}/*FLAIR.nii.gz",
#                  "T1FS": "{subject_id}/mri/orig*gz",
#                  "ASEG": "{subject_id}/mri/aseg*gz",
#                  }
#     # if (not opp) and (not oseg):  # if you want to compute lobar parcels, then include other freesurfer data
#     #     templates.update({
#     #                  "RIBBON": "{subject_id}/mri/ribbon.mgz",
#     #                  "ANNOT_LH": "{subject_id}/label/lh.aparc.annot",
#     #                  "ANNOT_RH": "{subject_id}/label/rh.aparc.annot",
#     #                  "WHITE_LH": "{subject_id}/surf/lh.white",
#     #                  "WHITE_RH": "{subject_id}/surf/rh.white",
#     #                  "PIAL_LH": "{subject_id}/surf/lh.pial",
#     #                  "PIAL_RH": "{subject_id}/surf/rh.pial",
#     #                  })
#
#     fileselector = pe.Node(SelectFiles(templates), name='fileselect')
#     fileselector.inputs.base_directory = scans_dir
#
#
#     #%% step-1a convert T1 mgz to T1.nii.gz if mgz
#     convert_t1_mgz = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
#                                                           function=convert_mgz), name='convert_t1_mgz')
#
#     # step-1b convert aseg mgz to aseg.nii.gz if in mgz format
#     convert_aseg_mgz = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
#                                                      function=convert_mgz), name='convert_aseg_mgz')
#
#     # step 1-c reorient 2 std
#     reorient2std_fst1 = pe.Node(interface=Reorient2Std(), name= 'reorient2std_fst1')
#     reorient2std_aseg = pe.Node(interface=Reorient2Std(), name= 'reorient2std_aseg')
#
#
#
#     #%% step-3: N4BiasFieldCorrect low res images
#     #step 3a: N4 FLAIR
#     n4biasfieldcorrect_fl = pe.Node(interface=N4BiasFieldCorrection(),  name='n4biascorrect_fl')
#     n4biasfieldcorrect_fl.inputs.dimension = 3
#     n4biasfieldcorrect_fl.inputs.n_iterations = [50, 50, 30, 20]
#     # n4biasfieldcorrect_fl.inputs.n_iterations = [5, 5, 3, 2]
#     n4biasfieldcorrect_fl.inputs.convergence_threshold = 1e-6
#     n4biasfieldcorrect_fl.inputs.bspline_fitting_distance = 300
#     n4biasfieldcorrect_fl.inputs.output_image='FLAIR_n4.nii.gz'
#
#
#     #%% step-4: Register FST1, T1, T2 to FLAIR using FLIRT
#     #step-4a flirt T1FS to FLAIR
#     t1fs_to_flair = pe.Node(interface=fsl.FLIRT(), name='t1fs_to_flair')
#     t1fs_to_flair.inputs.cost = 'mutualinfo'
#     t1fs_to_flair.inputs.dof = 12
#     t1fs_to_flair.inputs.bins = 256
#     t1fs_to_flair.inputs.searchr_x = [-25, 25]
#     t1fs_to_flair.inputs.searchr_y = [-25, 25]
#     t1fs_to_flair.inputs.searchr_z = [-25, 25]
#     t1fs_to_flair.inputs.rigid2D=True
#     t1fs_to_flair.inputs.interp='trilinear'
#     t1fs_to_flair.inputs.out_matrix_file='FST1.mat'
#     t1fs_to_flair.inputs.out_file='FST1Warped.nii.gz'
#
#     #%% step-4d flirt aseg to FLAIR
#     aseg_to_flair = pe.Node(interface=fsl.FLIRT(), name='aseg_to_flair')
#     aseg_to_flair.inputs.interp='nearestneighbour'
#     aseg_to_flair.inputs.apply_xfm=True
#     aseg_to_flair.inputs.out_file='FLAIRaseg.nii.gz'
#
#
#     #%%step-5: Create brainmask in FLAIR space
#     compute_mask_from_aseg = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
#                                                           function=compute_mask), name='compute_mask_from_aseg')
#
#
#     #%%step-6a: Convert XFM T1
#     convert_t1_xfm = pe.Node(interface=fsl.ConvertXFM(), name='convert_t1_xfm')
#     convert_t1_xfm.inputs.invert_xfm=True
#     convert_t1_xfm.inputs.out_file='T1inv.mat'
#
#     #step-6b: Convert XFM T2
#     convert_t2_xfm = pe.Node(interface=fsl.ConvertXFM(), name='convert_t2_xfm')
#     convert_t2_xfm.inputs.invert_xfm=True
#     convert_t2_xfm.inputs.out_file='T2inv.mat'
#
#
#
#     #%%step 14a maskout image FLAIR
#     maskout_fl = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['maskoutfile'],
#                                                           function=maskout_image), name='maskout_fl')
#
#     #%%step 15a normalize image FLAIR
#     norm_fl = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['norm_outfile'],
#                                                           function=normalize_image), name='norm_fl')
#
#     #%%step 16 create inclusion mask to maskout all detections outside white matter using aseg in FLAIR space
#     inclusion_mask_from_aseg = pe.Node(interface=util.Function(input_names=['aseg_file'], output_names=['out_file'],
#                                                           function=inclusion_mask), name='inclusion_mask_from_aseg')
#
#
#     if not opp:  # if only preproc, then do not define deepmedic nodes
#
#         create_flair_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
#                                                             function=create_deepmedic_channel_file), name='create_flair_channel_config')
#         create_flair_channel_config.inputs.channel_name='flair'
#
#
#         #create_bianca_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
#         #                                                    function=create_deepmedic_channel_file), name='create_bianca_channel_config')
#         #create_bianca_channel_config.inputs.channel_name='bianca'
#
#
#         create_roi_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
#                                                             function=create_deepmedic_channel_file), name='create_roi_channel_config')
#         create_roi_channel_config.inputs.channel_name='roi'
#
#
#         create_pred_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
#                                                             function=create_deepmedic_channel_file), name='create_pred_channel_config')
#
#         create_pred_channel_config.inputs.channel_name='NamesOfPredictions'
#
#
#         create_dm_test_config = pe.Node(interface=util.Function(input_names=['flair_channel_file', 'roi_channel_file','pred_channel_file'], output_names=['test_config_file'],
#                                                             function=create_deepmedic_config_file), name='create_dm_test_config')
#
#
#         deepmedicrun = pe.Node(interface=DeepMedic(), name='deepmedicrun')
#         deepmedicrun.inputs.device=device
#         deepmedicrun.inputs.model_config_file = os.path.join(DM_MODEL_DIR, 'modelConfig_1ch.cfg')
#         # deepmedicrun.inputs.load_saved_model =  os.path.join(DM_MODEL_DIR, 'best_mod_1ch.ckpt.index')
#         if device=='cuda':
#             #this will set the use_gpu attribute for the nodes input
#             deepmedicrun.inputs.use_gpu = True
#
#         maskout_deepmedic_output = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['maskoutfile'],
#                                                               function=maskout_image), name='maskout_deepmedic_output')
#
#
#     #%% 18 collect outputs
#     datasinkout = pe.Node(interface=DataSink(), name='datasinkout')
#     datasinkout.inputs.parameterization=False
#
#     # %% workflow connections
#
#     #step 1a
#     wmhsppwf.connect(inputnode        , 'subject_ids',      fileselector,'subject_id')
#     wmhsppwf.connect(fileselector     , 'T1FS',             convert_t1_mgz, 'in_file')
#     #step 1b
#     wmhsppwf.connect(fileselector     , 'ASEG',             convert_aseg_mgz,'in_file')
#
#     #step 1c
#     wmhsppwf.connect(convert_aseg_mgz ,'out_file',          reorient2std_aseg,'in_file')
#     wmhsppwf.connect(convert_t1_mgz   ,'out_file',          reorient2std_fst1,'in_file')
#
#     #step 2a
#     wmhsppwf.connect(fileselector     , 'FLAIR',            n4biasfieldcorrect_fl  , 'input_image')
#
#     #step 4a
#     wmhsppwf.connect(reorient2std_fst1       , 'out_file',         t1fs_to_flair  , 'in_file')
#     wmhsppwf.connect(n4biasfieldcorrect_fl   , 'output_image',     t1fs_to_flair  , 'reference')
#     #step 4d
#     wmhsppwf.connect(reorient2std_aseg       ,'out_file',          aseg_to_flair  , 'in_file')
#     wmhsppwf.connect(n4biasfieldcorrect_fl   , 'output_image',     aseg_to_flair  , 'reference')
#     wmhsppwf.connect(t1fs_to_flair           , 'out_matrix_file',  aseg_to_flair  , 'in_matrix_file')
#
#     #step 5
#     wmhsppwf.connect(aseg_to_flair           , 'out_file',         compute_mask_from_aseg,'in_file')
#
#
#     #step 14
#     wmhsppwf.connect(compute_mask_from_aseg     , 'out_file',          maskout_fl, 'mask_file')
#     wmhsppwf.connect(n4biasfieldcorrect_fl   , 'output_image',      maskout_fl, 'image_file')
#
#     #step 15
#
#     wmhsppwf.connect(compute_mask_from_aseg     , 'out_file',         norm_fl, 'mask_file')
#     wmhsppwf.connect(maskout_fl                 , 'maskoutfile',      norm_fl, 'image_file')
#
#     #step 16
#     wmhsppwf.connect(aseg_to_flair              , 'out_file',         inclusion_mask_from_aseg,'aseg_file')
#
#
#     # outputs
#     wmhsppwf.connect(inputnode               , 'subject_ids',       datasinkout, 'container')
#     wmhsppwf.connect(inputnode               , 'outputdir',         datasinkout, 'base_directory')
#
#
#     #step 17 if create training set
#     wmhsppwf.connect(norm_fl   , 'norm_outfile',    datasinkout, '@flair_maskout_norm')
#     wmhsppwf.connect(compute_mask_from_aseg, 'out_file', datasinkout, '@brainmask')
#     wmhsppwf.connect(inclusion_mask_from_aseg, 'out_file', datasinkout, '@inclmask')
#
#     if not opp:
#
#         #outputs for deepmedic
#         wmhsppwf.connect(norm_fl                 , 'norm_outfile',       create_flair_channel_config, 'channel_file_path')
#         wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',           create_roi_channel_config, 'channel_file_path')
#
#         #just a dummy input to create_pred_channel_config node
#         wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',           create_pred_channel_config, 'channel_file_path')
#
#
#         wmhsppwf.connect(create_flair_channel_config  , 'channel_config_file', create_dm_test_config, 'flair_channel_file')
#         wmhsppwf.connect(create_roi_channel_config    , 'channel_config_file', create_dm_test_config, 'roi_channel_file')
#         wmhsppwf.connect(create_pred_channel_config   , 'channel_config_file', create_dm_test_config, 'pred_channel_file')
#
#         wmhsppwf.connect(create_dm_test_config        , 'test_config_file',    deepmedicrun, 'test_config_file')
#
#         wmhsppwf.connect(deepmedicrun                 , 'out_segmented_file',  maskout_deepmedic_output, 'image_file')
#         wmhsppwf.connect(inclusion_mask_from_aseg     , 'out_file',            maskout_deepmedic_output, 'mask_file')
#
#         wmhsppwf.connect(deepmedicrun                 , 'out_segmented_file',  datasinkout,'deepmedic.@predictions')
#         wmhsppwf.connect(maskout_deepmedic_output     , 'maskoutfile',         datasinkout,'deepmedic.@pred_maskout')
#
#     return wmhsppwf
#

