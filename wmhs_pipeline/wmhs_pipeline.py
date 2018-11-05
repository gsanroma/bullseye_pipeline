from __future__ import division

import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util

from nipype.interfaces import fsl
from nipype.interfaces.fsl.utils import Reorient2Std
from nipype.interfaces.fsl.maths import ApplyMask

from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.ants import DenoiseImage
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.ants import Registration

from nipype import IdentityInterface, DataSink

from .utils import *

from .configoptions import BIANCA_CLASSIFIER_DATA
from .configoptions import DM_MODEL_DIR
import os


def wmhs_pipeline(scans_dir, work_dir, outputdir, subject_ids, num_threads, device, cts=False,  name='wmhs_preproc'):
    wmhsppwf = pe.Workflow(name=name)
    wmhsppwf.base_dir = work_dir

    inputnode = pe.Node(interface=IdentityInterface(fields=['subject_ids', 'outputdir']), name='inputnode')
    inputnode.iterables = [('subject_ids', subject_ids)]
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.outputdir = outputdir

    #template for input files
    templates = {"FLAIR": "{subject_id}/*FLAIR.nii.gz",
                 "T1": "{subject_id}/*T1*.nii.gz",
                 "T2": "{subject_id}/*T2*.nii.gz",
                 "T1FS": "{subject_id}/orig*gz",
                 "ASEG": "{subject_id}/aseg*gz"}
                 
    fileselector = pe.Node(SelectFiles(templates), name='fileselect')
    fileselector.inputs.base_directory = scans_dir


    #%% step-1a convert T1 mgz to T1.nii.gz if mgz
    convert_t1_mgz = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
                                                          function=convert_mgz), name='convert_t1_mgz')
    
    # step-1b convert aseg mgz to aseg.nii.gz if in mgz format
    convert_aseg_mgz = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
                                                     function=convert_mgz), name='convert_aseg_mgz')
    
    # step 1-c reorient 2 std
    reorient2std_fst1 = pe.Node(interface=Reorient2Std(), name= 'reorient2std_fst1')
    reorient2std_aseg = pe.Node(interface=Reorient2Std(), name= 'reorient2std_aseg')
    
    
    #%%step 2: Denoise low resolution images
    # step 2a: Denoise FLAIR
    denoise_fl = pe.Node(interface=DenoiseImage(), name="denoise_fl")
    denoise_fl.inputs.dimension = 3
    denoise_fl.inputs.shrink_factor=2
    denoise_fl.inputs.output_image="FLAIR_denoised.nii.gz"
    
    #%% step-2b denoise T1
    denoise_t1 = pe.Node(interface=DenoiseImage(), name="denoise_t1")
    denoise_t1.inputs.dimension = 3
    denoise_t1.inputs.shrink_factor=2
    denoise_t1.inputs.output_image="T1_denoised.nii.gz"
        
    #%% step-2c denoise T2
    denoise_t2 = pe.Node(interface=DenoiseImage(), name="denoise_t2")
    denoise_t2.inputs.dimension = 3
    denoise_t2.inputs.shrink_factor=2
    denoise_t2.inputs.output_image="T2_denoised.nii.gz"


    #%% step-3: N4BiasFieldCorrect low res images
    #step 3a: N4 FLAIR
    n4biasfieldcorrect_fl = pe.Node(interface=N4BiasFieldCorrection(),  name='n4biascorrect_fl')
    n4biasfieldcorrect_fl.inputs.dimension = 3
    n4biasfieldcorrect_fl.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_fl.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_fl.inputs.bspline_fitting_distance = 300
    n4biasfieldcorrect_fl.inputs.output_image='FLAIR_denoised_n4.nii.gz'
    
    #%% step-3b N4 T1
    n4biasfieldcorrect_t1 = pe.Node(interface=N4BiasFieldCorrection(),  name='n4biascorrect_t1')
    n4biasfieldcorrect_t1.inputs.dimension = 3
    n4biasfieldcorrect_t1.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_t1.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_t1.inputs.bspline_fitting_distance = 300
    n4biasfieldcorrect_t1.inputs.output_image='T1_denoised_n4.nii.gz'

    #%% step-3c N4 T2
    n4biasfieldcorrect_t2 = pe.Node(interface=N4BiasFieldCorrection(),  name='n4biascorrect_t2')
    n4biasfieldcorrect_t2.inputs.dimension = 3
    n4biasfieldcorrect_t2.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_t2.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_t2.inputs.bspline_fitting_distance = 300
    n4biasfieldcorrect_t2.inputs.output_image='T2_denoised_n4.nii.gz'
    

    #%% step-4: Register FST1, T1, T2 to FLAIR using FLIRT
    #step-4a flirt T1FS to FLAIR 
    t1fs_to_flair = pe.Node(interface=fsl.FLIRT(), name='t1fs_to_flair')
    t1fs_to_flair.inputs.cost = 'mutualinfo'
    t1fs_to_flair.inputs.dof = 12
    t1fs_to_flair.inputs.bins = 256
    t1fs_to_flair.inputs.searchr_x = [-25, 25]
    t1fs_to_flair.inputs.searchr_y = [-25, 25]
    t1fs_to_flair.inputs.searchr_z = [-25, 25]
    t1fs_to_flair.inputs.rigid2D=True
    t1fs_to_flair.inputs.interp='trilinear'
    t1fs_to_flair.inputs.out_matrix_file='FST1.mat'
    t1fs_to_flair.inputs.out_file='FST1Warped.nii.gz'

    #%% step-4b flirt T1 to FLAIR 
    t1_to_flair = pe.Node(interface=fsl.FLIRT(), name='t1_to_flair')
    t1_to_flair.inputs.cost = 'mutualinfo'
    t1_to_flair.inputs.dof = 12
    t1_to_flair.inputs.bins = 256
    t1_to_flair.inputs.searchr_x = [-25, 25]
    t1_to_flair.inputs.searchr_y = [-25, 25]
    t1_to_flair.inputs.searchr_z = [-25, 25]
    t1_to_flair.inputs.rigid2D=True
    t1_to_flair.inputs.interp='trilinear'
    t1_to_flair.inputs.out_matrix_file='T1.mat'
    t1_to_flair.inputs.out_file='T1Warped.nii.gz'
    
    #%% step-4c flirt T2 to FLAIR 
    t2_to_flair = pe.Node(interface=fsl.FLIRT(), name='t2_to_flair')
    t2_to_flair.inputs.cost = 'mutualinfo'
    t2_to_flair.inputs.dof = 12
    t2_to_flair.inputs.bins = 256
    t2_to_flair.inputs.searchr_x = [-25, 25]
    t2_to_flair.inputs.searchr_y = [-25, 25]
    t2_to_flair.inputs.searchr_z = [-25, 25]
    t2_to_flair.inputs.rigid2D=True
    t2_to_flair.inputs.interp='trilinear'
    t2_to_flair.inputs.out_matrix_file='T2.mat'
    t2_to_flair.inputs.out_file='T2Warped.nii.gz'
    
    
    #%% step-4d flirt aseg to FLAIR
    aseg_to_flair = pe.Node(interface=fsl.FLIRT(), name='aseg_to_flair')
    aseg_to_flair.inputs.interp='nearestneighbour'
    aseg_to_flair.inputs.apply_xfm=True
    aseg_to_flair.inputs.out_file='FLAIRaseg.nii.gz'
    
    
    #%%step-5: Create brainmask in FLAIR space
    compute_mask_from_aseg = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
                                                          function=compute_mask), name='compute_mask_from_aseg')
    
    
    #%%step-6a: Convert XFM T1
    convert_t1_xfm = pe.Node(interface=fsl.ConvertXFM(), name='convert_t1_xfm')
    convert_t1_xfm.inputs.invert_xfm=True
    convert_t1_xfm.inputs.out_file='T1inv.mat'
    
    #step-6b: Convert XFM T2
    convert_t2_xfm = pe.Node(interface=fsl.ConvertXFM(), name='convert_t2_xfm')
    convert_t2_xfm.inputs.invert_xfm=True
    convert_t2_xfm.inputs.out_file='T2inv.mat'
    
    
    #%%step-7: warp brainmask to T1, T2
    #step 7a: brainmask to T1
    warp_bm_to_t1 = pe.Node(interface=fsl.FLIRT(), name='brainmask_to_t1')
    warp_bm_to_t1.inputs.interp='nearestneighbour'
    warp_bm_to_t1.inputs.apply_xfm=True
    warp_bm_to_t1.inputs.out_file='brainmaskT1.nii.gz'
    
    #step 7b: brainmask to T2
    warp_bm_to_t2 = pe.Node(interface=fsl.FLIRT(), name='brainmask_to_t2')
    warp_bm_to_t2.inputs.interp='nearestneighbour'
    warp_bm_to_t2.inputs.apply_xfm=True
    warp_bm_to_t2.inputs.out_file='brainmaskT2.nii.gz'
    
    


    #%%step 8: Denoise with High resolution mask
    # step 8a: Denoise FLAIR
    denoise_hi_fl = pe.Node(interface=DenoiseImageWithMask(), name="denoise_hi_fl")
    denoise_hi_fl.inputs.dimension = 3
    denoise_hi_fl.inputs.output_image='FLAIR_denoised_hires.nii.gz'
    
    #%% step-8b denoise T1
    denoise_hi_t1 = pe.Node(interface=DenoiseImageWithMask(), name="denoise_hi_t1")
    denoise_hi_t1.inputs.dimension = 3
    denoise_hi_t1.inputs.output_image='T1_denoised_hires.nii.gz'
        
    #%% step-8c denoise T2
    denoise_hi_t2 = pe.Node(interface=DenoiseImageWithMask(), name="denoise_hi_t2")
    denoise_hi_t2.inputs.dimension = 3
    denoise_hi_t2.inputs.output_image='T2_denoised_hires.nii.gz'




    #%% step-9: N4BiasFieldCorrect hi res images
    #step 9a: N4 FLAIR
    n4biasfieldcorrect_hi_fl = pe.Node(interface=N4BiasFieldCorrection(),  name='n4biascorrect_hi_fl')
    n4biasfieldcorrect_hi_fl.inputs.dimension = 3
    n4biasfieldcorrect_hi_fl.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_hi_fl.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_hi_fl.inputs.bspline_fitting_distance = 300
    n4biasfieldcorrect_hi_fl.inputs.output_image='FLAIR_denoised_n4_hires.nii.gz'
    
    #%% step-9b N4 T1
    n4biasfieldcorrect_hi_t1 = pe.Node(interface=N4BiasFieldCorrection(),  name='n4biascorrect_hi_t1')
    n4biasfieldcorrect_hi_t1.inputs.dimension = 3
    n4biasfieldcorrect_hi_t1.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_hi_t1.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_hi_t1.inputs.bspline_fitting_distance = 300
    n4biasfieldcorrect_hi_t1.inputs.output_image='T1_denoised_n4_hires.nii.gz'

    #%% step-9c N4 T2
    n4biasfieldcorrect_hi_t2 = pe.Node(interface=N4BiasFieldCorrection(),  name='n4biascorrect_hi_t2')
    n4biasfieldcorrect_hi_t2.inputs.dimension = 3
    n4biasfieldcorrect_hi_t2.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_hi_t2.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_hi_t2.inputs.bspline_fitting_distance = 300
    n4biasfieldcorrect_hi_t2.inputs.output_image='T2_denoised_n4_hires.nii.gz'



    #%% step 10: warp T1, T2 to FLAIR hires 
    #step-10a flirt T1 to FLAIR 
    t1_to_flair_hi = pe.Node(interface=fsl.FLIRT(), name='t1_to_flair_hi')
    t1_to_flair_hi.inputs.out_file='T1Warped_denoised_n4_hires.nii.gz'
    t1_to_flair_hi.inputs.apply_xfm=True
    t1_to_flair_hi.inputs.interp='trilinear'
    
    #%% step-10b flirt T2 to FLAIR 
    t2_to_flair_hi = pe.Node(interface=fsl.FLIRT(), name='t2_to_flair_hi')
    t2_to_flair_hi.inputs.out_file='T2Warped_denoised_n4_hires.nii.gz'
    t2_to_flair_hi.inputs.apply_xfm=True
    t2_to_flair_hi.inputs.interp='trilinear'
    
    
    #%%step 11: register FLAIR to MNI
    
    ants_register_flair2mni = pe.Node(interface=Registration(), name='flair2mni')
    ants_register_flair2mni.inputs.dimension=3
    ants_register_flair2mni.inputs.output_transform_prefix = "FLAIR_to_template_"
    ants_register_flair2mni.inputs.float=True
    ants_register_flair2mni.inputs.use_histogram_matching=False
    ants_register_flair2mni.inputs.write_composite_transform = False
    ants_register_flair2mni.inputs.collapse_output_transforms = True
    ants_register_flair2mni.inputs.initialize_transforms_per_stage = False
    ants_register_flair2mni.inputs.interpolation = 'Linear'
    ants_register_flair2mni.inputs.metric = ['MI']*2
    ants_register_flair2mni.inputs.smoothing_sigmas = [[4,2,1,0],[4,2,1,0]]
    ants_register_flair2mni.inputs.shrink_factors = [[8,4,2,1],[8,4,2,1]]
    ants_register_flair2mni.inputs.transforms = ['Rigid','Affine']
    ants_register_flair2mni.inputs.transform_parameters = [(0.1,),(0.1,)] 
    ants_register_flair2mni.inputs.number_of_iterations = [[1000, 500, 250, 0],[1000,500,250,0]]
    ants_register_flair2mni.inputs.metric_weight=[1]*2
    ants_register_flair2mni.inputs.radius_or_number_of_bins = [32]*2
    ants_register_flair2mni.inputs.sampling_strategy = ['Regular','Regular']
    ants_register_flair2mni.inputs.sampling_percentage = [0.25, 0.25]
    ants_register_flair2mni.inputs.convergence_threshold = [1.e-8, 1.e-8]
    ants_register_flair2mni.inputs.convergence_window_size = [10]*2
    ants_register_flair2mni.inputs.output_warped_image='FLAIRWarped_to_template.nii.gz'
    ants_register_flair2mni.inputs.fixed_image = fsl.Info.standard_image('MNI152_T1_1mm.nii.gz')
    ants_register_flair2mni.inputs.num_threads=num_threads
    
  
    #%%step 12: convert transform .mat file to .txt
    convert_transform_file = pe.Node(interface=ConvertTransformFile(), name='convert_transform_file')
    convert_transform_file.inputs.dimension=3
    
    #%%step 13: convert ANTs .txt transform to FSL
    c3d_affine=pe.Node(interface=C3DAffineTool(), name='c3d_affine')
    c3d_affine.inputs.reference_file=fsl.Info.standard_image('MNI152_T1_1mm.nii.gz')
    c3d_affine.inputs.ras2fsl=True
    c3d_affine.inputs.out_filename='FLAIR_to_template_fsl.mat'
    
    #%%step 14a maskout image FLAIR
    maskout_fl = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['maskoutfile'],
                                                          function=maskout_image), name='maskout_fl')
    
    #step 14b maskout image T1warped
    maskout_t1w = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['maskoutfile'],
                                                          function=maskout_image), name='maskout_t1w')
    #step 14c maskout image T2warped
    maskout_t2w = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['maskoutfile'],
                                                          function=maskout_image), name='maskout_t2w')
    
    
    #%%step 15a normalize image FLAIR
    norm_fl = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['norm_outfile'],
                                                          function=normalize_image), name='norm_fl')
    
    #step 15b normalize image T1warped
    norm_t1w = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['norm_outfile'],
                                                          function=normalize_image), name='norm_t1w')
     #step 15c normalize image T2warped
    norm_t2w = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['norm_outfile'],
                                                          function=normalize_image), name='norm_t2w')
    
    #%%step 16 create inclusion mask to maskout all detections outside white matter using aseg in FLAIR space
    inclusion_mask_from_aseg = pe.Node(interface=util.Function(input_names=['aseg_file'], output_names=['out_file'],
                                                          function=inclusion_mask), name='inclusion_mask_from_aseg')
    
    
    
    #%% step-17 create master file for bianca
    if cts:
        create_masterfile_tr = pe.Node(interface=util.Function(input_names=['flair', 't1w','t2w', 'fl2mni_matrix_file'], output_names=['master_file'],
                                                        function=create_master_file_train), name='create_masterfile_tr')
    else:
        create_masterfile_qr = pe.Node(interface=util.Function(input_names=['flair', 't1w','t2w', 'fl2mni_matrix_file'], output_names=['master_file'],
                                                        function=create_master_file_query), name='create_masterfile_qr')
        
        bianca = pe.Node(interface=Bianca(), name='bianca')
        bianca.inputs.querysubjectnum=1
        bianca.inputs.brainmaskfeaturenum=1
        bianca.inputs.featuresubset="1,2,3"
        bianca.inputs.matfeaturenum=5
        bianca.inputs.spatialweight=1.0
        bianca.inputs.trainingpts=2000
        bianca.inputs.nonlespts=10000
        bianca.inputs.out_filename='bianca_wmhseg.nii.gz'
        bianca.inputs.loadclassifierdata=BIANCA_CLASSIFIER_DATA
        
        threshold_bianca_output=pe.Node(interface=util.Function(input_names=['biancasegfile'], output_names=['thresholded_file'],
                                                        function=threshold_bianca), name='threshold_bianca_output')
        
        maskout_bianca_output = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['maskoutfile'],
                                                          function=maskout_image), name='maskout_bianca_output')
    
    
    #%% step-18 DeepMedic run
    
    create_t1_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
                                                        function=create_deepmedic_channel_file), name='create_t1_channel_config')
    create_t1_channel_config.inputs.channel_name='t1'
    

    create_t2_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
                                                        function=create_deepmedic_channel_file), name='create_t2_channel_config')
    create_t2_channel_config.inputs.channel_name='t2'
    

    create_flair_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
                                                        function=create_deepmedic_channel_file), name='create_flair_channel_config')
    create_flair_channel_config.inputs.channel_name='flair'
    

    #create_bianca_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
    #                                                    function=create_deepmedic_channel_file), name='create_bianca_channel_config')
    #create_bianca_channel_config.inputs.channel_name='bianca'
    

    create_roi_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
                                                        function=create_deepmedic_channel_file), name='create_roi_channel_config')
    create_roi_channel_config.inputs.channel_name='roi'
    

    create_pred_channel_config = pe.Node(interface=util.Function(input_names=['channel_name', 'channel_file_path'], output_names=['channel_config_file'],
                                                        function=create_deepmedic_channel_file), name='create_pred_channel_config')
    
    create_pred_channel_config.inputs.channel_name='NamesOfPredictions'
    
    
    create_dm_test_config = pe.Node(interface=util.Function(input_names=['flair_channel_file', 't1_channel_file','t2_channel_file','roi_channel_file','pred_channel_file'],
                                                         output_names=['test_config_file'],
                                                        function=create_deepmedic_config_file), name='create_dm_test_config')
    
    
    deepmedicrun = pe.Node(interface=DeepMedic(), name='deepmedicrun')
    deepmedicrun.inputs.device=device
    deepmedicrun.inputs.model_config_file = os.path.join(DM_MODEL_DIR, 'modelConfig_3ch_tf.cfg') 
    deepmedicrun.inputs.load_saved_model =  os.path.join(DM_MODEL_DIR, 'generic_model_3ch_tf.all_onlydm_shahid_tf.final.2018-10-26.02.52.04.107352.model.ckpt.index')
    if device=='cuda':
        deepmedicrun.inputs.use_gpu = True
    else:
        deepmedicrun.inputs.use_gpu = False
    
    maskout_deepmedic_output = pe.Node(interface=util.Function(input_names=['mask_file','image_file'], output_names=['maskoutfile'],
                                                          function=maskout_image), name='maskout_deepmedic_output')




    #%% 18 collect outputs
    datasinkout = pe.Node(interface=DataSink(), name='datasinkout')
    datasinkout.inputs.parameterization=False

    # %% workflow connections
    
    #step 1a
    wmhsppwf.connect(inputnode        , 'subject_ids',      fileselector,'subject_id')
    wmhsppwf.connect(fileselector     , 'T1FS',             convert_t1_mgz, 'in_file')
    #step 1b
    wmhsppwf.connect(fileselector     , 'ASEG',             convert_aseg_mgz,'in_file')
    
    #step 1c
    wmhsppwf.connect(convert_aseg_mgz ,'out_file',          reorient2std_aseg,'in_file')
    wmhsppwf.connect(convert_t1_mgz   ,'out_file',          reorient2std_fst1,'in_file')
    
    #step 2a
    wmhsppwf.connect(fileselector     , 'FLAIR',            denoise_fl  , 'input_image')
    #step 2b
    wmhsppwf.connect(fileselector     , 'T1',               denoise_t1  , 'input_image')
    #step 2c
    wmhsppwf.connect(fileselector     , 'T2',               denoise_t2  , 'input_image')    
    
    #step 3a
    wmhsppwf.connect(denoise_fl       , 'output_image',           n4biasfieldcorrect_fl  , 'input_image')   
    #step 3b
    wmhsppwf.connect(denoise_t1       , 'output_image',           n4biasfieldcorrect_t1  , 'input_image')   
    #step 3c
    wmhsppwf.connect(denoise_t2       , 'output_image',           n4biasfieldcorrect_t2  , 'input_image')   
    
    
    #step 4a
    wmhsppwf.connect(reorient2std_fst1       , 'out_file',         t1fs_to_flair  , 'in_file')
    wmhsppwf.connect(n4biasfieldcorrect_fl   , 'output_image',     t1fs_to_flair  , 'reference')
    #step 4b
    wmhsppwf.connect(n4biasfieldcorrect_t1   , 'output_image',     t1_to_flair  , 'in_file')
    wmhsppwf.connect(n4biasfieldcorrect_fl   , 'output_image',     t1_to_flair  , 'reference')
    #step 4c
    wmhsppwf.connect(n4biasfieldcorrect_t2   , 'output_image',     t2_to_flair  , 'in_file')
    wmhsppwf.connect(n4biasfieldcorrect_fl   , 'output_image',     t2_to_flair  , 'reference')
    #step 4d
    wmhsppwf.connect(reorient2std_aseg       ,'out_file',          aseg_to_flair  , 'in_file')
    wmhsppwf.connect(n4biasfieldcorrect_fl   , 'output_image',     aseg_to_flair  , 'reference')
    wmhsppwf.connect(t1fs_to_flair           , 'out_matrix_file',  aseg_to_flair  , 'in_matrix_file')
    
    #step 5
    wmhsppwf.connect(aseg_to_flair           , 'out_file',         compute_mask_from_aseg,'in_file')
    
    
    #step 6a
    wmhsppwf.connect(t1_to_flair             , 'out_matrix_file',  convert_t1_xfm, 'in_file')
    #step 6b
    wmhsppwf.connect(t2_to_flair             , 'out_matrix_file',  convert_t2_xfm, 'in_file')
    
    
    #step 7a
    wmhsppwf.connect(convert_t1_xfm          , 'out_file',     warp_bm_to_t1,   'in_matrix_file')
    wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',     warp_bm_to_t1,   'in_file')
    wmhsppwf.connect(n4biasfieldcorrect_t1   , 'output_image', warp_bm_to_t1, 'reference')
    
    #step 7b
    wmhsppwf.connect(convert_t2_xfm          , 'out_file',     warp_bm_to_t2,   'in_matrix_file')
    wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',     warp_bm_to_t2,   'in_file')
    wmhsppwf.connect(n4biasfieldcorrect_t2   , 'output_image', warp_bm_to_t2,   'reference')
    
    
    #step 8a
    wmhsppwf.connect(fileselector            , 'FLAIR',            denoise_hi_fl  , 'input_image')
    wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',         denoise_hi_fl  , 'mask_image')
    #step 8b
    wmhsppwf.connect(fileselector            , 'T1',               denoise_hi_t1  , 'input_image')
    wmhsppwf.connect(warp_bm_to_t1           , 'out_file',         denoise_hi_t1  , 'mask_image')
    #step 8c
    wmhsppwf.connect(fileselector            , 'T2',               denoise_hi_t2  , 'input_image')  
    wmhsppwf.connect(warp_bm_to_t2           , 'out_file',         denoise_hi_t2  , 'mask_image')
    
    #step 9a
    wmhsppwf.connect(denoise_hi_fl           , 'output_image',     n4biasfieldcorrect_hi_fl  , 'input_image')   
    #step 9b
    wmhsppwf.connect(denoise_hi_t1           , 'output_image',     n4biasfieldcorrect_hi_t1  , 'input_image')   
    #step 9c
    wmhsppwf.connect(denoise_hi_t2           , 'output_image',     n4biasfieldcorrect_hi_t2  , 'input_image')
    
    #step 10a
    wmhsppwf.connect(n4biasfieldcorrect_hi_fl , 'output_image',    t1_to_flair_hi, 'reference')
    wmhsppwf.connect(n4biasfieldcorrect_hi_t1 , 'output_image',    t1_to_flair_hi, 'in_file')
    wmhsppwf.connect(t1_to_flair              , 'out_matrix_file', t1_to_flair_hi, 'in_matrix_file')
    
    #step 10b
    wmhsppwf.connect(n4biasfieldcorrect_hi_fl , 'output_image',    t2_to_flair_hi, 'reference')
    wmhsppwf.connect(n4biasfieldcorrect_hi_t2 , 'output_image',    t2_to_flair_hi, 'in_file')
    wmhsppwf.connect(t2_to_flair              , 'out_matrix_file', t2_to_flair_hi, 'in_matrix_file')
    
    
    #step 11   
    wmhsppwf.connect(n4biasfieldcorrect_hi_fl  , 'output_image',       ants_register_flair2mni, 'moving_image')
    
    #step 12
    wmhsppwf.connect(ants_register_flair2mni, ('forward_transforms', get_first_file), convert_transform_file, 'in_file')
    
    #step 13
    wmhsppwf.connect(n4biasfieldcorrect_hi_fl   , 'output_image',      c3d_affine, 'source_file')
    wmhsppwf.connect(convert_transform_file     , 'out_file',          c3d_affine, 'transform_file')
    
    #step 14
    
    wmhsppwf.connect(compute_mask_from_aseg     , 'out_file',          maskout_fl, 'mask_file')
    wmhsppwf.connect(n4biasfieldcorrect_hi_fl   , 'output_image',      maskout_fl, 'image_file')
        
    wmhsppwf.connect(compute_mask_from_aseg      , 'out_file',         maskout_t1w, 'mask_file')
    wmhsppwf.connect(t1_to_flair_hi             , 'out_file',          maskout_t1w, 'image_file')
        
    wmhsppwf.connect(compute_mask_from_aseg      , 'out_file',         maskout_t2w, 'mask_file')
    wmhsppwf.connect(t2_to_flair_hi             , 'out_file',          maskout_t2w, 'image_file')
    

    #step 15
    
    wmhsppwf.connect(compute_mask_from_aseg     , 'out_file',         norm_fl, 'mask_file')
    wmhsppwf.connect(maskout_fl                 , 'maskoutfile',      norm_fl, 'image_file')
        
    wmhsppwf.connect(compute_mask_from_aseg     , 'out_file',         norm_t1w, 'mask_file')
    wmhsppwf.connect(maskout_t1w                , 'maskoutfile',      norm_t1w, 'image_file')
        
    wmhsppwf.connect(compute_mask_from_aseg     , 'out_file',         norm_t2w, 'mask_file')
    wmhsppwf.connect(maskout_t2w                , 'maskoutfile',      norm_t2w, 'image_file')
    
    #step 16
    wmhsppwf.connect(aseg_to_flair              , 'out_file',         inclusion_mask_from_aseg,'aseg_file')
    
               
    # outputs
    wmhsppwf.connect(inputnode               , 'subject_ids',       datasinkout, 'container')
    wmhsppwf.connect(inputnode               , 'outputdir',         datasinkout, 'base_directory')


    #step 17 if create training set
    if cts:
        wmhsppwf.connect(maskout_fl              , 'maskoutfile',       create_masterfile_tr, 'flair')
        wmhsppwf.connect(maskout_t1w             , 'maskoutfile',       create_masterfile_tr, 't1w')
        wmhsppwf.connect(maskout_t2w             , 'maskoutfile',       create_masterfile_tr, 't2w')
        wmhsppwf.connect(c3d_affine              , 'fsl_transform',     create_masterfile_tr, 'fl2mni_matrix_file')
        
        wmhsppwf.connect(create_masterfile_tr    , 'master_file',       datasinkout, 'BIANCA.@masterfile')
    else:
        wmhsppwf.connect(maskout_fl              , 'maskoutfile',       create_masterfile_qr, 'flair')
        wmhsppwf.connect(maskout_t1w             , 'maskoutfile',       create_masterfile_qr, 't1w')
        wmhsppwf.connect(maskout_t2w             , 'maskoutfile',       create_masterfile_qr, 't2w')
        wmhsppwf.connect(c3d_affine              , 'fsl_transform',     create_masterfile_qr, 'fl2mni_matrix_file')
        
        wmhsppwf.connect(c3d_affine              , 'fsl_transform',     datasinkout, 'BIANCA.@flair2mnimat')        
        #bianca
        wmhsppwf.connect(create_masterfile_qr    , 'master_file',        bianca,  'master_file')
        wmhsppwf.connect(create_masterfile_qr    , 'master_file',        datasinkout,'BIANCA.@masterfile')
        
        wmhsppwf.connect(bianca                  , 'out_file',           threshold_bianca_output, 'biancasegfile')
        wmhsppwf.connect(threshold_bianca_output , 'thresholded_file',   datasinkout, 'BIANCA.@threholded_file')
        
        wmhsppwf.connect(threshold_bianca_output , 'thresholded_file',   maskout_bianca_output, 'image_file')
        wmhsppwf.connect(inclusion_mask_from_aseg, 'out_file',           maskout_bianca_output, 'mask_file')
        
        wmhsppwf.connect(maskout_bianca_output   , 'maskoutfile',        datasinkout, 'BIANCA.@bianca_thresh_maskout')
        wmhsppwf.connect(inclusion_mask_from_aseg, 'out_file',           datasinkout, 'BIANCA.@inclmask')
        
        wmhsppwf.connect(bianca                  , 'out_file',           datasinkout, 'BIANCA.@biancasegfile')
        wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',           datasinkout, '@brainmask')
        
        wmhsppwf.connect(n4biasfieldcorrect_hi_fl   , 'output_image',    datasinkout, '@flair')
        wmhsppwf.connect(t1_to_flair_hi             , 'out_file',        datasinkout, '@t1w')
        wmhsppwf.connect(t2_to_flair_hi             , 'out_file',        datasinkout, '@t2w')
        
        #outputs for deepmedic
        wmhsppwf.connect(norm_fl                 , 'norm_outfile',       create_flair_channel_config, 'channel_file_path')
        wmhsppwf.connect(norm_t1w                , 'norm_outfile',       create_t1_channel_config, 'channel_file_path')
        wmhsppwf.connect(norm_t2w                , 'norm_outfile',       create_t2_channel_config, 'channel_file_path')
        #wmhsppwf.connect(bianca                  , 'out_file',           create_bianca_channel_config, 'channel_file_path')
        wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',           create_roi_channel_config, 'channel_file_path')
        
        #just a dummy input to create_pred_channel_config node
        wmhsppwf.connect(compute_mask_from_aseg  , 'out_file',           create_pred_channel_config, 'channel_file_path')
        
        
        wmhsppwf.connect(create_flair_channel_config  , 'channel_config_file', create_dm_test_config, 'flair_channel_file')
        wmhsppwf.connect(create_t1_channel_config     , 'channel_config_file', create_dm_test_config, 't1_channel_file')
        wmhsppwf.connect(create_t2_channel_config     , 'channel_config_file', create_dm_test_config, 't2_channel_file')
        #wmhsppwf.connect(create_bianca_channel_config , 'channel_config_file', create_dm_test_config, 'bianca_channel_file')
        wmhsppwf.connect(create_roi_channel_config    , 'channel_config_file', create_dm_test_config, 'roi_channel_file')
        wmhsppwf.connect(create_pred_channel_config   , 'channel_config_file', create_dm_test_config, 'pred_channel_file')
        
        wmhsppwf.connect(create_dm_test_config        , 'test_config_file',    deepmedicrun, 'test_config_file')
        
        wmhsppwf.connect(deepmedicrun                 , 'out_segmented_file',  maskout_deepmedic_output, 'image_file')
        wmhsppwf.connect(inclusion_mask_from_aseg     , 'out_file',            maskout_deepmedic_output, 'mask_file')
        
        wmhsppwf.connect(deepmedicrun                 , 'out_segmented_file',  datasinkout,'deepmedic.@predictions')
        wmhsppwf.connect(maskout_deepmedic_output     , 'maskoutfile',         datasinkout,'deepmedic.@pred_maskout')        
        wmhsppwf.connect(inclusion_mask_from_aseg     , 'out_file',            datasinkout,'deepmedic.@inclmask')
        
        
           
    
    return wmhsppwf
    

