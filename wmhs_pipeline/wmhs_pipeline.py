from __future__ import division

import nipype.pipeline.engine as pe
from nipype import SelectFiles
import nipype.interfaces.utility as util

from nipype.interfaces import fsl
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.ants import DenoiseImage
from nipype.interfaces.ants import N4BiasFieldCorrection

from nipype import IdentityInterface, DataSink

from .configoptions import WMHS_MASKS_DIR
from .utils import convert_mgz, create_master_file


def create_wmhs_pipeline(scans_dir, work_dir, outputdir, subject_ids, name='wmhs_pipeline'):
    wmhswf = pe.Workflow(name=name)
    wmhswf.base_dir = work_dir

    inputnode = pe.Node(interface=IdentityInterface(fields=['subject_ids', 'outputdir']), name='inputnode')
    inputnode.iterables = [('subject_ids', subject_ids)]
    inputnode.inputs.subject_ids = subject_ids
    inputnode.inputs.outputdir = outputdir

    #template for input files
    templates = {"FLAIR": "{subject_id}/*FLAIR.nii.gz",
                 "T1": "{subject_id}/*T1*.nii.gz",
                 "T2": "{subject_id}/*T2*.nii.gz",
                 "T1FS": "{subject_id}/orig*gz",
                 "BMASK": "{subject_id}/brainmask*gz"}
                 
    fileselector = pe.Node(SelectFiles(templates), name='fileselect')
    fileselector.inputs.base_directory = scans_dir


    #%% step-1a convert T1 mgz to T1.nii.gz if mgz
    convert_t1_mgz = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
                                                          function=convert_mgz), name='convert_t1_mgz')
    
    # step-1b convert brainmask mgz to brainmask.nii.gz if in mgz format
    convert_bm_mgz = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file'],
                                                     function=convert_mgz), name='convert_bm_mgz')
    
    
    
    #%% step-2a flirt T1FS to FLAIR 
    t1fs_to_flair = pe.Node(interface=fsl.FLIRT(), name='t1fs_to_flair')
    t1fs_to_flair.inputs.cost = 'mutualinfo'
    t1fs_to_flair.inputs.dof = 12
    t1fs_to_flair.inputs.bins = 256
    t1fs_to_flair.inputs.searchr_x = [-25.0, 25.0]
    t1fs_to_flair.inputs.searchr_y = [-25.0, 25.0]
    t1fs_to_flair.inputs.searchr_z = [-25.0, 25.0]
    
    #%% step-2b flirt T1 to FLAIR 
    t1_to_flair = pe.Node(interface=fsl.FLIRT(), name='t1_to_flair')
    t1_to_flair.inputs.cost = 'mutualinfo'
    t1_to_flair.inputs.dof = 12
    t1_to_flair.inputs.bins = 256
    t1_to_flair.inputs.searchr_x = [-25.0, 25.0]
    t1_to_flair.inputs.searchr_y = [-25.0, 25.0]
    t1_to_flair.inputs.searchr_z = [-25.0, 25.0]
    
    #%% step-2c flirt T2 to FLAIR 
    t2_to_flair = pe.Node(interface=fsl.FLIRT(), name='t2_to_flair')
    t2_to_flair.inputs.cost = 'mutualinfo'
    t2_to_flair.inputs.dof = 12
    t2_to_flair.inputs.bins = 256
    t2_to_flair.inputs.searchr_x = [-25.0, 25.0]
    t2_to_flair.inputs.searchr_y = [-25.0, 25.0]
    t2_to_flair.inputs.searchr_z = [-25.0, 25.0]
        
   
    
    #%% step-3 binarize brainmask 
    binarize_bm = pe.Node(interface=Binarize(), name='binarize_bm')
    binarize_bm.inputs.max = 1000
    binarize_bm.inputs.min = 1
    


    #%% step-4a applyxfms BM
    applyxfm_bm = pe.Node(interface=fsl.FLIRT(), name='applyxfm_bm')
    applyxfm_bm.inputs.apply_xfm = True
    applyxfm_bm.inputs.interp = 'nearestneighbour'
    
    #%% step-4b applyxfms BM
    applyxfm_t1 = pe.Node(interface=fsl.FLIRT(), name='applyxfm_t1')
    applyxfm_t1.inputs.apply_xfm = True
    applyxfm_t1.inputs.interp = 'trilinear'

    #%% step-4c applyxfms BM
    applyxfm_t2 = pe.Node(interface=fsl.FLIRT(), name='applyxfm_t2')
    applyxfm_t2.inputs.apply_xfm = True
    applyxfm_t2.inputs.interp = 'trilinear'
            
    
    #%% step-5a apply mask to flair
    applymask_fl = pe.Node(interface=ApplyMask(), name='applymask_fl')
    
    #%% step-5b apply mask to t1
    applymask_t1 = pe.Node(interface=ApplyMask(), name='applymask_t1')
        
    #%% step-5c apply mask to t2
    applymask_t2 = pe.Node(interface=ApplyMask(), name='applymask_t2')
    

    #%% step-6a denoise fl
    denoise_fl = pe.Node(interface=DenoiseImage(), name="denoise_fl")
    denoise_fl.inputs.dimension = 3    
    
    #%% step-6b denoise fl
    denoise_t1 = pe.Node(interface=DenoiseImage(), name="denoise_t1")
    denoise_t1.inputs.dimension = 3    
        
    #%% step-6c denoise fl
    denoise_t2 = pe.Node(interface=DenoiseImage(), name="denoise_t2")
    denoise_t2.inputs.dimension = 3    
    

    #%% step-7a n4 fl
    n4biasfieldcorrect_fl = pe.Node(interface=N4BiasFieldCorrection(), name='n4biascorrect_fl')
    n4biasfieldcorrect_fl.inputs.dimension = 3
    n4biasfieldcorrect_fl.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_fl.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_fl.inputs.bspline_fitting_distance = 300
    
    #%% step-7b n4 t1
    n4biasfieldcorrect_t1 = pe.Node(interface=N4BiasFieldCorrection(), name='n4biascorrect_t1')
    n4biasfieldcorrect_t1.inputs.dimension = 3
    n4biasfieldcorrect_t1.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_t1.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_t1.inputs.bspline_fitting_distance = 300

    #%% step-7c n4 t2
    n4biasfieldcorrect_t2 = pe.Node(interface=N4BiasFieldCorrection(), name='n4biascorrect_t2')
    n4biasfieldcorrect_t2.inputs.dimension = 3
    n4biasfieldcorrect_t2.inputs.n_iterations = [50, 50, 30, 20]
    n4biasfieldcorrect_t2.inputs.convergence_threshold = 1e-6
    n4biasfieldcorrect_t2.inputs.bspline_fitting_distance = 300


    
    #%% step-8 fl to mni
    flair2mni = pe.Node(interface=fsl.FLIRT(), name='flair2mni')
    flair2mni.inputs.cost = 'mutualinfo'
    flair2mni.inputs.dof = 12
    flair2mni.inputs.reference = fsl.Info.standard_image('MNI152_T1_1mm_brain.nii.gz')
    
    

    #%% step-9 create master file for bianca
    create_master_file = pe.Node(interface=util.Function(input_names=['querysubject_file'], output_names=['master_file'],
                                                        function=create_master_file), name='create_master_file')
    


    #%% 17 collect outputs
    datasink = pe.Node(interface=DataSink(), name='datasinker')
    datasink.inputs.parameterization=False
    

    # %% workflow connections
    
    #step 1a
    wmhswf.connect(inputnode               , 'subject_ids',      fileselector, 'subject_id')
    wmhswf.connect(fileselector            , 'T1FS',             convert_t1_mgz, 'in_file')
    #step 1b
    wmhswf.connect(fileselector            , 'BMASK',            convert_bm_mgz, 'in_file')
    
    
    #step 2a
    wmhswf.connect(convert_t1_mgz          , 'out_file',         t1fs_to_flair  , 'in_file')
    wmhswf.connect(fileselector            , 'FLAIR',            t1fs_to_flair  , 'reference')
    
    #step 2b
    wmhswf.connect(fileselector            , 'T1',               t1_to_flair  , 'in_file')
    wmhswf.connect(fileselector            , 'FLAIR',            t1_to_flair  , 'reference')

    #step 2c
    wmhswf.connect(fileselector            , 'T2',               t2_to_flair  , 'in_file')
    wmhswf.connect(fileselector            , 'FLAIR',            t2_to_flair  , 'reference')
        
    
    #step 3
    wmhswf.connect(convert_bm_mgz          ,'out_file',          binarize_bm  , 'in_file')
    
  

    #step 4a
    wmhswf.connect(binarize_bm             , 'binary_file',      applyxfm_bm  , 'in_file')
    wmhswf.connect(fileselector            , 'FLAIR',            applyxfm_bm  , 'reference')
    wmhswf.connect(t1fs_to_flair           , 'out_matrix_file',  applyxfm_bm  , 'in_matrix_file')

    #step 4b
    wmhswf.connect(binarize_bm             , 'binary_file',      applyxfm_t1  , 'in_file')
    wmhswf.connect(fileselector            , 'FLAIR',            applyxfm_t1  , 'reference')
    wmhswf.connect(t1_to_flair             , 'out_matrix_file',  applyxfm_t1  , 'in_matrix_file')


    #step 4c
    wmhswf.connect(binarize_bm             , 'binary_file',      applyxfm_t2  , 'in_file')
    wmhswf.connect(fileselector            , 'FLAIR',            applyxfm_t2  , 'reference')
    wmhswf.connect(t2_to_flair             , 'out_matrix_file',  applyxfm_t2  , 'in_matrix_file')

    
    #step 5a
    wmhswf.connect(applyxfm_bm             , 'out_file',         applymask_fl  , 'mask_file')
    wmhswf.connect(fileselector            , 'FLAIR',            applymask_fl  , 'in_file')

    #step 5b
    wmhswf.connect(applyxfm_bm             , 'out_file',         applymask_t1  , 'mask_file')
    wmhswf.connect(applyxfm_t1             , 'out_file',         applymask_t1  , 'in_file')

    #step 5c
    wmhswf.connect(applyxfm_bm             , 'out_file',         applymask_t2  , 'mask_file')
    wmhswf.connect(applyxfm_t2             , 'out_file',         applymask_t2  , 'in_file')


    #step 6a
    wmhswf.connect(applymask_fl             , 'out_file',         denoise_fl  , 'in_file')

    #step 6b
    wmhswf.connect(applymask_t1             , 'out_file',         denoise_t1  , 'in_file')

    #step 6c
    wmhswf.connect(applymask_t2             , 'out_file',         denoise_t2  , 'in_file')    
    
    
    #step 7a
    wmhswf.connect(denoise_fl             , 'denoised',           n4biasfieldcorrect_fl  , 'input_image')   
    
    #step 7b
    wmhswf.connect(denoise_t1             , 'denoised',           n4biasfieldcorrect_t1  , 'input_image')   

    #step 7c
    wmhswf.connect(denoise_t2             , 'denoised',           n4biasfieldcorrect_t2  , 'input_image')   
    
    
    #step 8
    wmhswf.connect(n4biasfieldcorrect_fl  , 'output_image',       flair2mni   ,  'in_file')
    
    wmhswf.connect(flair2mni              , 'out_matrix_file',     
    
                   
    # outputs
    wmhswf.connect(inputnode               , 'subject_ids',       datasink, 'container')
    wmhswf.connect(inputnode               , 'outputdir',         datasink, 'base_directory')

    wmhswf.connect(eddy_cuda               , 'out_movement_rms',  datasink, 'Eddy.@out_movement_rms')
    wmhswf.connect(eddy_cuda               , 'out_outlier_report',datasink, 'Eddy.@out_outlier_report')
    wmhswf.connect(eddy_cuda               , 'out_parameter',     datasink, 'Eddy.@out_parameter')
    wmhswf.connect(eddy_cuda               , 'out_restricted_movement_rms',     datasink, 'Eddy.@out_restricted_movement_rms')
    wmhswf.connect(eddy_cuda               , 'out_rotated_bvecs',     datasink, 'Eddy.@out_rotated_bvecs')
    
    
    return wmhswf
    

