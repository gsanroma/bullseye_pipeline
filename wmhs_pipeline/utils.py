# from nipype.interfaces.fsl.base import CommandLine, CommandLineInputSpec
# from nipype.interfaces.base import (traits, TraitedSpec, File, isdefined,InputMultiPath)
# from nipype.interfaces.ants.base import ANTSCommand, ANTSCommandInputSpec
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    CommandLineInputSpec,
    CommandLine,
    File
)

#from nipype.utils.filemanip import copyfile
import os

###### set environment vars

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

#######

def get_first_file(file_list):
    return file_list[0]

def compute_mask(in_file):
    """
    create a brainmask from the input image
    """
    import nibabel as nib
    import numpy as np
    import os
    from scipy.ndimage.morphology import binary_dilation
    
    dilat_rad = [5, 5, 5]
    struct = np.ones(dilat_rad, dtype=np.bool)
    img_nib = nib.load(in_file)
    img = img_nib.get_data()
    
    mask = np.zeros(img.shape, dtype=np.bool)
    mask[img > 0] = True
    mask = binary_dilation(mask, struct)
    mask_nib = nib.Nifti1Image(mask, img_nib.affine, img_nib.header)
    nib.save(mask_nib, 'brainmask.nii.gz')
    return os.path.abspath('brainmask.nii.gz')

    
def convert_mgz(in_file):
    """
    convert mgz to nii.gz 
    """
    from nipype.interfaces.freesurfer.preprocess import MRIConvert
    import os.path as op

    fname, ext = op.splitext(op.basename(in_file))
    if ext == ".gz":
        return in_file
        #fname, ext2 = op.splitext(fname)
        #ext = ext2 + ext
    else:
        mc = MRIConvert()
        mc.inputs.in_file = in_file
        mc.inputs.out_type = 'niigz'
        mc.inputs.out_file = fname + '.nii.gz'
        res = mc.run()
        outputs = res.outputs.get()
        return outputs['out_file']


def maskout_image(mask_file, image_file):
    import nibabel as nib
    import numpy as np
    import os
    
    # read mask
    mask_nib = nib.load(mask_file)
    mask = mask_nib.get_data().astype(np.bool)
    
    # maskout images
    img_nib = nib.load(image_file)
    img = img_nib.get_data()
    img[~mask] = img.min()

    aux = nib.Nifti1Image(img, img_nib.affine, img_nib.header)
    fname, ext = os.path.splitext(os.path.basename(image_file))
    maskoutname = fname.replace('.nii','') + '_maskout.nii.gz'
    maskoutfile = os.path.join(os.getcwd(),maskoutname)
    nib.save(aux, maskoutfile)

    return maskoutfile

def normalize_image(mask_file, image_file):
    import nibabel as nib
    import numpy as np
    import os
    from scipy.stats.mstats import zscore
    
    # read mask
    mask_nib = nib.load(mask_file)
    mask = mask_nib.get_data().astype(np.bool)
    
    # normalize
    img_nib = nib.load(image_file)
    img = img_nib.get_data()
    
    img2 = np.zeros(img.shape, dtype=np.float32)
    img2[mask] = zscore(img[mask].astype(np.float32))
    aux = nib.Nifti1Image(img2, img_nib.affine, img_nib.header.set_data_dtype(np.float32))
    fname,ext = os.path.splitext(os.path.basename(image_file))
    normname = fname.replace('.nii','') + '_norm.nii.gz'
    normoutfile = os.path.join(os.getcwd(),normname)
    nib.save(aux, normoutfile)
    
    return normoutfile
    
    
def inclusion_mask(aseg_file):
    
    import numpy as np
    import nibabel as nib
    import os
    
    in_nib = nib.load(aseg_file)
    
    in0 = in_nib.get_data()
    
    out0 = np.zeros(in0.shape, dtype=np.bool)
    
    for label in [2, 41, 77, 11, 50]:
        out0[in0 == label] = True
        
    out_nib = nib.Nifti1Image(out0, in_nib.affine, in_nib.header)
    nib.save(out_nib, 'inclmask.nii.gz')
    
    return os.path.abspath('inclmask.nii.gz')

    
    
def create_deepmedic_channel_file(channel_name, channel_file_path):
    """
    create channel configuration file required by deepMedicRun
    """
    
    import os
    
    if channel_name == 'NamesOfPredictions':
        channel_file_path='pred'
        
    channel_config_file = 'testChannel_'+channel_name+'.cfg'
    with open(channel_config_file, 'w') as fid:
        fid.write(channel_file_path+'\n')
    
    return os.path.abspath(channel_config_file)

def create_deepmedic_config_file(flair_channel_file, roi_channel_file,pred_channel_file):
    
    import os

    # from wmhs_pipeline.configoptions import DM_MODEL_DIR
    from configoptions import DM_MODEL_DIR

    test_config_file = 'testConfig.cfg'
    #this workaround to set the output path to the deepmedic run folder
    folder_for_output= os.path.join(os.path.abspath(os.path.join(os.getcwd(),os.pardir)), 'deepmedicrun')
    model_file_path = os.path.join(DM_MODEL_DIR, 'best_mod_1ch.ckpt')
    channels = '["'+ flair_channel_file + '"]'
    
    
    with open(test_config_file, 'w') as fid:
        fid.write('sessionName = "deepmed_v1"'+'\n\n')
        fid.write('folderForOutput = "'+ folder_for_output +'"\n\n')
        fid.write('cnnModelFilePath = "'+ model_file_path +'"\n\n')
        fid.write('channels = '+ channels + '\n\n')
        fid.write('namesForPredictionsPerCase = "'+ pred_channel_file +'"\n\n')
        fid.write('roiMasks = "' + roi_channel_file +'"\n\n')
        fid.write('batchsize = 10\n\n')
        fid.write('saveSegmentation = True\n')
        fid.write('saveProbMapsForEachClass = [False, False]\n')
        fid.write('saveIndividualFms = False\n')
        fid.write('saveAllFmsIn4DimImage = False\n')
        # fid.write('minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathway = []\n')
        # fid.write('minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathway = [[],[],[],[],[],[],[],[]]\n')
        # fid.write('minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathway = [[],[0,150],[]]\n')
        fid.write('padInputImagesBool = True\n')
        
    return os.path.abspath(test_config_file)
        
    

class DeepMedicInputSpec(CommandLineInputSpec):
    """
    interface for DeepMedic
    """
    model_config_file = File(exists=True, desc='deepMedic model config file.', argstr='-model %s', position=0, mandatory=True) 
    test_config_file  = File(exists=True, desc='deepMedic test config file.',   argstr='-test %s',  position=1, mandatory=True)
    # load_saved_model  = File(exists=True, desc='deepMedic saved model file.',   argstr='-load %s',  position=2, mandatory=True)
    device            = traits.String(desc='device name', argstr='-dev %s', position=3, mandatory=True)
    use_gpu = traits.Bool(desc='set  the flag to use gpu')

class DeepMedicOutputSpec(TraitedSpec):
    
    out_segmented_file = File(exists=True, desc='Output files from deepMedicRun' )


class DeepMedic(CommandLine):
    
    _cmd='deepMedicRun'
    input_spec = DeepMedicInputSpec
    output_spec = DeepMedicOutputSpec
    
    def __init__(self, **inputs):
        return super(DeepMedic, self).__init__(**inputs)

    def _format_arg(self, name, spec, value):
        if(name=='load_saved_model'):
             #remove the .index extension here, the file will be existing file
             return spec.argstr %( self.inputs.load_saved_model.replace('.index', ''))


        return super(DeepMedic,
                     self)._format_arg(name, spec, value)

    
    def _run_interface(self, runtime):
        
        runtime = super(DeepMedic, self)._run_interface(runtime)        
        
        if runtime.stderr:
            self.raise_exception(runtime)
            
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_segmented_file'] = os.path.abspath(os.getcwd()+'/predictions/deepmed_v1/predictions/pred_Segm.nii.gz')
        
        return outputs

        
class Annot2LabelInputSpec(CommandLineInputSpec):
    subject = traits.String(desc='subject id', argstr='--subject %s', position=0, mandatory=True)
    hemi = traits.Enum("rh", "lh", desc="hemisphere [rh | lh]", position=1, argstr="--hemi %s", mandatory=True)
    lobes = traits.Enum("lobes", desc='lobes type', argstr='--lobesStrict %s', position=2)
    in_annot = traits.File(desc='input annotation file', mandatory=True)

class Annot2LabelOutputSpec(TraitedSpec):
    out_annot_file = File(desc = "lobes annotation file", exists = True)

class Annot2Label(CommandLine):
    input_spec = Annot2LabelInputSpec
    output_spec = Annot2LabelOutputSpec
    _cmd = os.path.join(os.environ['FREESURFER_HOME'], 'bin', 'mri_annotation2label')
    # cmd = 'mri_annotation2label'

    def _list_outputs(self):
            outputs = self.output_spec().get()
            outputs['out_annot_file'] = os.path.join(os.path.dirname(self.inputs.in_annot), self.inputs.hemi + ".lobes.annot")
            return outputs

    def _format_arg(self, name, spec, value):
        if(name=='subject'):
             # take only the last part of the subject path
             return spec.argstr % ( os.path.basename(os.path.normpath(self.inputs.subject)))

        return super(Annot2Label, self)._format_arg(name, spec, value)


class Aparc2AsegInputSpec(CommandLineInputSpec):
    subject = traits.String(desc='subject id', argstr='--subject %s', position=0, mandatory=True)
    annot = traits.String(desc='name of annot file', argstr='--annot %s', position=1, mandatory=True)
    labelwm = traits.Bool(desc='percolate white matter', argstr='--labelwm', position=2)
    dmax = traits.Int(desc='depth to percolate', argstr='--wmparc-dmax %d', position=3)
    rip = traits.Bool(desc='rip unknown label', argstr='--rip-unknown', position=4)
    hypo = traits.Bool(desc='hypointensities as wm', argstr='--hypo-as-wm', position=5)
    out_file = traits.File(desc='output aseg file', argstr='-o %s', position=6)

class Aparc2AsegOutputSpec(TraitedSpec):
    out_aseg_file = File(desc = "lobes aseg file", exists = True)

class Aparc2Aseg(CommandLine):
    input_spec = Annot2LabelInputSpec
    output_spec = Annot2LabelOutputSpec
    _cmd = os.path.join(os.environ['FREESURFER_HOME'], 'bin', 'mri_annotation2label')
    # cmd = 'mri_annotation2label'

    def _list_outputs(self):
            outputs = self.output_spec().get()
            outputs['out_annot_file'] = os.path.join(os.path.dirname(self.inputs.in_annot), self.inputs.hemi + ".lobes.annot")
            return outputs

    def _format_arg(self, name, spec, value):
        if(name=='subject'):
             # take only the last part of the subject path
             return spec.argstr % ( os.path.basename(os.path.normpath(self.inputs.subject)))

        return super(Annot2Label, self)._format_arg(name, spec, value)


