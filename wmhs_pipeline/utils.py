from nipype.interfaces.fsl.base import CommandLine, CommandLineInputSpec
from nipype.interfaces.base import (traits, TraitedSpec, File, isdefined,InputMultiPath)
from nipype.interfaces.ants.base import ANTSCommand, ANTSCommandInputSpec

#from nipype.utils.filemanip import copyfile
import os


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
        
