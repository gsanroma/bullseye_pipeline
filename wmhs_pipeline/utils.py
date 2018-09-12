from nipype.interfaces.fsl.base import CommandLine, CommandLineInputSpec
from nipype.interfaces.base import (traits, TraitedSpec, File, isdefined)
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

    
    
def create_master_file_train(flair, t1w,t2w, fl2mni_matrix_file):
    """
    create master file required by bianca
    """
    
    import os
    import glob
    
    from wmhs_pipeline.configoptions import WMHS_MASKS_DIR

    masterfile=""
    
    WMHmaskbin_file_list = glob.glob(WMHS_MASKS_DIR.rstrip()+'/*WMHmaskbin.nii.gz')

    for fname in WMHmaskbin_file_list:
        fbname=os.path.basename(fname)
        sid=fbname.split('_')[0]
        if sid in flair:
                masterfile = flair + ' ' + t1w + ' ' + t2w + ' ' + fname +  ' ' + fl2mni_matrix_file
                break
    
    with open('masterfile.txt', 'w') as fid:
        fid.write(masterfile+'\n')
        
    
    return os.path.abspath('masterfile.txt')

def create_master_file_query(flair, t1w,t2w, fl2mni_matrix_file):
    """
    create master file required by bianca
    """
    
    import os
    
    masterfile = flair + ' ' + t1w + ' ' + t2w + ' '  + 'NOLABEL' +  ' ' + fl2mni_matrix_file
    
    with open('masterfile.txt', 'w') as fid:
        fid.write(masterfile+'\n')
        
    
    return os.path.abspath('masterfile.txt')


def threshold_bianca(biancasegfile):
    import numpy as np
    import nibabel as nib
    import os
    
    prlab_nib = nib.load(biancasegfile)
    prlab = prlab_nib.get_data()
    
    # output mask
    mask = np.zeros(prlab.shape, dtype=np.uint8)
    mask[prlab > 0.95] = 1
    
    # save
    mask_nib = nib.Nifti1Image(mask, prlab_nib.affine, prlab_nib.header)
    mask_nib.set_data_dtype(np.uint8)
    
    fname,ext = os.path.splitext(os.path.basename(biancasegfile))
    thresh_name = fname.replace('.nii','') + '_thr95.nii.gz'
    thresholded_file = os.path.join(os.getcwd(),thresh_name)
    nib.save(mask_nib, thresholded_file)
    return thresholded_file
    
    
class BiancaInputSpec(CommandLineInputSpec):
    """
    interface for bianca
    """
    
    master_file = File(exists=True, desc='bianca master file.', argstr='--singlefile=%s', position=0, mandatory=True)
    querysubjectnum = traits.Int(desc='row no of qury subject', argstr='--querysubjectnum=%d', position=1, mandatory=True)
    brainmaskfeaturenum = traits.Int(desc='col no of file used as brainmask', argstr='--brainmaskfeaturenum=%d', position=2, mandatory=True)
    featuresubset = traits.String(desc='comma separated set of col nos used as features', argstr='--featuresubset=%s', position=3, mandatory=True)
    matfeaturenum = traits.Int(desc='feature no of xfm matrix', argstr='--matfeaturenum=%d', position=4, mandatory=True)
    spatialweight = traits.Float(desc='spatial weight', argstr='--spatialweight=%.2f', position=5, mandatory=True)
    selectpts = traits.Enum('noborder', 'any', 'surround', argstr='--selectpts=%s', desc='select points option', position=6, usedefault=True)
    trainingpts = traits.Int(desc='training points', argstr='--trainingpts=%d', position=7, mandatory=True)
    nonlespts = traits.Int(desc='no of lesion points', argstr='--nonlespts=%d', position=8, mandatory=True)
    out_filename = traits.String(desc='output file name', argstr='-o %s', default='bianca_wmhseg.nii.gz', position=9, mandatory=True)
    loadclassifierdata = traits.String(desc='classifier data name', argstr='--loadclassifierdata=%s', position=10, mandatory=True)


class BiancaOutputSpec(TraitedSpec):
    
    out_file = File(exists=True, desc='bianca output WMH segmented file')


class Bianca(CommandLine):
    
    _cmd='bianca'
    input_spec = BiancaInputSpec
    output_spec = BiancaOutputSpec
    
    def __init__(self, **inputs):
        return super(Bianca, self).__init__(**inputs)
    
    def _run_interface(self, runtime):
        
        runtime = super(Bianca, self)._run_interface(runtime)        
        
        if runtime.stderr:
            self.raise_exception(runtime)
            
        return runtime
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(os.path.basename(self.inputs.out_filename))
        
        return outputs



class DenoiseImageWithMaskInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(2, 3, 4, argstr='-d %d', usedefault=False,
                            desc='This option forces the image to be treated '
                                 'as a specified-dimensional image. If not '
                                 'specified, the program tries to infer the '
                                 'dimensionality from the input image.')
    input_image = File(exists=True, argstr="-i %s", mandatory=True,
                       desc='A scalar image is expected as input for noise correction.')
    mask_image = File(exists=True, argstr="-x %s", mandatory=True,
                       desc='A mask to perform denoise within.')
    
    noise_model = traits.Enum('Gaussian', 'Rician', argstr='-n %s', usedefault=True,
                              desc=('Employ a Rician or Gaussian noise model.'))
    shrink_factor = traits.Int(default_value=1, usedefault=True, argstr='-s %s',
                               desc=('Running noise correction on large images can '
                                     'be time consuming. To lessen computation time, '
                                     'the input image can be resampled. The shrink '
                                     'factor, specified as a single integer, describes '
                                     'this resampling. Shrink factor = 1 is the default.'))
    output_image = File(argstr="-o %s", name_source=['input_image'], hash_files=False,
                        keep_extension=True, name_template='%s_noise_corrected',
                        desc='The output consists of the noise corrected '
                             'version of the input image.')
    save_noise = traits.Bool(False, mandatory=True, usedefault=True,
                             desc=('True if the estimated noise should be saved '
                                   'to file.'), xor=['noise_image'])
    noise_image = File(name_source=['input_image'], hash_files=False,
                       keep_extension=True, name_template='%s_noise',
                       desc='Filename for the estimated noise.')
    verbose = traits.Bool(False, argstr="-v", desc=('Verbose output.'))


class DenoiseImageWithMaskOutputSpec(TraitedSpec):
    output_image = File(exists=True)
    noise_image = File()


class DenoiseImageWithMask(ANTSCommand):
    """
    Examples
    --------
    >>> import copy
    >>> from nipype.interfaces.ants import DenoiseImage
    >>> denoise = DenoiseImage()
    >>> denoise.inputs.dimension = 3
    >>> denoise.inputs.input_image = 'im1.nii'
    >>> denoise.cmdline # doctest: +ALLOW_UNICODE
    'DenoiseImage -d 3 -i im1.nii -n Gaussian -o im1_noise_corrected.nii -s 1'

    >>> denoise_2 = copy.deepcopy(denoise)
    >>> denoise_2.inputs.output_image = 'output_corrected_image.nii.gz'
    >>> denoise_2.inputs.noise_model = 'Rician'
    >>> denoise_2.inputs.shrink_factor = 2
    >>> denoise_2.cmdline # doctest: +ALLOW_UNICODE
    'DenoiseImage -d 3 -i im1.nii -n Rician -o output_corrected_image.nii.gz -s 2'

    >>> denoise_3 = DenoiseImage()
    >>> denoise_3.inputs.input_image = 'im1.nii'
    >>> denoise_3.inputs.save_noise = True
    >>> denoise_3.cmdline # doctest: +ALLOW_UNICODE
    'DenoiseImage -i im1.nii -n Gaussian -o [ im1_noise_corrected.nii, im1_noise.nii ] -s 1'
    """
    input_spec = DenoiseImageWithMaskInputSpec
    output_spec = DenoiseImageWithMaskOutputSpec
    _cmd = 'DenoiseImage'

    def _format_arg(self, name, trait_spec, value):
        if ((name == 'output_image') and
                (self.inputs.save_noise or isdefined(self.inputs.noise_image))):
            newval = '[ %s, %s ]' % (self._filename_from_source('output_image'),
                                     self._filename_from_source('noise_image'))
            return trait_spec.argstr % newval

        return super(DenoiseImageWithMask,
                     self)._format_arg(name, trait_spec, value)



class ConvertTransformFileInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(2, 3, 4, argstr='%d', usedefault=False,
                            desc='This option forces the image to be treated '
                                 'as a specified-dimensional image. If not '
                                 'specified, the program tries to infer the '
                                 'dimensionality from the input image.', position=0)
    in_file = File(exists=True, argstr='%s', mandatory=True,
                       desc='input transform .mat file.', position=1)
    out_filename = File(argstr='%s', position=2,desc='name of output file')
    


class ConvertTransformFileOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output transform .txt file")
    


class ConvertTransformFile(ANTSCommand):

    input_spec = ConvertTransformFileInputSpec
    output_spec = ConvertTransformFileOutputSpec
    _cmd = 'ConvertTransformFile'

    def _format_arg(self, name, trait_spec, value):
        if(name=='in_file'):
             fname,ext=os.path.splitext(os.path.basename(self.inputs.in_file))
             self.inputs.out_filename = os.path.join(os.getcwd(),fname+'.txt')
             
        
        return super(ConvertTransformFile,
                     self)._format_arg(name, trait_spec, value)
    
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_filename)
        return outputs
        

class C3DAffineToolInputSpec(CommandLineInputSpec):
    reference_file = File(exists=True, argstr="-ref %s", position=1)
    source_file = File(exists=True, argstr='-src %s', position=2)
    transform_file = File(exists=True, argstr='-itk %s', position=3)
    ras2fsl = traits.Bool(argstr='-ras2fsl', position=4)
    out_filename = File(argstr='-o %s', position=5)


class C3DAffineToolOutputSpec(TraitedSpec):
    fsl_transform = File(exists=True)


class C3DAffineTool(CommandLine):
    input_spec = C3DAffineToolInputSpec
    output_spec = C3DAffineToolOutputSpec
    
    _cmd = 'c3d_affine_tool'
    
    def _list_outputs(self):
        outputs=self.output_spec().get()
        outputs['fsl_transform'] = os.path.abspath(self.inputs.out_filename)
        return outputs
    