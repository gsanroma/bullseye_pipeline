from nipype.interfaces.fsl.base import CommandLine, CommandLineInputSpec
from nipype.interfaces.base import (traits, TraitedSpec, File)

from nipype.utils.filemanip import copyfile
import os

   
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
                masterfile=flair + ' ' + t1w + ' ' + t2w + ' ' + fname +  ' ' + fl2mni_matrix_file
                break
    
    with open('masterfile.txt', 'w') as fid:
        fid.write(masterfile+'\n')
        
    
    return os.path.abspath('masterfile.txt')

def create_master_file_query(flair, t1w,t2w, fl2mni_matrix_file):
    """
    create master file required by bianca
    """
    
    import os
    
    masterfile=flair + ' ' + t1w + ' ' + t2w + ' '  + 'NOLABEL' +  ' ' + fl2mni_matrix_file
    
    with open('masterfile.txt', 'w') as fid:
        fid.write(masterfile+'\n')
        
    
    return os.path.abspath('masterfile.txt')



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
