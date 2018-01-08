from nipype.interfaces.freesurfer.preprocess import MRIConvert
from .configoptions import WMHS_MASKS_DIR

   
def convert_mgz(in_file):
    """
    convert mgz to nii.gz 
    """
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


def create_master_file(matrix_file):
    """
    create master file required by bianca
    """
    
    import csv
        
    return os.path.abspath(matrix_file)


