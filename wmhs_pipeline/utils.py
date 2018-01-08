from nipype.interfaces.freesurfer.preprocess import MRIConvert
from nipype.interfaces.base import (traits, TraitedSpec, File)


   
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


def create_master_file(in_bvals):
    """
    create acqparams file using input bvals <= 50
    """
    
    import numpy as np
    import os
    from dsi_pipeline.dtiutil import get_b0_indices
    from dsi_pipeline.configoptions import ECHO_SPACING_MSEC, ECHO_TRAIN_LENGTH,PA_NUM
    
    b0_indices = get_b0_indices(in_bvals)
    
    #predefined constants
    echo_train_duration_sec = ECHO_SPACING_MSEC * ( ECHO_TRAIN_LENGTH - 1 ) * 0.001

    AP_string = "0 -1 0 %.3f" % (echo_train_duration_sec)
    PA_string = "0 1 0 %.3f" % (echo_train_duration_sec)
    
    acqparams_file="acqparams.txt"
    
    acqparams = np.repeat([AP_string], len(b0_indices))
    acqparams = np.append(acqparams, np.repeat([PA_string], PA_NUM))
    #nipye will complain if fmt="%..." is used, so convert to str("%...")
    np.savetxt(acqparams_file,acqparams, delimiter=" ",fmt=str("%s"))
        
    return os.path.abspath(acqparams_file)


