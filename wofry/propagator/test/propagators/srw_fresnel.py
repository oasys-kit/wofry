# TODO: REMOVE THIS!!!!
try:
    from oasys_srw.srwlib import *
    SRWLIB_AVAILABLE = True
except:
    try:
        from srwlib import *
        SRWLIB_AVAILABLE = True
    except:
        SRWLIB_AVAILABLE = False
        print("SRW is not available")

import scipy.constants as codata
angstroms_to_eV = codata.h*codata.c/codata.e*1e10

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D
from wofry.propagator.examples import WOSRWWavefront

class FresnelSRW(Propagator2D):

    HANDLER_NAME = "FRESNEL_SRW"

    def get_handler_name(self):
        return self.HANDLER_NAME

    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance:
    :param srw_autosetting:set to 1 for automatic SRW redimensionate wavefront
    :return:
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters):
        if not SRWLIB_AVAILABLE: raise ImportError("SRW is not available")

        if not parameters.has_additional_parameter("srw_autosetting"):
            srw_autosetting = 0
        else:
            srw_autosetting = parameters.get_additional_parameter("srw_autosetting")

        is_generic_wavefront = isinstance(wavefront, GenericWavefront2D)

        if is_generic_wavefront:
            wavefront = WOSRWWavefront.fromGenericWavefront(wavefront)
        else:
            if not isinstance(wavefront, WOSRWWavefront): raise ValueError("wavefront cannot be managed by this propagator")

        #
        # propagation
        #
        optDrift = SRWLOptD(propagation_distance) #Drift space


        #Wavefront Propagation Parameters:
        #[0]: Auto-Resize (1) or not (0) Before propagation
        #[1]: Auto-Resize (1) or not (0) After propagation
        #[2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
        #[3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
        #[4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
        #[5]: Horizontal Range modification factor at Resizing (1. means no modification)
        #[6]: Horizontal Resolution modification factor at Resizing
        #[7]: Vertical Range modification factor at Resizing
        #[8]: Vertical Resolution modification factor at Resizing
        #[9]: Type of wavefront Shift before Resizing (not yet implemented)
        #[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
        #[11]: New Vertical wavefront Center position after Shift (not yet implemented)

        if srw_autosetting:
            #                 0  1  2   3  4  5   6   7   8   9 10 11
            propagParDrift = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
        else:
            #                 0  1  2   3  4  5   6   7   8   9 10 11
            propagParDrift = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

        optBL = SRWLOptC([optDrift], [propagParDrift]) #"Beamline" - Container of Optical Elements (together with the corresponding wavefront propagation instructions)

        print('   Simulating Electric Field Wavefront Propagation by SRW ... ', end='\n')
        srwl.PropagElecField(wavefront, optBL)

        if is_generic_wavefront:
            return wavefront.toGenericWavefront()
        else:
            return wavefront
