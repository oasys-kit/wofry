#   propagate_2D_fresnel               \
#   propagate_2D_fresnel_convolution   | Near field Fresnel propagators via convolution in Fourier space. Three methods
#   propagate_2D_fresnel_srw          /
#
#          three methods available: 'fft': fft -> multiply by kernel in freq -> ifft
#                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
#                                   'srw': use the SRW package
#
#
#
# *********************************** IMPORTANT *******************************************
#                RECOMMENDATIONS:
#
#     >>> Prefer propagate_2D_fresnel <<<
#       Prefer EVEN number of bins.
#       Set shift_half_pixel=1 (now the default)
#    Under these circumstances, the results agree very well with SRW
#
#
#

# TODO: REMOVE THIS!!!!
try:
    from srwlib import *
    SRWLIB_AVAILABLE = True
except:
    try:
        from wpg.srwlib import *
        SRWLIB_AVAILABLE = True
    except:
        SRWLIB_AVAILABLE = False
        print("SRW is not available")

import numpy
import scipy.constants as codata
angstroms_to_eV = codata.h*codata.c/codata.e*1e10

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class Fresnel2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME


    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters):
        if not parameters.has_additional_parameter("shift_half_pixel"):
            shift_half_pixel = True
        else:
            shift_half_pixel = parameters.get_additional_parameter("shift_half_pixel")

        shift_half_pixel = parameters.get_additional_parameter("shift_half_pixel")

        wavelength = wavefront.get_wavelength()

        #
        # convolving with the Fresnel kernel via FFT multiplication
        #
        fft = numpy.fft.fft2(wavefront.get_complex_amplitude())

        # frequency for axis 1
        shape = wavefront.size()
        delta = wavefront.delta()

        pixelsize = delta[0] # p_x[1] - p_x[0]
        npixels = shape[0]
        freq_nyquist = 0.5/pixelsize
        freq_n = numpy.linspace(-1.0,1.0,npixels)
        freq_x = freq_n * freq_nyquist

        # frequency for axis 2
        pixelsize = delta[1]
        npixels = shape[1]
        freq_nyquist = 0.5/pixelsize
        freq_n = numpy.linspace(-1.0,1.0,npixels)
        freq_y = freq_n * freq_nyquist

        if shift_half_pixel:
            freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])
            freq_y = freq_y - 0.5 * numpy.abs(freq_y[1] - freq_y[0])

        freq_xy = numpy.array(numpy.meshgrid(freq_y,freq_x))
        fft *= numpy.exp((-1.0j) * numpy.pi * wavelength * propagation_distance *
                      numpy.fft.fftshift(freq_xy[0]*freq_xy[0] + freq_xy[1]*freq_xy[1]) )

        ifft = numpy.fft.ifft2(fft)

        wf_propagated = GenericWavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x(),
                                                                            wavefront.get_coordinate_y(),
                                                                            ifft,
                                                                            wavelength=wavelength)
        return wf_propagated

class FresnelConvolution2D(Propagator2D):

    HANDLER_NAME = "FRESNEL_CONVOLUTION_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation(self, wavefront, propagation_distance, parameters):
        if not parameters.has_additional_parameter("shift_half_pixel"):
            shift_half_pixel = True
        else:
            shift_half_pixel = parameters.get_additional_parameter("shift_half_pixel")

        from scipy.signal import fftconvolve

        wavelength = wavefront.get_wavelength()

        X = wavefront.get_mesh_x()
        Y = wavefront.get_mesh_y()

        if shift_half_pixel:
            x = wavefront.get_coordinate_x()
            y = wavefront.get_coordinate_y()
            X += 0.5 * numpy.abs( x[0] - x[1] )
            Y += 0.5 * numpy.abs( y[0] - y[1] )

        kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() *
                           (X**2 + Y**2) / 2 / propagation_distance)
        kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
        kernel /=  1j * wavefront.get_wavelength() * propagation_distance
        tmp = fftconvolve(wavefront.get_complex_amplitude(),kernel,mode='same')

        wf_propagated = GenericWavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x(),
                                                                     wavefront.get_coordinate_y(),
                                                                     tmp,
                                                                     wavelength=wavelength)
        return wf_propagated

from wofry.propagator.wavefront2D.wavefront_srw import WOSRWWavefront

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