import numpy

from srxraylib.util.data_structures import ScaledArray
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D

class FresnelConvolution1D(Propagator1D):

    HANDLER_NAME = "FRESNEL_CONVOLUTION_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters=None, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters=parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters=None, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters=parameters, element_index=element_index)

    """
    1D Fresnel propagator using direct convolution
    :param wavefront:
    :param propagation_distance:
    :return:
    """
    def do_specific_progation(self, wavefront, propagation_distance, parameters=None, element_index=None):
        # instead of numpy.convolve, this can be used:
        # from scipy.signal import fftconvolve
        return self.propagate_wavefront(wavefront,propagation_distance)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance):

        kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * wavefront.get_abscissas()**2 / 2 / propagation_distance)
        kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
        kernel /=  1j * wavefront.get_wavelength() * propagation_distance
        tmp = numpy.convolve(wavefront.get_complex_amplitude(),kernel,mode='same')

        wavefront_out =  GenericWavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(tmp,
                                    wavefront.offset(), wavefront.delta()))

        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        wavefront_out.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    wavefront_out.get_intensity().sum()))

        return wavefront_out