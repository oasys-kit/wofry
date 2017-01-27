
import numpy

from wofry.propagator.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Generic1DPropagator, PropagationParameters, PropagationElements

class Fraunhofer1DPropagationParameters(PropagationParameters):
    def __init__(self,
                 wavefront = GenericWavefront1D(),
                 propagation_elements = PropagationElements(),
                 shift_half_pixel=True):
        PropagationParameters.__init__(wavefront, propagation_elements)

        self._shift_half_pixel=shift_half_pixel

    def get_shift_half_pixel(self):
        return self._shift_half_pixel

class Fraunhofer1D(Generic1DPropagator):

    def get_handler_name(self):
        return "FRAUNHOFER_1D"

    """
    1D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :return: a new 1D wavefront object with propagated wavefront
    """

    # TODO: check resulting amplitude normalization
    def do_specific_progation(self, wavefront, propagation_distance, parameters):
        if not isinstance(parameters, Fraunhofer1DPropagationParameters):
            raise ValueError("parameters are not " + Fraunhofer1DPropagationParameters.__name__)

        if propagation_distance == 0.0: return wavefront

        fft = numpy.fft.fft(wavefront.get_complex_amplitude())
        fft2 = numpy.fft.fftshift(fft)

        # frequency for axis 1

        freq_nyquist = 0.5/wavefront.delta()
        freq_n = numpy.linspace(-1.0,1.0,wavefront.size())
        freq_x = freq_n * freq_nyquist
        freq_x *= wavefront.get_wavelength()

        if parameters.get_shift_half_pixel():
            freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])

        #if propagation_distance == 0:
        #    wf = GenericWavefront1D.initialize_wavefront_from_arrays(freq_x,fft2,wavelength=wavefront.get_wavelength())
        #    return wf
        #else:
        #    wf = GenericWavefront1D.initialize_wavefront_from_arrays(freq_x*propagation_distance,fft2,wavelength=wavefront.get_wavelength())
        #    return wf

        return GenericWavefront1D.initialize_wavefront_from_arrays(freq_x*propagation_distance,fft2,wavelength=wavefront.get_wavelength())
