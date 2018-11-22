

import numpy

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D

class FresnelZoom1D(Propagator1D):

    HANDLER_NAME = "FRESNEL_ZOOM_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)


    def do_specific_progation(self, wavefront1, propagation_distance, parameters, element_index=None):

        magnification_x = self.get_additional_parameter("magnification_x",1.0,parameters,element_index=element_index)

        return self.propagate_wavefront(wavefront1,propagation_distance,magnification_x=magnification_x)

    @classmethod
    def propagate_wavefront(cls,wavefront1,propagation_distance,magnification_x=1.0):

        wavefront = wavefront1.duplicate()
        shape = wavefront.size()
        delta = wavefront.delta()
        wavenumber = wavefront.get_wavenumber()
        wavelength = wavefront.get_wavelength()

        fft_scale = numpy.fft.fftfreq(shape)/delta

        x = wavefront.get_abscissas()

        x_rescaling = wavefront.get_abscissas() * magnification_x

        r1sq = x ** 2 * (1 - magnification_x)
        r2sq = x_rescaling ** 2 * ((magnification_x - 1) / magnification_x)
        fsq = (fft_scale ** 2 / magnification_x)

        Q1 = wavenumber / 2 / propagation_distance * r1sq
        Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance * fsq)
        Q3 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * r2sq)

        wavefront.add_phase_shift(Q1)

        fft = numpy.fft.fft(wavefront.get_complex_amplitude())
        ifft = numpy.fft.ifft(fft * Q2) * Q3 / numpy.sqrt(magnification_x)

        wf_propagated = GenericWavefront1D.initialize_wavefront_from_arrays(x_rescaling,
                                                                            ifft,
                                                                            wavelength=wavelength)

        return wf_propagated

