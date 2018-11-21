

import numpy

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D

class FresnelZoomScaling1D(Propagator1D):

    HANDLER_NAME = "FRESNEL_ZOOM_SCALING_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=None)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=None)


    def do_specific_progation(self, wavefront1, propagation_distance1, parameters, element_index=None):


        try:
            m = parameters.get_additional_parameter("magnification_x")
        except:
            m = 1.0

        radius = parameters.get_additional_parameter("radius")

        wavefront = wavefront1.duplicate()
        shape = wavefront.size()
        delta = wavefront.delta()
        wavenumber = wavefront.get_wavenumber()
        wavelength = wavefront.get_wavelength()

        # radius = -50.0
        magnification = (propagation_distance1 + radius) / radius
        propagation_distance = propagation_distance1 / magnification

        #
        # make plane wave by adding a spherical wavefront with radius -R
        #
        new_phase = 1.0 * wavefront.get_wavenumber() * (wavefront.get_abscissas()**2) / (-2 * radius)
        wavefront.add_phase_shifts(new_phase)


        #
        # main 2 FFT block
        #
        fft_scale = numpy.fft.fftfreq(shape)/delta

        x = wavefront.get_abscissas()

        x_rescaling = wavefront.get_abscissas() * m

        r1sq = x ** 2 * (1 - m)
        r2sq = x_rescaling ** 2 * ((m - 1) / m)
        fsq = (fft_scale ** 2 / m)

        Q1 = wavenumber / 2 / propagation_distance * r1sq
        Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance * fsq)
        Q3 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * r2sq)

        wavefront.add_phase_shift(Q1)

        fft = numpy.fft.fft(wavefront.get_complex_amplitude())
        ifft = numpy.fft.ifft(fft * Q2) * Q3 / numpy.sqrt(m)
        #
        #
        #

        #
        # calculate new phase term and scale coordinates
        #
        k = wavefront.get_wavenumber()

        PHASE1 = numpy.ones(wavefront.get_abscissas().size) * (k * propagation_distance * (1 - 1/magnification))
        PHASE2 = k /2 / propagation_distance * (magnification - 1)/magnification * (wavefront.get_abscissas()*magnification)**2
        PHASE = PHASE2 + PHASE1


        wf_propagated = GenericWavefront1D.initialize_wavefront_from_arrays(
                        x_rescaling*magnification,
                        ifft/numpy.sqrt(magnification)*numpy.exp(1j*PHASE),
                        wavelength=wavelength)

        #
        # wf_propagated = GenericWavefront1D.initialize_wavefront_from_arrays(x_rescaling,
        #                                                                     ifft,
        #                                                                     wavelength=wavelength)

        return wf_propagated

