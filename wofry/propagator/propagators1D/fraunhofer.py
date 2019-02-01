
import numpy

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import Propagator1D

class Fraunhofer1D(Propagator1D):

    HANDLER_NAME = "FRAUNHOFER_1D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation(wavefront, propagation_distance, parameters, element_index=element_index)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        return self.do_specific_progation( wavefront, propagation_distance, parameters, element_index=element_index)

    """
    1D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :return: a new 1D wavefront object with propagated wavefront
    """

    # TODO: check resulting amplitude normalization
    def do_specific_progation(self, wavefront, propagation_distance, parameters, element_index=None):

        shift_half_pixel = self.get_additional_parameter("shift_half_pixel",False,parameters,element_index=element_index)

        return self.propagate_wavefront(wavefront,propagation_distance,shift_half_pixel=shift_half_pixel)

    @classmethod
    def propagate_wavefront(cls,wavefront,propagation_distance,shift_half_pixel=False):

        shape = wavefront.size()
        delta = wavefront.delta()
        wavelength = wavefront.get_wavelength()
        wavenumber = wavefront.get_wavenumber()
        fft_scale = numpy.fft.fftfreq(shape, d=delta)
        fft_scale = numpy.fft.fftshift(fft_scale)
        x2 = fft_scale * propagation_distance * wavelength

        if shift_half_pixel:
            x2 = x2 - 0.5 * numpy.abs(x2[1] - x2[0])


        p1 = numpy.exp(1.0j * wavenumber * propagation_distance)
        p2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * x2**2)
        p3 = 1.0j*wavelength*propagation_distance

        fft = numpy.fft.fft(wavefront.get_complex_amplitude())
        fft = fft * p1 * p2 / p3
        fft2 = numpy.fft.fftshift(fft)

        wavefront_out =  GenericWavefront1D.initialize_wavefront_from_arrays(x2, fft2, wavelength=wavefront.get_wavelength())


        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        wavefront_out.rescale_amplitude( numpy.sqrt(wavefront.get_intensity().sum() /
                                                    wavefront_out.get_intensity().sum()))

        return wavefront_out

    #todo not yet working...
    @classmethod
    def propagate_wavefront_new(cls,wavefront,
                            propagation_distance,
                            fraunhofer_kernel=True, # set to False for far field propagator
                            shift_half_pixel=None, # not more used, kept for version compatibility
                            ):
        #
        # check validity
        #
        x = wavefront.get_abscissas()
        deltax = wavefront.delta()

        #
        # compute Fourier transform
        #

        # frequency for axis 1
        npixels = wavefront.size()
        pixelsize = wavefront.delta()
        wavenumber = wavefront.get_wavenumber()
        wavelength = wavefront.get_wavelength()

        freq_nyquist = 0.5 / pixelsize
        if numpy.mod(npixels, 2) == 0:
            freq_n = numpy.arange(-npixels // 2, npixels // 2, 1) / (npixels // 2)
        else:
            freq_n = numpy.arange(-(npixels - 1) // 2, (npixels + 1) // 2, 1) / ((npixels - 1) // 2)

        freq_x = freq_n * freq_nyquist


        x2 = freq_x * propagation_distance * wavelength

        P1 = numpy.exp(1.0j * wavenumber * propagation_distance)

        # fsq = freq_x ** 2
        # P2 = numpy.exp(-1.0j * wavenumber / 2 / propagation_distance * fsq)

        P2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * x**2)
        P3 = 1.0j * wavelength * propagation_distance

        if fraunhofer_kernel:
            exponential = 1.0 + 0j
        else:
            exponential = numpy.exp(1j * wavenumber / 2 / propagation_distance * x ** 2)

        F1 = numpy.fft.fft(exponential * wavefront.get_complex_amplitude())  # Take the fourier transform of the image.
        #  Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F1 *= P1
        F1 *= P2
        F1 /= numpy.sqrt(P3)  # this is 1D -> no sqrt for 2D
        F2 = numpy.fft.fftshift(F1)
        F2 *= deltax  # why??

        wavefront_out = GenericWavefront1D.initialize_wavefront_from_arrays(x_array=x2,
                                                                            y_array=F2,
                                                                            wavelength=wavelength)

        # added srio@esrf.eu 2018-03-23 to conserve energy - TODO: review method!
        # wavefront_out.rescale_amplitude(numpy.sqrt(wavefront.get_intensity().sum() /
        #                                            wavefront_out.get_intensity().sum()))

        return wavefront_out