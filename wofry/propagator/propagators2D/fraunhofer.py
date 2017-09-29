#
#   propagate_2D_fraunhofer: Far field Fraunhofer propagator. TODO: Check phases, not to be used for downstream propagation
#
#    The fraunhoffer method cannot be used in a compound system (more than one element) and in connection with lenses
#

import numpy

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import Propagator2D

class Fraunhofer2D(Propagator2D):

    HANDLER_NAME = "FRAUNHOFER_2D"

    def get_handler_name(self):
        return self.HANDLER_NAME

    """
    2D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters):
        return self.do_specific_progation(wavefront, propagation_distance, parameters)

    def do_specific_progation(self, wavefront, propagation_distance, parameters):
        if not parameters.has_additional_parameter("shift_half_pixel"):
            raise ValueError("Missing Parameter shift_half_pixel")

        shift_half_pixel = parameters.get_additional_parameter("shift_half_pixel")

        wavelength = wavefront.get_wavelength()

        #
        # check validity
        #
        x =  wavefront.get_coordinate_x()
        y =  wavefront.get_coordinate_y()
        half_max_aperture = 0.5 * numpy.array((x[-1]-x[0], y[-1]-y[0])).max()
        far_field_distance = half_max_aperture**2/wavelength
        if propagation_distance < far_field_distance:
            print("WARNING: Fraunhoffer diffraction valid for distances > > half_max_aperture^2/lambda = %f m (propagating at %4.1f)"%
                        (far_field_distance,propagation_distance))
        #
        #compute Fourier transform
        #
        F1 = numpy.fft.fft2(wavefront.get_complex_amplitude())  # Take the fourier transform of the image.
        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = numpy.fft.fftshift( F1 )

        # frequency for axis 1
        shape = wavefront.size()
        delta = wavefront.delta()

        pixelsize = delta[0] # p_x[1] - p_x[0]
        npixels = shape[0]
        freq_nyquist = 0.5/pixelsize
        freq_n = numpy.linspace(-1.0, 1.0, npixels)
        freq_x = freq_n * freq_nyquist
        freq_x *= wavelength

        # frequency for axis 2
        pixelsize = delta[1]
        npixels = shape[1]
        freq_nyquist = 0.5/pixelsize
        freq_n = numpy.linspace(-1.0, 1.0, npixels)
        freq_y = freq_n * freq_nyquist
        freq_y *= wavelength


        if shift_half_pixel:
            freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])
            freq_y = freq_y - 0.5 * numpy.abs(freq_y[1] - freq_y[0])

        if propagation_distance != 1.0:
            freq_x *= propagation_distance
            freq_y *= propagation_distance

        wf_propagated = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=freq_x,
                                                                            y_array=freq_y,
                                                                            z_array=F2,
                                                                            wavelength=wavelength)

        return  wf_propagated
