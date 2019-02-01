import unittest
import numpy


#
# Note that the tests for the Fraunhofer phase do not make any assert, because a good matching has not yet been found.
#


from syned.beamline.shape import Rectangle, Ellipse
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement

from wofry.propagator.propagator import PropagationElements

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagator import PropagationManager, PropagationParameters

from wofry.beamline.optical_elements.absorbers.slit import WOSlit1D, WOGaussianSlit1D


do_plot = False

if do_plot:
    from srxraylib.plot.gol import plot

from wofry.propagator.propagators1D.fresnel_zoom import FresnelZoom1D

from wofry.propagator.propagators1D.fraunhofer import Fraunhofer1D
from wofry.propagator.propagators1D.fresnel import Fresnel1D
from wofry.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
from wofry.propagator.propagators1D.integral import Integral1D
from wofry.propagator.propagators1D import initialize_default_propagator_1D

from scipy.special import fresnel



propagator = PropagationManager.Instance()
initialize_default_propagator_1D()


def fraunhofer_analytical_rectangle(
    fresnel_number=None,propagation_distance=1.140,
    aperture_half=1e-3,wavelength=639e-9,
    detector_array=None,npoints=1000,
    ):


    if fresnel_number is None:
        fresnel_number = aperture_half**2 / (wavelength * propagation_distance)

    print("Fresnel number: ",fresnel_number)


    if detector_array is None:
        if fresnel_number > 1.0:
            window_aperture_ratio = 2.0
        else:
            window_aperture_ratio = 1.0 / fresnel_number
        x = numpy.linspace(-window_aperture_ratio*aperture_half,window_aperture_ratio*aperture_half,npoints)
    else:
        x = detector_array.copy()

    argument_sinc = 2.0 * aperture_half * numpy.pi / wavelength / propagation_distance * x # TODO: check the 2??
    alpha = 2.0 * aperture_half / (wavelength*propagation_distance)**(1.0/2.0) * \
            numpy.exp(1j*numpy.pi/wavelength/propagation_distance * x**2) * \
            numpy.sin(argument_sinc) / argument_sinc

    # TODO note that the global phase (Goldman 4-59) is missing

    return x,alpha

def fresnel_analytical_rectangle(
    fresnel_number=None,propagation_distance=1.140,
    aperture_half=1e-3,wavelength=639e-9,
    detector_array=None,npoints=1000,
    ):


    if fresnel_number is None:
        fresnel_number = aperture_half**2 / (wavelength * propagation_distance)

    print("Fresnel number: ",fresnel_number)


    if detector_array is None:
        if fresnel_number > 1.0:
            window_aperture_ratio = 2.0
        else:
            window_aperture_ratio = 1.0 / fresnel_number
        x = numpy.linspace(-window_aperture_ratio*aperture_half,window_aperture_ratio*aperture_half,npoints)
    else:
        x = detector_array.copy()

    s_plus  = numpy.sqrt(2.0 * fresnel_number) * ( 1.0 + x / aperture_half)
    s_minus = numpy.sqrt(2.0 * fresnel_number) * ( 1.0 - x / aperture_half)

    fs_plus,fc_plus = fresnel(s_plus)
    fs_minus,fc_minus = fresnel(s_minus)

    Ux = (fc_minus + fc_plus) + 1j*(fs_minus + fs_plus)
    Ux *= 1.0/numpy.sqrt(2.0)

    # TODO note that the global phase (Goldman 4-59) is missing

    return x,Ux  # note that wavefield is being returned, not intensity!


class propagatorTest(unittest.TestCase):

    #
    # Common interface for all 1D methods :
    #                                   'fraunhofer':
    #                                   'fft': fft -> multiply by kernel in freq -> ifft
    #                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
    #                                   'zoom': fft -> multiply by kernel in freq -> ifft
    # valid apertute_type: square, gaussian

    def propagate_1D(self,do_plot=do_plot,
                                method='fft',
                                wavelength=1.24e-10,
                                aperture_type='square',
                                aperture_diameter=40e-6,
                                wavefront_length=100e-6,
                                npoints=500,
                                propagation_distance = 30.0,
                                normalization=True,  # TODO put False
                                show=1,
                                amplitude=(0.0+1.0j)):



        print("\n#                                                            ")
        print("# 1D (%s) propagation from a %s aperture  "%(method,aperture_type))
        print("#                                                            ")

        wf = GenericWavefront1D.initialize_wavefront_from_range(x_min=-wavefront_length/2, x_max=wavefront_length/2,
                                                                number_of_points=npoints,wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude(amplitude) # an arbitraty value

        deltax = wf.get_abscissas()[1] - wf.get_abscissas()[0]

        propagation_elements = PropagationElements()

        slit = None

        if aperture_type == 'square':
            slit = WOSlit1D(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, 0, 0))
        elif aperture_type == 'gaussian':
            slit = WOGaussianSlit1D(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, 0, 0))
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")

        propagation_elements.add_beamline_element(BeamlineElement(optical_element=slit,
                                                                  coordinates=ElementCoordinates(p=0, q=propagation_distance)  ))


        propagator = PropagationManager.Instance()
        propagation_parameters = PropagationParameters(wavefront=wf,
                                                       propagation_elements=propagation_elements)

        print("Using propagator method:  ",method)
        fresnel_analytical = True
        if method == 'fft':
            wf1 = propagator.do_propagation(propagation_parameters, Fresnel1D.HANDLER_NAME)
        elif method == 'convolution':
            wf1 = propagator.do_propagation(propagation_parameters, FresnelConvolution1D.HANDLER_NAME)
        elif method == 'integral':
            propagation_parameters.set_additional_parameters("magnification_x", 1.5)
            propagation_parameters.set_additional_parameters("magnification_N", 2.0)
            wf1 = propagator.do_propagation(propagation_parameters, Integral1D.HANDLER_NAME)
        elif method == 'fraunhofer':
            fresnel_analytical = False
            # propagation_parameters.set_additional_parameters("shift_half_pixel", 0)
            wf1 = propagator.do_propagation(propagation_parameters, Fraunhofer1D.HANDLER_NAME)
        elif method == 'zoom':
            propagation_parameters.set_additional_parameters("magnification_x", 1.5)
            wf1 = propagator.do_propagation(propagation_parameters, FresnelZoom1D.HANDLER_NAME)
        else:
            raise Exception("Not implemented method: %s"%method)


        if fresnel_analytical:
            xx, alpha = fresnel_analytical_rectangle(
                fresnel_number = None, propagation_distance = propagation_distance,
                aperture_half = 0.5*aperture_diameter, wavelength = wavelength,
                detector_array = wf1.get_abscissas(), npoints = None)
        else:
            xx, alpha = fraunhofer_analytical_rectangle(
                fresnel_number = None, propagation_distance = propagation_distance,
                aperture_half = 0.5*aperture_diameter, wavelength = wavelength,
                detector_array = wf1.get_abscissas(), npoints = None)

        angle_x = xx / propagation_distance
        intensity_theory = numpy.abs(amplitude*alpha)**2

        intensity_calculated =  wf1.get_intensity()

        if normalization:
            intensity_calculated /= intensity_calculated.max()
            intensity_theory /= intensity_theory.max()

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1.get_abscissas()*1e6/propagation_distance,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["%s "%method,"analytical"],
                 legend_position=(0.95, 0.95),
                 title="1D (%s) diffraction from a %s aperture of %3.1f um at \n wavelength of %3.1f A"%
                       (method,aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",
                 xrange=[-20,20],ylog=True,
                 show=show)

            plot(wf1.get_abscissas()*1e6,wf1.get_phase(unwrap=True),
                 1e6 * xx, numpy.unwrap(numpy.angle(alpha)),
                 legend=["%s " % method, "analytical"],
                 title="1D (%s) diffraction from a %s aperture of %3.1f um at \n wavelength of %3.1f A NOT ASSERTED!!"%
                       (method, aperture_type, aperture_diameter * 1e6, wavelength * 1e10),
                 xtitle="X (urad)", ytitle="Phase",
                 # xrange=[-20, 20],
                 )

        return wf1.get_abscissas()/propagation_distance,intensity_calculated,intensity_theory


    #
    # tests/example cases for 1D propagators
    #


    # @unittest.skip("classing skipping")
    def test_propagate_1D_fraunhofer_bis(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        npoints=1024

        print("\n#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="fraunhofer",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = 1.0, show=1)

        # works for normalized only
        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)

    # @unittest.skip("classing skipping")
    def test_propagate_1D_fraunhofer_phase(self,do_plot=do_plot):

        # aperture_type="square"
        # aperture_diameter = 40e-6
        # wavefront_length = 800e-6
        # wavelength = 1.24e-10
        # npoints=1024
        # propagation_distance=40
        show = 1

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024


        print("\n#                                                            ")
        print("# far field 1D (fraunhofer and zoom) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")

        wf = GenericWavefront1D.initialize_wavefront_from_range(x_min=-wavefront_length/2, x_max=wavefront_length/2,
                                                                number_of_points=npoints,wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((2.0+1.0j)) # an arbitraty value

        propagation_elements = PropagationElements()

        if aperture_type == 'square':
            slit = WOSlit1D(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, 0, 0))
        else:
            raise Exception("Not implemented! ")

        propagation_elements.add_beamline_element(BeamlineElement(optical_element=slit,
                                                                  coordinates=ElementCoordinates(p=0, q=propagation_distance)  ))


        propagator = PropagationManager.Instance()
        propagation_parameters = PropagationParameters(wavefront=wf,
                                                       propagation_elements=propagation_elements)



        wf1_franuhofer = propagator.do_propagation(propagation_parameters, Fraunhofer1D.HANDLER_NAME)

        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        propagation_parameters.set_additional_parameters("magnification_x", 1.5)
        wf1_zoom = propagator.do_propagation(propagation_parameters, FresnelZoom1D.HANDLER_NAME)


        intensity_fraunhofer = wf1_franuhofer.get_intensity() / wf1_franuhofer.get_intensity().max()
        intensity_zoom = wf1_zoom.get_intensity() / wf1_zoom.get_intensity().max()

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1_franuhofer.get_abscissas()*1e6/propagation_distance,intensity_fraunhofer,
                 wf1_zoom.get_abscissas()*1e6/propagation_distance,intensity_zoom,
                 legend=["Fraunhofer","Zoom"],
                 legend_position=(0.95, 0.95),
                 title="1D  INTENSITY diffraction from aperture of %3.1f um at wavelength of %3.1f A"%
                       (aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
                 show=show)
            plot(wf1_franuhofer.get_abscissas()*1e6/propagation_distance,wf1_franuhofer.get_phase(unwrap=1),
                 wf1_zoom.get_abscissas()*1e6/propagation_distance,wf1_zoom.get_phase(unwrap=1),
                 legend=["Fraunhofer","Zoom"],
                 legend_position=(0.95, 0.95),
                 title="1D  diffraction from a %s aperture of %3.1f um at wavelength of %3.1f A"%
                       (aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
                 show=show)

        # TODO assert phase
        #numpy.testing.assert_almost_equal(1e3*intensity_fraunhofer,1e3*intensity_zoom,1)

    # @unittest.skip("classing skipping")
    def test_propagate_1D_fft(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024

        # print("\n#                                                            ")
        # print("# near field 1D (fft) diffraction from a %s aperture  "%aperture_type)
        # print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="fft",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = propagation_distance,
                                show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)

    # @unittest.skip("classing skipping")
    def test_propagate_1D_fft_zoom(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024

        print("\n#                                                            ")
        print("# fft zoom diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="zoom",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = propagation_distance,
                                show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)

    # @unittest.skip("classing skipping")
    def test_propagate_1D_fresnel_convolution(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024

        print("\n#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="convolution",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = propagation_distance,
                                show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)


    # @unittest.skip("classing skipping")
    def test_propagate_1D_integral(self,do_plot=do_plot):

        aperture_type="square"
        aperture_diameter = 40e-6
        wavefront_length = 800e-6
        wavelength = 1.24e-10
        propagation_distance = 30.0
        npoints=1024

        print("\n#                                                            ")
        print("# near field 1D (integral) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")



        angle, intensity_calculated,intensity_theory = self.propagate_1D(do_plot=do_plot,method="integral",
                                wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                wavefront_length=wavefront_length,npoints=npoints,
                                propagation_distance = propagation_distance,
                                show=1)

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)


    # @unittest.skip("classing skipping")
    def test_propagate_1D_compare_amplitudes(self,do_plot=do_plot):

        # aperture_type="square"
        # aperture_diameter = 40e-6
        # wavefront_length = 800e-6
        # wavelength = 1.24e-10
        # propagation_distance = 30.0
        # npoints=1024

        aperture_type="square"
        aperture_diameter = 100e-6
        wavefront_length = 1000e-6
        wavelength = 1.5e-10
        propagation_distance = 55.0
        npoints=2048


        print("\n#                                                            ")
        print("# near field 1D (integral) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")


        #
        # angle, intensity_calculated,intensity_theory_integral = self.propagate_1D(do_plot=do_plot,method="integral",
        #                         wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
        #                         wavefront_length=wavefront_length,npoints=npoints,
        #                         propagation_distance = propagation_distance, show=1)


        # compare with this one, the reference
        # angle_fft, intensity_calculated_fft,intensity_theory = self.propagate_1D(do_plot=False,method="fft",
        #                         wavelength=wavelength,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
        #                         wavefront_length=wavefront_length,npoints=npoints,
        #                         propagation_distance = propagation_distance,
        #                         show=False)


        for method in ["fft","zoom","convolution","integral","fraunhofer"]:

            angle_1, intensity_calculated_1,intensity_theory = self.propagate_1D(
                                    do_plot=do_plot,
                                    method=method,
                                    wavelength=wavelength,aperture_type=aperture_type,
                                    aperture_diameter=aperture_diameter,
                                    wavefront_length=wavefront_length,npoints=npoints,
                                    propagation_distance = propagation_distance,
                                    show=False)


            # x_fft = angle_fft * propagation_distance
            # x_1 = angle_1 * propagation_distance
            #
            # ymax_all = [intensity_calculated_fft.max(),intensity_calculated_1.max()]
            # ymax = numpy.max(ymax_all)
            # print(">>>",ymax_all,numpy.sqrt(intensity_calculated_fft.max()/intensity_calculated_1.max()))
            # if do_plot:
            #     plot(x_fft,intensity_calculated_fft,x_1,intensity_calculated_1,
            #          legend=["fft",method],yrange=[0,ymax],
            #          title="Comparing intensities - Not yet a test!",
            #          ylog=True,
            #          show=True)

            # numpy.testing.assert_almost_equal(intensity_calculated_fft,intensity_calculated_1,1)




