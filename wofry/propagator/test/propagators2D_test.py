import unittest
import numpy


#
# Note that the tests for the Fraunhofer phase do not make any assert, because a good matching has not yet been found.
#


from syned.beamline.shape import Rectangle, Ellipse
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement

from wofry.propagator.propagator import PropagationElements

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import PropagationManager, PropagationParameters

from wofry.beamline.optical_elements.absorbers.slit import WOSlit, WOGaussianSlit
from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen
from wofry.beamline.optical_elements.ideal_elements.lens import WOIdealLens

do_plot = False

if do_plot:
    from srxraylib.plot.gol import plot,plot_image,plot_table


from wofry.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofry.propagator.propagators2D.fresnel import Fresnel2D
from wofry.propagator.propagators2D.fresnel_convolution import FresnelConvolution2D
from wofry.propagator.propagators2D.integral import Integral2D
from wofry.propagator.propagators2D import initialize_default_propagator_2D
from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
#
# some common tools
#
# from propagators1D_test import get_theoretical_diffraction_pattern


propagator = PropagationManager.Instance()
initialize_default_propagator_2D()


#TODO replace by the pne used in 1D
def get_theoretical_diffraction_pattern(angle_x,
                                        aperture_type='square',aperture_diameter=40e-6,
                                        wavelength=1.24e-10,normalization=False):

    # a quick and dirty trick to avoid 0/0
    indices_bad = numpy.where(angle_x == 0)
    angle_x[indices_bad] += 1e-15

    # get the theoretical value
    if aperture_type == 'circle': #circular, also display analytical values
        from scipy.special import jv
        x = (2*numpy.pi/wavelength) * (aperture_diameter/2) * angle_x
        amplitude_theory = 2*jv(1,x)/x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'square':

        x = (2*numpy.pi / wavelength) * (aperture_diameter / 2) * angle_x
        amplitude_theory = 2 * numpy.sin(x)  / x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'gaussian':
        sigma = aperture_diameter/2.35
        sigma_ft = 1.0 / sigma * wavelength / (2.0 * numpy.pi)
        # Factor 2.0 is because we wwant intensity (amplitude**2)
        intensity_theory = numpy.exp( -2.0*(angle_x**2/sigma_ft**2/2) )
    else:
        raise Exception("Undefined aperture type (accepted: circle, square, gaussian)")

    if normalization:
        intensity_theory /= intensity_theory.max()

    return intensity_theory


def line_image(image,horizontal_or_vertical='H'):
    if horizontal_or_vertical == "H":
        tmp = image[:,image.shape[1]/2]
    else:
        tmp = image[image.shape[0]/2,:]
    return tmp

def line_fwhm(line):
    #
    #CALCULATE fwhm in number of abscissas bins (supposed on a regular grid)
    #
    tt = numpy.where(line>=max(line)*0.5)
    if line[tt].size > 1:
        # binSize = x[1]-x[0]
        FWHM = (tt[0][-1]-tt[0][0])
        return FWHM
    else:
        return -1

class propagator2DTest(unittest.TestCase):
    #
    # TOOLS
    #


    #
    # Common interface for all methods using fresnel, via convolution in FT space
    #          three methods available: 'fft': fft -> multiply by kernel in freq -> ifft
    #                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)


    def propagate_2D_fresnel(self,do_plot=do_plot,method='fft',
                                wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance = 30.0,show=1,
                                use_additional_parameters_input_way=1, # 0=default (common), 1=new (specific))
                             ):


        method_label = "fresnel (%s)"%method
        print("\n#                                                             ")
        print("# 2D near field fresnel (%s) diffraction from a %s aperture  "%(method_label,aperture_type))
        print("#                                                             ")


        # wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
        #                                                         x_step=pixelsize_x,
        #                                                         y_start=-pixelsize_y*npixels_y/2,
        #                                                         y_step=pixelsize_y,
        #                                                         wavelength=wavelength,
        #                                                         number_of_points=(npixels_x,npixels_y))

        wf = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((1.0+0j))


        propagation_elements = PropagationElements()

        slit = None

        if aperture_type == 'square':
            slit = WOSlit(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, -aperture_diameter/2, aperture_diameter/2))
        elif aperture_type == 'gaussian':
            slit = WOGaussianSlit(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, -aperture_diameter/2, aperture_diameter/2))
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")


        if use_additional_parameters_input_way == 0:
            propagation_elements.add_beamline_element(BeamlineElement(optical_element=slit,
                                                                      coordinates=ElementCoordinates(p=0, q=propagation_distance)))


            propagator = PropagationManager.Instance()
            propagation_parameters = PropagationParameters(wavefront=wf,
                                                           propagation_elements=propagation_elements)

            if method == 'fft':
                propagation_parameters.set_additional_parameters("shift_half_pixel", True)
                wf1 = propagator.do_propagation(propagation_parameters, Fresnel2D.HANDLER_NAME)
            elif method == 'convolution':
                propagation_parameters.set_additional_parameters("shift_half_pixel", True)
                wf1 = propagator.do_propagation(propagation_parameters, FresnelConvolution2D.HANDLER_NAME)
            elif method == 'integral':
                propagation_parameters.set_additional_parameters("shuffle_interval", 0)
                propagation_parameters.set_additional_parameters("calculate_grid_only", 1)
                wf1 = propagator.do_propagation(propagation_parameters, Integral2D.HANDLER_NAME)
            elif method == 'zoom':
                propagation_parameters.set_additional_parameters("shift_half_pixel", True)
                propagation_parameters.set_additional_parameters("magnification_x", 2.0)
                propagation_parameters.set_additional_parameters("magnification_y", 0.5)
                wf1 = propagator.do_propagation(propagation_parameters, FresnelZoomXY2D.HANDLER_NAME)
            else:
                raise Exception("Not implemented method: %s"%method)
        else:

            if method == 'fft':
                additional_parameters = {"shift_half_pixel":True}
            elif method == 'convolution':
                additional_parameters = {"shift_half_pixel":True}
            elif method == 'integral':
                additional_parameters = {"shuffle_interval":0,
                                         "calculate_grid_only":1}
            elif method == 'zoom':
                additional_parameters = {"shift_half_pixel":True,
                                         "magnification_x":2.0,
                                         "magnification_y":0.5}
            else:
                raise Exception("Not implemented method: %s"%method)


            propagation_elements.add_beamline_element(
                        BeamlineElement(optical_element=slit,coordinates=ElementCoordinates(p=0, q=propagation_distance)),
                        element_parameters=additional_parameters)


            propagator = PropagationManager.Instance()
            propagation_parameters = PropagationParameters(wavefront=wf,
                                                           propagation_elements=propagation_elements)

            if method == 'fft':
                wf1 = propagator.do_propagation(propagation_parameters, Fresnel2D.HANDLER_NAME)
            elif method == 'convolution':
                wf1 = propagator.do_propagation(propagation_parameters, FresnelConvolution2D.HANDLER_NAME)
            elif method == 'integral':
                wf1 = propagator.do_propagation(propagation_parameters, Integral2D.HANDLER_NAME)
            elif method == 'zoom':
                wf1 = propagator.do_propagation(propagation_parameters, FresnelZoomXY2D.HANDLER_NAME)
            else:
                raise Exception("Not implemented method: %s"%method)


        if do_plot:
            from srxraylib.plot.gol import plot_image
            plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                       title="aperture intensity (%s), Diameter=%5.1f um"%
                             (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                       show=0)

            plot_image(wf1.get_intensity(),
                       1e6*wf1.get_coordinate_x()/propagation_distance,
                       1e6*wf1.get_coordinate_y()/propagation_distance,
                       title="Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                             (aperture_type,method_label,1e6*aperture_diameter),
                       xtitle="X [urad]",ytitle="Y [urad]",
                       show=0)

        # get the theoretical value
        angle_x = wf1.get_coordinate_x() / propagation_distance

        intensity_theory = get_theoretical_diffraction_pattern(angle_x,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                            wavelength=wavelength,normalization=True)

        intensity_calculated =  wf1.get_intensity()[:,int(wf1.size()[1]/2)]
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1.get_coordinate_x()*1e6/propagation_distance,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["%s H profile"%method_label,"Theoretical (far field)"],
                 legend_position=(0.95, 0.95),
                 title="%s diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (method_label,aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
                 show=show)

        return wf1.get_coordinate_x()/propagation_distance,intensity_calculated,angle_x,intensity_theory

    #
    #
    #
    def propagation_with_lens(self,do_plot=do_plot,method='fft',
                                wavelength=1.24e-10,
                                pixelsize_x=1e-6,npixels_x=2000,pixelsize_y=1e-6,npixels_y=2000,
                                propagation_distance=30.0,defocus_factor=1.0,propagation_steps=1,show=1):


        method_label = "fresnel (%s)"%method
        print("\n#                                                             ")
        print("# near field fresnel (%s) diffraction and focusing  "%(method_label))
        print("#                                                             ")

        #                               \ |  /
        #   *                           | | |                      *
        #                               / | \
        #   <-------    d  ---------------><---------   d   ------->
        #   d is propagation_distance

        # wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
        #                                                         x_step=pixelsize_x,
        #                                                         y_start=-pixelsize_y*npixels_y/2,
        #                                                         y_step=pixelsize_y,
        #                                                         wavelength=wavelength,
        #                                                         number_of_points=(npixels_x,npixels_y))

        wf = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)
        propagation_elements = PropagationElements()

        spherical_or_plane_and_lens = 1
        if spherical_or_plane_and_lens == 0:
            # set spherical wave at the lens entrance (radius=distance)
            wf.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)

            propagation_elements.add_beamline_element(BeamlineElement(optical_element=WOScreen(),
                                                                      coordinates=ElementCoordinates(p=0, q=propagation_distance)))

        else:
            # apply lens that will focus at propagation_distance downstream the lens.
            # Note that the vertical is a bit defocused
            wf.set_plane_wave_from_complex_amplitude(1.0+0j)

            focal_length = propagation_distance # / 2

            propagation_elements.add_beamline_element(BeamlineElement(optical_element=
                WOIdealLens("IdealLens",focal_x=focal_length, focal_y=focal_length),
                coordinates=ElementCoordinates(p=0, q=propagation_distance)))

        print("Incident intensity: ", wf.get_intensity().sum())

        propagator = PropagationManager.Instance()
        propagation_parameters = PropagationParameters(wavefront=wf,
                                                       propagation_elements=propagation_elements)

        if method == 'fft':
            propagation_parameters.set_additional_parameters("shift_half_pixel", True)
            wf1 = propagator.do_propagation(propagation_parameters, Fresnel2D.HANDLER_NAME)
        elif method == 'convolution':
            propagation_parameters.set_additional_parameters("shift_half_pixel", True)
            wf1 = propagator.do_propagation(propagation_parameters, FresnelConvolution2D.HANDLER_NAME)
        elif method == 'fraunhofer':
            propagation_parameters.set_additional_parameters("shift_half_pixel", True)
            wf1 = propagator.do_propagation(propagation_parameters, Fraunhofer2D.HANDLER_NAME)
        elif method == 'zoom':
            propagation_parameters.set_additional_parameters("shift_half_pixel", True)
            propagation_parameters.set_additional_parameters("magnification_x", 1.5)
            propagation_parameters.set_additional_parameters("magnification_y", 2.5)
            wf1 = propagator.do_propagation(propagation_parameters, FresnelZoomXY2D.HANDLER_NAME)
        else:
            raise Exception("Not implemented method: %s"%method)

        horizontal_profile = wf1.get_intensity()[:, int(wf.size()[1]/2)]
        horizontal_profile /= horizontal_profile.max()
        print("FWHM of the horizontal profile: %g um"%(1e6*line_fwhm(horizontal_profile)*wf1.delta()[0]))
        vertical_profile = wf1.get_intensity()[int(wf1.size()[0]/2),:]
        vertical_profile /= vertical_profile.max()
        print("FWHM of the vertical profile: %g um"%(1e6*line_fwhm(vertical_profile)*wf1.delta()[1]))

        if do_plot:
            from srxraylib.plot.gol import plot,plot_image
            plot_image(wf1.get_intensity(),wf1.get_coordinate_x(),wf1.get_coordinate_y(),title='intensity (%s)'%method,show=0)
            plot_image(wf1.get_phase(),wf1.get_coordinate_x(),wf1.get_coordinate_y(),title='phase (%s)'%method,show=0)

            plot(wf1.get_coordinate_x(),horizontal_profile,
                 wf1.get_coordinate_y(),vertical_profile,
                 legend=['Horizontal profile','Vertical profile'],title="%s"%method,show=show)

        print("Output intensity: ",wf1.get_intensity().sum())
        return wf1.get_coordinate_x(),horizontal_profile

    #
    # TESTS
    #
    # @unittest.skip("classing skipping")
    def test_propagate_2D_fraunhofer(self,do_plot=do_plot,aperture_type='square',aperture_diameter=40e-6,
                    pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,wavelength=1.24e-10):
        """

        :param do_plot: 0=No plot, 1=Do plot
        :param aperture_type: 'circle' 'square' 'gaussian' (Gaussian sigma = aperture_diameter/2.35)
        :param aperture_diameter:
        :param pixelsize_x:
        :param pixelsize_y:
        :param npixels_x:
        :param npixels_y:
        :param wavelength:
        :return:
        """

        print("\n#                                                            ")
        print("# far field 2D (fraunhofer) diffraction from a square aperture  ")
        print("#                                                            ")

        method = "fraunhofer"

        print("Fraunhoffer diffraction valid for distances > > a^2/lambda = %f m"%((aperture_diameter/2)**2/wavelength))

        wf = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((1.0+0j))


        propagation_elements = PropagationElements()

        slit = None

        if aperture_type == 'square':
            slit = WOSlit(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, -aperture_diameter/2, aperture_diameter/2))
        elif aperture_type == 'gaussian':
            slit = WOGaussianSlit(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, -aperture_diameter/2, aperture_diameter/2))
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")

        propagation_elements.add_beamline_element(BeamlineElement(optical_element=slit,
                                                                  coordinates=ElementCoordinates(p=0, q=1.0)))


        propagator = PropagationManager.Instance()
        propagation_parameters = PropagationParameters(wavefront=wf,
                                                       propagation_elements=propagation_elements)
        propagation_parameters.set_additional_parameters("shift_half_pixel", True)

        wf1 = propagator.do_propagation(propagation_parameters, Fraunhofer2D.HANDLER_NAME)

        if aperture_type == 'circle':
            wf.clip_circle(aperture_diameter/2)
        elif aperture_type == 'square':
            wf.clip_square(-aperture_diameter/2, aperture_diameter/2,-aperture_diameter/2, aperture_diameter/2)
        elif aperture_type == 'gaussian':
            X = wf.get_mesh_x()
            Y = wf.get_mesh_y()
            window = numpy.exp(- (X*X + Y*Y)/2/(aperture_diameter/2.35)**2)
            wf.rescale_amplitudes(window)
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")

        if do_plot:
            plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                       title="aperture intensity (%s), Diameter=%5.1f um"%
                             (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                       show=0)

            plot_image(wf1.get_intensity(),1e6*wf1.get_coordinate_x(),1e6*wf1.get_coordinate_y(),
                       title="2D Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                             (aperture_type,method,1e6*aperture_diameter),
                       xtitle="X [urad]",ytitle="Y [urad]",
                       show=0)

        angle_x = wf1.get_coordinate_x() # + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        intensity_theory = get_theoretical_diffraction_pattern(angle_x,
                                            aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                            wavelength=wavelength,normalization=True)


        intensity_calculated =  wf1.get_intensity()[:,int(wf1.size()[1]/2)]
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            plot(wf1.get_coordinate_x()*1e6,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["Calculated (FT) H profile","Theoretical"],legend_position=(0.95, 0.95),
                 title="2D Fraunhofer Diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-80,80])

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)

    # @unittest.skip("classing skipping")
    def test_propagate_2D_fraunhofer_phase(self,do_plot=do_plot,aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=30.0,wavelength=1.24e-10):


        print("\n#                                                            ")
        print("# far field 2D (fraunhofer) diffraction from a square aperture  ")
        print("#                                                            ")

        method = "fraunhofer"

        print("Fraunhoffer diffraction valid for distances > > a^2/lambda = %f m"%((aperture_diameter/2)**2/wavelength))

        wf = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((1.0+0j))


        propagation_elements = PropagationElements()

        slit = None

        if aperture_type == 'square':
            slit = WOSlit(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, -aperture_diameter/2, aperture_diameter/2))
        elif aperture_type == 'gaussian':
            slit = WOGaussianSlit(boundary_shape=Rectangle(-aperture_diameter/2, aperture_diameter/2, -aperture_diameter/2, aperture_diameter/2))
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")

        propagation_elements.add_beamline_element(BeamlineElement(optical_element=slit,
                                                                  coordinates=ElementCoordinates(p=0, q=propagation_distance)))


        propagator = PropagationManager.Instance()
        propagation_parameters = PropagationParameters(wavefront=wf,
                                                       propagation_elements=propagation_elements)


        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        wf1_fraunhofer = propagator.do_propagation(propagation_parameters, Fraunhofer2D.HANDLER_NAME)

        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        propagation_parameters.set_additional_parameters("magnification_x", 1.5)
        propagation_parameters.set_additional_parameters("magnification_y", 2.5)
        wf1_zoom = propagator.do_propagation(propagation_parameters, FresnelZoomXY2D.HANDLER_NAME)


        if aperture_type == 'circle':
            wf.clip_circle(aperture_diameter/2)
        elif aperture_type == 'square':
            wf.clip_square(-aperture_diameter/2, aperture_diameter/2,-aperture_diameter/2, aperture_diameter/2)
        elif aperture_type == 'gaussian':
            X = wf.get_mesh_x()
            Y = wf.get_mesh_y()
            window = numpy.exp(- (X*X + Y*Y)/2/(aperture_diameter/2.35)**2)
            wf.rescale_amplitudes(window)
        else:
            raise Exception("Not implemented! (accepted: circle, square, gaussian)")

        if do_plot:
            plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                       title="aperture intensity (%s), Diameter=%5.1f um"%
                             (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                       show=0)

            plot_image(wf1_fraunhofer.get_intensity(),1e6*wf1_fraunhofer.get_coordinate_x(),1e6*wf1_fraunhofer.get_coordinate_y(),
                       title="2D Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                             (aperture_type,"Fraunhofer",1e6*aperture_diameter),
                       xtitle="X [urad]",ytitle="Y [urad]",
                       show=0)

            plot_image(wf1_zoom.get_intensity(),1e6*wf1_zoom.get_coordinate_x(),1e6*wf1_zoom.get_coordinate_y(),
                       title="2D Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                             (aperture_type,"Zoom",1e6*aperture_diameter),
                       xtitle="X [urad]",ytitle="Y [urad]",
                       show=0)

        intensity_calculated_fraunhofer =  wf1_fraunhofer.get_intensity()[:,int(wf1_fraunhofer.size()[1]/2)]
        intensity_calculated_fraunhofer /= intensity_calculated_fraunhofer.max()

        intensity_calculated_zoom =  wf1_zoom.get_intensity()[:,int(wf1_fraunhofer.size()[1]/2)]
        intensity_calculated_zoom /= intensity_calculated_zoom.max()

        phase_calculated_fraunhofer =  wf1_fraunhofer.get_phase()[:,int(wf1_fraunhofer.size()[1]/2)]
        phase_calculated_fraunhofer = numpy.unwrap(phase_calculated_fraunhofer)

        phase_calculated_zoom =  wf1_zoom.get_phase()[:,int(wf1_zoom.size()[1]/2)]
        phase_calculated_zoom = numpy.unwrap(phase_calculated_zoom)

        if do_plot:
            plot(wf1_fraunhofer.get_coordinate_x()*1e6/propagation_distance, intensity_calculated_fraunhofer,
                 wf1_zoom.get_coordinate_x()*1e6      /propagation_distance, intensity_calculated_zoom,
                 legend=["Fraunhofer H profile","Zoom H profile"],legend_position=(0.95, 0.95),
                 title="2D Diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-80,80],show=0)

            plot(wf1_fraunhofer.get_coordinate_x()*1e6/propagation_distance, phase_calculated_fraunhofer,
                 wf1_zoom.get_coordinate_x()*1e6      /propagation_distance, phase_calculated_zoom,
                 legend=["Fraunhofer H profile","Zoom H profile"],legend_position=(0.95, 0.95),
                 title="2D Diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Phase",xrange=[-80,80])


        numpy.testing.assert_almost_equal(1e-3*intensity_calculated_fraunhofer,1e-3*intensity_calculated_zoom,1)
        # TODO: assert for phase


    # @unittest.skip("classing skipping")
    def test_propagate_2D_fresnel_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='fft',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=30.0,wavelength=1.24e-10,use_additional_parameters_input_way=0)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    # @unittest.skip("classing skipping")
    def test_propagate_2D_fresnel_convolution_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='convolution',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=30.0,wavelength=1.24e-10,use_additional_parameters_input_way=0)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    # @unittest.skip("classing skipping")
    def test_propagate_2D_fresnel_integral_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='integral',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6*2,pixelsize_y=1e-6*4,npixels_x=int(1024/2),npixels_y=int(1024/4),
                                propagation_distance=30.0,wavelength=1.24e-10,use_additional_parameters_input_way=0)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    # @unittest.skip("classing skipping")
    def test_propagate_2D_zoom_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='zoom',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=30.0,wavelength=1.24e-10,use_additional_parameters_input_way=0)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=do_plot,method='zoom',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance=30.0,wavelength=1.24e-10,use_additional_parameters_input_way=1)

        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    # @unittest.skip("classing skipping")
    def test_lens(self):

        lens_diameter = 0.002
        npixels_x = int(2048)
        pixelsize_x = lens_diameter / npixels_x
        print("pixelsize: ",pixelsize_x)

        pixelsize_y = pixelsize_x
        npixels_y = npixels_x

        wavelength = 1.24e-10
        propagation_distance = 30.0
        defocus_factor = 1.0 # 1.0 is at focus
        propagation_steps = 1

        x_fft, y_fft = self.propagation_with_lens(do_plot=0,method='fft',
                                propagation_steps=propagation_steps,
                                wavelength=wavelength,
                                pixelsize_x=pixelsize_x,npixels_x=npixels_x,pixelsize_y=pixelsize_y,npixels_y=npixels_y,
                                propagation_distance = propagation_distance, defocus_factor=defocus_factor)


        x_convolution, y_convolution = self.propagation_with_lens(do_plot=0,method='convolution',
                                propagation_steps=propagation_steps,
                                wavelength=wavelength,
                                pixelsize_x=pixelsize_x,npixels_x=npixels_x,pixelsize_y=pixelsize_y,npixels_y=npixels_y,
                                propagation_distance = propagation_distance, defocus_factor=defocus_factor)

        x_zoom, y_zoom = self.propagation_with_lens(do_plot=0,method='zoom',
                                propagation_steps=propagation_steps,
                                wavelength=wavelength,
                                pixelsize_x=pixelsize_x,npixels_x=npixels_x,pixelsize_y=pixelsize_y,npixels_y=npixels_y,
                                propagation_distance = propagation_distance, defocus_factor=defocus_factor)

        if do_plot:
            x = x_fft
            y = numpy.vstack((y_fft,y_convolution,y_zoom))

            plot_table(1e6*x,y,legend=["fft","convolution","zoom"],ytitle="Intensity",xtitle="x coordinate [um]",
                       title="Comparison 1:1 focusing")

        numpy.testing.assert_almost_equal(y_fft,y_convolution,1)

