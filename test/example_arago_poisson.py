from srxraylib.plot.gol import plot,plot_image


# source
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
# beamline
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement
from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen as Screen
# propagator
from wofry.propagator.propagator import PropagationManager, PropagationParameters
from wofry.propagator.propagator import PropagationElements
from wofry.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofry.propagator.propagators2D.fresnel import Fresnel2D, FresnelConvolution2D
from wofry.propagator.propagators2D import initialize_default_propagator_2D

propagator = PropagationManager.Instance()
initialize_default_propagator_2D()

# try:
#     from wofry.propagator.test.propagators.srw_fresnel import FresnelSRW
#     propagator.add_propagator(FresnelSRW())
# except:
#     print("FresnelSRW cannot be added")


if __name__ == "__main__":

    wavelength=0.15e-9
    aperture_diameter=50e-6
    pixelsize_x=1e-7
    pixelsize_y=1e-7
    npixels_x=2024
    npixels_y=2024
    propagation_distance = 1.0
    show=1

    method_label = "fresnel (fft)"
    print("\n#                                                             ")
    print("# 2D near field fresnel (%s) diffraction from a a circular stop  "%(method_label))
    print("#                                                             ")


    wf = GenericWavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                     y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                     number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

    wf.set_plane_wave_from_complex_amplitude((1.0+0j))


    wf.clip_circle(aperture_diameter/2,negative=True)

    plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
               title="intensity at screen/aperture plane, Diameter=%5.1f um"%
                     (1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
               show=0)

    #
    # define image plane
    #
    propagation_elements = PropagationElements()
    #
    propagation_elements.add_beamline_element(BeamlineElement(optical_element=Screen(),
                                                              coordinates=ElementCoordinates(p=0, q=propagation_distance)))
    propagation_parameters = PropagationParameters(wavefront=wf,
                                                   propagation_elements=propagation_elements)
    #
    method = 'fft'
    #
    if method == 'fft':
        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        wf1 = propagator.do_propagation(propagation_parameters, Fresnel2D.HANDLER_NAME)
    elif method == 'srw':
        # wf1 = propagator.do_propagation(propagation_parameters, FresnelSRW.HANDLER_NAME)
        raise Exception("To be implemented using wofrysrw")
    elif method == 'convolution':
        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        wf1 = propagator.do_propagation(propagation_parameters, FresnelConvolution2D.HANDLER_NAME)
    elif method == 'fraunhofer':
        propagation_parameters.set_additional_parameters("shift_half_pixel", True)
        wf1 = propagator.do_propagation(propagation_parameters, Fraunhofer2D.HANDLER_NAME)
    else:
        raise Exception("Not implemented method: %s"%method)




    plot_image(wf1.get_intensity(),
               1e6*wf1.get_coordinate_x()/propagation_distance,
               1e6*wf1.get_coordinate_y()/propagation_distance,
               title="Diffracted intensity by a circular stop %3.1f um"%
                     (1e6*aperture_diameter),
               xtitle="X [urad]",ytitle="Y [urad]",
               show=0)


    intensity_calculated =  wf1.get_intensity()[:,int(wf1.size()[1]/2)]

    intensity_calculated /= intensity_calculated.max()

    plot(wf1.get_coordinate_x()*1e6/propagation_distance,intensity_calculated,
         legend=["%s H profile"%method_label],
         legend_position=(0.95, 0.95),
         title="%s diffraction of a cirlular stop %3.1f um at wavelength of %3.1f A"%
               (method_label,aperture_diameter*1e6,wavelength*1e10),
         xtitle="X (urad)", ytitle="Intensity",xrange=[-100,100],
         show=show)

