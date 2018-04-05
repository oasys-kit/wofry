
from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
from wofry.propagator.propagators1D.fresnel import Fresnel1D
from wofry.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
from wofry.propagator.propagators1D.fraunhofer import Fraunhofer1D
from wofry.propagator.propagators1D.integral import Integral1D
from wofry.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
from wofry.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D

from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen
from wofry.beamline.optical_elements.absorbers.slit import WOSlit1D
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.shape import Rectangle

from srxraylib.plot.gol import plot


def initialize_default_propagator_1D():
    propagator = PropagationManager.Instance()

    propagator.add_propagator(Fraunhofer1D())
    propagator.add_propagator(Fresnel1D())
    propagator.add_propagator(FresnelConvolution1D())
    propagator.add_propagator(Integral1D())
    propagator.add_propagator(FresnelZoom1D())
    propagator.add_propagator(FresnelZoomScaling1D())


def propagate_wavefront(wavefront,distance,handler_name=Fresnel1D.HANDLER_NAME,zoom=0.005):

    wavefront_inside = wavefront.duplicate()

    slit = WOSlit1D(name="PIRRONE2",boundary_shape=Rectangle(-0.00005,0.00005,-0.00005,0.00005,))
    coordinates = ElementCoordinates(p = 0.0, q=0.0)
    propagation_elements = PropagationElements()
    propagation_elements.add_beamline_element(BeamlineElement(optical_element=slit,
                                                              coordinates=coordinates))
    parameters = PropagationParameters(wavefront=wavefront_inside,
                                       propagation_elements=propagation_elements)
    parameters.set_additional_parameters("shift_half_pixel", 1)
    parameters.set_additional_parameters("magnification_x", zoom)
    output_wf_1 = propagator.do_propagation(propagation_parameters=parameters,
                                            handler_name=handler_name)



    screen = WOScreen(name="PIRRONE")
    coordinates = ElementCoordinates(p = 0.0, q=distance)
    propagation_elements = PropagationElements()
    propagation_elements.add_beamline_element(BeamlineElement(optical_element=screen,
                                                              coordinates=coordinates))

    parameters = PropagationParameters(wavefront=output_wf_1,
                                       propagation_elements=propagation_elements)
    parameters.set_additional_parameters("shift_half_pixel", 1)
    parameters.set_additional_parameters("magnification_x", zoom)
    output_wf_2 = propagator.do_propagation(propagation_parameters=parameters,
                                            handler_name=handler_name)

    return output_wf_2


def compare(in_object_1,in_object_2,legend=["Spherical","Plane"]):


    xR = in_object_1.get_abscissas()
    yR = in_object_1.get_intensity()
    pR = in_object_1.get_phase(unwrap=1)


    xInf = in_object_2.get_abscissas()
    yInf = in_object_2.get_intensity()
    pInf = in_object_2.get_phase(unwrap=1)

    plot(xR,yR,xInf,yInf,legend=legend,title="Intensity")
    plot(xR,pR-pR.min(),xInf,pInf-pInf.min(),legend=legend,title="Phase")
    print("Total intensity: ",yR.sum())

#
#
#
#
#
#
def propagate_wavefront_scaling_new(wavefront,distance,handler_name=Fresnel1D.HANDLER_NAME,zoom=0.005,
                                using_radius=None):

    wavefront_inside = wavefront.duplicate()

    #
    # if R of curvature to be remove is not defined, guess it
    #
    if using_radius is None:
        plot_scan = True
        if plot_scan:
            radii,fig_of_mer = wavefront_inside.scan_wavefront_curvature(rmin=-1000,rmax=1000,rpoints=100)
            plot(radii,fig_of_mer)
        using_radius = wavefront_inside.guess_wavefront_curvature(rmin=-1000,rmax=1000,rpoints=100)

    #
    # propagate flattened wavefront to a new distance:  distance/magnification
    #

    slit = WOSlit1D(name="PIRRONE2",boundary_shape=Rectangle(-0.00005,0.00005,-0.00005,0.00005,))
    coordinates = ElementCoordinates(p = 0.0, q=0.0)
    propagation_elements = PropagationElements()
    propagation_elements.add_beamline_element(BeamlineElement(optical_element=slit,
                                                              coordinates=coordinates))
    parameters = PropagationParameters(wavefront=wavefront_inside,
                                       propagation_elements=propagation_elements)
    parameters.set_additional_parameters("shift_half_pixel", 1)
    parameters.set_additional_parameters("magnification_x", zoom)
    output_wf_1 = propagator.do_propagation(propagation_parameters=parameters,
                                            handler_name=handler_name)



    screen = WOScreen(name="PIRRONE")
    coordinates = ElementCoordinates(p = 0.0, q=distance)
    propagation_elements = PropagationElements()
    propagation_elements.add_beamline_element(BeamlineElement(optical_element=screen,
                                                              coordinates=coordinates))

    parameters = PropagationParameters(wavefront=output_wf_1,
                                       propagation_elements=propagation_elements)
    parameters.set_additional_parameters("shift_half_pixel", 1)
    parameters.set_additional_parameters("magnification_x", zoom)
    parameters.set_additional_parameters("radius", using_radius)
    output_wf_2 = propagator.do_propagation(propagation_parameters=parameters,
                                            handler_name=handler_name)


    return output_wf_2



if __name__ == "__main__":

    try:
        initialize_default_propagator_1D()
    except:
        print("Problems initializing 1D propagators")
        # pass

    propagator = PropagationManager.Instance()


    #
    # create source
    #
    wavefront = GenericWavefront1D.initialize_wavefront_from_range(x_min=-0.5e-3,
                                                                   x_max=0.5e-3,
                                                                   number_of_points=2048,
                                                                   wavelength=1.5e-10)
    radius = 50.0 # 50.
    wavefront.set_spherical_wave(radius=radius)


    distance = 5.0
    zoom = 1.0

    output_wf_2 = propagate_wavefront(wavefront,distance=distance,handler_name=FresnelZoom1D.HANDLER_NAME,zoom=zoom)
    output_wf_3 = propagate_wavefront_scaling_new(wavefront,distance=distance,handler_name=FresnelZoomScaling1D.HANDLER_NAME,
                                              zoom=zoom,using_radius=None)

    # plot(output_wf_2.get_abscissas()*1e6, output_wf_2.get_intensity(), title="WF2 - "+FresnelZoom1D.HANDLER_NAME)
    # plot(output_wf_3.get_abscissas()*1e6, output_wf_2.get_intensity(), title="SCALING!! WF2 - "+FresnelZoom1D.HANDLER_NAME)

    compare(output_wf_2,output_wf_3,legend=["zoom","zoom+scaling theorem"])
