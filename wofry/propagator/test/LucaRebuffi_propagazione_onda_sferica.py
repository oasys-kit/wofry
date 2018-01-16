
import matplotlib.pylab as plt

from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagators2D.fresnel import Fresnel2D, FresnelConvolution2D
from wofry.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofry.propagator.propagators2D.integral import Integral2D
from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement

def initialize_default_propagator_2D():
    propagator = PropagationManager.Instance()

    propagator.add_propagator(Fraunhofer2D())
    propagator.add_propagator(Fresnel2D())
    propagator.add_propagator(FresnelConvolution2D())
    propagator.add_propagator(Integral2D())
    propagator.add_propagator(FresnelZoomXY2D())

try:
    initialize_default_propagator_2D()
except:
    pass

def plot_image(mymode,theta,psi,title="TITLE",xtitle=r"X [$\mu m$]",ytitle=r"Y [$\mu m$]",cmap=None,show=1):


    fig = plt.figure(1)


    plt.imshow(mymode.T,origin='lower',extent=[theta[0],theta[-1],psi[0],psi[-1]],cmap=cmap)
    plt.colorbar()
    ax = fig.gca()
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    plt.title(title)

    if show: plt.show()








propagator = PropagationManager.Instance()


wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=-2.5e-3,
                                                               x_max=2.5e-3,
                                                               y_min=-1e-3,
                                                               y_max=1e-3,
                                                               number_of_points=(1024, 1024),
                                                               wavelength=73e-12)

radius = 28.3

wavefront.set_spherical_wave(radius=radius)

scale_factor = 1

screen = WOScreen(name="PIRRONE")
coordinates = ElementCoordinates(p = 0.0, q=-scale_factor*radius)

propagation_elements = PropagationElements()
propagation_elements.add_beamline_element(BeamlineElement(optical_element=screen,
                                                          coordinates=coordinates))

parameters = PropagationParameters(wavefront=wavefront,
                                   propagation_elements=propagation_elements)
parameters.set_additional_parameters("shift_half_pixel", 1)
parameters.set_additional_parameters("magnification_x", 1.5)
parameters.set_additional_parameters("magnification_y", 2.0)

output_wf_1 = propagator.do_propagation(propagation_parameters=parameters,
                                        handler_name=Fresnel2D.HANDLER_NAME)

# output_wf_2 = propagator.do_propagation(propagation_parameters=parameters,
#                                         handler_name=FresnelConvolution2D.HANDLER_NAME)

output_wf_2 = propagator.do_propagation(propagation_parameters=parameters,
                                        handler_name=FresnelZoomXY2D.HANDLER_NAME)

#output_wf_1 = output_wf_1.rebin(expansion_range_vertical=0.01,
#                                expansion_range_horizontal=0.01,
#                                expansion_points_vertical=1.0,
#                                expansion_points_horizontal=1.0)

#output_wf_2 = output_wf_2.rebin(expansion_range_vertical=0.01,
   #                             expansion_range_horizontal=0.01,
  #                              expansion_points_vertical=1.0,
 #                               expansion_points_horizontal=1.0)

plot_image(output_wf_1.get_intensity(), output_wf_1.get_coordinate_x()*1e6 , output_wf_1.get_coordinate_y()*1e6, title="WF1 - "+Fresnel2D.HANDLER_NAME)
plot_image(output_wf_2.get_intensity(), output_wf_2.get_coordinate_x()*1e6 , output_wf_2.get_coordinate_y()*1e6, title="WF2 - "+FresnelZoomXY2D.HANDLER_NAME)

