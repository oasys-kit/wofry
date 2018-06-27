

def create_wavefront():

    #
    # create input_wavefront
    #
    #
    from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
    input_wavefront = GenericWavefront2D.initialize_wavefront_from_range(x_min=-0.001200,x_max=0.001200,y_min=-0.001200,y_max=0.001200,number_of_points=(2048,2048))
    input_wavefront.set_photon_energy(17225)
    input_wavefront.set_spherical_wave(radius=28.3,complex_amplitude=complex(1, 0))
    return input_wavefront

def add_gaussian_profile(wf):
    import numpy

    A = wf.get_complex_amplitude()
    und_length = 1.40
    p = 28.3
    wavelength = wf.get_wavelength()
    print("wavelength = ",wavelength)

    sigma_photons = .69 * numpy.sqrt(wavelength/und_length)
    print(A.shape)
    X = wf.get_mesh_x()
    Y = wf.get_mesh_y()
    print(X.shape,Y.shape)
    print(X[-1,-1],Y[-1,-1])
    sigma = p * sigma_photons
    sigma *= numpy.sqrt(2)
    #sigmax = 2 * X[-1,-1] / 10
    #sigmay = 2 * Y[-1,-1] / 10
    #print("Sigmas: ",sigmax,sigmay)
    print("sigma_photons: ",sigma_photons)
    print("sigma: ",sigma)
    Gx = numpy.exp(-X*X/2/sigma**2)
    Gy = numpy.exp(-Y*Y/2/sigma**2)
    wf.set_complex_amplitude(A*Gx*Gy)

    return wf

def slit_ml_size(input_wavefront):

    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D

    #
    # info on current oe
    #
    #
    #    -------WOSlit---------
    #        -------Rectangle---------
    #        x_left    : -0.0018849 m # x (width) minimum (signed)
    #        x_right   : 0.0018849 m # x (width) maximum (signed)
    #        y_bottom  : -0.0018849 m # y (length) minimum (signed)
    #        y_top     : 0.0018849 m # y (length) maximum (signed)
    #

    #
    # define current oe
    #
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(x_left=-0.001885,x_right=0.001885,y_bottom=-0.001885,y_top=0.001885)
    from wofry.beamline.optical_elements.absorbers.slit import WOSlit
    optical_element = WOSlit(boundary_shape=boundary_shape)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=0.000000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 1.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 1.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')


    return output_wavefront,beamline_element


def ideal_lens_ml(input_wavefront):





    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D



    #
    # info on current oe
    #
    #
    #    -------WOIdealLens---------
    #        focal_x: 8.319 m # Focal length in x [horizontal]
    #        focal_y: 99999999999999.0 m # Focal length in y [vertical]
    #

    #
    # define current oe
    #
    from wofry.beamline.optical_elements.ideal_elements.lens import WOIdealLens

    optical_element = WOIdealLens(name='',focal_x=8.319000,focal_y=99999999999999.000000)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=0.000000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 1.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 1.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront,beamline_element

def slit_aperture_40m(input_wavefront):


    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D


    #
    # info on current oe
    #
    #
    #    -------WOSlit---------
    #        -------Rectangle---------
    #        x_left    : -2.5e-05 m # x (width) minimum (signed)
    #        x_right   : 2.5e-05 m # x (width) maximum (signed)
    #        y_bottom  : -0.5 m # y (length) minimum (signed)
    #        y_top     : 0.5 m # y (length) maximum (signed)
    #

    #
    # define current oe
    #
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(x_left=-0.000025,x_right=0.000025,y_bottom=-0.500000,y_top=0.500000)
    from wofry.beamline.optical_elements.absorbers.slit import WOSlit
    optical_element = WOSlit(boundary_shape=boundary_shape)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=11.700000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 0.010000)
    propagation_parameters.set_additional_parameters('magnification_y', 1.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront,beamline_element

def slit_KBv_size(input_wavefront):

    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D



    #
    # info on current oe
    #
    #
    #    -------WOSlit---------
    #        -------Rectangle---------
    #        x_left    : -0.025 m # x (width) minimum (signed)
    #        x_right   : 0.025 m # x (width) maximum (signed)
    #        y_bottom  : -0.00045 m # y (length) minimum (signed)
    #        y_top     : 0.00045 m # y (length) maximum (signed)
    #

    #
    # define current oe
    #
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(x_left=-0.025000,x_right=0.025000,y_bottom=-0.000450,y_top=0.000450)
    from wofry.beamline.optical_elements.absorbers.slit import WOSlit
    optical_element = WOSlit(boundary_shape=boundary_shape)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=144.900000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 440.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 5.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront,beamline_element

def ideal_lens_KBv(input_wavefront):


    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D

    #
    # info on current oe
    #
    #
    #    -------WOIdealLens---------
    #        focal_x: 100000000.0 m # Focal length in x [horizontal]
    #        focal_y: 0.09994594594594594 m # Focal length in y [vertical]
    #

    #
    # define current oe
    #
    from wofry.beamline.optical_elements.ideal_elements.lens import WOIdealLens

    optical_element = WOIdealLens(name='',focal_x=100000000.000000,focal_y=0.099946)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=0.000000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 1.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 1.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront,beamline_element

def slit_KBh_size(input_wavefront):

    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D


    #
    # info on current oe
    #
    #
    #    -------WOSlit---------
    #        -------Rectangle---------
    #        x_left    : -0.000195 m # x (width) minimum (signed)
    #        x_right   : 0.000195 m # x (width) maximum (signed)
    #        y_bottom  : -0.0065 m # y (length) minimum (signed)
    #        y_top     : 0.0065 m # y (length) maximum (signed)
    #

    #
    # define current oe
    #
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(x_left=-0.000195,x_right=0.000195,y_bottom=-0.006500,y_top=0.006500)
    from wofry.beamline.optical_elements.absorbers.slit import WOSlit
    optical_element = WOSlit(boundary_shape=boundary_shape)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=0.050000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 1.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 0.500000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')


    return output_wavefront,beamline_element

def ideal_lens_KBh(input_wavefront):

    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D



    #
    # info on current oe
    #
    #
    #    -------WOIdealLens---------
    #        focal_x: 0.049982758620701014 m # Focal length in x [horizontal]
    #        focal_y: 100000000.0 m # Focal length in y [vertical]
    #

    #
    # define current oe
    #
    from wofry.beamline.optical_elements.ideal_elements.lens import WOIdealLens

    optical_element = WOIdealLens(name='',focal_x=0.049983,focal_y=100000000.000000)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=0.000000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 1.000000)
    propagation_parameters.set_additional_parameters('magnification_y', 1.000000)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront,beamline_element

def slit_schermo(input_wavefront):

    #
    # ===== Example of python code to create propagate current element =====
    #

    #
    # Import section
    #
    import numpy
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D


    #
    # info on current oe
    #
    #
    #    -------WOSlit---------
    #        -------Rectangle---------
    #        x_left    : -0.5 m # x (width) minimum (signed)
    #        x_right   : 0.5 m # x (width) maximum (signed)
    #        y_bottom  : -0.5 m # y (length) minimum (signed)
    #        y_top     : 0.5 m # y (length) maximum (signed)
    #

    #
    # define current oe
    #
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(x_left=-0.500000,x_right=0.500000,y_bottom=-0.500000,y_top=0.500000)
    from wofry.beamline.optical_elements.absorbers.slit import WOSlit
    optical_element = WOSlit(boundary_shape=boundary_shape)

    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,coordinates=ElementCoordinates(p=0.050000,q=0.000000,angle_radial=numpy.radians(0.000000),angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront.duplicate(),propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 0.000070)
    propagation_parameters.set_additional_parameters('magnification_y', 0.000090)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')

    return output_wavefront,beamline_element




def propagate_full_beamline(wfr):

    # wfr = slit_ml_size(wfr)
    # wfr = ideal_lens_ml(wfr)
    # wfr = slit_aperture_40m(wfr)
    # wfr = slit_KBv_size(wfr)
    # wfr = ideal_lens_KBv(wfr)
    # wfr = slit_KBh_size(wfr)
    # wfr = ideal_lens_KBh(wfr)
    # wfr = slit_schermo(wfr)

    wfr,beamline_element = slit_ml_size(wfr)
    beamline_element.to_json("oe1.json")

    wfr,beamline_element = ideal_lens_ml(wfr)
    beamline_element.to_json("oe2.json")

    wfr,beamline_element = slit_aperture_40m(wfr)
    beamline_element.to_json("oe3.json")

    wfr,beamline_element = slit_KBv_size(wfr)
    beamline_element.to_json("oe4.json")

    wfr,beamline_element = ideal_lens_KBv(wfr)
    beamline_element.to_json("oe5.json")

    wfr,beamline_element = slit_KBh_size(wfr)
    beamline_element.to_json("oe6.json")

    wfr,beamline_element = ideal_lens_KBh(wfr)
    beamline_element.to_json("oe7.json")

    wfr,beamline_element = slit_schermo(wfr)
    beamline_element.to_json("oe8.json")

    return wfr

def create_json_full_beamline():


    from syned.util.json_tools import load_from_json_file
    from syned.beamline.beamline import Beamline
    from syned.storage_ring.empty_light_source import EmptyLightSource


    beamline1 = Beamline()

    src = EmptyLightSource()
    beamline1.set_light_source(src)

    for i in range(8):
        file_in = "oe%1d.json"%(1+i)
        print(">>>> getting file",file_in )
        tmp = load_from_json_file(file_in)
        print("\n-----------Info on: \n",tmp.info(),"----------------\n\n")
        print("returned class: ",type(tmp))
        beamline1.append_beamline_element(tmp)


    beamline1.to_json("beamline.json")




if __name__ == "__main__":
    #
    # source
    #
    wfr = create_wavefront()
    wfr = add_gaussian_profile(wfr)
    wfr.save_h5_file("source.h5")

    #
    # beamline
    #
    wfr = propagate_full_beamline(wfr)
    wfr.save_h5_file("image.h5")

    #
    # merge json-elements in json-beamline
    create_json_full_beamline()

    #
    # plot image
    #
    from srxraylib.plot.gol import plot_image
    plot_image(wfr.get_intensity(),1e6*wfr.get_coordinate_x(),1e6*wfr.get_coordinate_y())









