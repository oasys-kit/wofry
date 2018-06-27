

if __name__ == "__main__":

    from syned.util.json_tools import load_from_json_file
    from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
    from srxraylib.plot.gol import plot_image

    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D

    wfr = GenericWavefront2D.load_h5_file("source.h5","wfr")
    plot_image(wfr.get_intensity(),1e6*wfr.get_coordinate_x(),1e6*wfr.get_coordinate_y(),title="Source",xtitle="X [um]",ytitle="Y [um]")


    bl = load_from_json_file("beamline.json")


    #
    # propagate elements using Fresnel Zoom propagator
    #
    magnification_x = [1,1,0.01,440,1,  1,1,0.00007]
    magnification_y = [1,1,1,   5,  1,0.5,1,0.00009]

    # define propagator to be used
    propagator = PropagationManager.Instance()

    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass

    for i,beamline_element in enumerate(bl.get_beamline_elements()): #bl.get_beamline_elements_number()):
        #
        # propagating single element
        #
        print(">> Propagating element %d of %d"%(i+1,bl.get_beamline_elements_number()))
        propagation_elements = PropagationElements()
        propagation_elements.add_beamline_element(beamline_element)
        propagation_parameters = PropagationParameters(wavefront=wfr.duplicate(),propagation_elements = propagation_elements)

        propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
        propagation_parameters.set_additional_parameters('magnification_x', magnification_x[i])
        propagation_parameters.set_additional_parameters('magnification_y', magnification_y[i])

        output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,handler_name='FRESNEL_ZOOM_XY_2D')
        wfr = output_wavefront


    plot_image(wfr.get_intensity(),1e6*wfr.get_coordinate_x(),1e6*wfr.get_coordinate_y(),title="Image",xtitle="X [um]",ytitle="Y [um]")

    # compare results
    assert (wfr.is_identical(GenericWavefront2D.load_h5_file("image.h5","wfr")))







