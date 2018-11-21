###################################################################
# DO NOT TOUCH THIS CODE -- BEGIN
###################################################################
import threading

def synchronized_method(method):

    outer_lock = threading.Lock()
    lock_name = "__"+method.__name__+"_lock"+"__"

    def sync_method(self, *args, **kws):
        with outer_lock:
            if not hasattr(self, lock_name): setattr(self, lock_name, threading.Lock())
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)

    return sync_method

class Singleton:

    def __init__(self, decorated):
        self._decorated = decorated

    @synchronized_method
    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)

###################################################################
# DO NOT TOUCH THIS CODE -- END
###################################################################


from syned.beamline.beamline_element import BeamlineElement

from wofry.propagator.wavefront  import Wavefront, WavefrontDimension

class PropagationElements(object):
    
    INSERT_AFTER = 0
    INSERT_BEFORE = 1

    def __init__(self):
        self.__propagation_elements = []
        self.__propagation_element_parameters = []

    def add_beamline_element(self, beamline_element=BeamlineElement(), element_parameters=None):
        if beamline_element is None: raise ValueError("Beamline is None")

        self.__propagation_elements.append(beamline_element)
        self.__propagation_element_parameters.append(element_parameters)

    def add_beamline_elements(self, beamline_elements=[], element_parameters_list=None):
        if beamline_elements is None: raise ValueError("Beamline is None")
        
        if not element_parameters_list is None:
            if len(beamline_elements) != len(element_parameters_list): raise ValueError("Specific Parameters list does not match Beamline Elements list")
        else:
            element_parameters_list = [None]*len(beamline_elements)
            
        for beamline_element, element_parameters in zip(beamline_elements, element_parameters_list):
            self.add_beamline_element(beamline_element, element_parameters)
        
    def insert_beamline_element(self, index, new_element=BeamlineElement(), mode=INSERT_BEFORE, new_element_parameters=None):
        if mode == PropagationElements.INSERT_BEFORE:
            if index == 0:
                self.__propagation_elements = [new_element] + self.__propagation_elements
                self.__propagation_element_parameters = [new_element_parameters]
            else:
                self.__propagation_elements.insert(index, new_element)
                self.__propagation_element_parameters.insert(index, new_element_parameters)
        elif mode == PropagationElements.INSERT_AFTER:
            if index == len(self.__propagation_elements) - 1:
                self.__propagation_elements = self.__propagation_elements + [new_element]
                self.__propagation_element_parameters = self.__propagation_element_parameters + [new_element_parameters]
            else:
                self.__propagation_elements.insert(index+1, new_element)
                self.__propagation_element_parameters.insert(index+1, new_element_parameters)

    def get_propagation_elements_number(self):
        return len(self.__propagation_elements)

    def get_propagation_elements(self):
        return self.__propagation_elements

    def get_propagation_element(self, index):
        return self.__propagation_elements[index]

    def get_propagation_elements_parameters(self):
        return self.__propagation_element_parameters

    def get_propagation_element_parameter(self, index):
        return self.__propagation_element_parameters[index]



class PropagationParameters(object):
    def __init__(self,
                 wavefront = Wavefront(),
                 propagation_elements = PropagationElements()):
        self._wavefront = wavefront
        self._propagation_elements = propagation_elements
        self._additional_parameters = None

    def get_wavefront(self):
        return self._wavefront

    def get_PropagationElements(self):
        return self._propagation_elements

    def set_additional_parameters(self, key, value):
        if self._additional_parameters is None:
            self._additional_parameters = {key : value}
        else:
            self._additional_parameters[key] = value

    def get_additional_parameter(self, key):
        return self._additional_parameters[key]

    def has_additional_parameter(self, key):
        return key in self._additional_parameters

class AbstractPropagator(object):

    def __init__(self):
        super().__init__()

    def get_dimension(self):
        raise NotImplementedError("This method is abstract")

    def get_handler_name(self):
        raise NotImplementedError("This method is abstract")

    def is_handler(self, handler_name):
        return handler_name == self.get_handler_name()

    def do_propagation(self, parameters=PropagationParameters()):
        raise NotImplementedError("This method is abstract" +
                                  "\n\naccepts " + PropagationParameters.__module__ + "." + PropagationParameters.__name__ +
                                  "\nreturns " + Wavefront.__module__ + "." + Wavefront.__name__)


class PropagationMode:
    STEP_BY_STEP = 0
    WHOLE_BEAMLINE = 1

class PropagationApplication:
    ALL = "All"

@Singleton
class PropagationManager(object):

    def __init__(self):
        self.__chains_hashmap = {WavefrontDimension.ONE : [],
                                 WavefrontDimension.TWO : []}

        self.__propagation_mode_hashmap = {PropagationApplication.ALL : PropagationMode.STEP_BY_STEP}
        self.__is_initialized_hashmap = {PropagationApplication.ALL : False}

    @synchronized_method
    def set_initialized(self, application = PropagationApplication.ALL, initialized=True):
        self.__is_initialized_hashmap[application] = initialized

    @synchronized_method
    def is_initialized(self, application = PropagationApplication.ALL):
        if application in self.__is_initialized_hashmap.keys():
            return self.__is_initialized_hashmap[application]
        else:
            return False

    @synchronized_method
    def set_propagation_mode(self, application = PropagationApplication.ALL, mode=PropagationMode.STEP_BY_STEP):
        self.__propagation_mode_hashmap[application] = mode

    @synchronized_method
    def get_propagation_mode(self, application = PropagationApplication.ALL):
        return self.__propagation_mode_hashmap[application]

    @synchronized_method
    def add_propagator(self, propagator=AbstractPropagator()):
        if propagator is None: raise ValueError("Given propagator is None")
        if not isinstance(propagator, AbstractPropagator): raise ValueError("Given propagator is not a compatible object")

        dimension = propagator.get_dimension()

        if not (dimension == WavefrontDimension.ONE or dimension == WavefrontDimension.TWO):
            raise ValueError("Wrong propagator dimension")

        propagation_chain_of_responsibility = self.__chains_hashmap.get(dimension)

        for existing in propagation_chain_of_responsibility:
            if existing.is_handler(propagator.get_handler_name()):
                raise ValueError("Propagator already in the Chain")

        propagation_chain_of_responsibility.append(propagator)

    @synchronized_method
    def has_propagator(self, handler_name="<Propagator Name>", dimension=WavefrontDimension.ONE):
        propagation_chain_of_responsibility = self.__chains_hashmap.get(dimension)

        for existing in propagation_chain_of_responsibility:
            if existing.get_handler_name() == handler_name: return True

        return False

    @synchronized_method
    def get_propagators_number(self, dimension=None):
        propagation_chain_of_responsibility_1D = self.__chains_hashmap.get(WavefrontDimension.ONE)
        propagation_chain_of_responsibility_2D = self.__chains_hashmap.get(WavefrontDimension.TWO)

        if dimension == None:
            return len(propagation_chain_of_responsibility_1D), len(propagation_chain_of_responsibility_2D)
        elif dimension == WavefrontDimension.ONE:
            return len(propagation_chain_of_responsibility_1D)
        elif dimension == WavefrontDimension.TWO:
            return len(propagation_chain_of_responsibility_2D)
        else:
            raise ValueError("Dimension not valid " + str(dimension))

    def do_propagation(self, propagation_parameters, handler_name):
        for propagator in self.__chains_hashmap.get(propagation_parameters.get_wavefront().get_dimension()):
            if propagator.is_handler(handler_name):
                return propagator.do_propagation(parameters=propagation_parameters)

        raise Exception("Handler not found: "+handler_name)

# ---------------------------------------------------------------

class Propagator(AbstractPropagator):

    def do_propagation(self, parameters=PropagationParameters()):
        wavefront = parameters.get_wavefront()

        for index in range(0, parameters.get_PropagationElements().get_propagation_elements_number()):
            element = parameters.get_PropagationElements().get_propagation_element(index)
            coordinates = element.get_coordinates()

            if coordinates.p() != 0.0: wavefront = self.do_specific_progation_before(wavefront, coordinates.p(), parameters, element_index=index)
            wavefront = element.get_optical_element().applyOpticalElement(wavefront, parameters, element_index=index)
            if coordinates.q() != 0.0: wavefront = self.do_specific_progation_after(wavefront, coordinates.q(), parameters, element_index=index)


        return wavefront

    def do_specific_progation_before(self, wavefront, propagation_distance, parameters, element_index=None):
        raise NotImplementedError("This method is abstract")

    def do_specific_progation_after(self, wavefront, propagation_distance, parameters, element_index=None):
        raise NotImplementedError("This method is abstract")

    def get_additional_parameter(self, parameter_name, default_value, propagation_parameters, element_index=None, ):
        value = default_value
        try:
            value = propagation_parameters.get_additional_parameter(parameter_name)
        except:
            pass

        if element_index is None:
            myindex = 0
        else:
            myindex = element_index

        parameters_dict = propagation_parameters.get_PropagationElements().get_propagation_element_parameter(myindex)

        if parameters_dict is not None:
            if parameter_name in parameters_dict.keys():
                value = parameters_dict[parameter_name]

        return value


class Propagator1D(Propagator):

    def get_dimension(self):
        return WavefrontDimension.ONE

    def do_propagation(self, parameters=PropagationParameters()):
        if not parameters.get_wavefront().get_dimension() == WavefrontDimension.ONE:
            raise Exception("wrong wavefront!  it is not 1D")

        return super().do_propagation(parameters)

class Propagator2D(Propagator):

    def get_dimension(self):
        return WavefrontDimension.TWO

    def do_propagation(self, parameters=PropagationParameters()):
        if not parameters.get_wavefront().get_dimension() == WavefrontDimension.TWO:
            raise Exception("wrong wavefront!  it is not 2D")

        return super().do_propagation(parameters)

