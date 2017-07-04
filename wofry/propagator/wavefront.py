
class Wavefront(object):

    def __init__(self):
        super().__init__()

    def get_dimension(self):
        raise NotImplementedError("method is abstract")

    def duplicate(self):
        raise NotImplementedError("method is abstract")

class WavefrontDimension:
    ONE = "1"
    TWO = "2"
