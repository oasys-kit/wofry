
import pickle

class Wavefront(object):

    def __init__(self):
        super().__init__()

    def get_dimension(self):
        raise NotImplementedError("method is abstract")

    def duplicate(self):
        raise NotImplementedError("method is abstract")

    def to_hex_tring(self):
        return pickle.dumps(self).hex()

    @classmethod
    def from_hex_tring(cls, hex_string):
        return pickle.loads(bytes.fromhex(hex_string))


class WavefrontDimension:
    ONE = "1"
    TWO = "2"
