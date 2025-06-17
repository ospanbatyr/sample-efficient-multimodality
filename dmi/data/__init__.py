from .coco import COCOLoader
from .audiocaps import AudioCapsLoader
from .openvid import OpenvidLoader
from .sharegpt4v import ShareGPT4VLoader
from .clothodetail import ClothoDetailLoader
from .sharegpt4video import ShareGPT4VideoLoader
from .chebi20 import CHEBI20Loader
from .candels import CANDELSLoader
from .sydney import SydneyLoader

NAMES_LOADERS = {
    'coco': COCOLoader,
    'audiocaps': AudioCapsLoader,
    'openvid': OpenvidLoader,
    'sharegpt4v': ShareGPT4VLoader,
    'clothodetail': ClothoDetailLoader,
    'sharegpt4video': ShareGPT4VideoLoader,
    'chebi20': CHEBI20Loader,
    'candels': CANDELSLoader,
    'sydney': SydneyLoader,
}