from cldm.cldm import *


class BaseModel(ControlLDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
