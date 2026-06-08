import numpy as np


class CompositeBase:
    xp = np

    def __init__(self, bases):
        assert isinstance(bases, (list, tuple)), (
            f"You have to supply the bases for a composite base as tuple or list but gave {type(bases)}"
        )
        self.bases = bases

    @property
    def ndim(self):
        return len(self.bases)

    def get_grid(self):
        """
        Get grid in physical space
        """
        grids = [self.bases[i].get_grid() for i in range(self.ndim)]
        return self.xp.meshgrid(*grids, indexing="ij")
