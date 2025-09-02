"""
Deep learning tools for deformable medical image registration. This package
offers reference implementations of core registration networks, loss
functions, and utilities, with PyTorch and TensorFlow backends.

## Subpackages (overview)

- **nn**: Torch-based neural network components for Voxelmorph.
- **py**: Python utilities for Voxelmorph.

???+ quote "Citation"

    === "APA"
        Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca,
        A. V. (2018). VoxelMorph: A Learning Framework for Deformable
        Medical Image Registration. *arXiv:1809.05231*.
        <https://arxiv.org/abs/1809.05231>

    === "BibTeX"
        ```bibtex
        @article{balakrishnan2018voxelmorph,
          title   = {VoxelMorph: A Learning Framework for Deformable Medical
                     Image Registration},
          author  = {Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R and
                     Guttag, John and Dalca, Adrian V},
          journal = {arXiv preprint arXiv:1809.05231},
          year    = {2018},
          url     = {https://arxiv.org/abs/1809.05231}
        }
        ```
"""

# set version
__version__ = '0.2'


from packaging import version

# ensure valid neurite version is available
import neurite
minv = '0.2'
curv = getattr(neurite, '__version__', None)
if curv is None or version.parse(curv) < version.parse(minv):
    raise ImportError(f'voxelmorph requires neurite version {minv} or greater, '
                      f'but found version {curv}')

from . import nn
from . import py

__all__ = ['nn', 'py']
