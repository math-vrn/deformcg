from pkg_resources import get_distribution, DistributionNotFound

from deformcg.deform import *
from deformcg.solver_deform import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass