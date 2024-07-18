import os

from astropy import units as u
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from tardis.transport.montecarlo.packet_source import (
    weighted_packet_source,
)
