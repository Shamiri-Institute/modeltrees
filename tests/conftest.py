"""Single-threaded BLAS for the test process.

Parallel split search promises results identical to the serial search run
with single-threaded BLAS (docs/api-design.md, section 6), so the tests run
that way. A BLAS library reads these variables once, when numpy loads it;
pytest loads this file before any test module imports numpy.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
