"""Optional mutant: MICROIMPUTE_MUTANT=shared-seed hands every variable the base seed.

This reintroduces the pre-3.1.3 behaviour so the properties can be seen to fail
on it:  MICROIMPUTE_MUTANT=shared-seed pytest tests/
"""

import os

if os.environ.get("MICROIMPUTE_MUTANT") == "shared-seed":
    from microimpute.models.qrf import QRF

    QRF._seed_for_variable = lambda self, variable: self.seed
