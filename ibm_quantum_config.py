"""IBM Quantum configuration helpers.

This repo intentionally does NOT hard-code IBM Cloud instance identifiers or API keys.
Provide them via environment variables:

- IBM_QUANTUM_INSTANCE_CRN: IBM Quantum instance CRN (required for hardware runs)
- QISKIT_IBM_TOKEN: IBM Quantum API token (optional; Qiskit may also use stored credentials)

Keeping this in one module avoids accidental leaks and makes configuration consistent.
"""

from __future__ import annotations

import os
from typing import Optional


_ENV_INSTANCE_CRN = "IBM_QUANTUM_INSTANCE_CRN"
_ENV_QISKIT_TOKEN = "QISKIT_IBM_TOKEN"


def get_ibm_instance_crn(*, required: bool = False) -> Optional[str]:
    """Return the IBM Quantum instance CRN from the environment."""
    value = os.environ.get(_ENV_INSTANCE_CRN)
    if required and not value:
        raise RuntimeError(
            f"Missing {_ENV_INSTANCE_CRN}. Set it in your environment (or .env.local) "
            "to run on IBM Quantum hardware."
        )
    return value


def get_qiskit_ibm_token() -> Optional[str]:
    """Return the IBM Quantum API token from the environment (if set)."""
    return os.environ.get(_ENV_QISKIT_TOKEN)
