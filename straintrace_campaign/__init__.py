"""Reusable, headless strain-tracking campaigns.

Importing this package has no GUI, OpenCV, pandas, or scientific-core side
effects. Concrete processing modules load only when their workflow stage runs.
"""

from .models import CampaignError
from .workflow import (
    align,
    approve_rois,
    compute_roi_proposal_hash,
    export,
    import_keystone,
    init_project,
    load_project,
    propose_rois,
    run,
    scan,
    update_acquisition,
    update_roi,
    verify,
)

__all__ = [
    "CampaignError",
    "align",
    "approve_rois",
    "compute_roi_proposal_hash",
    "export",
    "import_keystone",
    "init_project",
    "load_project",
    "propose_rois",
    "run",
    "scan",
    "update_acquisition",
    "update_roi",
    "verify",
]
