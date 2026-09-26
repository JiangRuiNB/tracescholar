"""Non-destructive normalization of bibliographic records into independent studies."""

from tracescholar.studies.service import (
    StudySummary, decide_study_link, get_study_details, normalize_studies,
    set_version_result_relation,
)

__all__ = ["StudySummary", "decide_study_link", "get_study_details", "normalize_studies",
           "set_version_result_relation"]
