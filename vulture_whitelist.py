# Vulture whitelist for false positives
# register_all is used for side effects (node registration)

from arbitrium_core.domain.workflow.nodes import register_all

_ = register_all
