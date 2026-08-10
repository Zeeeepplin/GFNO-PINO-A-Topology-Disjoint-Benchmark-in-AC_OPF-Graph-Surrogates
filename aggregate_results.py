"""Compatibility entry point for the five-seed revision aggregator.

The former single-seed pilot aggregator was removed because it used obsolete
KCL/KVL labels and could silently regenerate tables inconsistent with the
reviewed manuscript.
"""

from aggregate_revision_results import main

if __name__ == "__main__":
    main()
