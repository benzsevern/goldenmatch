"""GoldenMatch -- entity resolution toolkit.

Quick start:
    import goldenmatch as gm

    # Deduplicate a CSV
    result = gm.dedupe("customers.csv", exact=["email"], fuzzy={"name": 0.85})
    result.golden.write_csv("deduped.csv")

    # Match across files
    result = gm.match("targets.csv", "reference.csv", fuzzy={"name": 0.85})

    # Privacy-preserving linkage
    result = gm.pprl_link("hospital_a.csv", "hospital_b.csv", fields=["name", "dob", "zip"])

    # Evaluate accuracy
    metrics = gm.evaluate("data.csv", config="config.yaml", ground_truth="gt.csv")
"""
__version__ = "0.7.1"

from goldenmatch._api import dedupe, match, pprl_link, evaluate, load_config, DedupeResult, MatchResult

__all__ = [
    "__version__",
    "dedupe",
    "match",
    "pprl_link",
    "evaluate",
    "load_config",
    "DedupeResult",
    "MatchResult",
]
