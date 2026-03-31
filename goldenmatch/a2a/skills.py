"""Skill dispatch for the A2A protocol server.

Routes incoming skill requests to the appropriate AgentSession methods
or GoldenMatch API functions.
"""

from __future__ import annotations

from typing import Any

import yaml

from goldenmatch.core.agent import AgentSession, _decision_to_config, profile_for_agent, select_strategy


def dispatch_skill(skill_id: str, params: dict) -> dict:
    """Dispatch an A2A skill request to the appropriate handler.

    Parameters
    ----------
    skill_id : str
        One of: analyze_data, configure, deduplicate, match, explain,
        review, compare_strategies, pprl.
    params : dict
        Skill-specific parameters.

    Returns
    -------
    dict
        Result payload.

    Raises
    ------
    ValueError
        If *skill_id* is not recognised.
    """
    session = AgentSession()

    if skill_id == "analyze_data":
        return session.analyze(params["file_path"])

    if skill_id == "deduplicate":
        result = session.deduplicate(
            params["file_path"],
            config=params.get("config"),
        )
        # Serialise -- strip non-JSON-friendly objects
        return _serialise_result(result)

    if skill_id == "match":
        result = session.match_sources(
            params["file_a"],
            params["file_b"],
            config=params.get("config"),
        )
        return _serialise_result(result)

    if skill_id == "compare_strategies":
        return session.compare_strategies(
            params["file_path"],
            ground_truth=params.get("ground_truth"),
        )

    if skill_id == "explain":
        import polars as pl
        from goldenmatch import explain_pair_df

        record_a = pl.DataFrame([params["record_a"]])
        record_b = pl.DataFrame([params["record_b"]])
        mk_cfg = params["matchkey"]
        explanation = explain_pair_df(record_a, record_b, mk_cfg)
        return {"explanation": explanation}

    if skill_id == "review":
        from goldenmatch.core.review_queue import ReviewQueue

        queue = ReviewQueue(backend="memory")
        return {"pending": queue.list_pending()}

    if skill_id == "configure":
        import polars as pl

        analysis = session.analyze(params["file_path"])
        decision = select_strategy(
            profile_for_agent(
                pl.read_csv(params["file_path"], encoding="utf8-lossy", ignore_errors=True)
            )
        )
        cfg = _decision_to_config(decision)
        return {"config_yaml": yaml.dump(cfg.model_dump(), default_flow_style=False)}

    if skill_id == "pprl":
        from goldenmatch import pprl_link

        result = pprl_link(
            params["file_a"],
            params["file_b"],
            fields=params.get("fields", []),
        )
        return _serialise_result({"result": result})

    raise ValueError(f"Unknown skill: {skill_id}")


def _serialise_result(obj: Any) -> dict:
    """Best-effort serialisation of pipeline results to JSON-safe dict."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                import polars as pl

                if isinstance(v, pl.DataFrame):
                    out[k] = {"rows": v.height, "columns": v.columns}
                    continue
            except Exception:
                pass
            if isinstance(v, dict):
                out[k] = _serialise_result(v)
            elif isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, list):
                out[k] = str(v)[:500]
            else:
                out[k] = str(v)[:500]
        return out
    return {"value": str(obj)[:500]}
