"""Compatibility wrappers for the method 2 episode pipeline.

The old session-anchor calibration path has been retired. Use src.m2 for the
method 2 grid-only candidate generator.
"""

from .m2 import (
    build_m2_scores,
    build_m2_episodes,
    build_m2_sensitivity,
    build_m2_candidates,
    build_m2_eval,
    build_m2_stability,
)


def build_episode_calibration(minute_df, out, cfg):
    scores = build_m2_scores(minute_df, out, cfg)
    episodes = build_m2_episodes(scores, out, cfg)
    sens = build_m2_sensitivity(episodes, scores, out, cfg)
    candidates = build_m2_candidates(episodes, scores, out, cfg)
    eval_df = build_m2_eval(episodes, candidates, scores, sens, out, cfg)
    stability = build_m2_stability(episodes, candidates, scores, out, cfg)
    return {
        "m2_scores": scores,
        "m2_ep": episodes,
        "m2_sens": sens,
        "m2_candidates": candidates,
        "m2_eval": eval_df,
        "m2_stability": stability,
    }

