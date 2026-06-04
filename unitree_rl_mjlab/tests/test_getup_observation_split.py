import unittest

from src.tasks.getup.getup_env_cfg import (
    _FORBIDDEN_ACTOR_TERMS,
    _get_actor_obs_terms,
    _get_critic_obs_terms,
    make_getup_env_cfg,
)


class GetupObservationSplitTest(unittest.TestCase):
    def test_actor_has_no_privileged_terms(self):
        actor_terms = _get_actor_obs_terms()
        overlap = set(actor_terms) & _FORBIDDEN_ACTOR_TERMS
        self.assertFalse(overlap, f"Actor contains privileged terms: {sorted(overlap)}")

    def test_critic_contains_actor_terms(self):
        actor_terms = _get_actor_obs_terms()
        critic_terms = _get_critic_obs_terms()
        missing = set(actor_terms) - set(critic_terms)
        self.assertFalse(missing, f"Critic is missing actor terms: {sorted(missing)}")

    def test_public_config_matches_split(self):
        cfg = make_getup_env_cfg()
        actor_terms = set(cfg.observations["actor"].terms)
        critic_terms = set(cfg.observations["critic"].terms)
        self.assertTrue(actor_terms.issubset(critic_terms))
        self.assertFalse(actor_terms & _FORBIDDEN_ACTOR_TERMS)


if __name__ == "__main__":
    unittest.main()
