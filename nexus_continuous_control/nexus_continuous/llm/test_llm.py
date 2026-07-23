"""Test suite for nexus_continuous.llm.

Run with:
    python -m unittest nexus_continuous.llm.test_llm -v
or:
    pytest nexus_continuous/llm/test_llm.py
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from dataclasses import asdict

from nexus_continuous.llm.schema import NexusSkillSet, SkillSpec, RewardTerm
from nexus_continuous.llm.client import LLMClient, LLMConfig, MockSkillGenerator
from nexus_continuous.llm.pipeline import (
    build_skill_prompt,
    validate_and_build,
    LLMSkillPipeline,
    generate_skillset,
    save_skillset,
    skillset_to_dict,
)
from nexus_continuous.llm.refinement_loop import (
    RefinementConfig,
    LLMRefinementLoop,
    summarize_metrics,
)
from nexus_continuous.llm.mock_training import MockTrainer, mock_train_fn, hand_written_baseline_metrics

from nexus_continuous.llm.jax_bootstrap import ensure_jax


ensure_jax()
try:
    import jax.numpy as jnp  # noqa: F401
    from nexus_continuous.llm import interpreter
    from nexus_continuous.policies import registry

    HAVE_JAX = True
    _SKIP_REASON = ""
except Exception as _e:  # pragma: no cover - environment dependent
    HAVE_JAX = False
    _SKIP_REASON = f"jax / nexus_continuous.policies.common not importable ({_e})"


FIELDS = ("cart_position", "pole_angle", "cart_velocity", "pole_angular_velocity")


# schema.py
# ---------------------------------------------------------------------------
class TestSchema(unittest.TestCase):
    def test_reward_term_defaults(self):
        t = RewardTerm(type="negative_distance", weight=1.0)
        self.assertIsNone(t.lhs)
        self.assertIsNone(t.rhs)
        self.assertIsNone(t.threshold)

    def test_frozen_dataclasses_are_immutable(self):
        t = RewardTerm(type="action_penalty", weight=0.1)
        with self.assertRaises(Exception):
            t.weight = 0.5

    def test_nexus_skillset_asdict_is_deep(self):
        skill = SkillSpec("s0", "d", "true", [RewardTerm(type="action_penalty", weight=0.1)])
        skillset = NexusSkillSet("Env", "schema", [skill], "notes")
        d = asdict(skillset)
        self.assertIsInstance(d["skills"][0], dict)
        self.assertIsInstance(d["skills"][0]["reward_terms"][0], dict)


# client.py
# ---------------------------------------------------------------------------
class TestExtractJson(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(LLMConfig(backend="mock"))

    def test_extract_plain_json(self):
        self.assertEqual(self.client.extract_json('{"a": 1}'), {"a": 1})

    def test_extract_json_with_preamble(self):
        text = 'Sure, here you go:\n{"a": 1}\nEnjoy!'
        self.assertEqual(self.client.extract_json(text), {"a": 1})

    def test_no_brace_raises(self):
        with self.assertRaises(RuntimeError):
            self.client.extract_json("no json here")

    def test_malformed_json_raises(self):
        with self.assertRaises(RuntimeError):
            self.client.extract_json("{not valid json,,}")


class TestMockBackend(unittest.TestCase):
    def test_default_mock_response_valid_shape(self):
        client = LLMClient(LLMConfig(backend="mock"))
        out = client.generate_json("sys", "user")
        self.assertGreaterEqual(len(out.get("skills", [])), 1)

    def test_mock_generator_deterministic(self):
        a = LLMClient(LLMConfig(backend="mock"), mock_generator=MockSkillGenerator(FIELDS, seed=42))
        b = LLMClient(LLMConfig(backend="mock"), mock_generator=MockSkillGenerator(FIELDS, seed=42))
        self.assertEqual(a.generate_json("s", "u"), b.generate_json("s", "u"))

    def test_mock_generator_varies_with_seed(self):
        a = LLMClient(LLMConfig(backend="mock"), mock_generator=MockSkillGenerator(FIELDS, seed=1))
        b = LLMClient(LLMConfig(backend="mock"), mock_generator=MockSkillGenerator(FIELDS, seed=2))
        self.assertNotEqual(a.generate_json("s", "u"), b.generate_json("s", "u"))

    def test_mock_generator_reacts_to_refinement_prompt(self):
        client = LLMClient(LLMConfig(backend="mock"), mock_generator=MockSkillGenerator(FIELDS, seed=0))
        first = client.generate_json("s", "no feedback here")
        second = client.generate_json("s", "... Previous skillset: {...} ...")
        self.assertNotEqual(first, second)

    def test_unknown_backend_raises(self):
        client = LLMClient(LLMConfig(backend="nope"))
        with self.assertRaises(ValueError):
            client.generate_json("s", "u")


# pipeline.py
# ---------------------------------------------------------------------------
class TestBuildPrompt(unittest.TestCase):
    def test_prompt_contains_env_and_task(self):
        sys_p, user_p = build_skill_prompt("CartpoleBalance", "cart_position", "Balance the pole.")
        self.assertIn("CartpoleBalance", user_p)
        self.assertIn("Balance the pole.", user_p)


class TestValidateAndBuild(unittest.TestCase):
    def test_missing_skills_raises(self):
        with self.assertRaises(ValueError):
            validate_and_build({})

    def test_empty_skills_raises(self):
        with self.assertRaises(ValueError):
            validate_and_build({"skills": []})

    def test_valid_minimal_skillset(self):
        raw = {"skills": [{"name": "s0", "activation_rule": "true", "reward_terms": []}]}
        result = validate_and_build(raw)
        self.assertIsInstance(result, NexusSkillSet)

    def test_unknown_field_warns(self):
        raw = {
            "skills": [
                {"name": "s0", "activation_rule": "true", "reward_terms": [{"type": "negative_distance", "weight": 1.0, "lhs": "bogus_field"}]}
            ]
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_and_build(raw, allowed_fields={"cart_position"})
            self.assertTrue(any("bogus_field" in str(wi.message) for wi in w))


class TestPipelineRetries(unittest.TestCase):
    class FlakyClient:
        def __init__(self, fail_times):
            self.fail_times, self.calls = fail_times, 0

        def generate_json(self, s, u):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise RuntimeError("bad json")
            return {"skills": [{"name": "ok", "activation_rule": "true", "reward_terms": []}]}

    def test_retries_then_succeeds(self):
        client = self.FlakyClient(2)
        result = LLMSkillPipeline(client).generate_skillset("E", "s", "t", max_retries=2)
        self.assertEqual(result.skills[0].name, "ok")

    def test_exhausts_retries_raises(self):
        client = self.FlakyClient(99)
        with self.assertRaises(RuntimeError):
            LLMSkillPipeline(client).generate_skillset("E", "s", "t", max_retries=1)


class TestSaveSkillset(unittest.TestCase):
    def test_nested_directory(self):
        skillset = NexusSkillSet("E", "s", [SkillSpec("s0", "", "true", [])], "")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a", "b", "skills.json")
            save_skillset(skillset, path)
            with open(path) as f:
                data = json.load(f)
            self.assertIsInstance(data["skills"][0], dict)

    def test_bare_filename_no_directory_component(self):
        skillset = NexusSkillSet("E", "s", [SkillSpec("s0", "", "true", [])], "")
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                save_skillset(skillset, "skills.json")
                self.assertTrue(os.path.exists("skills.json"))
            finally:
                os.chdir(cwd)


# refinement_loop.py
# ---------------------------------------------------------------------------
class TestRefinementConfig(unittest.TestCase):
    def test_task_description(self):
        cfg = RefinementConfig(env_name="E", observation_schema="s", task_description="do it")
        self.assertEqual(cfg.task_description, "do it")

    def test_legacy_typo_alias(self):
        cfg = RefinementConfig(env_name="E", observation_schema="s", task_decription="legacy")
        self.assertEqual(cfg.task_description, "legacy")


class TestSummarizeMetrics(unittest.TestCase):
    def test_known_keys(self):
        s = summarize_metrics({"returns/env_reward_mean": 1.5, "junk": 1})
        self.assertIn("returns/env_reward_mean: 1.5", s)
        self.assertNotIn("junk", s)


class TestRefinementLoop(unittest.TestCase):
    def _client(self, seed):
        return LLMClient(LLMConfig(backend="mock", seed=seed), mock_generator=MockSkillGenerator(FIELDS, seed=seed))

    def test_full_run_reproducible(self):
        cfg_kwargs = dict(
            env_name="CartpoleBalance",
            observation_schema="\n".join(FIELDS),
            task_description="Balance the pole.",
            num_iterations=3,
            allowed_fields=set(FIELDS),
        )
        loop_a = LLMRefinementLoop(LLMSkillPipeline(self._client(7)), self._client(7))
        result_a = loop_a.run(RefinementConfig(**cfg_kwargs), MockTrainer(seed=7))

        loop_b = LLMRefinementLoop(LLMSkillPipeline(self._client(7)), self._client(7))
        result_b = loop_b.run(RefinementConfig(**cfg_kwargs), MockTrainer(seed=7))

        self.assertEqual([r.metrics for r in result_a.history], [r.metrics for r in result_b.history])
        self.assertFalse(result_a.stopped_early)

    def test_records_failure_without_losing_history(self):
        class FailsAfterFirst:
            def __init__(self):
                self.calls = 0

            def generate_json(self, s, u):
                self.calls += 1
                if self.calls == 1:
                    return {"skills": [{"name": "s0", "activation_rule": "true", "reward_terms": []}]}
                raise RuntimeError("broken")

        client = FailsAfterFirst()
        loop = LLMRefinementLoop(LLMSkillPipeline(client), client)
        cfg = RefinementConfig(env_name="E", observation_schema="s", task_description="t", num_iterations=4, max_retries=0)
        result = loop.run(cfg, MockTrainer(seed=0))
        self.assertTrue(result.stopped_early)
        self.assertFalse(result.history[-1].refinement_ok)


# mock_training.py
# ---------------------------------------------------------------------------
class TestMockTraining(unittest.TestCase):
    def test_deterministic_given_seed(self):
        skillset = {"skills": [{"name": "a", "activation_rule": "abs(x)>0.1", "reward_terms": [{"type": "action_penalty", "weight": 0.1}]}]}
        self.assertEqual(MockTrainer(seed=1)(skillset), MockTrainer(seed=1)(skillset))

    def test_metrics_keys(self):
        m = mock_train_fn({"skills": [{"name": "a", "activation_rule": "true", "reward_terms": []}]}, seed=0)
        for k in ("returns/env_reward_mean", "policy_diag/primary_success_rate"):
            self.assertIn(k, m)

    def test_hand_written_baseline_deterministic(self):
        self.assertEqual(
            hand_written_baseline_metrics("CartpoleBalance", seed=0),
            hand_written_baseline_metrics("CartpoleBalance", seed=0),
        )


# interpreter.py + registry.py integration (skipped if jax unavailable)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_JAX, _SKIP_REASON)
class TestInterpreter(unittest.TestCase):
    def _skillset(self):
        return {
            "skills": [
                {
                    "name": "recover",
                    "activation_rule": "abs(pole_angle) > 0.2",
                    "reward_terms": [{"type": "negative_distance", "weight": 1.0, "lhs": "pole_angle"}],
                },
                {"name": "hold", "activation_rule": "true", "reward_terms": [{"type": "action_penalty", "weight": 0.1}]},
            ]
        }

    def test_eval_rule_basic(self):
        import numpy as np

        fields = {"cart_position": jnp.asarray([0.0, 0.5, -0.5])}
        out = np.asarray(interpreter.eval_rule("abs(cart_position) > 0.2", fields))
        np.testing.assert_array_equal(out, [False, True, True])

    def test_malformed_rule_fails_closed(self):
        import numpy as np

        fields = {"cart_position": jnp.asarray([0.0, 0.5])}
        out = np.asarray(interpreter.eval_rule("not ) valid (", fields))
        self.assertFalse(bool(np.any(out)))

    def test_make_policy_module_end_to_end(self):
        import numpy as np

        mod = interpreter.make_policy_module(self._skillset(), field_names=FIELDS)
        rng = np.random.default_rng(0)
        obs = jnp.asarray(rng.standard_normal((6, 4)).astype("float32"))
        action = jnp.asarray(rng.standard_normal((6, 1)).astype("float32"))
        done = jnp.zeros((6,), dtype=bool)
        rewards = mod.skill_rewards(obs, obs, action, None, done, None)
        self.assertEqual(rewards.shape, (6, 2))

    def test_make_policy_module_empty_skills_raises(self):
        with self.assertRaises(ValueError):
            interpreter.make_policy_module({"skills": []}, field_names=FIELDS)


@unittest.skipUnless(HAVE_JAX, _SKIP_REASON)
class TestRegistryIntegration(unittest.TestCase):
    def test_missing_llm_skillset_key(self):
        with self.assertRaisesRegex(KeyError, "LLM_SKILLSET"):
            registry.load_policy_module({"USE_LLM_SKILLS": True, "OBS_FIELDS": FIELDS})

    def test_missing_obs_fields_key(self):
        skillset = asdict(NexusSkillSet("E", "s", [SkillSpec("s0", "", "true", [])], ""))
        with self.assertRaisesRegex(KeyError, "OBS_FIELDS"):
            registry.load_policy_module({"USE_LLM_SKILLS": True, "LLM_SKILLSET": skillset})

    def test_full_llm_config_builds_module(self):
        skillset = asdict(NexusSkillSet("E", "s", [SkillSpec("s0", "", "true", [])], ""))
        cfg = {"USE_LLM_SKILLS": True, "LLM_SKILLSET": skillset, "OBS_FIELDS": FIELDS}
        mod = registry.load_policy_module(cfg)
        self.assertEqual(mod.NUM_SKILLS, 1)

    def test_accepts_raw_dataclass_instance(self):
        skillset = NexusSkillSet("E", "s", [SkillSpec("s0", "", "true", [])], "")
        cfg = {"USE_LLM_SKILLS": True, "LLM_SKILLSET": skillset, "OBS_FIELDS": FIELDS}
        mod = registry.load_policy_module(cfg)
        self.assertEqual(mod.NUM_SKILLS, 1)


if __name__ == "__main__":
    unittest.main()
