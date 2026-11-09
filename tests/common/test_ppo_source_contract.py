from __future__ import annotations

import ast
import unittest
from pathlib import Path


PPO_PATH = (
    Path(__file__).resolve().parents[2]
    / "drone_control"
    / "algorithms"
    / "ppo.py"
)


class TestPPOActionBoundsSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PPO_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def get_class(self, name: str) -> ast.ClassDef:
        return next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        )

    def get_init_argument_names(self, class_name: str) -> list[str]:
        class_node = self.get_class(class_name)
        init_node = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        return [argument.arg for argument in init_node.args.args]

    def test_actor_critic_exposes_configurable_bounds(self) -> None:
        arguments = self.get_init_argument_names("ActorCritic")
        self.assertIn("action_low", arguments)
        self.assertIn("action_high", arguments)

    def test_ppo_exposes_configurable_bounds_at_end_of_signature(self) -> None:
        arguments = self.get_init_argument_names("PPO")
        self.assertEqual(arguments[-2:], ["action_low", "action_high"])

    def test_old_hardcoded_scaling_and_clamp_are_removed(self) -> None:
        self.assertNotIn("ScaleAction(1.5)", self.source)
        self.assertNotIn("action.clamp(-1.5, 1.5)", self.source)

    def test_bounds_do_not_register_new_checkpoint_buffers(self) -> None:
        scale_action = self.get_class("ScaleAction")
        register_buffer_calls = [
            node
            for node in ast.walk(scale_action)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_buffer"
        ]
        self.assertEqual(register_buffer_calls, [])


if __name__ == "__main__":
    unittest.main()
