from __future__ import annotations

import unittest

import numpy as np

try:
    import torch

    from drone_control.algorithms.ppo import ActorCritic, PPO, ScaleAction, device

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch chưa được cài trong interpreter này")
class TestScaleAction(unittest.TestCase):
    def test_symmetric_scalar_bounds(self) -> None:
        layer = ScaleAction(-1.5, 1.5, action_dim=3)
        values = torch.tensor([[-1.0, 0.0, 1.0]])
        output = layer(values)
        torch.testing.assert_close(output, torch.tensor([[-1.5, 0.0, 1.5]]))

    def test_asymmetric_vector_bounds(self) -> None:
        layer = ScaleAction([-2.0, 0.0], [2.0, 10.0], action_dim=2)
        values = torch.tensor([[0.0, 0.0]])
        output = layer(values)
        torch.testing.assert_close(output, torch.tensor([[0.0, 5.0]]))

    def test_invalid_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScaleAction([0.0, 0.0], [1.0], action_dim=3)
        with self.assertRaises(ValueError):
            ScaleAction(1.0, -1.0, action_dim=1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch chưa được cài trong interpreter này")
class TestActorCriticActionBounds(unittest.TestCase):
    def test_velocity_forward_shape_and_default_bounds(self) -> None:
        model = ActorCritic(13, 3, action_std_init=0.01)
        states = torch.zeros((5, 13))
        means = model.actor(model.state_norm(states))
        self.assertEqual(tuple(means.shape), (5, 3))
        self.assertTrue(torch.all(means >= -1.5))
        self.assertTrue(torch.all(means <= 1.5))
        self.assertEqual(model.action_low, (-1.5, -1.5, -1.5))
        self.assertEqual(model.action_high, (1.5, 1.5, 1.5))

    def test_motor_forward_shape_and_normalized_bounds(self) -> None:
        model = ActorCritic(
            17,
            4,
            action_std_init=0.01,
            action_low=-1.0,
            action_high=1.0,
        )
        states = torch.zeros((5, 17))
        means = model.actor(model.state_norm(states))
        self.assertEqual(tuple(means.shape), (5, 4))
        self.assertTrue(torch.all(means >= -1.0))
        self.assertTrue(torch.all(means <= 1.0))

    def test_sampled_action_is_clipped_to_configured_bounds(self) -> None:
        model = ActorCritic(
            17,
            4,
            action_std_init=100.0,
            action_low=-1.0,
            action_high=1.0,
        ).to(device)
        for _ in range(10):
            action, _, _ = model.act(torch.zeros(17, device=device))
            self.assertTrue(torch.all(action >= -1.0))
            self.assertTrue(torch.all(action <= 1.0))

    def test_bounds_do_not_add_state_dict_keys(self) -> None:
        velocity_model = ActorCritic(13, 3, action_std_init=0.6)
        clone = ActorCritic(13, 3, action_std_init=0.6)
        clone.load_state_dict(velocity_model.state_dict())
        self.assertFalse(
            any("action_low" in key or "action_high" in key for key in velocity_model.state_dict())
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch chưa được cài trong interpreter này")
class TestPPOConfigurations(unittest.TestCase):
    @staticmethod
    def make_ppo(state_dim: int, action_dim: int, **kwargs) -> PPO:
        return PPO(
            state_dim,
            action_dim,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            K_epochs=2,
            eps_clip=0.2,
            action_std_init=0.01,
            **kwargs,
        )

    def test_existing_velocity_constructor_keeps_old_default(self) -> None:
        agent = self.make_ppo(13, 3)
        self.assertEqual(agent.action_low, (-1.5, -1.5, -1.5))
        self.assertEqual(agent.action_high, (1.5, 1.5, 1.5))
        action = agent.select_action(np.zeros(13, dtype=np.float32))
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(action >= -1.5))
        self.assertTrue(np.all(action <= 1.5))

    def test_motor_constructor_uses_normalized_bounds(self) -> None:
        agent = self.make_ppo(17, 4, action_low=-1.0, action_high=1.0)
        action = agent.select_action(np.zeros(17, dtype=np.float32))
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_vector_bounds_are_supported(self) -> None:
        agent = self.make_ppo(
            4,
            2,
            action_low=[-2.0, 0.0],
            action_high=[2.0, 1.0],
        )
        action = agent.select_action(np.zeros(4, dtype=np.float32))
        self.assertGreaterEqual(action[0], -2.0)
        self.assertLessEqual(action[0], 2.0)
        self.assertGreaterEqual(action[1], 0.0)
        self.assertLessEqual(action[1], 1.0)


if __name__ == "__main__":
    unittest.main()
