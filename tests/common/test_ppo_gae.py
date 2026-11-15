from __future__ import annotations

import unittest

try:
    import torch

    from drone_control.algorithms.ppo import compute_gae

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch chưa được cài trong interpreter này")
class TestGAESemantics(unittest.TestCase):
    def test_physical_terminal_does_not_bootstrap(self) -> None:
        advantages, returns = compute_gae(
            torch.tensor([1.0]),
            torch.tensor([0.5]),
            torch.tensor([10.0]),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            gamma=0.99,
            gae_lambda=0.95,
        )
        torch.testing.assert_close(advantages, torch.tensor([0.5]))
        torch.testing.assert_close(returns, torch.tensor([1.0]))

    def test_time_limit_bootstraps_but_stops_episode_recursion(self) -> None:
        advantages, returns = compute_gae(
            torch.tensor([1.0, 100.0]),
            torch.tensor([0.5, 0.0]),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
            gamma=0.9,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages[0], torch.tensor(2.3))
        torch.testing.assert_close(returns[0], torch.tensor(2.8))

    def test_non_boundary_transition_propagates_next_advantage(self) -> None:
        advantages, _ = compute_gae(
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 0.0]),
            gamma=0.5,
            gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages, torch.tensor([2.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
