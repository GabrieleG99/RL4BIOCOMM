import torch
import numpy as np
import gymnasium.spaces as spaces
import pytest
import tempfile
import os
from typing import Dict, Any


# Import your policy classes (adjust import paths as needed)
from ..models.policies import DictActorCriticPolicy, ActorCriticPolicy
from ..models.layers import FeedForwardNN


def create_test_action_spaces():
    """Create various Dict action spaces for testing"""

    # Simple mixed space
    simple_space = spaces.Dict({
        'discrete': spaces.Discrete(4),
        'continuous': spaces.Box(low=-1, high=1, shape=(2,))
    })

    # Complex mixed space
    complex_space = spaces.Dict({
        'move': spaces.Discrete(8),
        'shoot': spaces.Discrete(2),
        'aim': spaces.Box(low=-1, high=1, shape=(2,)),
        'reload': spaces.Discrete(3),
        'inventory': spaces.MultiDiscrete([5, 3, 7])
    })

    # All discrete
    all_discrete_space = spaces.Dict({
        'action1': spaces.Discrete(3),
        'action2': spaces.Discrete(5),
        'action3': spaces.Discrete(2)
    })

    # All continuous
    all_continuous_space = spaces.Dict({
        'velocity': spaces.Box(low=-10, high=10, shape=(3,)),
        'force': spaces.Box(low=0, high=1, shape=(1,)),
        'angle': spaces.Box(low=-np.pi, high=np.pi, shape=(2,))
    })

    return {
        'simple': simple_space,
        'complex': complex_space,
        'all_discrete': all_discrete_space,
        'all_continuous': all_continuous_space
    }


def create_test_observation_space():
    """Create observation space for testing"""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(10,))


class TestDictActorCriticPolicy:
    """Test suite for DictActorCriticPolicy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.action_spaces = create_test_action_spaces()
        self.observation_space = create_test_observation_space()
        self.batch_size = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_policy(self, action_space_name='simple', **kwargs):
        """Helper to create policy instance"""
        default_kwargs = {
            'action_space': self.action_spaces[action_space_name],
            'observation_space': self.observation_space,
            'net_arch': {'pi': [32, 32], 'vf': [32, 32]},
            'device': self.device
        }
        default_kwargs.update(kwargs)

        return DictActorCriticPolicy(**default_kwargs)

    def create_random_observation(self, batch_size=None):
        """Create random observation tensor"""
        if batch_size is None:
            batch_size = self.batch_size
        return torch.randn(batch_size, self.observation_space.shape[0], device=self.device)

    def test_initialization(self):
        """Test policy initialization with different action spaces"""
        print("Testing initialization...")

        for space_name, action_space in self.action_spaces.items():
            print(f"  Testing {space_name} action space...")
            policy = self.create_policy(space_name)

            # Check that actor heads are created correctly
            assert hasattr(policy, 'output_size')
            assert isinstance(policy.output_size, dict)
            assert len(policy.output_size) == len(action_space.spaces)

            # Check output sizes match action space
            for key, space in action_space.spaces.items():
                if isinstance(space, spaces.Discrete):
                    assert policy.output_size[key] == space.n
                elif isinstance(space, spaces.Box):
                    assert policy.output_size[key] == space.shape[0]
                elif isinstance(space, spaces.MultiDiscrete):
                    assert policy.output_size[key] == np.sum(space.nvec)

            print(f"    ✓ {space_name} action space initialized correctly")

    def test_invalid_action_space(self):
        """Test that non-Dict action spaces are rejected"""
        print("Testing invalid action space rejection...")

        invalid_spaces = [
            spaces.Discrete(4),
            spaces.Box(low=-1, high=1, shape=(2,)),
            spaces.MultiDiscrete([3, 4])
        ]

        for invalid_space in invalid_spaces:
            try:
                policy = DictActorCriticPolicy(
                    action_space=invalid_space,
                    observation_space=self.observation_space
                )
                assert False, f"Should have raised ValueError for {type(invalid_space)}"
            except ValueError as e:
                assert "Dict action space" in str(e)
                print(f"    ✓ Correctly rejected {type(invalid_space).__name__}")

    def test_forward_pass(self):
        """Test forward pass with different action spaces"""
        print("Testing forward pass...")

        for space_name in self.action_spaces.keys():
            print(f"  Testing {space_name} action space...")
            policy = self.create_policy(space_name)
            obs = self.create_random_observation()

            # Test forward pass
            actions, log_probs = policy.forward(obs)

            # Check output types
            assert isinstance(actions, dict)
            assert isinstance(log_probs, torch.Tensor)

            # Check action dict structure
            action_space = self.action_spaces[space_name]
            assert set(actions.keys()) == set(action_space.spaces.keys())

            # Check tensor shapes
            for key, action_tensor in actions.items():
                assert action_tensor.shape[0] == self.batch_size
                space = action_space.spaces[key]

                if isinstance(space, spaces.Discrete):
                    assert action_tensor.dtype == torch.long
                    assert len(action_tensor.shape) == 1
                elif isinstance(space, spaces.Box):
                    assert action_tensor.dtype == torch.float32
                    assert action_tensor.shape[1:] == space.shape
                elif isinstance(space, spaces.MultiDiscrete):
                    assert action_tensor.dtype == torch.long
                    assert action_tensor.shape[1] == len(space.nvec)

            # Check log probabilities shape
            assert log_probs.shape == (self.batch_size,1)

            print(f"    ✓ Forward pass works for {space_name}")

    def test_prediction(self):
        """Test prediction method"""
        print("Testing prediction...")

        for space_name in self.action_spaces.keys():
            print(f"  Testing {space_name} action space...")
            policy = self.create_policy(space_name)
            obs = np.random.randn(self.observation_space.shape[0])

            # Test deterministic prediction
            actions_det = policy.predict(obs, deterministic=True)
            assert isinstance(actions_det, dict)

            # Test stochastic prediction
            actions_stoch = policy.predict(obs, deterministic=False)
            assert isinstance(actions_stoch, dict)

            # Check output structure
            action_space = self.action_spaces[space_name]
            assert set(actions_det.keys()) == set(action_space.spaces.keys())
            assert set(actions_stoch.keys()) == set(action_space.spaces.keys())

            # Check output types are numpy arrays
            for key in actions_det.keys():
                assert isinstance(actions_det[key], np.ndarray)
                assert isinstance(actions_stoch[key], np.ndarray)

            print(f"    ✓ Prediction works for {space_name}")

    def test_evaluation(self):
        """Test policy evaluation"""
        print("Testing policy evaluation...")

        for space_name in self.action_spaces.keys():
            print(f"  Testing {space_name} action space...")
            policy = self.create_policy(space_name)
            obs = self.create_random_observation()

            # Get actions from forward pass
            actions, _ = policy.forward(obs)

            # Test evaluation
            log_probs, entropies = policy.evaluate(obs, actions)

            # Check output shapes
            assert log_probs.shape == (self.batch_size,1)
            assert entropies.shape == (self.batch_size,1)

            # Check that log probs are finite
            assert torch.all(torch.isfinite(log_probs))
            assert torch.all(torch.isfinite(entropies))

            print(f"    ✓ Evaluation works for {space_name}")

    def test_value_function(self):
        """Test critic value function"""
        print("Testing value function...")

        policy = self.create_policy('simple')
        obs = self.create_random_observation()

        values = policy.get_state_value(obs)

        assert values.shape == (self.batch_size, 1)
        assert torch.all(torch.isfinite(values))

        print("    ✓ Value function works correctly")

    def test_optimizer_creation(self):
        """Test optimizer creation with different configurations"""
        print("Testing optimizer creation...")

        # Test shared optimizer
        policy_shared = self.create_policy('simple')
        assert hasattr(policy_shared, 'optimizer')
        print("    ✓ Shared optimizer created")

        # Test separate optimizers
        optim_kwargs = {
            'actor': {'lr': 1e-3},
            'critic': {'lr': 1e-4}
        }
        policy_separate = self.create_policy('simple', optim_kwargs=optim_kwargs)
        assert hasattr(policy_separate, 'actor_optimizer')
        assert hasattr(policy_separate, 'critic_optimizer')
        print("    ✓ Separate optimizers created")

    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        print("Testing gradient flow...")

        policy = self.create_policy('simple')
        obs = self.create_random_observation()

        # Forward pass
        actions, log_probs = policy.forward(obs)
        values = policy.get_state_value(obs)

        # Compute dummy loss
        loss = -log_probs.mean() + values.pow(2).mean()
        loss.backward()

        # Check that gradients exist
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for parameter {name}"

        print("    ✓ Gradients flow correctly")

    def test_action_space_constraints(self):
        """Test that actions respect action space constraints"""
        print("Testing action space constraints...")

        for space_name in self.action_spaces.keys():
            print(f"  Testing {space_name} action space constraints...")
            policy = self.create_policy(space_name)
            obs = np.random.randn(self.observation_space.shape[0])

            # Get deterministic actions
            actions = policy.predict(obs, deterministic=True)
            action_space = self.action_spaces[space_name]

            # Check constraints for each action component
            for key, action in actions.items():
                space = action_space.spaces[key]

                if isinstance(space, spaces.Discrete):
                    assert 0 <= action < space.n, f"Discrete action {key} out of bounds"
                elif isinstance(space, spaces.Box):
                    # Note: For continuous actions, we might want to add clipping in the policy
                    assert action.shape == space.shape, f"Box action {key} wrong shape"
                elif isinstance(space, spaces.MultiDiscrete):
                    assert len(action) == len(space.nvec), f"MultiDiscrete action {key} wrong length"
                    for i, (a, n) in enumerate(zip(action, space.nvec)):
                        assert 0 <= a < n, f"MultiDiscrete action {key}[{i}] out of bounds"

            print(f"    ✓ {space_name} constraints satisfied")


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\nRunning performance benchmark...")

    action_space = spaces.Dict({
        'move': spaces.Discrete(8),
        'shoot': spaces.Discrete(2),
        'aim': spaces.Box(low=-1, high=1, shape=(2,)),
        'inventory': spaces.MultiDiscrete([5, 3, 7])
    })

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,))

    policy = DictActorCriticPolicy(
        action_space=action_space,
        observation_space=observation_space,
        net_arch={'pi': [128, 128], 'vf': [128, 128]},
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    batch_sizes = [32, 128, 512]
    n_iterations = 100

    for batch_size in batch_sizes:
        obs = torch.randn(batch_size, 64, device=policy.device)

        # Warm up
        for _ in range(10):
            with torch.no_grad():
                policy.forward(obs)

        # Benchmark
        import time
        start_time = time.time()

        for _ in range(n_iterations):
            with torch.no_grad():
                actions, log_probs = policy.forward(obs)

        end_time = time.time()
        avg_time = (end_time - start_time) / n_iterations
        throughput = batch_size / avg_time

        print(f"  Batch size {batch_size:3d}: {avg_time * 1000:.2f}ms/batch, {throughput:.0f} samples/sec")


def main():
    """Main test runner"""
    print("=" * 60)
    print("Testing DictActorCriticPolicy")
    print("=" * 60)

    # Create test instance
    test_suite = TestDictActorCriticPolicy()
    test_suite.setup_method()

    # Run tests
    tests = [
        test_suite.test_initialization,
        test_suite.test_invalid_action_space,
        test_suite.test_forward_pass,
        test_suite.test_prediction,
        test_suite.test_evaluation,
        test_suite.test_value_function,
        test_suite.test_optimizer_creation,
        test_suite.test_gradient_flow,
        test_suite.test_action_space_constraints,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        run_performance_benchmark()
    else:
        print(f"❌ {failed} tests failed")

    print("=" * 60)


if __name__ == "__main__":
    main()