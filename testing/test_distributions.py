import torch
import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete
from ..dists import MultiCategoricalDistribution

class TestMultiCategoricalDistribution:
    
    def setup_method(self):
        """Setup eseguito prima di ogni test"""
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_initialization(self):
        """Test inizializzazione base"""
        # Caso semplice: 2 dimensioni con 3 e 4 categorie
        action_space = MultiDiscrete([3, 4])
        batch_size = 5
        total_dims = sum(action_space.nvec)  # 3 + 4 = 7
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        assert len(dist.categorical_distributions) == 2
        assert dist.action_dims.tolist() == [3, 4]
        assert dist.logits.shape == (batch_size, total_dims)
        
    def test_initialization_single_dim(self):
        """Test con una singola dimensione"""
        action_space = MultiDiscrete([5])
        batch_size = 3
        logits = torch.randn(batch_size, 5)
        
        dist = MultiCategoricalDistribution(logits, action_space)
        
        assert len(dist.categorical_distributions) == 1
        assert dist.action_dims.tolist() == [5]
        
    def test_initialization_multiple_dims(self):
        """Test con multiple dimensioni"""
        action_space = MultiDiscrete([2, 3, 4, 5])
        batch_size = 2
        total_dims = sum(action_space.nvec)  # 2 + 3 + 4 + 5 = 14
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        assert len(dist.categorical_distributions) == 4
        assert dist.action_dims.tolist() == [2, 3, 4, 5]
        
    def test_sample_shape(self):
        """Test che sample() restituisca la forma corretta"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 5
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        samples = dist.sample()
        
        assert samples.shape == (batch_size, len(action_space.nvec))
        assert samples.dtype == torch.long
        
    def test_sample_values_in_range(self):
        """Test che i valori campionati siano nel range corretto"""
        action_space = MultiDiscrete([3, 4, 2])
        batch_size = 100
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        samples = dist.sample()
        
        # Verifica che ogni dimensione sia nel range corretto
        for i, max_val in enumerate(action_space.nvec):
            assert torch.all(samples[:, i] >= 0)
            assert torch.all(samples[:, i] < max_val)
            
    def test_log_prob_shape(self):
        """Test che log_prob() restituisca la forma corretta"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 5
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        actions = torch.tensor([[0, 1], [2, 3], [1, 0], [0, 2], [2, 1]])
        log_probs = dist.log_prob(actions)
        
        assert log_probs.shape == (batch_size, len(action_space.nvec))
        assert torch.all(log_probs <= 0)  # Le log-prob sono sempre <= 0
        
    def test_log_prob_values(self):
        """Test che i valori di log_prob siano ragionevoli"""
        action_space = MultiDiscrete([2, 2])
        batch_size = 1
        
        # Logits che favoriscono azioni [0, 1]
        logits = torch.tensor([[5.0, -5.0, -5.0, 5.0]])  # Prima dim: [5, -5], Seconda dim: [-5, 5]
        dist = MultiCategoricalDistribution(logits, action_space)
        
        # Test azione favorita
        favored_action = torch.tensor([[0, 1]])
        log_prob_favored = dist.log_prob(favored_action)
        
        # Test azione sfavorita
        unfavored_action = torch.tensor([[1, 0]])
        log_prob_unfavored = dist.log_prob(unfavored_action)
        
        # L'azione favorita dovrebbe avere log_prob maggiore
        assert torch.all(log_prob_favored > log_prob_unfavored)
        
    def test_entropy_shape(self):
        """Test che entropy() restituisca la forma corretta"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 5
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        entropies = dist.entropy()
        
        assert entropies.shape == (batch_size, len(action_space.nvec))
        assert torch.all(entropies >= 0)  # L'entropia è sempre >= 0
        
    def test_entropy_uniform_distribution(self):
        """Test entropia per distribuzione uniforme"""
        action_space = MultiDiscrete([4, 4])
        batch_size = 1
        
        # Logits uniformi (tutti zero)
        logits = torch.zeros(batch_size, 8)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        entropies = dist.entropy()
        expected_entropy = np.log(4, dtype=np.float32)  # ln(4) per distribuzione uniforme su 4 categorie
        
        assert torch.allclose(entropies, torch.tensor(expected_entropy), atol=1e-6)
        
    def test_entropy_deterministic_distribution(self):
        """Test entropia per distribuzione deterministica"""
        action_space = MultiDiscrete([2, 3])
        batch_size = 1
        
        # Logits molto sbilanciati
        logits = torch.tensor([[100.0, -100.0, 100.0, -100.0, -100.0]])
        dist = MultiCategoricalDistribution(logits, action_space)
        
        entropies = dist.entropy()
        
        # L'entropia dovrebbe essere vicina a 0 per distribuzioni deterministiche
        assert torch.all(entropies < 0.01)
        
    def test_gradient_flow(self):
        """Test che i gradienti fluiscano correttamente"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 2
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims, requires_grad=True)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        # Test gradient flow attraverso sample (dovrebbe funzionare con reparameterization)
        entropies = dist.entropy()
        loss = entropies.sum()
        loss.backward()
        
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)
        
    def test_consistency_sample_log_prob(self):
        """Test consistenza tra sample e log_prob"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 100
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        # Campiona azioni
        actions = dist.sample()
        
        # Calcola log_prob per le azioni campionate
        log_probs = dist.log_prob(actions)
        
        # Verifica che non ci siano NaN o inf
        assert torch.all(torch.isfinite(log_probs))
        assert torch.all(log_probs <= 0)
        
    def test_batch_independence(self):
        """Test che i batch siano indipendenti"""
        action_space = MultiDiscrete([2, 3])
        batch_size = 5
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        # Campiona più volte
        samples1 = dist.sample()
        samples2 = dist.sample()
        
        # I campioni dovrebbero essere potenzialmente diversi
        # (Non possiamo garantire che siano sempre diversi a causa della natura stocastica)
        assert samples1.shape == samples2.shape
        
    def test_edge_cases(self):
        """Test casi limite"""
        # Caso con una sola categoria per dimensione
        action_space = MultiDiscrete([1, 1])
        batch_size = 3
        logits = torch.randn(batch_size, 2)
        
        dist = MultiCategoricalDistribution(logits, action_space)
        samples = dist.sample()
        
        # Tutte le azioni dovrebbero essere [0, 0]
        expected = torch.zeros(batch_size, 2, dtype=torch.long)
        assert torch.equal(samples, expected)
        
        # L'entropia dovrebbe essere 0 (distribuzione deterministica)
        entropies = dist.entropy()
        assert torch.allclose(entropies, torch.zeros_like(entropies))
        
    def test_large_action_space(self):
        """Test con spazio delle azioni grande"""
        action_space = MultiDiscrete([10, 15, 20])
        batch_size = 2
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        samples = dist.sample()
        log_probs = dist.log_prob(samples)
        entropies = dist.entropy()
        
        # Verifica forme
        assert samples.shape == (batch_size, 3)
        assert log_probs.shape == (batch_size, 3)
        assert entropies.shape == (batch_size, 3)
        
        # Verifica range
        assert torch.all(samples[:, 0] < 10)
        assert torch.all(samples[:, 1] < 15)
        assert torch.all(samples[:, 2] < 20)
        
    def test_reproducibility(self):
        """Test riproducibilità con seed"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 5
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        
        # Primo campionamento
        torch.manual_seed(123)
        dist1 = MultiCategoricalDistribution(logits, action_space)
        samples1 = dist1.sample()
        
        # Secondo campionamento con stesso seed
        torch.manual_seed(123)
        dist2 = MultiCategoricalDistribution(logits, action_space)
        samples2 = dist2.sample()
        
        assert torch.equal(samples1, samples2)

# Test aggiuntivi per errori e casi di input non validi
class TestMultiCategoricalDistributionErrors:
    
    def test_invalid_action_dimensions(self):
        """Test comportamento con azioni non valide"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 2
        total_dims = sum(action_space.nvec)
        
        logits = torch.randn(batch_size, total_dims)
        dist = MultiCategoricalDistribution(logits, action_space)
        
        # Azioni con valori fuori range
        invalid_actions = torch.tensor([[3, 2], [1, 4]])  # 3 >= 3, 4 >= 4
        
        # Questo dovrebbe causare un errore o comportamento indefinito
        with pytest.raises((RuntimeError, IndexError, ValueError)):
            dist.log_prob(invalid_actions)
            
    def test_wrong_logits_shape(self):
        """Test con forma dei logits errata"""
        action_space = MultiDiscrete([3, 4])
        batch_size = 2
        
        # Logits con dimensioni sbagliate
        wrong_logits = torch.randn(batch_size, 5)  # Dovrebbe essere 7
        
        # Questo dovrebbe causare un errore
        with pytest.raises((RuntimeError, ValueError)):
            MultiCategoricalDistribution(wrong_logits, action_space)

if __name__ == "__main__":
    # Esegui i test
    pytest.main([__file__, "-v"])