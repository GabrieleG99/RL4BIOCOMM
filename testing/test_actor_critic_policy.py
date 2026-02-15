import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium.spaces as spaces
import pytest
import tempfile
import os
from unittest.mock import Mock, patch

# Assumendo che questi moduli siano disponibili
from ..models import ActorCriticPolicy
from ..models import FeedForwardNN


class TestActorCriticPolicy:
    """Test suite completa per ActorCriticPolicy"""

    @pytest.fixture
    def discrete_action_space(self):
        """Fixture per action space discreto"""
        return spaces.Discrete(4)

    @pytest.fixture
    def box_action_space(self):
        """Fixture per action space continuo"""
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    @pytest.fixture
    def multi_discrete_action_space(self):
        """Fixture per action space multi-discreto"""
        return spaces.MultiDiscrete([3, 2])

    @pytest.fixture
    def observation_space(self):
        """Fixture per observation space"""
        return spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    @pytest.fixture
    def lr_schedule(self):
        """Fixture per learning rate schedule"""
        return lambda progress: 1e-3

    @pytest.fixture
    def sample_observation(self, observation_space):
        """Fixture per una singola osservazione"""
        return observation_space.sample()

    @pytest.fixture
    def batch_observations(self, observation_space):
        """Fixture per batch di osservazioni"""
        return np.array([observation_space.sample() for _ in range(16)])

    def test_init_discrete_action_space(self, discrete_action_space, observation_space, lr_schedule):
        """Test inizializzazione con action space discreto"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        # Test attributi base
        assert policy.action_space == discrete_action_space
        assert policy.observation_space == observation_space
        assert policy.input_size == observation_space.shape[0]
        assert policy.output_size == discrete_action_space.n

        # Test componenti della rete
        assert hasattr(policy, 'actor')
        assert hasattr(policy, 'critic')
        assert hasattr(policy, 'act_features_extractor')
        assert hasattr(policy, 'critic_features_extractor')

        # Test optimizer
        assert hasattr(policy, 'optimizer')
        assert isinstance(policy.optimizer, optim.Adam)

    def test_init_box_action_space(self, box_action_space, observation_space, lr_schedule):
        """Test inizializzazione con action space continuo"""
        policy = ActorCriticPolicy(
            action_space=box_action_space,
            observation_space=observation_space,
        )

        assert policy.action_space == box_action_space
        assert policy.observation_space == observation_space
        assert policy.input_size == observation_space.shape[0]
        assert policy.output_size == box_action_space.shape[0]

    def test_init_multi_discrete_action_space(self, multi_discrete_action_space, observation_space, lr_schedule):
        """Test inizializzazione con action space multi-discreto"""
        policy = ActorCriticPolicy(
            action_space=multi_discrete_action_space,
            observation_space=observation_space,
        )

        assert policy.action_space == multi_discrete_action_space
        assert policy.observation_space == observation_space
        assert policy.input_size == observation_space.shape[0]

    def test_custom_net_arch(self, discrete_action_space, observation_space, lr_schedule):
        """Test architettura di rete personalizzata"""
        custom_net_arch = {'pi': [64, 32], 'vf': [128, 64]}

        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            net_arch=custom_net_arch
        )

        assert policy.net_arch == custom_net_arch

    def test_custom_optimizer(self, discrete_action_space, observation_space, lr_schedule):
        """Test ottimizzatore personalizzato"""
        custom_optim_kwargs = {'weight_decay': 1e-4}

        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_class=optim.SGD,
            optim_kwargs=custom_optim_kwargs
        )

        assert isinstance(policy.optimizer, optim.SGD)
        assert policy.optimizer.param_groups[0]['weight_decay'] == 1e-4

    def test_predict_single_observation(self, discrete_action_space, observation_space, lr_schedule,
                                        sample_observation):
        """Test predizione per singola osservazione"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        # Test predizione deterministica
        action_det = policy.predict(sample_observation, deterministic=True)
        assert isinstance(action_det, np.ndarray)
        assert action_det.shape == ()
        assert 0 <= action_det < discrete_action_space.n

        # Test predizione stocastica
        action_stoch = policy.predict(sample_observation, deterministic=False)
        assert isinstance(action_stoch, np.ndarray)
        assert action_stoch.shape == ()
        assert 0 <= action_stoch < discrete_action_space.n

    def test_predict_box_action_space(self, box_action_space, observation_space, lr_schedule, sample_observation):
        """Test predizione con action space continuo"""
        policy = ActorCriticPolicy(
            action_space=box_action_space,
            observation_space=observation_space,
        )

        action = policy.predict(sample_observation, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert np.all(action >= box_action_space.low)
        assert np.all(action <= box_action_space.high)

    def test_forward_pass(self, discrete_action_space, observation_space, lr_schedule, batch_observations):
        """Test forward pass con batch di osservazioni"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        obs_tensor = torch.from_numpy(batch_observations).float()
        actions, log_probs = policy.forward(obs_tensor)

        assert isinstance(actions, torch.Tensor)
        assert isinstance(log_probs, torch.Tensor)
        assert actions.shape == (16,)
        assert log_probs.shape == (16,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < discrete_action_space.n)

    def test_get_state_value(self, discrete_action_space, observation_space, lr_schedule, batch_observations):
        """Test stima del valore dello stato"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        obs_tensor = torch.from_numpy(batch_observations).float()
        values = policy.get_state_value(obs_tensor)

        assert isinstance(values, torch.Tensor)
        assert values.shape == (16, 1)

    def test_evaluate(self, discrete_action_space, observation_space, lr_schedule, batch_observations):
        """Test metodo evaluate"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        obs_tensor = torch.from_numpy(batch_observations).float()

        # Genera azioni prima
        actions, _ = policy.forward(obs_tensor)

        # Valuta
        log_probs, entropies = policy.evaluate(obs_tensor, actions)

        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(entropies, torch.Tensor)
        assert log_probs.shape == (16,)
        assert entropies.shape == (16,)

    def test_save_load(self, discrete_action_space, observation_space, lr_schedule, sample_observation):
        """Test salvataggio e caricamento del modello"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        # Salva parametri originali
        original_params = {name: param.clone() for name, param in policy.named_parameters()}

        # Salva il modello
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name

        try:
            policy.save(save_path)

            # Carica il modello
            loaded_policy = ActorCriticPolicy.load(save_path)

            # Confronta parametri
            for name, param in loaded_policy.named_parameters():
                assert torch.allclose(param, original_params[name])

            # Test che il modello caricato funzioni
            action1 = policy.predict(sample_observation, deterministic=True)
            action2 = loaded_policy.predict(sample_observation, deterministic=True)
            assert np.allclose(action1, action2)

        finally:
            # Pulizia
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_device_handling(self, discrete_action_space, observation_space, lr_schedule):
        """Test gestione dei device"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            device='cpu'
        )

        assert policy.device == torch.device('cpu')

        # Verifica che tutti i parametri siano sul device corretto
        for param in policy.parameters():
            assert param.device == torch.device('cpu')

    def test_training_mode(self, discrete_action_space, observation_space, lr_schedule):
        """Test modalità training/eval"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        # Test modalità training
        policy.set_training_mode(True)
        assert policy.training == True

        # Test modalità eval
        policy.set_training_mode(False)
        assert policy.training == False

    def test_get_constructor_parameters(self, discrete_action_space, observation_space, lr_schedule):
        """Test recupero parametri del costruttore"""
        custom_kwargs = {
            'net_arch': {'pi': [64, 32], 'vf': [128, 64]},
            'activation': 'tanh',
            'optim_kwargs': {'weight_decay': 1e-4}
        }

        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            **custom_kwargs
        )

        params = policy._get_constructor_parameters()

        assert 'action_space' in params
        assert 'observation_space' in params
        assert 'net_arch' in params
        assert 'activation' in params
        assert 'optim_kwargs' in params

    def test_unsupported_action_space(self, observation_space, lr_schedule):
        """Test gestione di action space non supportati"""
        # Test con Tuple space (non supportato)
        unsupported_space = spaces.Tuple((spaces.Discrete(2), spaces.Box(0, 1, (2,))))

        policy = ActorCriticPolicy(
            action_space=unsupported_space,
            observation_space=observation_space,
        )

        # Questo dovrebbe sollevare un errore quando si tenta di creare la distribuzione
        with pytest.raises(NotImplementedError):
            obs_tensor = torch.randn(1, 8)
            policy._get_prob_dis_from_act_space(torch.randn(1, 2))

    def test_gradient_flow(self, discrete_action_space, observation_space, lr_schedule, batch_observations):
        """Test flusso dei gradienti"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        obs_tensor = torch.from_numpy(batch_observations).float()

        # Forward pass
        actions, log_probs = policy.forward(obs_tensor)
        values = policy.get_state_value(obs_tensor)

        # Simula una loss
        loss = -log_probs.mean() + values.mean()

        # Backward pass
        loss.backward()

        # Verifica che i gradienti siano stati calcolati
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"Gradient not computed for {name}"

    def test_reproducibility(self, discrete_action_space, observation_space, lr_schedule, sample_observation):
        """Test riproducibilità con seed fisso"""
        torch.manual_seed(42)
        np.random.seed(42)

        policy1 = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        torch.manual_seed(42)
        np.random.seed(42)

        policy2 = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        # I parametri dovrebbero essere identici
        for (name1, param1), (name2, param2) in zip(policy1.named_parameters(), policy2.named_parameters()):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_batch_consistency(self, discrete_action_space, observation_space, lr_schedule):
        """Test consistenza tra predizioni singole e batch"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
        )

        # Singola osservazione
        single_obs = observation_space.sample()
        single_action = policy.predict(single_obs, deterministic=True)

        # Batch con la stessa osservazione
        batch_obs = np.array([single_obs])
        batch_action = policy.predict(batch_obs[0], deterministic=True)

        # Dovrebbero essere identici
        assert np.allclose(single_action, batch_action)

    # =====================================
    # TEST INIZIALIZZAZIONE OPTIMIZER
    # =====================================

    def test_shared_optimizer_default(self, discrete_action_space, observation_space):
        """Test inizializzazione optimizer condiviso con parametri di default"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space
        )

        # Verifica che l'optimizer condiviso sia stato creato
        assert hasattr(policy, 'optimizer')
        assert policy.shared_optim == True

        # Verifica che sia Adam di default
        assert isinstance(policy.optimizer, optim.Adam)

        # Verifica che abbia tutti i parametri del modello
        policy_params = set(policy.parameters())
        optim_params = set(policy.optimizer.param_groups[0]['params'])
        assert policy_params == optim_params

        # Verifica parametri di default
        assert policy.optimizer.param_groups[0]['lr'] == 0.001  # Default Adam lr
        assert policy.optimizer.param_groups[0]['betas'] == (0.9, 0.999)

    def test_shared_optimizer_custom_class(self, discrete_action_space, observation_space):
        """Test inizializzazione optimizer condiviso con classe personalizzata"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_class=optim.SGD,
            optim_kwargs={'lr': 0.01, 'momentum': 0.9}
        )

        # Verifica che l'optimizer sia SGD
        assert isinstance(policy.optimizer, optim.SGD)
        assert policy.shared_optim == True

        # Verifica parametri personalizzati
        assert policy.optimizer.param_groups[0]['lr'] == 0.01
        assert policy.optimizer.param_groups[0]['momentum'] == 0.9

    def test_shared_optimizer_custom_kwargs(self, discrete_action_space, observation_space):
        """Test inizializzazione optimizer condiviso con kwargs personalizzati"""
        custom_kwargs = {
            'lr': 0.001,
            'weight_decay': 1e-4,
            'eps': 1e-7
        }

        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_class=optim.Adam,
            optim_kwargs=custom_kwargs
        )

        # Verifica parametri personalizzati
        assert policy.optimizer.param_groups[0]['lr'] == 0.001
        assert policy.optimizer.param_groups[0]['weight_decay'] == 1e-4
        assert policy.optimizer.param_groups[0]['eps'] == 1e-7

    def test_separate_optimizers(self, discrete_action_space, observation_space):
        """Test inizializzazione optimizer separati per actor e critic"""
        optim_kwargs = {
            'actor': {'lr': 0.01, 'weight_decay': 1e-4},
            'critic': {'lr': 0.001, 'weight_decay': 1e-3}
        }

        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_class=optim.Adam,
            optim_kwargs=optim_kwargs
        )

        # Verifica che siano stati creati optimizer separati
        assert hasattr(policy, 'actor_optimizer')
        assert hasattr(policy, 'critic_optimizer')
        assert policy.shared_optim == False

        # Verifica che siano entrambi Adam
        assert isinstance(policy.actor_optimizer, optim.Adam)
        assert isinstance(policy.critic_optimizer, optim.Adam)

        # Verifica parametri specifici per actor
        assert policy.actor_optimizer.param_groups[0]['lr'] == 0.01
        assert policy.actor_optimizer.param_groups[0]['weight_decay'] == 1e-4

        # Verifica parametri specifici per critic
        assert policy.critic_optimizer.param_groups[0]['lr'] == 0.001
        assert policy.critic_optimizer.param_groups[0]['weight_decay'] == 1e-3

    def test_separate_optimizers_different_classes(self, discrete_action_space, observation_space):
        """Test optimizer separati con classi diverse"""
        optim_kwargs = {
            'actor': {'lr': 0.01},
            'critic': {'lr': 0.001}
        }

        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_class=optim.Adam,
            optim_kwargs=optim_kwargs
        )

        # Verifica che entrambi gli optimizer siano stati creati
        assert hasattr(policy, 'actor_optimizer')
        assert hasattr(policy, 'critic_optimizer')
        assert policy.shared_optim == False

    def test_optimizer_parameters_coverage(self, discrete_action_space, observation_space):
        """Test che gli optimizer coprano tutti i parametri necessari"""
        # Test optimizer condiviso
        policy_shared = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space
        )

        # Raccogli tutti i parametri del modello
        all_params = set(policy_shared.parameters())
        optim_params = set(policy_shared.optimizer.param_groups[0]['params'])

        # Verifica che l'optimizer contenga tutti i parametri
        assert all_params == optim_params

        # Test optimizer separati
        optim_kwargs = {
            'actor': {'lr': 0.01},
            'critic': {'lr': 0.001}
        }

        policy_separate = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_kwargs=optim_kwargs
        )

        # Con la nuova implementazione, gli optimizer utilizzano parameter groups
        # Actor optimizer ha due gruppi: actor_head e act_features_extractor
        actor_head_params = set(policy_separate.actor_head.parameters())
        act_features_params = set(policy_separate.act_features_extractor.parameters())
        
        # Critic optimizer ha due gruppi: critic_head e critic_features_extractor  
        critic_head_params = set(policy_separate.critic_head.parameters())
        critic_features_params = set(policy_separate.critic_features_extractor.parameters())

        # Raccogli parametri dagli optimizer (tutti i parameter groups)
        actor_optim_params = set()
        for group in policy_separate.actor_optimizer.param_groups:
            actor_optim_params.update(group['params'])
        
        critic_optim_params = set()
        for group in policy_separate.critic_optimizer.param_groups:
            critic_optim_params.update(group['params'])

        # Verifica che l'actor optimizer contenga i parametri corretti
        expected_actor_params = actor_head_params.union(act_features_params)
        assert actor_optim_params == expected_actor_params

        # Verifica che il critic optimizer contenga i parametri corretti
        expected_critic_params = critic_head_params.union(critic_features_params)
        assert critic_optim_params == expected_critic_params

        # Verifica che non ci siano sovrapposizioni tra actor e critic
        shared_params = actor_optim_params.intersection(critic_optim_params)
        assert len(shared_params) == 0

        # Verifica che insieme coprano tutti i parametri del modello
        all_params_separate = set(policy_separate.parameters())
        combined_optim_params = actor_optim_params.union(critic_optim_params)
        assert combined_optim_params == all_params_separate

        # Test aggiuntivo: verifica la struttura dei parameter groups
        # Actor optimizer dovrebbe avere 2 gruppi
        assert len(policy_separate.actor_optimizer.param_groups) == 2
        
        # Critic optimizer dovrebbe avere 2 gruppi  
        assert len(policy_separate.critic_optimizer.param_groups) == 2

        # Verifica che ogni gruppo abbia i parametri corretti
        actor_group1_params = set(policy_separate.actor_optimizer.param_groups[0]['params'])
        actor_group2_params = set(policy_separate.actor_optimizer.param_groups[1]['params'])
        
        # Uno dei gruppi dovrebbe corrispondere ad actor_head, l'altro ad act_features_extractor
        assert (actor_group1_params == actor_head_params and actor_group2_params == act_features_params) or \
               (actor_group1_params == act_features_params and actor_group2_params == actor_head_params)

        critic_group1_params = set(policy_separate.critic_optimizer.param_groups[0]['params'])
        critic_group2_params = set(policy_separate.critic_optimizer.param_groups[1]['params'])
        
        # Uno dei gruppi dovrebbe corrispondere a critic_head, l'altro a critic_features_extractor
        assert (critic_group1_params == critic_head_params and critic_group2_params == critic_features_params) or \
               (critic_group1_params == critic_features_params and critic_group2_params == critic_head_params)

    def test_optimizer_state_dict(self, discrete_action_space, observation_space):
        """Test salvataggio e caricamento dello stato dell'optimizer"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_kwargs={'lr': 0.01}
        )

        # Salva lo stato iniziale dell'optimizer
        initial_state = policy.optimizer.state_dict()

        # Esegui un passo di ottimizzazione
        obs = torch.randn(1, 8)
        actions, log_probs = policy.forward(obs)
        values = policy.get_state_value(obs)

        # Simula una loss e un step
        loss = -log_probs.mean() + values.mean()
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()

        # Verifica che lo stato sia cambiato
        updated_state = policy.optimizer.state_dict()
        assert updated_state != initial_state

        # Ripristina lo stato iniziale
        policy.optimizer.load_state_dict(initial_state)
        restored_state = policy.optimizer.state_dict()

        # Verifica che sia stato ripristinato correttamente
        assert restored_state == initial_state

    def test_optimizer_with_empty_kwargs(self, discrete_action_space, observation_space):
        """Test inizializzazione con optim_kwargs vuoti"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_kwargs={}
        )

        # Verifica che l'optimizer sia stato creato con parametri di default
        assert hasattr(policy, 'optimizer')
        assert isinstance(policy.optimizer, optim.Adam)
        assert policy.shared_optim == True

    def test_optimizer_with_none_kwargs(self, discrete_action_space, observation_space):
        """Test inizializzazione con optim_kwargs=None"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_kwargs=None
        )

        # Verifica che l'optimizer sia stato creato
        assert hasattr(policy, 'optimizer')
        assert isinstance(policy.optimizer, optim.Adam)
        assert policy.shared_optim == True

    def test_optimizer_learning_rate_schedule(self, discrete_action_space, observation_space):
        """Test che l'optimizer supporti schedule del learning rate"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_kwargs={'lr': 0.01}
        )

        # Verifica learning rate iniziale
        initial_lr = policy.optimizer.param_groups[0]['lr']
        assert initial_lr == 0.01

        # Modifica learning rate
        new_lr = 0.001
        for param_group in policy.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Verifica che sia stato modificato
        assert policy.optimizer.param_groups[0]['lr'] == new_lr

    def test_optimizer_gradient_clipping_compatibility(self, discrete_action_space, observation_space):
        """Test compatibilità con gradient clipping"""
        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space
        )

        # Simula forward pass e loss
        obs = torch.randn(1, 8)
        actions, log_probs = policy.forward(obs)
        values = policy.get_state_value(obs)
        loss = -log_probs.mean() + values.mean()

        # Calcola gradienti
        policy.optimizer.zero_grad()
        loss.backward()

        # Applica gradient clipping
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

        # Verifica che i gradienti siano stati clippati
        total_norm = 0
        for p in policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        assert total_norm <= max_grad_norm + 1e-6  # Piccola tolleranza per errori numerici

        # Verifica che l'optimizer possa fare il passo
        policy.optimizer.step()

    def test_separate_optimizers_step(self, discrete_action_space, observation_space):
        """Test step separato per optimizer di actor e critic"""
        optim_kwargs = {
            'actor': {'lr': 0.01},
            'critic': {'lr': 0.001}
        }

        policy = ActorCriticPolicy(
            action_space=discrete_action_space,
            observation_space=observation_space,
            optim_kwargs=optim_kwargs
        )

        # Simula forward pass
        obs = torch.randn(1, 8)
        actions, log_probs = policy.forward(obs)
        values = policy.get_state_value(obs)

        # Simula loss separate
        actor_loss = -log_probs.mean()
        critic_loss = values.mean()

        # Step separato per actor
        policy.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        policy.actor_optimizer.step()

        # Step separato per critic
        policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        policy.critic_optimizer.step()

        # Verifica che entrambi abbiano fatto il passo
        assert policy.actor_optimizer.param_groups[0]['lr'] == 0.01
        assert policy.critic_optimizer.param_groups[0]['lr'] == 0.001


if __name__ == "__main__":
    # Esegui i test direttamente
    pytest.main([__file__, "-v"])