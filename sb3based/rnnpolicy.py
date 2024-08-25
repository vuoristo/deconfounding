from stable_baselines3.common.policies import ActorCriticPolicy

from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch as th
from torchvision import transforms
from gym import spaces
from torch import nn
import tree

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule, TensorDict
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space, preprocess_obs
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class KeyCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        obs_keys: Optional[List[str]] = None,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if obs_keys is not None and not key in obs_keys:
                continue
            if key not in observation_space.spaces.keys():
                raise ValueError(f"Key {key} is missing from the observation space")
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


class SimpleMLP(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        input_size = observation_space["obs"].shape[0]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.ELU(),
            nn.Linear(64, features_dim),
            nn.ELU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)


class RNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Union[
            List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None
        ] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        hidden_size: int = 250,
        action_embedding_dim: int = 250,
        rnn_hidden_size: int = 1000,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.action_embedding_dim = action_embedding_dim
        self.features_extractor = self.make_features_extractor()
        if hasattr(action_space, "n"):
            self.policy_action_embedding = nn.Embedding(
                action_space.n, self.action_embedding_dim
            )
        else:
            self.policy_action_embedding = nn.Linear(
                action_space.low.size, self.action_embedding_dim
            )
        self.rnn_prenetwork = nn.Sequential(
            nn.Linear(self.features_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
        )
        self.rnn = nn.GRU(
            self.hidden_size + self.action_embedding_dim, self.rnn_hidden_size
        )
        self.policy_state_extractor = self.make_features_extractor()
        self.policy_output_projection = nn.Sequential(
            nn.Linear(self.rnn_hidden_size + self.features_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
        )
        self.pi_parameters = [
            *self.rnn_prenetwork.parameters(),
            *self.features_extractor.parameters(),
            *self.policy_state_extractor.parameters(),
            *self.rnn.parameters(),
            *self.policy_action_embedding.parameters(),
            *self.policy_output_projection.parameters(),
        ]
        if hasattr(action_space, "n"):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=self.hidden_size
            )
            self.pi_parameters.extend([*self.action_net.parameters()])
        else:
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.hidden_size
            )
            self.pi_parameters.extend([*self.action_net.parameters(), self.log_std])

        for m in self.modules():
            init_weights_tf2(m)

    def init_hidden(self, batch_size):
        return th.zeros(1, batch_size, self.rnn_hidden_size).to(self.device)

    def forward_encoder(self, rnn_state, obs_t, action_tm1):
        T, B = action_tm1.shape[:2]
        obs_t = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), obs_t)
        # Classes: BaseModel <- BasePolicy <- ActorCriticPolicy <- RNNPolicy
        # Calls super on ActorCriticPolicy to skip the conflicting extract_features there
        state_features = super(ActorCriticPolicy, self).extract_features(obs_t, self.features_extractor)
        action_features = self.policy_action_embedding(action_tm1).reshape(T * B, -1)
        x = self.rnn_prenetwork(state_features)
        x = th.cat([x, action_features], dim=-1).reshape(T, B, -1)
        x, new_rnn_state = self.rnn(x, rnn_state)
        x = x.reshape(-1, x.shape[-1])
        return x, new_rnn_state

    def forward(self, rnn_state, obs_t, action_tm1):
        hidden, rnn_state = self.forward_encoder(rnn_state, obs_t, action_tm1)
        obs_t = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), obs_t)
        state_features = super(ActorCriticPolicy, self).extract_features(obs_t, self.policy_state_extractor)
        x = th.cat(
            [state_features, hidden.reshape(state_features.shape[0], -1)],
            dim=-1,
        )
        x = self.policy_output_projection(x)
        distribution = self._get_action_dist_from_latent(
            x.reshape(-1, self.hidden_size)
        )
        return distribution, rnn_state


def init_weights_tf2(m):
    # Match TF2 initializations
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)
    if type(m) == nn.GRU:
        for k, v in m.named_parameters():
            if "weight_ih" in k:
                nn.init.xavier_uniform_(v.data)
            elif "weight_hh" in k:
                nn.init.orthogonal_(v.data)
            elif "bias" in k:
                nn.init.zeros_(v.data)


class DeconfoundedRNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Union[
            List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None
        ] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        latent_dim: int = 64,
        hidden_size: int = 128,
        action_embedding_dim: int = 128,
        rnn_hidden_size: int = 500,
        latent_pred_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.action_embedding_dim = action_embedding_dim

        self.decoder_obs_keys = features_extractor_kwargs["obs_keys"]
        self.latent_pred_kwargs = latent_pred_kwargs

        self.enc_state_extractor = self.make_features_extractor()
        if hasattr(action_space, "n"):
            self.enc_action_embedding = nn.Embedding(
                action_space.n, self.action_embedding_dim
            )
        else:
            self.enc_action_embedding = nn.Linear(
                action_space.low.size, self.action_embedding_dim
            )
        self.rnn_prenetwork = nn.Sequential(
            nn.Linear(self.features_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
        )
        self.rnn = nn.GRU(
            self.hidden_size + self.action_embedding_dim, self.rnn_hidden_size
        )
        self.latent_param_projection = nn.Linear(self.rnn_hidden_size, latent_dim * 2)

        # Decoder
        decoder_params = self.build_decoder()

        self.enc_parameters = [
            *self.enc_state_extractor.parameters(),
            *self.enc_action_embedding.parameters(),
            *self.rnn_prenetwork.parameters(),
            *self.rnn.parameters(),
            *self.latent_param_projection.parameters(),
            *decoder_params,
        ]

        # Policy
        self.policy_state_extractor = self.make_features_extractor()
        self.policy_output_projection = nn.Sequential(
            nn.Linear(self.features_dim + latent_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
        )
        self.pi_parameters = [
            *self.policy_state_extractor.parameters(),
            *self.policy_output_projection.parameters(),
        ]
        if hasattr(action_space, "n"):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=self.hidden_size
            )
            self.pi_parameters.extend([*self.action_net.parameters()])
        else:
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.hidden_size
            )
            self.pi_parameters.extend([*self.action_net.parameters(), self.log_std])

        for m in self.modules():
            init_weights_tf2(m)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                latent_dim=self.latent_dim,
            )
        )
        return data

    def build_decoder(self):
        if hasattr(self.action_space, "n"):
            self.dec_action_embedding = nn.Embedding(
                self.action_space.n, self.action_embedding_dim
            )
        else:
            self.dec_action_embedding = nn.Linear(
                self.action_space.low.size, self.action_embedding_dim
            )
        self.dec_state_extractor = self.make_features_extractor()
        output_dim = self.observation_space["obs"].shape[0]
        self.state_decoder = nn.Sequential(
            nn.Linear(
                self.dec_state_extractor.features_dim
                + self.action_embedding_dim
                + self.latent_dim,
                self.hidden_size,
            ),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-3),
            nn.ELU(),
        )
        heads = {}
        for key in self.decoder_obs_keys:
            output_dim = spaces.utils.flatdim(self.observation_space[key])
            heads[key] = nn.Linear(self.hidden_size, output_dim)
        self.heads = nn.ModuleDict(heads)
        decoder_params = [
            *self.dec_action_embedding.parameters(),
            *self.dec_state_extractor.parameters(),
            *self.state_decoder.parameters(),
            *self.heads.parameters(),
        ]
        if self.latent_pred_kwargs is not None:
            latent_pred_size = self.latent_pred_kwargs['pred_size']
        else:
            latent_pred_size = 1
        self.debug_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, latent_pred_size),
        )
        decoder_params += self.debug_decoder.parameters()
        return decoder_params

    def init_hidden(self, batch_size):
        return th.zeros(1, batch_size, self.rnn_hidden_size).to(self.device)

    def forward_encoder(self, rnn_state, obs_t, action_tm1):
        T, B = action_tm1.shape[:2]
        obs_t = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), obs_t)
        x  = super(ActorCriticPolicy, self).extract_features(obs_t, self.enc_state_extractor)
        state_features = self.rnn_prenetwork(x)
        action_features = self.enc_action_embedding(action_tm1).reshape(T * B, -1)
        x = th.cat([state_features, action_features], dim=-1).reshape(T, B, -1)
        x, new_rnn_state = self.rnn(x, rnn_state)
        x = x.reshape(-1, x.shape[-1])
        latent_params = self.latent_param_projection(x)
        latent_mean = latent_params[:, : latent_params.shape[1] // 2]
        latent_std = th.exp(0.5 * latent_params[:, latent_params.shape[1] // 2 :])
        m = th.distributions.normal.Normal(
            latent_mean.reshape(T, B, -1),
            latent_std.reshape(T, B, -1),
        )
        return m, new_rnn_state

    def forward_policy(self, z, obs_t):
        obs_t = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), obs_t)
        state_features = super(ActorCriticPolicy, self).extract_features(obs_t, self.policy_state_extractor)
        x = th.cat(
            [state_features, z.reshape(state_features.shape[0], -1)],
            dim=-1,
        )
        x = self.policy_output_projection(x)
        return x

    def forward_decoder(self, z, obs_t, action_t):
        T, B = action_t.shape[:2]
        obs_t = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), obs_t)
        state_features = super(ActorCriticPolicy, self).extract_features(obs_t, self.dec_state_extractor)
        action_features = self.dec_action_embedding(action_t).reshape(T * B, -1)
        x = th.cat(
            [
                state_features,
                action_features,
                z.reshape(state_features.shape[0], -1),
            ],
            dim=-1,
        )
        x = self.state_decoder(x)
        outputs = {k: self.heads[k](x) for k in self.decoder_obs_keys}
        outputs = tree.map_structure(lambda x: x.reshape(T, B, -1), outputs)
        predicted_latent = self.debug_decoder(
            z.detach().reshape(-1, z.shape[-1])
        ).reshape(T, B, -1)
        return outputs, predicted_latent

    def forward(self, rnn_state, obs_t, action_tm1, z=None):
        if z is None:
            m, rnn_state = self.forward_encoder(rnn_state, obs_t, action_tm1)
            z = m.mean
        x = self.forward_policy(z.detach(), obs_t)
        distribution = self._get_action_dist_from_latent(
            x.reshape(-1, self.hidden_size)
        )
        return distribution, rnn_state


class RewardNet(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        hidden_size: int = 32,
        action_embedding_dim: int = 32,
        inputs: List[str] = ["obs", "action"],
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_embedding_dim = action_embedding_dim
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)
        discriminator_input_size = 0
        self.inputs = inputs
        if "action" in inputs:
            if hasattr(action_space, "n"):
                self.action_embedding = nn.Embedding(
                    action_space.n, self.action_embedding_dim
                )
            else:
                self.action_embedding = nn.Linear(
                    action_space.low.size, self.action_embedding_dim
                )
            discriminator_input_size += self.action_embedding_dim
        if "state" in inputs:
            discriminator_input_size += self.features_extractor.features_dim
        if "next_state" in inputs:
            discriminator_input_size += self.features_extractor.features_dim
        self.discriminator_network = nn.Sequential(
            nn.Linear(
                discriminator_input_size,
                self.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        for m in self.modules():
            init_weights_tf2(m)

    @property
    def device(self) -> th.device:
        return next(self.parameters()).device

    def forward(self, state, action, next_state, done):
        T, B = action.shape[:2]
        features = []
        if "state" in self.inputs:
            state = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), state)
            preprocessed_state = preprocess_obs(state, self.observation_space)
            state_features = self.features_extractor(preprocessed_state)
            features.append(state_features)
        if "action" in self.inputs:
            action_features = self.action_embedding(action).reshape(T * B, -1)
            features.append(action_features)
        if "next_state" in self.inputs:
            next_state = tree.map_structure(lambda x: x.reshape(-1, *x.shape[2:]), next_state)
            preprocessed_next_state = preprocess_obs(next_state, self.observation_space)
            next_state_features = self.features_extractor(preprocessed_next_state)
            features.append(next_state_features)
        x = th.cat(features, dim=-1).reshape(T, B, -1)
        x = self.discriminator_network(x)
        return x

    def reward(self, state, action, next_state, done):
        with th.no_grad():
            action = th.tensor(action, device=self.device).unsqueeze(0)
            state = tree.map_structure(lambda x: th.tensor(x, device=self.device).unsqueeze(0), state)
            next_state = tree.map_structure(lambda x: th.tensor(x, device=self.device).unsqueeze(0), next_state)
            logits = self.forward(state, action, next_state, done).squeeze(-1)
            r = -th.nn.functional.logsigmoid(-logits).squeeze(0).detach().cpu().numpy()
        return r
