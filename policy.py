"""
The network that decides where each box should go.

It has to score every possible move. Rather than learn 18,000 unrelated scores,
it learns how to describe a box and how to describe a spot, then scores a pair
by how well the two descriptions match. That way, what it learns about one spot
carries over to similar ones, which is what makes this trainable at all.

The boxes go through an attention layer so the network sees them as a set, and
the surface of the pile goes through a small CNN.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from environment import ROTATIONS


class PackingPolicy(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 n_rotations: int = 6, embed_dim: int = 64, n_heads: int = 4, **kwargs):
        self.n_rot = n_rotations
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_boxes = observation_space["boxes"].shape[0]
        self.grid = observation_space["heightmap"].shape[0]
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule) -> None:
        d = self.embed_dim

        # (l, w, h) of one box in one orientation -> a vector
        self.box_encoder = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d))
        # ...then let the boxes look at each other, so each one is embedded in
        # the context of what else is left to pack (this is the attention
        # encoder the papers use, and what makes variable box counts work).
        self.box_attn = nn.MultiheadAttention(d, self.n_heads, batch_first=True)
        self.box_norm = nn.LayerNorm(d)

        # The heightmap -> one vector per (x, y) cell. Convolutions mean a
        # cell is described by its local surface shape, and that description
        # is shared across every position on the floor.
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(1, d, 3, padding=1), nn.ReLU(),
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(),
            nn.Conv2d(d, d, 1),
        )

        # container scale -> a context vector added to BOTH the box and the
        # position embeddings, so "how big is this container" can modulate
        # what counts as a good box and a good spot
        self.container_encoder = nn.Sequential(nn.Linear(3, d), nn.ReLU(), nn.Linear(d, d))
        self.value_net = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.action_dist = MaskableCategoricalDistribution(int(self.action_space.n))

        # the 6 orientations as an index, so every (box, rotation) pair can be
        # built in one go rather than in a loop
        perms = torch.tensor(ROTATIONS[: self.n_rot], dtype=torch.long)
        self.register_buffer("perms", perms)

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _logits_values(self, obs):
        boxes = obs["boxes"].float()            # (B, N, 3)
        heightmap = obs["heightmap"].float()    # (B, G, G)
        B = boxes.shape[0]
        d = self.embed_dim

        # every box in every orientation -> (B, N*R, 3)
        idx = self.perms.view(1, 1, self.n_rot, 3).expand(B, self.n_boxes, self.n_rot, 3)
        oriented = boxes.unsqueeze(2).expand(B, self.n_boxes, self.n_rot, 3).gather(3, idx)
        tokens = oriented.reshape(B, self.n_boxes * self.n_rot, 3)

        ctx = self.container_encoder(obs["container"].float())  # (B, d)

        box_emb = self.box_encoder(tokens) + ctx.unsqueeze(1)
        attended, _ = self.box_attn(box_emb, box_emb, box_emb, need_weights=False)
        box_emb = self.box_norm(box_emb + attended)             # (B, N*R, d)

        pos_emb = self.pos_encoder(heightmap.unsqueeze(1))      # (B, d, G, G)
        pos_emb = pos_emb.flatten(2).transpose(1, 2)            # (B, G*G, d), index = x*G + y
        pos_emb = pos_emb + ctx.unsqueeze(1)

        # score every (box-rotation, position) pair, then flatten in exactly
        # the order encode_action() uses
        logits = torch.bmm(box_emb, pos_emb.transpose(1, 2)) / math.sqrt(d)
        logits = logits.reshape(B, -1)

        pooled = torch.cat([box_emb.mean(dim=1), pos_emb.mean(dim=1), ctx], dim=1)
        return logits, self.value_net(pooled)

    def _distribution(self, logits, action_masks):
        dist = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    # --- the four entry points SB3 actually calls ---

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self._logits_values(obs)
        dist = self._distribution(logits, action_masks)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self._logits_values(obs)
        dist = self._distribution(logits, action_masks)
        return values, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs, action_masks=None):
        logits, _ = self._logits_values(obs)
        return self._distribution(logits, action_masks)

    def predict_values(self, obs):
        return self._logits_values(obs)[1]
