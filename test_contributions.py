#!/usr/bin/env python3
"""
Test des 3 contributions PhD — sans ROS2, sans GPU obligatoire.

Usage :
    python test_contributions.py

Ce script vérifie :
  1. Multi-Scale CNN + Spatial Attention  (Contribution 1)
  2. GeoCommQMIX — Communication apprise  (Contribution 2)
  3. Geo-ICM — Curiosité intrinsèque      (Contribution 3)
  + StateEncoder CNN                      (Correction hypernetwork)
  + Intégration QMixLocalNetwork
"""

import sys
import os
import traceback

import torch
import torch.nn as nn
import numpy as np

# ── Chemin vers les modules du projet ────────────────────────────────────────
_pkg = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'src', 'ree_exploration_qmix', 'ree_exploration_qmix'
)
sys.path.insert(0, _pkg)

from networks import (
    CNNEncoder,
    MultiScaleCNNEncoder,
    CommModule,
    QMixLocalNetwork,
    StateEncoder,
)
from geo_icm import GeoICM, GeoICMForwardModel

# ── Helpers d'affichage ───────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_pass = 0
_fail = 0

def ok(msg):
    global _pass
    _pass += 1
    print(f"  {GREEN}✓{RESET} {msg}")

def fail(msg, detail=""):
    global _fail
    _fail += 1
    print(f"  {RED}✗{RESET} {msg}")
    if detail:
        print(f"    {RED}→ {detail}{RESET}")

def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def check(condition, msg_ok, msg_fail, detail=""):
    if condition:
        ok(msg_ok)
    else:
        fail(msg_fail, detail)


# ─────────────────────────────────────────────────────────────────────────────
#  CONTRIBUTION 1 — Multi-Scale CNN + Spatial Attention
# ─────────────────────────────────────────────────────────────────────────────
section("CONTRIBUTION 1 — Multi-Scale CNN + Spatial Attention")

try:
    B, C, H = 4, 6, 20
    encoder = MultiScaleCNNEncoder(input_channels=C, hidden_dim=64,
                                   local_size=H, regional_size=H)
    encoder.eval()

    local_map    = torch.randn(B, C, H, H)
    regional_map = torch.randn(B, C, H, H)

    # --- Test 1.1 : shape de sortie ----------------------------------------
    with torch.no_grad():
        out = encoder(local_map, regional_map)
    check(out.shape == (B, 64),
          f"Sortie shape correcte : {out.shape}",
          f"Shape incorrecte : attendu ({B}, 64), obtenu {out.shape}")

    # --- Test 1.2 : 25 tokens spatiaux depuis get_feature_maps() -----------
    local_maps = encoder.local_encoder.get_feature_maps(local_map)
    check(local_maps.shape == (B, 64, 5, 5),
          f"Feature maps locales : {local_maps.shape}  (25 tokens spatiaux)",
          f"Shape feature maps incorrecte : {local_maps.shape}")

    n_tokens = local_maps.shape[2] * local_maps.shape[3]
    check(n_tokens == 25,
          f"Nombre de tokens spatiaux : {n_tokens}  (attendu 25)",
          f"Tokens insuffisants : {n_tokens} ≠ 25")

    # --- Test 1.3 : attention non-dégénérée (la sortie varie avec l'entrée) -
    local_map2    = torch.randn(B, C, H, H)
    regional_map2 = torch.randn(B, C, H, H)
    with torch.no_grad():
        out2 = encoder(local_map2, regional_map2)
    diff = (out - out2).abs().mean().item()
    check(diff > 1e-4,
          f"Attention non-dégénérée : diff moyenne = {diff:.5f}",
          f"Attention dégénérée : les sorties sont identiques (diff={diff:.2e})")

    # --- Test 1.4 : encode_local_only (fallback sans régionale) -------------
    with torch.no_grad():
        out_local = encoder.encode_local_only(local_map)
    check(out_local.shape == (B, 64),
          f"encode_local_only shape : {out_local.shape}",
          f"encode_local_only shape incorrecte : {out_local.shape}")

    # --- Test 1.5 : multi-scale apporte de l'information régionale ----------
    diff_vs_local = (out - out_local).abs().mean().item()
    check(diff_vs_local > 1e-4,
          f"Multi-scale ≠ local seul (diff = {diff_vs_local:.5f})",
          f"Multi-scale = local seul — l'attention régionale est sans effet")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  {YELLOW}ℹ{RESET}  Paramètres MultiScaleCNNEncoder : {n_params:,}")

except Exception as e:
    fail("Contribution 1 — exception non gérée", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  CONTRIBUTION 2 — GeoCommQMIX (Communication apprise)
# ─────────────────────────────────────────────────────────────────────────────
section("CONTRIBUTION 2 — GeoCommQMIX (Communication apprise)")

try:
    B, N, feat_dim, comm_dim = 4, 4, 64, 32
    comm = CommModule(feature_dim=feat_dim, comm_dim=comm_dim, num_agents=N)
    comm.eval()

    own_features = torch.randn(B, feat_dim)
    other_msgs   = torch.randn(B, N - 1, comm_dim)

    # --- Test 2.1 : encode_message shape ------------------------------------
    with torch.no_grad():
        msg = comm.encode_message(own_features)
    check(msg.shape == (B, comm_dim),
          f"encode_message shape : {msg.shape}",
          f"Shape incorrecte : attendu ({B}, {comm_dim}), obtenu {msg.shape}")

    # --- Test 2.2 : process_messages shape ----------------------------------
    with torch.no_grad():
        enhanced = comm.process_messages(own_features, other_msgs)
    check(enhanced.shape == (B, feat_dim),
          f"process_messages shape : {enhanced.shape}",
          f"Shape incorrecte : attendu ({B}, {feat_dim}), obtenu {enhanced.shape}")

    # --- Test 2.3 : gate ∈ [0, 1] -------------------------------------------
    with torch.no_grad():
        own_msg         = comm.message_encoder(own_features).unsqueeze(1)
        attended_raw, _ = comm.msg_attention(own_msg, other_msgs, other_msgs)
        attended        = comm.attn_norm(attended_raw.squeeze(1))
        gate_val        = comm.gate(torch.cat([own_features, attended], dim=-1))
    gate_min = gate_val.min().item()
    gate_max = gate_val.max().item()
    check(0.0 <= gate_min and gate_max <= 1.0,
          f"Gate ∈ [0, 1] : min={gate_min:.4f}  max={gate_max:.4f}",
          f"Gate hors [0,1] : min={gate_min:.4f}  max={gate_max:.4f}")

    # --- Test 2.4 : la communication modifie les features ------------------
    diff_comm = (enhanced - own_features).abs().mean().item()
    check(diff_comm > 1e-6,
          f"Communication modifie les features (diff = {diff_comm:.6f})",
          f"La communication n'a aucun effet (diff = {diff_comm:.2e})")

    # --- Test 2.5 : messages différents → sorties différentes ---------------
    other_msgs2 = torch.randn(B, N - 1, comm_dim)
    with torch.no_grad():
        enhanced2 = comm.process_messages(own_features, other_msgs2)
    diff_msgs = (enhanced - enhanced2).abs().mean().item()
    check(diff_msgs > 1e-4,
          f"Messages différents → sorties différentes (diff = {diff_msgs:.5f})",
          f"La communication ignore les messages reçus (diff = {diff_msgs:.2e})")

    # --- Test 2.6 : messages nuls → influence réduite ----------------------
    zero_msgs = torch.zeros(B, N - 1, comm_dim)
    with torch.no_grad():
        enhanced_zero = comm.process_messages(own_features, zero_msgs)
    diff_zero = (enhanced_zero - own_features).abs().mean().item()
    print(f"  {YELLOW}ℹ{RESET}  Messages nuls → influence = {diff_zero:.6f} "
          f"(devrait être faible si le gate se ferme)")

    n_params = sum(p.numel() for p in comm.parameters())
    print(f"  {YELLOW}ℹ{RESET}  Paramètres CommModule : {n_params:,}")

except Exception as e:
    fail("Contribution 2 — exception non gérée", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  CONTRIBUTION 3 — Geo-ICM (Curiosité intrinsèque coopérative)
# ─────────────────────────────────────────────────────────────────────────────
section("CONTRIBUTION 3 — Geo-ICM (Curiosité intrinsèque coopérative)")

try:
    B, feat_dim, action_dim = 8, 64, 8
    icm = GeoICM(feature_dim=feat_dim, action_dim=action_dim,
                 hidden_dim=128, curiosity_weight=0.1)
    icm.eval()

    features_t  = torch.randn(B, feat_dim)
    features_t1 = torch.randn(B, feat_dim)
    actions     = torch.randint(0, action_dim, (B,))

    # --- Test 3.1 : predict_next_features shape ----------------------------
    with torch.no_grad():
        pred = icm.forward_model.predict_next_features(features_t, actions)
    check(pred.shape == (B, feat_dim),
          f"predict_next_features shape : {pred.shape}",
          f"Shape incorrecte : attendu ({B}, {feat_dim}), obtenu {pred.shape}")

    # --- Test 3.2 : compute_curiosity_reward shape -------------------------
    with torch.no_grad():
        curiosity = icm.compute_curiosity_reward(features_t, actions, features_t1,
                                                  normalize=False)
    check(curiosity.shape == (B,),
          f"Curiosité shape : {curiosity.shape}",
          f"Shape incorrecte : attendu ({B},), obtenu {curiosity.shape}")

    # --- Test 3.3 : curiosité ≥ 0 ------------------------------------------
    check((curiosity >= 0).all(),
          f"Curiosité ≥ 0 : min={curiosity.min().item():.6f}",
          f"Curiosité négative : min={curiosity.min().item():.6f}")

    # --- Test 3.4 : poids β appliqué ----------------------------------------
    raw_curiosity = icm.forward_model.compute_curiosity_reward(
        features_t, actions, features_t1
    )
    expected  = icm.curiosity_weight * raw_curiosity
    diff_beta = (curiosity - expected).abs().max().item()
    check(diff_beta < 1e-5,
          f"Poids β={icm.curiosity_weight} correctement appliqué",
          f"Poids β incorrect : diff={diff_beta:.2e}")

    # --- Test 3.5 : Welford running stats se mettent à jour ----------------
    count_before = icm.curiosity_count.item()
    _ = icm.compute_curiosity_reward(features_t, actions, features_t1)
    count_after  = icm.curiosity_count.item()
    check(count_after > count_before,
          f"Running stats mises à jour : count {count_before:.0f} → {count_after:.0f}",
          f"Running stats n'évoluent pas : count={count_after}")

    # --- Test 3.6 : normalisation active après 10+ samples -----------------
    for _ in range(3):
        _ = icm.compute_curiosity_reward(features_t, actions, features_t1)
    cur_norm   = icm.compute_curiosity_reward(features_t, actions, features_t1,
                                               normalize=True)
    cur_nonorm = icm.compute_curiosity_reward(features_t, actions, features_t1,
                                               normalize=False)
    diff_norm = (cur_norm - cur_nonorm).abs().mean().item()
    count_now = icm.curiosity_count.item()
    check(count_now > 10,
          f"Normalisation activable : curiosity_count={count_now:.0f} > 10",
          f"Pas assez d'échantillons pour activer la normalisation : {count_now:.0f}")
    print(f"  {YELLOW}ℹ{RESET}  Diff normalisé/brut = {diff_norm:.5f}")

    # --- Test 3.7 : loss ICM scalaire et différentiable --------------------
    icm.train()
    features_t_grad = features_t.detach().requires_grad_(True)
    loss = icm.compute_loss(features_t_grad, actions, features_t1)
    check(loss.shape == torch.Size([]),
          f"Loss ICM scalaire : {loss.item():.6f}",
          f"Loss ICM non scalaire : shape={loss.shape}")
    loss.backward()
    check(features_t_grad.grad is not None,
          "Gradients calculables (backprop OK)",
          "Pas de gradient — le forward model n'est pas différentiable")

    # --- Test 3.8 : la loss diminue après quelques étapes ------------------
    icm.train()
    optimizer = torch.optim.Adam(icm.parameters(), lr=1e-3)
    loss_before = icm.compute_loss(features_t, actions, features_t1).item()
    for _ in range(30):
        optimizer.zero_grad()
        l = icm.compute_loss(features_t, actions, features_t1)
        l.backward()
        optimizer.step()
    loss_after = icm.compute_loss(features_t, actions, features_t1).item()
    check(loss_after < loss_before,
          f"Loss ICM diminue après 30 étapes : {loss_before:.6f} → {loss_after:.6f}",
          f"Loss ICM n'a pas diminué : {loss_before:.6f} → {loss_after:.6f}")

    n_params = sum(p.numel() for p in icm.parameters())
    print(f"  {YELLOW}ℹ{RESET}  Paramètres GeoICM : {n_params:,}  (~13k attendus)")

except Exception as e:
    fail("Contribution 3 — exception non gérée", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  BONUS — StateEncoder CNN (remplacement flatten 60k)
# ─────────────────────────────────────────────────────────────────────────────
section("BONUS — StateEncoder CNN (hypernetwork −99.7% params)")

try:
    B, C, H = 4, 6, 100
    se = StateEncoder(input_channels=C)
    se.eval()

    state = torch.randn(B, C, H, H)
    with torch.no_grad():
        encoded = se(state)

    check(encoded.shape == (B, StateEncoder.STATE_ENCODED_DIM),
          f"StateEncoder shape : {encoded.shape}",
          f"Shape incorrecte : attendu ({B}, 256), obtenu {encoded.shape}")

    n_params = sum(p.numel() for p in se.parameters())
    # Hypernetwork naïf avec Linear(60000, 64) sur 4 sous-réseaux ≈ 11.6M params
    # StateEncoder (1.4M) + hypernetwork (70k) = 1.48M total → −87% vs ancienne arch.
    naive_hyper = (60_000 * 64 + 64) * 4  # 4 Linear(60k→64) dans l'hypernetwork
    total_new   = n_params + 70_401        # StateEncoder + nouveau hypernetwork
    saving_pct  = (1 - total_new / naive_hyper) * 100
    check(n_params < naive_hyper,
          f"Params StateEncoder : {n_params:,}  "
          f"(total système : {total_new:,} vs {naive_hyper:,} naïf, −{saving_pct:.0f}%)",
          f"StateEncoder dépasse le naïf : {n_params:,} > {naive_hyper:,}")

    check((encoded >= 0).all(),
          "Sortie ReLU ≥ 0 (activation finale correcte)",
          "Sortie contient des valeurs négatives — ReLU final manquant ?")

except Exception as e:
    fail("StateEncoder — exception non gérée", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  INTÉGRATION — QMixLocalNetwork avec toutes les contributions
# ─────────────────────────────────────────────────────────────────────────────
section("INTÉGRATION — QMixLocalNetwork (multi-scale + communication)")

try:
    B, C, H, N, n_act = 4, 6, 20, 4, 8
    net = QMixLocalNetwork(
        input_shape=(C, H, H),
        num_actions=n_act,
        hidden_dim=64,
        local_obs_size=H,
        use_multi_scale=True,
        use_comm=True,
        comm_dim=32,
        num_agents=N
    )
    net.eval()

    local_map    = torch.randn(B, C, H, H)
    regional_map = torch.randn(B, C, H, H)
    position     = torch.rand(B, 2)
    recv_msgs    = torch.randn(B, N - 1, 32)

    # --- Test intégration complète ------------------------------------------
    with torch.no_grad():
        q_vals = net(local_map, position, regional_map, recv_msgs)
    check(q_vals.shape == (B, n_act),
          f"Q-values shape (multi-scale + comm) : {q_vals.shape}",
          f"Shape incorrecte : attendu ({B}, {n_act}), obtenu {q_vals.shape}")

    # --- Test encode() pour ICM --------------------------------------------
    with torch.no_grad():
        features = net.encode(local_map, regional_map)
    check(features.shape == (B, 64),
          f"encode() shape (pour Geo-ICM) : {features.shape}",
          f"Shape encode() incorrecte : {features.shape}")

    # --- Test sans communication -------------------------------------------
    with torch.no_grad():
        q_no_comm = net(local_map, position, regional_map, received_messages=None)
    check(q_no_comm.shape == (B, n_act),
          f"Forward sans comm shape : {q_no_comm.shape}",
          f"Erreur forward sans comm : {q_no_comm.shape}")

    # --- ICM sur les features de QMixLocalNetwork -------------------------
    icm2 = GeoICM(feature_dim=64, action_dim=n_act, curiosity_weight=0.1)
    icm2.eval()
    features2  = net.encode(local_map, regional_map)
    features2b = net.encode(torch.randn(B, C, H, H), regional_map)
    actions2   = torch.randint(0, n_act, (B,))
    with torch.no_grad():
        curiosity2 = icm2.compute_curiosity_reward(
            features2.detach(), actions2, features2b.detach(), normalize=False
        )
    check(curiosity2.shape == (B,) and (curiosity2 >= 0).all(),
          f"ICM sur features QMixLocalNetwork : shape={curiosity2.shape}, "
          f"min={curiosity2.min():.4f}",
          f"ICM incompatible avec les features du réseau Q")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  {YELLOW}ℹ{RESET}  Paramètres QMixLocalNetwork (multi-scale+comm) : {n_params:,}")

except Exception as e:
    fail("Intégration QMixLocalNetwork — exception non gérée", traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  RÉSUMÉ
# ─────────────────────────────────────────────────────────────────────────────
total = _pass + _fail
print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  RÉSUMÉ : {_pass}/{total} tests réussis{RESET}")
if _fail == 0:
    print(f"{GREEN}{BOLD}  Toutes les contributions sont opérationnelles.{RESET}")
else:
    print(f"{RED}{BOLD}  {_fail} test(s) échoué(s) — voir détails ci-dessus.{RESET}")
print(f"{BOLD}{'═'*60}{RESET}\n")

sys.exit(0 if _fail == 0 else 1)
