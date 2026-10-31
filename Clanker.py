"""
Usage:
    python clanker.py train        # train and save clanker_ppo.zip
    python clanker.py eval         # load clanker_ppo.zip and test vs bots
    python clanker.py all          # train, then eval

Make sure the Showdown server is running first:
    node pokemon-showdown start --no-security
    
"""
import asyncio
import random
import sys

import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from poke_env import AccountConfiguration
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.data.gen_data import GenData
from poke_env.environment import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import (
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BATTLE_FORMAT = "gen4anythinggoes"
N_FEATURES = 12
NUM_ENVS = 4
TOTAL_STEPS = 98_304 
MODEL_PATH = "clanker_ppo"
N_EVAL_BATTLES = 500


# 1. RL Team (typical player team going into Cynthia)
rl_team = """Gyarados @ Leftovers  
Ability: Intimidate  
Level: 58
EVs: 1 Atk / 1 SpD / 252 Spe  
Jolly Nature  
- Dragon Dance  
- Waterfall  
- Ice Fang  
- Taunt  

Togekiss @ Leftovers  
Ability: Serene Grace  
Level: 57
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
- Air Slash  
- Nasty Plot  
- Roost  
- Aura Sphere  

Lucario @ Life Orb  
Ability: Inner Focus  
Level: 58
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Swords Dance  
- Close Combat  
- Extreme Speed  
- Ice Punch  

Magnezone @ Choice Scarf  
Ability: Magnet Pull  
Level: 56
EVs: 4 HP / 252 SpA / 252 Spe  
Timid Nature  
- Thunderbolt  
- Flash Cannon  
- Signal Beam  
- Magnet Rise  

Garchomp @ Choice Scarf  
Ability: Sand Veil  
Level: 60
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Earthquake  
- Outrage  
- Stone Edge  
- Fire Blast  

Roserade @ Focus Sash  
Ability: Natural Cure  
Level: 56
EVs: 4 HP / 252 SpA / 252 Spe  
Timid Nature  
- Leaf Storm  
- Sludge Bomb  
- Shadow Ball  
- Extrasensory"""

# 2. Cynthia (Diamond/Pearl, no EVs)
cynthia_dp_team = """


Spiritomb @ Leftovers  
Ability: Pressure  
Level: 61
EVs: 1 HP
IVs: 0 Atk  
- Dark Pulse  
- Psychic  
- Silver Wind  
- Embargo  

Roserade @ Life Orb  
Ability: Natural Cure  
Level: 60
EVs: 1 HP
IVs: 0 Atk  
- Energy Ball  
- Sludge Bomb  
- Shadow Ball  
- Extrasensory  

Gastrodon @ Rindo Berry  
Ability: Sticky Hold  
Level: 60
EVs: 1 HP
- Muddy Water  
- Earthquake  
- Stone Edge  
- Sludge Bomb  

Lucario @ Choice Specs  
Ability: Inner Focus  
Level: 63
EVs: 1 HP
- Aura Sphere  
- Dragon Pulse  
- Psychic  
- Extreme Speed  

Milotic @ Flame Orb  
Ability: Marvel Scale  
Level: 63
EVs: 1 HP
IVs: 0 Atk  
- Surf  
- Ice Beam  
- Mirror Coat  
- Aqua Ring  

Garchomp @ Yache Berry  
Ability: Sand Veil  
Level: 66
EVs: 1 HP
- Dragon Rush  
- Earthquake  
- Brick Break  
- Giga Impact  
"""

AGENT_TEAM = rl_team
CYNTHIA_TEAM = cynthia_dp_team
# ----------------------------------------------------------------------------
# Cynthia's logic (not really that smart honestly)
# ----------------------------------------------------------------------------
class CynthiaPlayer(Player):
    """
    Rule-based opponent (highest scoring move goes)

    Does not switch voluntarily. 
    Scoring = base power * STAB * type effectiveness * accuracy. 
    If tied, move selected at random.
    """
    def choose_move(self, battle):
        """
        Choose highest scoring move, or random order if no moves exist
        """
        if not battle.available_moves:
            return self.choose_random_move(battle)

        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        scored = [(self._score(m, me, opp), m) for m in battle.available_moves]
        best = max(s for s, _ in scored)
        return self.create_order(random.choice([m for s, m in scored if s == best]))

    def _score(self, move, me, opp):
        """
        Heuristic value of using particular move

        - Status moves: 30 if the opponent has no status, otherwise -10

        - Damaging moves: See above

        - Moves with 0x effectiveness heavily penalized (-100)
        """
        if move.category.name == "STATUS":
            if opp.status is not None:
                return -10
            return 30

        mult = opp.damage_multiplier(move)
        if mult == 0: # For moves that do not affect opponent
            return -100
        stab = 1.5 if move.type in me.types else 1.0
        return move.base_power * stab * mult * move.accuracy


# ----------------------------------------------------------------------------
# Observation embedding
# ----------------------------------------------------------------------------
def embed_battle(battle: AbstractBattle) -> np.ndarray:
    """
    The first task is to interpret the battle. To do this, we need to summarize the current state
    of the game into a vector (12 components in this case). The components are as follows:
    1. Base power of each move (4 components) and Dmg Multiplier of each move against opponent (4 components)
    2. Fraction of team eliminated (2 values)
    3. HP fraction of each active pokemon (2 values)
    """

    moves_base_power = -np.ones(4) # 4 Base power components
    moves_dmg_multiplier = np.ones(4) # 4 Dmg Multiplier components

    # 1. Evaluate every single move's power and multiplier 
    for i, move in enumerate(battle.available_moves):

        # 1a. Move base power
        moves_base_power[i] = move.base_power / 100

        # 1b. Move multiplier based on type
        if battle.opponent_active_pokemon is not None:
            moves_dmg_multiplier[i] = move.type.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
                battle.opponent_active_pokemon.type_2,
                type_chart=GenData.from_gen(battle.gen).type_chart,
            )

    # 2. Calculated fainted pokemon fraction
    fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
    fainted_mon_opponent = (len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6)

    # 3. HPs 
    our_hp = (battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0)
    opp_hp = (battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0.0)
    return np.concatenate(
        [
            moves_base_power,
            moves_dmg_multiplier,
            [fainted_mon_team, fainted_mon_opponent],
            [our_hp, opp_hp],
        ],
        dtype=np.float32,
    )


randomizer = random.randint(0,9999) # For naming purposes

# ----------------------------------------------------------------------------
# Training environment
# ----------------------------------------------------------------------------
class ExampleEnv(SinglesEnv):
    """
    Gymnasium environment for 1v1 battles on a locally run Showdown server.

    poke-env allows two agents to play against each other. 
    """
    possible_agents: list[str]
    observation_spaces: dict[str, Box]

    def __init__(self, **kwargs):
        """
        Create env and declare agent's observation space as a Box of N_FEATURES floats
        in [-1, 4] (4 comes from 4x type multiplier)
        """
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @classmethod

    def create_env(cls) -> Monitor:
        """
        Factory used by SubprocVecEnv

        Creates env, pairs with fixed SimpleHeuristicsPlayer using SingleAgentWrapper (only one model trained)
        Wraps in Monitor so SB3 logs episode reward and length


        """
        env = cls(
            battle_format=BATTLE_FORMAT,
            log_level=40,
            open_timeout=None,
            team=AGENT_TEAM,

        )
        opponent = SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            team=CYNTHIA_TEAM,
            start_listening=False,
        )
        return Monitor(SingleAgentWrapper(env, opponent))

    def calc_reward(self, battle) -> float:
        """
        Reward signal for PPO. Computed by poke-env helper as change in battle value between turns

        +2 for each defeated Pokemon
        +1 per HP fraction dealt
        +0.5 for status inflicted
        +30 for winning (vice versa for losing)
        
        Big bonus to reward winning over everything else
        """
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
        """
        Observation hook called by poke-env each step; delegates to shared
        embed_battle() function so training and evaluation match
        """
        return embed_battle(battle)


# ----------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------
class FeaturesExtractor(BaseFeaturesExtractor):
    """
    Fetches numeric observation from dictionary that poke-env spits out

    poke-env observations are dicts with "observation" (previously defined 12 features) and
    "action_mask" (legal moves)

    MLP should only account for the former, so this class passes it through unchanged and 
    declares features_dim so SB3 builds the network with the right input size.
    """
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    PPO actor-critic with illegal-action masking

    Uses two hidden layers of 64 units for actor and critic. Before sampling,
    logits for illegal actions set to -inf (to prevent them from happening). 
    Used to prevent wasted experience on moves like switching to fainted Pokemon

    Mask stashed on policy in forward()/evaluate_actions() because SB3's distrubtion hook
    doesn't receive observation
    """
    _mask: torch.Tensor | None

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[64, 64],
            features_extractor_class=FeaturesExtractor,
        )

    def forward(self, obs: dict[str, torch.Tensor], deterministic=False):
        """
        Sample (or if deterministic, argmax) an action; returns (actions, values, log_probs)
        Saves mask for distribution step
        """

        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)  # type: ignore

    def evaluate_actions(self, obs: dict[str, torch.Tensor], actions):
        """
        Used by PPO's update step to re-score stored actions
        Also stores mask so same masking applies during learning

        """
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        """
        Add 0 to legal logits and -inf to illegal ones, then create action distribution
        """
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + mask)


# ----------------------------------------------------------------------------
# Trained player
# ----------------------------------------------------------------------------
class EvolvedClanker(Player):
    """
    poke-env player whose decisions come from a trained policy.

    Training happens in Gymnasium env, now it needs to play normally.
    This class bridges the two by embedding each battle with the same 
    function used in training and querying the policy
    """
    def __init__(self, policy, *args, **kwargs):
        """
        Policy is trained MaskedActorCriticPolicy (ppo policy)
        """
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(self, battle):
        """
        Embed battle, mask illegal actions, return policy's highest-probability action as battle order
        """
        if battle.wait:
            return DefaultBattleOrder()

        obs = embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))  # type: ignore

        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict, deterministic=True)
        action = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action, battle)  # type: ignore


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------
def train() -> PPO:
    """
    Train a PPO agent for TOTAL_STEPS and save it to MODEL_PATH.
 
    Runs NUM_ENVS Showdown environments in parallel processes (SubprocVecEnv).
    Each PPO update uses 3072 total steps (n_steps per env x NUM_ENVS).
    Must be called from under `if __name__ == "__main__":` because worker
    processes re-import this file. Returns the trained PPO object.
    """

    env = SubprocVecEnv([ExampleEnv.create_env for _ in range(NUM_ENVS)])
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=3072 // NUM_ENVS,
        batch_size=128,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )
    ppo.learn(TOTAL_STEPS)
    ppo.save(MODEL_PATH)
    env.close()
    return ppo

async def evaluate(ppo: PPO):

    """
    Play the trained agent against each scripted bot and print win rates.
 
    Each opponent gets N_EVAL_BATTLES games. Players get readable usernames so
    they can be told apart in the Showdown client at localhost:8000, and the
    agent's battle history is reset between opponents so counts don't mix.
    Prints "name: win% (wins/total)" for each.
    """
    ppo.policy.set_training_mode(False)
    agent = EvolvedClanker(
        policy=ppo.policy,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
        team=AGENT_TEAM,
        account_configuration=AccountConfiguration(f"Clanker{randomizer}",None)
    )

    opps = {
        "RandomBot": RandomPlayer,
        "MaxBaseBot": MaxBasePowerPlayer,
        "HeuristicBot": SimpleHeuristicsPlayer,
        "CynthiaBot": CynthiaPlayer
    }

    print("--- Win rates vs bots ---")
    for name, cls in opps.items():
        opp = cls(battle_format=BATTLE_FORMAT, 
                  max_concurrent_battles=10, 
                  team=CYNTHIA_TEAM,
                  account_configuration=AccountConfiguration(f"{name}{randomizer}", None))
        print(agent.username, opp.username)

        agent.reset_battles()
        await agent.battle_against(opp, n_battles=N_EVAL_BATTLES)
        win_rate = round(100 * agent.n_won_battles / max(agent.n_finished_battles, 1))
        print(f"{cls.__name__}: {win_rate}%")



if __name__ == "__main__":
    # Required. SubprocVecEnv worker processes reimport this. Without it, workers try to train themselves.
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("train", "all"):
        ppo = train()
    else:
        ppo = PPO.load(MODEL_PATH, device="cpu")

    if mode in ("eval", "all"):
        asyncio.run(evaluate(ppo))