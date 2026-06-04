# Continuous-Control NEXUS Results

Commit: `2944438df595f4e235651d6886f4e4e1daf45ab8`

## Environment Info

- python/platform: Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
- jax: 0.10.1
- mujoco: 3.9.0
- mujoco_playground: unknown

## Method Summary

Continuous-control NEXUS keeps the NEXUS meta-policy over interpretable skills, but replaces discrete option values with deterministic skill actors and skill critics. Skill critics train from shared rollouts on skill-specific rewards, while learned meta policies train on environment reward with masked max-Q bootstraps for NeSy.

## Exact Configs

- `configs/cartpole_balance_nesy.yaml`
- `configs/cartpole_balance_neural.yaml`
- `configs/cartpole_balance_symbolic.yaml`
- `configs/cheetah_run_nesy.yaml`
- `configs/cheetah_run_neural.yaml`
- `configs/flat_cartpole_balance.yaml`
- `configs/flat_cheetah_run.yaml`
- `configs/flat_go1_joystick.yaml`
- `configs/flat_hopper_hop.yaml`
- `configs/flat_panda_pick_cube.yaml`
- `configs/flat_walker_walk.yaml`
- `configs/go1_joystick_nesy.yaml`
- `configs/go1_joystick_neural.yaml`
- `configs/hopper_hop_nesy.yaml`
- `configs/hopper_hop_neural.yaml`
- `configs/panda_pick_cube_nesy.yaml`
- `configs/panda_pick_cube_neural.yaml`
- `configs/panda_pick_cube_symbolic.yaml`
- `configs/walker_walk_nesy.yaml`
- `configs/walker_walk_neural.yaml`

## Environment and Skill Table

| Environment | Skills | Rule summary |
| --- | --- | --- |
| CartpoleBalance | recover_balance, center_cart, damp_motion | angle/velocity recovery, cart centering, velocity damping |
| CheetahRun | accelerate_forward, stabilize_posture, energy_efficient_run | speed, posture, and control-cost rewards |
| WalkerWalk | stand_recover, walk_forward, stabilize_gait, energy_efficient | height/uprightness, target speed, gait stability, torque efficiency |
| HopperHop | stand_recover, hop_forward, stabilize_landing, energy_efficient | survival/uprightness plus env-reward tracking skills |
| PandaPickCube | reach_cube, grasp_cube, lift_cube, place_or_stabilize | distance-to-cube, grasp, height, target placement phases |
| Go1JoystickFlatTerrain | stand, track_velocity, turn, recover | stance, command tracking, yaw tracking, fall recovery |

## Gate Summary

- Final matrix: 60 runs loaded; 3 seeds for every required final config plus Go1 replacement configs.
- Main environment set for success gates: CartpoleBalance, CheetahRun, WalkerWalk, PandaPickCube, Go1JoystickFlatTerrain.
- HopperHop: failure case, not counted in the five-environment success set; final returned episodes remain near zero after the repair/tuning pass.
- Positive learning trend: pass on 5/5 main environments using returned episode return.
- Neural vs flat >=80%: pass on 4/5; failed/weak: Go1JoystickFlatTerrain: 0.743.
- NeSy vs flat >=70%: pass on 4/5; failed/weak: Go1JoystickFlatTerrain: 0.416.
- Panda sequential usage: pass; max final-window grasp usage 0.600, lift usage 0.811.
- Hopper best final mean returned episode return: 0.616; treat this as an explicit limitation.

## Main Performance

| run_id | seed | env_name | meta_policy_type | final_env_step | last10pct_mean/env/returned_episode_returns | last10pct_mean/returns/env_reward_mean | last10pct_mean/train/critic_abs_td |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | 1966080.0 | 268.9896240234375 | 0.2374550104141235 | 10.586471239725748 |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | 1966080.0 | 296.3910217285156 | 0.1596535791953404 | 17.31588617960612 |
| cartpole_balance_nesy | 2 | CartpoleBalance | nesy | 1966080.0 | 287.7936096191406 | 0.2053687175114949 | 28.380118052164715 |
| cartpole_balance_neural | 0 | CartpoleBalance | neural | 1966080.0 | 283.1806335449219 | 0.1254439701636632 | 74.89314778645833 |
| cartpole_balance_neural | 1 | CartpoleBalance | neural | 1966080.0 | 259.1449279785156 | 0.2285505880912145 | 28.714205423990887 |
| cartpole_balance_neural | 2 | CartpoleBalance | neural | 1966080.0 | 276.0589294433594 | 0.2501213798920313 | 13.399890581766764 |
| cartpole_balance_symbolic | 0 | CartpoleBalance | symbolic | 1966080.0 | 229.5834197998047 | 0.1749711632728576 | 45.91887283325195 |
| cartpole_balance_symbolic | 1 | CartpoleBalance | symbolic | 1966080.0 | 239.9002685546875 | 0.0668687336146831 | 2.383378267288208 |
| cartpole_balance_symbolic | 2 | CartpoleBalance | symbolic | 1966080.0 | 250.1024932861328 | 0.1638177434603373 | 23.283535639444988 |
| cheetah_run_nesy | 0 | CheetahRun | nesy | 4980736.0 | 114.55664825439452 | 0.2196794301271438 | 0.5161305516958237 |
| cheetah_run_nesy | 1 | CheetahRun | nesy | 4980736.0 | 163.6274871826172 | 0.1352488957345485 | 0.6135088205337524 |
| cheetah_run_nesy | 2 | CheetahRun | nesy | 4980736.0 | 158.624755859375 | 0.2945824787020683 | 0.8784646391868591 |
| cheetah_run_neural | 0 | CheetahRun | neural | 4980736.0 | 186.03179931640625 | 0.2093558609485626 | 0.9888445436954498 |
| cheetah_run_neural | 1 | CheetahRun | neural | 4980736.0 | 194.50218200683597 | 0.1934170313179493 | 0.6309453248977661 |
| cheetah_run_neural | 2 | CheetahRun | neural | 4980736.0 | 204.74581909179688 | 0.3838729709386825 | 0.9909232258796692 |
| flat_cartpole_balance | 0 | CartpoleBalance | flat | 1966080.0 | 246.43881225585935 | 0.1574906706809997 | 0.1104694580038388 |
| flat_cartpole_balance | 1 | CartpoleBalance | flat | 1966080.0 | 257.4691467285156 | 0.1483662476142247 | 0.0823588321606318 |
| flat_cartpole_balance | 2 | CartpoleBalance | flat | 1966080.0 | 234.11648559570312 | 0.2339973300695419 | 0.109990989168485 |
| flat_cheetah_run | 0 | CheetahRun | flat | 4980736.0 | 152.89764404296875 | 0.0355510832741856 | 0.0613432638347148 |
| flat_cheetah_run | 1 | CheetahRun | flat | 4980736.0 | 72.87635803222656 | 0.0878434982150793 | 0.0638617910444736 |
| flat_cheetah_run | 2 | CheetahRun | flat | 4980736.0 | 157.65399169921875 | 0.0871614841744303 | 0.0942287147045135 |
| flat_go1_joystick | 0 | Go1JoystickFlatTerrain | flat | 19922944.0 | 10.610016584396362 | 0.0108608257141895 | 0.0164700150489807 |
| flat_go1_joystick | 1 | Go1JoystickFlatTerrain | flat | 19922944.0 | 9.2728830575943 | 0.0097903835703618 | 0.0156637953477911 |
| flat_go1_joystick | 2 | Go1JoystickFlatTerrain | flat | 19922944.0 | 9.579612493515016 | 0.0097367044654674 | 0.0153707686113193 |
| flat_hopper_hop | 0 | HopperHop | flat | 4980736.0 | 0.1491418033838272 | 0.0 | 0.0006564733121194 |
| flat_hopper_hop | 1 | HopperHop | flat | 4980736.0 | 1.0323060750961304 | 1.3864689321962942e-05 | 0.0044389824906829 |
| flat_hopper_hop | 2 | HopperHop | flat | 4980736.0 | 0.2906038165092468 | 0.0 | 0.0011763279544538 |
| flat_panda_pick_cube | 0 | PandaPickCube | flat | 9961472.0 | 472.1655101776123 | 3.1534070670604706 | 5.3608295656740665 |
| flat_panda_pick_cube | 1 | PandaPickCube | flat | 9961472.0 | 525.935432434082 | 3.512413442134857 | 6.325421318411827 |
| flat_panda_pick_cube | 2 | PandaPickCube | flat | 9961472.0 | 470.8203201293945 | 3.1390401124954224 | 5.523809500038624 |
| flat_walker_walk | 0 | WalkerWalk | flat | 4980736.0 | 42.32115936279297 | 0.0918950084596872 | 0.0990589642897248 |
| flat_walker_walk | 1 | WalkerWalk | flat | 4980736.0 | 347.7015380859375 | 0.6644994020462036 | 0.1758182235062122 |
| flat_walker_walk | 2 | WalkerWalk | flat | 4980736.0 | 80.47119903564453 | 0.130263403058052 | 0.0710623562335968 |
| go1_joystick_nesy | 0 | Go1JoystickFlatTerrain | nesy | 19922944.0 | 5.050708591938019 | 0.005582333455095 | 0.7263928223401308 |
| go1_joystick_nesy | 1 | Go1JoystickFlatTerrain | nesy | 19922944.0 | 2.119040533900261 | 0.0032027481065597 | 0.8787817806005478 |
| go1_joystick_nesy | 2 | Go1JoystickFlatTerrain | nesy | 19922944.0 | 5.09638261795044 | 0.0062230229959823 | 0.8003984093666077 |
| go1_joystick_neural | 0 | Go1JoystickFlatTerrain | neural | 19922944.0 | 7.1764649748802185 | 0.0085789620934519 | 0.6250250153243542 |
| go1_joystick_neural | 1 | Go1JoystickFlatTerrain | neural | 19922944.0 | 5.429533511400223 | 0.0063702031329739 | 0.6548472363501787 |
| go1_joystick_neural | 2 | Go1JoystickFlatTerrain | neural | 19922944.0 | 9.287592113018036 | 0.0102067103725858 | 0.7223912216722965 |
| hopper_hop_nesy | 0 | HopperHop | nesy | 4980736.0 | 0.604293942451477 | 0.0 | 0.1405522152781486 |

## Baseline Comparison

| env_name | meta_policy_type | metric | final_mean | final_std | num_seeds | flat_final_mean | ratio_to_flat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CartpoleBalance | flat | env/returned_episode_returns | 246.00814819335935 | 11.682285698742334 | 3 | 246.00814819335935 | 1.0 |
| CartpoleBalance | nesy | env/returned_episode_returns | 284.39141845703125 | 14.01393333706196 | 3 | 246.00814819335935 | 1.1560243859626271 |
| CartpoleBalance | neural | env/returned_episode_returns | 272.7948303222656 | 12.345830989568377 | 3 | 246.00814819335935 | 1.1088853451628449 |
| CartpoleBalance | symbolic | env/returned_episode_returns | 239.862060546875 | 10.259590102593242 | 3 | 246.00814819335935 | 0.9750167313903212 |
| CheetahRun | flat | env/returned_episode_returns | 127.80933125813802 | 47.6327552356129 | 3 | 127.80933125813802 | 1.0 |
| CheetahRun | nesy | env/returned_episode_returns | 145.60296376546225 | 27.00300187539729 | 3 | 127.80933125813802 | 1.1392201362151424 |
| CheetahRun | neural | env/returned_episode_returns | 195.09326680501303 | 9.37100154558758 | 3 | 127.80933125813802 | 1.5264399311422798 |
| Go1JoystickFlatTerrain | flat | env/returned_episode_returns | 9.820837378501892 | 0.7004452876401054 | 3 | 9.820837378501892 | 1.0 |
| Go1JoystickFlatTerrain | nesy | env/returned_episode_returns | 4.088710581262906 | 1.705937162023039 | 3 | 9.820837378501892 | 0.4163301380199224 |
| Go1JoystickFlatTerrain | neural | env/returned_episode_returns | 7.297863533099492 | 1.9318921426499431 | 3 | 9.820837378501892 | 0.7430999263947425 |
| HopperHop | flat | env/returned_episode_returns | 0.4906838983297348 | 0.4743614782665753 | 3 | 0.4906838983297348 | 1.0 |
| HopperHop | nesy | env/returned_episode_returns | 0.3745262697339058 | 0.275380182209566 | 3 | 0.4906838983297348 | 0.7632740161410958 |
| HopperHop | neural | env/returned_episode_returns | 0.6163811286290487 | 0.2851037251105192 | 3 | 0.4906838983297348 | 1.2561674241343184 |
| PandaPickCube | flat | env/returned_episode_returns | 489.6404209136963 | 31.439597325906647 | 3 | 489.6404209136963 | 1.0 |
| PandaPickCube | nesy | env/returned_episode_returns | 449.273328145345 | 17.13788744219946 | 3 | 489.6404209136963 | 0.9175576789738396 |
| PandaPickCube | neural | env/returned_episode_returns | 472.1247984568278 | 11.23126380324678 | 3 | 489.6404209136963 | 0.9642275806719892 |
| PandaPickCube | symbolic | env/returned_episode_returns | 329.34014479319256 | 106.2520685251238 | 3 | 489.6404209136963 | 0.6726163337957792 |
| WalkerWalk | flat | env/returned_episode_returns | 156.831298828125 | 166.3954403146642 | 3 | 156.831298828125 | 1.0 |
| WalkerWalk | nesy | env/returned_episode_returns | 139.47784932454428 | 9.076469295384594 | 3 | 156.831298828125 | 0.8893495773276815 |
| WalkerWalk | neural | env/returned_episode_returns | 169.0746053059896 | 51.74581989034527 | 3 | 156.831298828125 | 1.0780667288312284 |

## Learning Trends

| run_id | seed | env_name | meta_policy_type | first10pct_mean | last10pct_mean | delta | positive_learning_trend |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | 0.0 | 268.9896240234375 | 268.9896240234375 | True |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | 0.0 | 296.3910217285156 | 296.3910217285156 | True |
| cartpole_balance_nesy | 2 | CartpoleBalance | nesy | 0.0 | 287.7936096191406 | 287.7936096191406 | True |
| cartpole_balance_neural | 0 | CartpoleBalance | neural | 0.0 | 283.1806335449219 | 283.1806335449219 | True |
| cartpole_balance_neural | 1 | CartpoleBalance | neural | 0.0 | 259.1449279785156 | 259.1449279785156 | True |
| cartpole_balance_neural | 2 | CartpoleBalance | neural | 0.0 | 276.0589294433594 | 276.0589294433594 | True |
| cartpole_balance_symbolic | 0 | CartpoleBalance | symbolic | 0.0 | 229.5834197998047 | 229.5834197998047 | True |
| cartpole_balance_symbolic | 1 | CartpoleBalance | symbolic | 0.0 | 239.9002685546875 | 239.9002685546875 | True |
| cartpole_balance_symbolic | 2 | CartpoleBalance | symbolic | 0.0 | 250.1024932861328 | 250.1024932861328 | True |
| cheetah_run_nesy | 0 | CheetahRun | nesy | 0.0 | 114.55664825439452 | 114.55664825439452 | True |
| cheetah_run_nesy | 1 | CheetahRun | nesy | 0.0 | 163.6274871826172 | 163.6274871826172 | True |
| cheetah_run_nesy | 2 | CheetahRun | nesy | 0.0 | 158.624755859375 | 158.624755859375 | True |
| cheetah_run_neural | 0 | CheetahRun | neural | 0.0 | 186.03179931640625 | 186.03179931640625 | True |
| cheetah_run_neural | 1 | CheetahRun | neural | 0.0 | 194.50218200683597 | 194.50218200683597 | True |
| cheetah_run_neural | 2 | CheetahRun | neural | 0.0 | 204.74581909179688 | 204.74581909179688 | True |
| flat_cartpole_balance | 0 | CartpoleBalance | flat | 0.0 | 246.43881225585935 | 246.43881225585935 | True |
| flat_cartpole_balance | 1 | CartpoleBalance | flat | 0.0 | 257.4691467285156 | 257.4691467285156 | True |
| flat_cartpole_balance | 2 | CartpoleBalance | flat | 0.0 | 234.11648559570312 | 234.11648559570312 | True |
| flat_cheetah_run | 0 | CheetahRun | flat | 0.0 | 152.89764404296875 | 152.89764404296875 | True |
| flat_cheetah_run | 1 | CheetahRun | flat | 0.0 | 72.87635803222656 | 72.87635803222656 | True |
| flat_cheetah_run | 2 | CheetahRun | flat | 0.0 | 157.65399169921875 | 157.65399169921875 | True |
| flat_go1_joystick | 0 | Go1JoystickFlatTerrain | flat | 0.0072595832316437 | 10.610016584396362 | 10.60275700116472 | True |
| flat_go1_joystick | 1 | Go1JoystickFlatTerrain | flat | 0.0038699673496012 | 9.2728830575943 | 9.269013090244698 | True |
| flat_go1_joystick | 2 | Go1JoystickFlatTerrain | flat | 0.0040758921250017 | 9.579612493515016 | 9.575536601390011 | True |
| flat_hopper_hop | 0 | HopperHop | flat | 0.0 | 0.1491418033838272 | 0.1491418033838272 | True |
| flat_hopper_hop | 1 | HopperHop | flat | 0.0 | 1.0323060750961304 | 1.0323060750961304 | True |
| flat_hopper_hop | 2 | HopperHop | flat | 0.0 | 0.2906038165092468 | 0.2906038165092468 | True |
| flat_panda_pick_cube | 0 | PandaPickCube | flat | 60.6395366191864 | 472.1655101776123 | 411.5259735584259 | True |
| flat_panda_pick_cube | 1 | PandaPickCube | flat | 74.36377906799316 | 525.935432434082 | 451.57165336608887 | True |
| flat_panda_pick_cube | 2 | PandaPickCube | flat | 55.40943360328674 | 470.8203201293945 | 415.4108865261078 | True |
| flat_walker_walk | 0 | WalkerWalk | flat | 0.0 | 42.32115936279297 | 42.32115936279297 | True |
| flat_walker_walk | 1 | WalkerWalk | flat | 0.0 | 347.7015380859375 | 347.7015380859375 | True |
| flat_walker_walk | 2 | WalkerWalk | flat | 0.0 | 80.47119903564453 | 80.47119903564453 | True |
| go1_joystick_nesy | 0 | Go1JoystickFlatTerrain | nesy | 0.0046606482446804 | 5.050708591938019 | 5.046047943693338 | True |
| go1_joystick_nesy | 1 | Go1JoystickFlatTerrain | nesy | 0.0119595894439044 | 2.119040533900261 | 2.1070809444563565 | True |
| go1_joystick_nesy | 2 | Go1JoystickFlatTerrain | nesy | 0.0045467126601579 | 5.09638261795044 | 5.0918359052902815 | True |
| go1_joystick_neural | 0 | Go1JoystickFlatTerrain | neural | 0.0054384893460337 | 7.1764649748802185 | 7.171026485534185 | True |
| go1_joystick_neural | 1 | Go1JoystickFlatTerrain | neural | 0.0077214163929966 | 5.429533511400223 | 5.421812095007226 | True |
| go1_joystick_neural | 2 | Go1JoystickFlatTerrain | neural | 0.0046443830196949 | 9.287592113018036 | 9.28294772999834 | True |
| hopper_hop_nesy | 0 | HopperHop | nesy | 0.0 | 0.604293942451477 | 0.604293942451477 | True |

## Skill and Mask Diagnostics

| run_id | seed | env_name | meta_policy_type | num_usage_skills | usage_entropy | skill_reward_std | skill_rewards_nonconstant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | 3 | 0.9956568837765843 | 5.24910575871739 | True |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | 3 | 0.9736864215958192 | 8.666797834740798 | True |
| cartpole_balance_nesy | 2 | CartpoleBalance | nesy | 3 | 1.0248880508077332 | 14.441514428179012 | True |
| cartpole_balance_neural | 0 | CartpoleBalance | neural | 3 | 1.0126555133373685 | 38.24934546656035 | True |
| cartpole_balance_neural | 1 | CartpoleBalance | neural | 3 | 0.9446124847962954 | 14.33918168057842 | True |
| cartpole_balance_neural | 2 | CartpoleBalance | neural | 3 | 0.6190120770232654 | 7.071707646006277 | True |
| cartpole_balance_symbolic | 0 | CartpoleBalance | symbolic | 1 | 0.0032234373823657 | 21.88224999009527 | True |
| cartpole_balance_symbolic | 1 | CartpoleBalance | symbolic | 1 | -0.0 | 1.018835827621134 | True |
| cartpole_balance_symbolic | 2 | CartpoleBalance | symbolic | 1 | -0.0 | 11.94995299485772 | True |
| cheetah_run_nesy | 0 | CheetahRun | nesy | 3 | 0.8127449927185686 | 0.6344676212681744 | True |
| cheetah_run_nesy | 1 | CheetahRun | nesy | 3 | 0.9781821482788298 | 0.4038594299789239 | True |
| cheetah_run_nesy | 2 | CheetahRun | nesy | 3 | 0.8849985060918001 | 0.9618973190082056 | True |
| cheetah_run_neural | 0 | CheetahRun | neural | 3 | 1.0263410792269063 | 0.7147846447549986 | True |
| cheetah_run_neural | 1 | CheetahRun | neural | 3 | 1.0166473070098507 | 0.6798459352373462 | True |
| cheetah_run_neural | 2 | CheetahRun | neural | 3 | 0.7021940665765016 | 1.3026484231793674 | True |
| flat_cartpole_balance | 0 | CartpoleBalance | flat | 1 | -0.0 | 0.0 | False |
| flat_cartpole_balance | 1 | CartpoleBalance | flat | 1 | -0.0 | 0.0 | False |
| flat_cartpole_balance | 2 | CartpoleBalance | flat | 1 | -0.0 | 0.0 | False |
| flat_cheetah_run | 0 | CheetahRun | flat | 1 | -0.0 | 0.0 | False |
| flat_cheetah_run | 1 | CheetahRun | flat | 1 | -0.0 | 0.0 | False |
| flat_cheetah_run | 2 | CheetahRun | flat | 1 | -0.0 | 0.0 | False |
| flat_go1_joystick | 0 | Go1JoystickFlatTerrain | flat | 1 | -0.0 | 0.0 | False |
| flat_go1_joystick | 1 | Go1JoystickFlatTerrain | flat | 1 | -0.0 | 0.0 | False |
| flat_go1_joystick | 2 | Go1JoystickFlatTerrain | flat | 1 | -0.0 | 0.0 | False |
| flat_hopper_hop | 0 | HopperHop | flat | 1 | -0.0 | 0.0 | False |
| flat_hopper_hop | 1 | HopperHop | flat | 1 | -0.0 | 0.0 | False |
| flat_hopper_hop | 2 | HopperHop | flat | 1 | -0.0 | 0.0 | False |
| flat_panda_pick_cube | 0 | PandaPickCube | flat | 1 | -0.0 | 0.0 | False |
| flat_panda_pick_cube | 1 | PandaPickCube | flat | 1 | -0.0 | 0.0 | False |
| flat_panda_pick_cube | 2 | PandaPickCube | flat | 1 | -0.0 | 0.0 | False |
| flat_walker_walk | 0 | WalkerWalk | flat | 1 | -0.0 | 0.0 | False |
| flat_walker_walk | 1 | WalkerWalk | flat | 1 | -0.0 | 0.0 | False |
| flat_walker_walk | 2 | WalkerWalk | flat | 1 | -0.0 | 0.0 | False |
| go1_joystick_nesy | 0 | Go1JoystickFlatTerrain | nesy | 4 | 1.2115000911695708 | 0.9645405668826652 | True |
| go1_joystick_nesy | 1 | Go1JoystickFlatTerrain | nesy | 4 | 1.2833907679016463 | 0.9617767069410542 | True |
| go1_joystick_nesy | 2 | Go1JoystickFlatTerrain | nesy | 4 | 1.0068659200013808 | 0.9857081161431176 | True |
| go1_joystick_neural | 0 | Go1JoystickFlatTerrain | neural | 4 | 1.1882824729709502 | 0.9642866822519416 | True |
| go1_joystick_neural | 1 | Go1JoystickFlatTerrain | neural | 4 | 0.8744243499975745 | 0.8451354792586097 | True |
| go1_joystick_neural | 2 | Go1JoystickFlatTerrain | neural | 4 | 1.2049370654510128 | 0.9898410539119796 | True |
| hopper_hop_nesy | 0 | HopperHop | nesy | 4 | 1.0549891603682866 | 0.2103425349302603 | True |

| run_id | seed | env_name | meta_policy_type | kind | skill | last10pct_mean |
| --- | --- | --- | --- | --- | --- | --- |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 0_accelerate_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 0_flat_actor | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 0_reach_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 0_recover_balance | 0.999969482421875 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 0_stand | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 0_stand_recover | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 1_center_cart | 0.9601847330729166 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 1_grasp_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 1_hop_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 1_stabilize_posture | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 1_track_velocity | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 1_walk_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 2_damp_motion | 1.0 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 2_energy_efficient_run | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 2_lift_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 2_stabilize_gait | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 2_stabilize_landing | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 2_turn | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 3_energy_efficient | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 3_place_or_stabilize | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_available | 3_recover | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 0_accelerate_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 0_flat_actor | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 0_reach_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 0_recover_balance | 0.5514742136001587 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 0_stand | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 0_stand_recover | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 1_center_cart | 0.2061432649691899 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 1_grasp_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 1_hop_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 1_stabilize_posture | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 1_track_velocity | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 1_walk_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 2_damp_motion | 0.2505696614583333 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 2_energy_efficient_run | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 2_lift_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 2_stabilize_gait | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 2_stabilize_landing | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 2_turn | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 3_energy_efficient | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 3_place_or_stabilize | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_given_available | 3_recover | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 0_accelerate_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 0_flat_actor | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 0_reach_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 0_recover_balance | 0.5514577229817709 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 0_stand | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 0_stand_recover | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 1_center_cart | 0.1979726155598958 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 1_grasp_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 1_hop_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 1_stabilize_posture | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 1_track_velocity | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 1_walk_forward | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 2_damp_motion | 0.2505696614583333 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 2_energy_efficient_run | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 2_lift_cube | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 2_stabilize_gait | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 2_stabilize_landing | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | mask_selected_when_available | 2_turn | n/a |

## Raw Feature Diagnostics

| run_id | seed | env_name | meta_policy_type | feature | mean | std | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cartpole/cart_position | -0.2279624102908807 | 0.8862785455587916 | -1.7373988628387451 | 1.6453804969787598 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cartpole/cart_velocity | 0.0424950531373421 | 0.6023369436408725 | -2.127420425415039 | 1.1498976945877075 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cartpole/pole_angle | -3.7827037146625417 | 4.896289721280295 | -12.862241744995115 | 8.8507719039917 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cartpole/pole_angular_velocity | -0.1991146072745323 | 2.931151387143593 | -5.33479118347168 | 5.3555192947387695 |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cheetah/action_norm | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cheetah/forward_velocity | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cheetah/joint_speed | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | cheetah/torso_pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | flat/env_reward | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | go1/base_height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | go1/command_xy_norm | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | go1/command_yaw | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | go1/pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | go1/roll | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | hopper/done_fraction | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | hopper/forward_velocity | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | hopper/height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | hopper/joint_speed | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | hopper/pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | panda/cube_height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | panda/dist_cube_target | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | panda/dist_tcp_cube | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | panda/grasped | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | panda/gripper | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | walker/forward_velocity | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | walker/height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | walker/joint_speed | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 0 | CartpoleBalance | nesy | walker/pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cartpole/cart_position | -0.0811749539958934 | 0.5986737997662585 | -1.2059829235076904 | 1.1513638496398926 |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cartpole/cart_velocity | -0.0496937935551007 | 0.5283083863501403 | -1.2366344928741455 | 1.1503067016601562 |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cartpole/pole_angle | 2.175120093735556 | 5.264712749038154 | -1.6956455707550049 | 17.817031860351562 |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cartpole/pole_angular_velocity | 1.1215818583965302 | 1.6957481280325375 | -1.6843957901000977 | 4.427635192871094 |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cheetah/action_norm | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cheetah/forward_velocity | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cheetah/joint_speed | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | cheetah/torso_pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | flat/env_reward | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | go1/base_height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | go1/command_xy_norm | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | go1/command_yaw | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | go1/pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | go1/roll | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | hopper/done_fraction | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | hopper/forward_velocity | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | hopper/height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | hopper/joint_speed | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | hopper/pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | panda/cube_height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | panda/dist_cube_target | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | panda/dist_tcp_cube | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | panda/grasped | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | panda/gripper | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | walker/forward_velocity | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | walker/height | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | walker/joint_speed | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 1 | CartpoleBalance | nesy | walker/pitch | n/a | n/a | n/a | n/a |
| cartpole_balance_nesy | 2 | CartpoleBalance | nesy | cartpole/cart_position | -0.3134229616106798 | 0.8720939624444813 | -1.3837409019470217 | 1.519475340843201 |
| cartpole_balance_nesy | 2 | CartpoleBalance | nesy | cartpole/cart_velocity | -0.0499754262777666 | 0.580376728186929 | -1.3163893222808838 | 1.705938458442688 |
| cartpole_balance_nesy | 2 | CartpoleBalance | nesy | cartpole/pole_angle | -2.3073027761633664 | 2.8744077257536773 | -7.618255615234375 | 0.541570246219635 |
| cartpole_balance_nesy | 2 | CartpoleBalance | nesy | cartpole/pole_angular_velocity | -0.349374112021178 | 0.9698321411738596 | -2.572457790374756 | 1.2003183364868164 |

## Plots

- `plots/paper/main_return_curves.png`
- `plots/paper/final_performance_vs_flat.png`
- `plots/paper/skill_reward_curves_by_env.png`
- `plots/paper/skill_usage_by_env_variant.png`
- `plots/paper/mask_availability_vs_selection.png`
- `plots/paper/panda_phase_diagnostics.png`
- `plots/paper/loss_and_td_diagnostics.png`
- `plots/paper/raw_feature_diagnostics.png`

## Limitations and Failure Cases

# NEXUS result diagnostics

Generated: 2026-06-04T13:38:51
Runs loaded: 60
Pickle load failures: 0
Metric extraction errors: 0

## Inventory

By stage: {"final_research_matrix": 60}
By environment: {"CartpoleBalance": 12, "CheetahRun": 9, "Go1JoystickFlatTerrain": 9, "HopperHop": 9, "PandaPickCube": 12, "WalkerWalk": 9}
By variant: {"flat": 18, "nesy": 18, "neural": 18, "symbolic": 6}

## Finite-value checks

- OK: no non-finite numeric metric values detected.

## Skill usage checks

- Max |sum(skill_usage)-1|: 0
- WARNING: low final-window skill usage detected:
  - cartpole_balance_symbolic seed0 last10pct_mean/skill_usage/1_center_cart: 0.0003611
  - cartpole_balance_symbolic seed1 last10pct_mean/skill_usage/1_center_cart: 0
  - cartpole_balance_symbolic seed2 last10pct_mean/skill_usage/1_center_cart: 0
  - panda_pick_cube_symbolic seed0 last10pct_mean/skill_usage/1_grasp_cube: 0.007215
  - cartpole_balance_symbolic seed0 last10pct_mean/skill_usage/2_damp_motion: 0
  - cartpole_balance_symbolic seed1 last10pct_mean/skill_usage/2_damp_motion: 0
  - cartpole_balance_symbolic seed2 last10pct_mean/skill_usage/2_damp_motion: 0
  - panda_pick_cube_nesy seed0 last10pct_mean/skill_usage/3_place_or_stabilize: 1.907e-06
  - panda_pick_cube_nesy seed1 last10pct_mean/skill_usage/3_place_or_stabilize: 0
  - panda_pick_cube_nesy seed2 last10pct_mean/skill_usage/3_place_or_stabilize: 1.907e-06
  - panda_pick_cube_symbolic seed0 last10pct_mean/skill_usage/3_place_or_stabilize: 0
  - panda_pick_cube_symbolic seed1 last10pct_mean/skill_usage/3_place_or_stabilize: 0
  - panda_pick_cube_symbolic seed2 last10pct_mean/skill_usage/3_place_or_stabilize: 1.431e-05

## Learning-signal checks

- CHECK: cartpole_balance_nesy seed0 returns/env_reward_mean: first=0.6018, last=0.2375, delta=-0.3644
- CHECK: cartpole_balance_nesy seed1 returns/env_reward_mean: first=0.5941, last=0.1597, delta=-0.4345
- CHECK: cartpole_balance_nesy seed2 returns/env_reward_mean: first=0.6202, last=0.2054, delta=-0.4148
- CHECK: cartpole_balance_neural seed0 returns/env_reward_mean: first=0.6279, last=0.1254, delta=-0.5024
- CHECK: cartpole_balance_neural seed1 returns/env_reward_mean: first=0.5913, last=0.2286, delta=-0.3627
- CHECK: cartpole_balance_neural seed2 returns/env_reward_mean: first=0.5969, last=0.2501, delta=-0.3468
- CHECK: cartpole_balance_symbolic seed0 returns/env_reward_mean: first=0.5717, last=0.175, delta=-0.3967
- CHECK: cartpole_balance_symbolic seed1 returns/env_reward_mean: first=0.5599, last=0.06687, delta=-0.493
- CHECK: cartpole_balance_symbolic seed2 returns/env_reward_mean: first=0.5981, last=0.1638, delta=-0.4342
- OK: cheetah_run_nesy seed0 returns/env_reward_mean: first=0.01256, last=0.2197, delta=0.2071
- OK: cheetah_run_nesy seed1 returns/env_reward_mean: first=0.01017, last=0.1352, delta=0.1251
- OK: cheetah_run_nesy seed2 returns/env_reward_mean: first=0.0139, last=0.2946, delta=0.2807
- OK: cheetah_run_neural seed0 returns/env_reward_mean: first=0.009482, last=0.2094, delta=0.1999
- OK: cheetah_run_neural seed1 returns/env_reward_mean: first=0.01093, last=0.1934, delta=0.1825
- OK: cheetah_run_neural seed2 returns/env_reward_mean: first=0.01348, last=0.3839, delta=0.3704
- CHECK: flat_cartpole_balance seed0 returns/env_reward_mean: first=0.5652, last=0.1575, delta=-0.4077
- CHECK: flat_cartpole_balance seed1 returns/env_reward_mean: first=0.5659, last=0.1484, delta=-0.4175
- CHECK: flat_cartpole_balance seed2 returns/env_reward_mean: first=0.5837, last=0.234, delta=-0.3497
- OK: flat_cheetah_run seed0 returns/env_reward_mean: first=0.02333, last=0.03555, delta=0.01222
- OK: flat_cheetah_run seed1 returns/env_reward_mean: first=0.01043, last=0.08784, delta=0.07742
- OK: flat_cheetah_run seed2 returns/env_reward_mean: first=0.009238, last=0.08716, delta=0.07792
- OK: flat_go1_joystick seed0 returns/env_reward_mean: first=0.0004654, last=0.01086, delta=0.0104
- OK: flat_go1_joystick seed1 returns/env_reward_mean: first=0.000121, last=0.00979, delta=0.009669
- OK: flat_go1_joystick seed2 returns/env_reward_mean: first=0.0001019, last=0.009737, delta=0.009635
- CHECK: flat_hopper_hop seed0 returns/env_reward_mean: first=0.0001094, last=0, delta=-0.0001094
- CHECK: flat_hopper_hop seed1 returns/env_reward_mean: first=0.0002596, last=1.386e-05, delta=-0.0002457
- CHECK: flat_hopper_hop seed2 returns/env_reward_mean: first=7.556e-05, last=0, delta=-7.556e-05
- OK: flat_panda_pick_cube seed0 returns/env_reward_mean: first=0.6519, last=3.153, delta=2.501
- OK: flat_panda_pick_cube seed1 returns/env_reward_mean: first=0.7027, last=3.512, delta=2.81
- OK: flat_panda_pick_cube seed2 returns/env_reward_mean: first=0.5194, last=3.139, delta=2.62
- OK: flat_walker_walk seed0 returns/env_reward_mean: first=0.02872, last=0.0919, delta=0.06318
- OK: flat_walker_walk seed1 returns/env_reward_mean: first=0.04519, last=0.6645, delta=0.6193
- OK: flat_walker_walk seed2 returns/env_reward_mean: first=0.02879, last=0.1303, delta=0.1015
- OK: go1_joystick_nesy seed0 returns/env_reward_mean: first=0.0009271, last=0.005582, delta=0.004655
- OK: go1_joystick_nesy seed1 returns/env_reward_mean: first=0.0009851, last=0.003203, delta=0.002218
- OK: go1_joystick_nesy seed2 returns/env_reward_mean: first=0.0001311, last=0.006223, delta=0.006092
- OK: go1_joystick_neural seed0 returns/env_reward_mean: first=0.0006028, last=0.008579, delta=0.007976
- OK: go1_joystick_neural seed1 returns/env_reward_mean: first=0.001744, last=0.00637, delta=0.004626
- OK: go1_joystick_neural seed2 returns/env_reward_mean: first=0.0001491, last=0.01021, delta=0.01006
- CHECK: hopper_hop_nesy seed0 returns/env_reward_mean: first=8.429e-05, last=0, delta=-8.429e-05
- CHECK: hopper_hop_nesy seed1 returns/env_reward_mean: first=8.226e-05, last=3.601e-06, delta=-7.866e-05
- CHECK: hopper_hop_nesy seed2 returns/env_reward_mean: first=7.766e-05, last=1.471e-06, delta=-7.619e-05
- CHECK: hopper_hop_neural seed0 returns/env_reward_mean: first=0.0001053, last=0, delta=-0.0001053
- CHECK: hopper_hop_neural seed1 returns/env_reward_mean: first=8.035e-05, last=0, delta=-8.035e-05
- CHECK: hopper_hop_neural seed2 returns/env_reward_mean: first=7.775e-05, last=2.539e-05, delta=-5.236e-05
- OK: panda_pick_cube_nesy seed0 returns/env_reward_mean: first=0.3213, last=2.918, delta=2.596
- OK: panda_pick_cube_nesy seed1 returns/env_reward_mean: first=0.3578, last=3.146, delta=2.788
- OK: panda_pick_cube_nesy seed2 returns/env_reward_mean: first=0.3345, last=2.984, delta=2.649
- OK: panda_pick_cube_neural seed0 returns/env_reward_mean: first=0.9359, last=3.216, delta=2.28
- OK: panda_pick_cube_neural seed1 returns/env_reward_mean: first=1.223, last=3.168, delta=1.945
- OK: panda_pick_cube_neural seed2 returns/env_reward_mean: first=0.9519, last=3.077, delta=2.125
- OK: panda_pick_cube_symbolic seed0 returns/env_reward_mean: first=0.3242, last=1.476, delta=1.152
- OK: panda_pick_cube_symbolic seed1 returns/env_reward_mean: first=0.3515, last=2.732, delta=2.381
- OK: panda_pick_cube_symbolic seed2 returns/env_reward_mean: first=0.3431, last=2.514, delta=2.17
- OK: walker_walk_nesy seed0 returns/env_reward_mean: first=0.06442, last=0.1898, delta=0.1254
- OK: walker_walk_nesy seed1 returns/env_reward_mean: first=0.06655, last=0.2577, delta=0.1911
- OK: walker_walk_nesy seed2 returns/env_reward_mean: first=0.05534, last=0.2439, delta=0.1885
- OK: walker_walk_neural seed0 returns/env_reward_mean: first=0.06422, last=0.4, delta=0.3358
- OK: walker_walk_neural seed1 returns/env_reward_mean: first=0.06822, last=0.234, delta=0.1658
- OK: walker_walk_neural seed2 returns/env_reward_mean: first=0.05902, last=0.3806, delta=0.3216

## Required run coverage checklist

- [x] cartpole_balance_nesy: 3 seed rows (OK)
- [x] cartpole_balance_neural: 3 seed rows (OK)
- [x] cartpole_balance_symbolic: 3 seed rows (OK)
- [x] cheetah_run_nesy: 3 seed rows (OK)
- [x] cheetah_run_neural: 3 seed rows (OK)
- [x] flat_cartpole_balance: 3 seed rows (OK)
- [x] flat_cheetah_run: 3 seed rows (OK)
- [x] flat_hopper_hop: 3 seed rows (OK)
- [x] flat_panda_pick_cube: 3 seed rows (OK)
- [x] flat_walker_walk: 3 seed rows (OK)
- [x] hopper_hop_nesy: 3 seed rows (OK)
- [x] hopper_hop_neural: 3 seed rows (OK)
- [x] panda_pick_cube_nesy: 3 seed rows (OK)
- [x] panda_pick_cube_neural: 3 seed rows (OK)
- [x] panda_pick_cube_symbolic: 3 seed rows (OK)
- [x] walker_walk_nesy: 3 seed rows (OK)
- [x] walker_walk_neural: 3 seed rows (OK)
- [x] go1_joystick_nesy extension: 3 seed rows

## Checklist failure flags

- OK: flat baseline present for every loaded environment.
- OK: raw feature diagnostics found (28 metric columns).
- OK: no monotonic-scale critic TD explosion detected by threshold.

