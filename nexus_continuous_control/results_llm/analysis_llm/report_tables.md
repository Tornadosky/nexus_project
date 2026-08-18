### Environment comparison summary (hand-written vs LLM vs refined)

| env | backend | hand_env_reward_mean | hand_env_reward_std | hand_success_rate_mean | hand_success_rate_std | hand_n_skills | llm_env_reward_mean | llm_env_reward_std | llm_success_rate_mean | llm_success_rate_std | llm_n_skills | refined_env_reward_mean | refined_success_rate_mean | refined_n_skills | refinement_iterations | refinement_stopped_early | llm_vs_hand_reward_gap_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CartpoleBalance | hf | 0.5388 | 0.1279 | 0.4146 | 0.2137 | 3 | 0.1352 | 0.02625 | 0.03462 | 0.03218 | 3 | 0.3102 | 0.1076 | 5 | 4 | False | -74.91 |
| CheetahRun | hf | 0.8423 | 0.02865 | 0.8993 | 0.09494 | 3 | 0.7825 | 0.1332 | 0.9128 | 0.04218 | 3 | 0.8492 | 0.9291 | 5 | 4 | False | -7.096 |
| Go1JoystickFlatTerrain | hf | 0.006367 | 0.002133 | 0.2838 | 0.08395 | 4 | 8.513e-07 | 4.584e-07 | 0.03738 | 0.01137 | 3 | 5.871e-06 | 0.0173 | 5 | 4 | False | -99.99 |
| HopperHop | hf | 0.2057 | 0.09317 | 0.2409 | 0.05277 | 4 | 0 | 0 | 0.0007766 | 0.0002893 | 3 | 0 | 0 | 5 | 4 | False | -100 |
| WalkerWalk | hf | 0.9291 | 0.01127 | 0.9969 | 0.003827 | 4 | 0.1988 | 0.07766 | 0.009375 | 0.01875 | 3 | 0.4071 | 0.3634 | 5 | 4 | False | -78.6 |


### Skillset sizes and names

| env | hand_written_skills | hand_written_names | llm_initial_skills | llm_initial_names | llm_refined_skills | llm_refined_names |
|---|---|---|---|---|---|---|
| CartpoleBalance | 3 | 0_recover_balance; 1_center_cart; 2_damp_motion | 3 | Initial Balance Check; Leverage Gravity to Maintain Balance; Optimize Locomotion for Stability | 5 | Initial Balance Check; Pole Locomotion; Stability Enhancement; Dynamic Pole Control; Improved Stability |
| CheetahRun | 3 | 0_accelerate_forward; 1_stabilize_posture; 2_energy_efficient_run | 3 | Initial Stability; Lateral Locomotion; Optimal Performance | 5 | Initial Balance; Sideways Movement; High-Speed Forward Motion; Steering Maneuver; Dynamic Locomotion |
| Go1JoystickFlatTerrain | 4 | 0_stand; 1_track_velocity; 2_turn; 3_recover | 3 | BaseHeightSafety; LateralMovementControl; OptimalGait | 5 | SafeDistanceFromObstacles; AvoidCollisions; OptimalLandingParameters; SmoothSwinging; StabilizeBaseHeight |
| HopperHop | 4 | 0_stand_recover; 1_hop_forward; 2_stabilize_landing; 3_energy_efficient | 3 | Initial Stability; Forward Movement; Optimal Locomotion | 5 | Initial Balance; Forward Momentum; Efficient Locomotion; Stability Enhancement; Dynamic Stabilization |
| WalkerWalk | 4 | 0_stand_recover; 1_walk_forward; 2_stabilize_gait; 3_energy_efficient | 3 | Initial Balance Check; Forward Movement; Optimal Locomotion | 5 | Initial Balance Check; Steady Walking; Smooth Locomotion; Dynamic Stance; Stealthy Movement |


### Skill usage by condition (hand-written / LLM-initial / LLM-refined; hand-written and LLM-initial are averaged over 5 seeds, LLM-refined is the single final-iteration training run)

| env | condition | skill | mean_usage_fraction |
|---|---|---|---|
| CartpoleBalance | hand_written | 2_damp_motion | 0.7381 |
| CartpoleBalance | hand_written | 1_center_cart | 0.1484 |
| CartpoleBalance | hand_written | 0_recover_balance | 0.1134 |
| CheetahRun | hand_written | 2_energy_efficient_run | 0.6018 |
| CheetahRun | hand_written | 0_accelerate_forward | 0.3544 |
| CheetahRun | hand_written | 1_stabilize_posture | 0.0439 |
| Go1JoystickFlatTerrain | hand_written | 0_stand | 0.3588 |
| Go1JoystickFlatTerrain | hand_written | 3_recover | 0.2798 |
| Go1JoystickFlatTerrain | hand_written | 1_track_velocity | 0.2016 |
| Go1JoystickFlatTerrain | hand_written | 2_turn | 0.1597 |
| HopperHop | hand_written | 3_energy_efficient | 0.3727 |
| HopperHop | hand_written | 1_hop_forward | 0.2911 |
| HopperHop | hand_written | 0_stand_recover | 0.1927 |
| HopperHop | hand_written | 2_stabilize_landing | 0.1436 |
| WalkerWalk | hand_written | 3_energy_efficient | 0.8244 |
| WalkerWalk | hand_written | 0_stand_recover | 0.1201 |
| WalkerWalk | hand_written | 2_stabilize_gait | 0.0555 |
| WalkerWalk | hand_written | 1_walk_forward | 0 |
| CartpoleBalance | llm_initial | 0_Initial Balance Check | 1 |
| CartpoleBalance | llm_initial | 1_Leverage Gravity to Maintain Balance | 0 |
| CartpoleBalance | llm_initial | 2_Optimize Locomotion for Stability | 0 |
| CheetahRun | llm_initial | 1_Lateral Locomotion | 0.4731 |
| CheetahRun | llm_initial | 2_Optimal Performance | 0.4032 |
| CheetahRun | llm_initial | 0_Initial Stability | 0.1236 |
| Go1JoystickFlatTerrain | llm_initial | 0_BaseHeightSafety | 0.8573 |
| Go1JoystickFlatTerrain | llm_initial | 2_OptimalGait | 0.1202 |
| Go1JoystickFlatTerrain | llm_initial | 1_LateralMovementControl | 0.0226 |
| HopperHop | llm_initial | 2_Optimal Locomotion | 0.583 |
| HopperHop | llm_initial | 0_Initial Stability | 0.4083 |
| HopperHop | llm_initial | 1_Forward Movement | 0.0086 |
| WalkerWalk | llm_initial | 1_Forward Movement | 0.8577 |
| WalkerWalk | llm_initial | 0_Initial Balance Check | 0.1419 |
| WalkerWalk | llm_initial | 2_Optimal Locomotion | 0.0004 |
| CartpoleBalance | llm_refined | 0_Initial Balance Check | 0.6573 |
| CartpoleBalance | llm_refined | 3_Dynamic Pole Control | 0.1452 |
| CartpoleBalance | llm_refined | 2_Stability Enhancement | 0.1276 |
| CartpoleBalance | llm_refined | 1_Pole Locomotion | 0.0474 |
| CartpoleBalance | llm_refined | 4_Improved Stability | 0.0225 |
| CheetahRun | llm_refined | 2_High-Speed Forward Motion | 0.6085 |
| CheetahRun | llm_refined | 1_Sideways Movement | 0.2434 |
| CheetahRun | llm_refined | 0_Initial Balance | 0.07 |
| CheetahRun | llm_refined | 4_Dynamic Locomotion | 0.0399 |
| CheetahRun | llm_refined | 3_Steering Maneuver | 0.0382 |
| Go1JoystickFlatTerrain | llm_refined | 0_SafeDistanceFromObstacles | 0.6713 |
| Go1JoystickFlatTerrain | llm_refined | 3_SmoothSwinging | 0.1563 |
| Go1JoystickFlatTerrain | llm_refined | 4_StabilizeBaseHeight | 0.1327 |
| Go1JoystickFlatTerrain | llm_refined | 1_AvoidCollisions | 0.0395 |
| Go1JoystickFlatTerrain | llm_refined | 2_OptimalLandingParameters | 0.0002 |
| HopperHop | llm_refined | 1_Forward Momentum | 0.5851 |
| HopperHop | llm_refined | 3_Stability Enhancement | 0.2063 |
| HopperHop | llm_refined | 0_Initial Balance | 0.1914 |
| HopperHop | llm_refined | 4_Dynamic Stabilization | 0.0173 |
| HopperHop | llm_refined | 2_Efficient Locomotion | 0 |
| WalkerWalk | llm_refined | 0_Initial Balance Check | 0.639 |
| WalkerWalk | llm_refined | 2_Smooth Locomotion | 0.2448 |
| WalkerWalk | llm_refined | 1_Steady Walking | 0.071 |
| WalkerWalk | llm_refined | 3_Dynamic Stance | 0.0402 |
| WalkerWalk | llm_refined | 4_Stealthy Movement | 0.005 |


### LLM refinement loop, iteration by iteration

| env | iteration | n_skills | env_reward_mean | skill_reward_mean | primary_success_rate | mask_violation_rate | refinement_ok | refinement_error |
|---|---|---|---|---|---|---|---|---|
| CartpoleBalance | 0 | 3 | 0.1405 | -0.3184 | 2.441e-05 | 0 | True |  |
| CartpoleBalance | 1 | 3 | 0.367 | -0.4513 | 0.1251 | 0 | True |  |
| CartpoleBalance | 2 | 4 | 0.2602 | -0.9945 | 0.136 | 0 | True |  |
| CartpoleBalance | 3 | 5 | 0.3102 | -0.3969 | 0.1076 | 0 | True |  |
| CheetahRun | 0 | 3 | 0.7429 | 7.693 | 0.846 | 0 | True |  |
| CheetahRun | 1 | 3 | 0.7519 | 7.779 | 0.8774 | 0 | True |  |
| CheetahRun | 2 | 4 | 0.8055 | 6.238 | 0.8898 | 0 | True |  |
| CheetahRun | 3 | 5 | 0.8492 | 5.254 | 0.9291 | 0 | True |  |
| Go1JoystickFlatTerrain | 0 | 3 | 0.006233 | 0.1425 | 0.2973 | 0 | True |  |
| Go1JoystickFlatTerrain | 1 | 4 | 4.165e-07 | 0.6664 | 0.01112 | 0 | True |  |
| Go1JoystickFlatTerrain | 2 | 4 | 1.124e-05 | 0.664 | 0.01162 | 0 | True |  |
| Go1JoystickFlatTerrain | 3 | 5 | 5.871e-06 | 0.5682 | 0.0173 | 0 | True |  |
| HopperHop | 0 | 3 | 0 | -0.1299 | 0 | 0 | True |  |
| HopperHop | 1 | 4 | 0 | -0.4019 | 0 | 0 | True |  |
| HopperHop | 2 | 5 | 0 | -0.455 | 0 | 0 | True |  |
| HopperHop | 3 | 5 | 0 | -0.455 | 0 | 0 | True |  |
| WalkerWalk | 0 | 3 | 0.6374 | 4.399 | 0.6762 | 0 | True |  |
| WalkerWalk | 1 | 4 | 0.4442 | 1.055 | 0.4806 | 0 | True |  |
| WalkerWalk | 2 | 5 | 0.5769 | 0.7418 | 0.5443 | 0 | True |  |
| WalkerWalk | 3 | 5 | 0.4071 | 0.6028 | 0.3634 | 0 | True |  |


### Mean training wall-clock time (seconds)

| env | hand_written_wall_s_mean | llm_initial_wall_s_mean |
|---|---|---|
| CartpoleBalance | 63.14 | 67.76 |
| CheetahRun | 794 | 879.4 |
| Go1JoystickFlatTerrain | 450.4 | 370.1 |
| HopperHop | 1276 | 1150 |
| WalkerWalk | 1484 | 1348 |


### Per-seed raw results

| env | condition | seed | env_reward_mean | skill_reward_mean | eval_episode_return_mean | eval_success_rate | eval_goal_metric | mask_violation_rate | wall_s |
|---|---|---|---|---|---|---|---|---|---|
| CartpoleBalance | hand_written | 0 | 0.5327 | -0.3586 | 680.1 | 0.5252 | 0.8104 | 0 | 62.55 |
| CartpoleBalance | hand_written | 1 | 0.36 | -2.251 | 448.1 | 0.2664 | 0.2831 | 0 | 60.35 |
| CartpoleBalance | hand_written | 2 | 0.4779 | -2.09 | 354.9 | 0.1945 | 0.3112 | 0 | 63.17 |
| CartpoleBalance | hand_written | 3 | 0.5729 | -0.6994 | 871.8 | 0.7806 | 0.9992 | 0 | 64.2 |
| CartpoleBalance | hand_written | 4 | 0.7506 | 0.547 | 687.4 | 0.3063 | 0.9089 | 0 | 65.43 |
| CartpoleBalance | llm_initial | 0 | 0.1198 | -21.86 | 135.5 | 0.01854 | 0.06 | 0 | 64.88 |
| CartpoleBalance | llm_initial | 1 | 0.1871 | 11.67 | 258.9 | 0.09897 | 0.08607 | 0 | 66.73 |
| CartpoleBalance | llm_initial | 2 | 0.1297 | 17.17 | 144.8 | 0.01835 | 0.05767 | 0 | 67.63 |
| CartpoleBalance | llm_initial | 3 | 0.1198 | -21.87 | 135.5 | 0.01854 | 0.05999 | 0 | 69.09 |
| CartpoleBalance | llm_initial | 4 | 0.1196 | -21.81 | 135.9 | 0.0187 | 0.06009 | 0 | 70.49 |
| CheetahRun | hand_written | 0 | 0.8756 | 4.554 | 814.6 | 0.9389 | 8.151 | 0 | 641.6 |
| CheetahRun | hand_written | 1 | 0.8127 | 4.254 | 620.7 | 0.7161 | 6.216 | 0 | 764.2 |
| CheetahRun | hand_written | 2 | 0.827 | 4.198 | 890.8 | 0.9676 | 8.994 | 0 | 834.2 |
| CheetahRun | hand_written | 3 | 0.818 | 4.222 | 802.6 | 0.9019 | 8.039 | 0 | 865 |
| CheetahRun | hand_written | 4 | 0.8783 | 4.584 | 856.2 | 0.9719 | 8.571 | 0 | 864.9 |
| CheetahRun | llm_initial | 0 | 0.8663 | 11.16 | 808.9 | 0.9567 | 8.1 | 0 | 869.7 |
| CheetahRun | llm_initial | 1 | 0.8573 | 11.04 | 754.1 | 0.8661 | 7.55 | 0 | 886.3 |
| CheetahRun | llm_initial | 2 | 0.8437 | 10.87 | 867.1 | 0.965 | 8.681 | 0 | 883.4 |
| CheetahRun | llm_initial | 3 | 0.5175 | 6.675 | 724.6 | 0.8676 | 7.257 | 0 | 886.7 |
| CheetahRun | llm_initial | 4 | 0.8279 | 10.67 | 765.7 | 0.9083 | 7.669 | 0 | 870.8 |
| Go1JoystickFlatTerrain | hand_written | 0 | 0.007815 | 1.398 | 6.406 | 0.2536 | -0.9882 | 0 | 380.8 |
| Go1JoystickFlatTerrain | hand_written | 1 | 0.006942 | 1.382 | 5.787 | 0.3054 | -0.9055 | 0 | 449.8 |
| Go1JoystickFlatTerrain | hand_written | 2 | 0.006243 | 1.396 | 4.459 | 0.4364 | -0.8376 | 0 | 506.7 |
| Go1JoystickFlatTerrain | hand_written | 3 | 0.008458 | 1.408 | 6.708 | 0.2213 | -0.9628 | 0 | 529.9 |
| Go1JoystickFlatTerrain | hand_written | 4 | 0.002375 | 1.301 | 2.58 | 0.2022 | -0.9259 | 0 | 384.9 |
| Go1JoystickFlatTerrain | llm_initial | 0 | 1.384e-06 | -1.374 | 0.0003916 | 0.03664 | -1.923 | 0 | 327.4 |
| Go1JoystickFlatTerrain | llm_initial | 1 | 5.177e-08 | -1.623 | 0.0002278 | 0.01949 | -2.221 | 0 | 330.6 |
| Go1JoystickFlatTerrain | llm_initial | 2 | 1.041e-06 | -1.438 | 0.001771 | 0.03853 | -2.022 | 0 | 330.2 |
| Go1JoystickFlatTerrain | llm_initial | 3 | 1.101e-06 | -1.575 | 0.0002984 | 0.05538 | -1.731 | 0 | 371.9 |
| Go1JoystickFlatTerrain | llm_initial | 4 | 6.789e-07 | -1.407 | 0.0009291 | 0.03685 | -1.968 | 0 | 490.5 |
| HopperHop | hand_written | 0 | 0.1311 | 0.3237 | 147.5 | 0.1908 | 0.6218 | 0 | 1170 |
| HopperHop | hand_written | 1 | 0.1296 | 0.2903 | 147.8 | 0.1987 | 0.9993 | 0 | 1281 |
| HopperHop | hand_written | 2 | 0.2822 | 0.5599 | 244.8 | 0.3139 | 1.024 | 0 | 1305 |
| HopperHop | hand_written | 3 | 0.351 | 0.668 | 358.3 | 0.296 | 1.192 | 0 | 1306 |
| HopperHop | hand_written | 4 | 0.1345 | 0.3083 | 161.8 | 0.2053 | 1.039 | 0 | 1317 |
| HopperHop | llm_initial | 0 | 0 | 0.1382 | 0.6471 | 0.0006016 | 0.3323 | 0 | 1151 |
| HopperHop | llm_initial | 1 | 0 | 0.1269 | 0.4297 | 0.0004922 | 0.3212 | 0 | 1152 |
| HopperHop | llm_initial | 2 | 0 | 0.1111 | 0.6958 | 0.0007969 | 0.2866 | 0 | 1147 |
| HopperHop | llm_initial | 3 | 0 | 0.08384 | 1.021 | 0.00132 | 0.2963 | 0 | 1148 |
| HopperHop | llm_initial | 4 | 0 | 0.0989 | 0.5818 | 0.0006719 | 0.2992 | 0 | 1153 |
| WalkerWalk | hand_written | 0 | 0.9264 | 2.777 | 904.1 | 1 | 7.056 | 0 | 1376 |
| WalkerWalk | hand_written | 1 | 0.9453 | 2.415 | 921.5 | 1 | 5.832 | 0 | 1478 |
| WalkerWalk | hand_written | 2 | 0.9127 | 2.308 | 881.7 | 0.9922 | 5.601 | 0 | 1521 |
| WalkerWalk | hand_written | 3 | 0.9374 | 2.475 | 909.7 | 1 | 6.444 | 0 | 1524 |
| WalkerWalk | hand_written | 4 | 0.9238 | 2.36 | 893.9 | 0.9922 | 6.563 | 0 | 1523 |
| WalkerWalk | llm_initial | 0 | 0.3459 | -1.807 | 342.1 | 0 | 4.23 | 0 | 1356 |
| WalkerWalk | llm_initial | 1 | 0.1661 | -2.05 | 169.9 | 0 | 5.373 | 0 | 1348 |
| WalkerWalk | llm_initial | 2 | 0.1559 | -2.278 | 157 | 0.04688 | 3.12 | 0 | 1345 |
| WalkerWalk | llm_initial | 3 | 0.2021 | -1.934 | 194.1 | 0 | 4.281 | 0 | 1345 |
| WalkerWalk | llm_initial | 4 | 0.1241 | -1.836 | 123.3 | 0 | 3.596 | 0 | 1350 |
