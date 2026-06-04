# Go1 Phase-2 Tuning Decision

Two Go1JoystickFlatTerrain NeSy seed-0 tuning runs were completed before the phase-2 matrix.

| Candidate | Deterministic primary success | Deterministic return | Velocity tracking error | Yaw tracking error | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| A: reduced noise, all-states actor update | 0.0182 | 6.4077 | 0.8678 | 0.4543 | rejected |
| B: reduced noise, active-only actor update | 0.0343 | 6.7261 | 0.8942 | 0.3612 | selected |

Selection followed the prescribed ordering: deterministic `primary_success_rate`, then deterministic return, then lower velocity tracking error, then lower fall rate. Candidate B wins on the first two criteria and has lower yaw error, although velocity tracking remains weak.

The selected config is `configs/go1_joystick_nesy_phase2.yaml`. Go1 remains a weak robotics stress-test limitation unless the full final matrix improves the deterministic tracking-success rate substantially.
