# Offline tendon control bootstrap

The contact-projected smooth candidate does not produce a walking initialization
under this short controller search. This is a controller outcome, not a proof
that every controller or every permitted reference is infeasible.

The prototype uses native MuJoCo mass, bias, passive force, tendon moment arms,
ground-contact EFC edge Jacobians, and passive equality/limit Jacobians. Tendon
tensions are nonnegative and limited both by actuator force ranges and by the
force reachable at current tendon length within each actuator control range.
Root forward velocity is tracked toward 1.39 m/s; other root and joint targets
follow the candidate. Passive MTP and patella joints have no acceleration target
task and no virtual motors. All root motion comes from native tendon and ground
forces: no applied generalized/body forces are written.

Each step solves a bounded acceleration tracking problem and translates the
allocated tension into a tendon length command with the unchanged kp = 50000.
Actual trajectories are integrated by native mj_step, including its actual
contact-force law. Planned contact forces do not enter the simulator directly.
The optimizer is only an approximate controller and its contact-force allocation
cannot establish feasibility by itself. Recorded actual qacc and predicted
qacc expose this approximation.

Twenty-five trials at the training 5 ms timestep varied joint gains 100/150/400/
1000, root weighting 2/8/15, contact weighting 0.1/2/5, and starting phases 0/45.
Every trial crossed the fall threshold (pelvis below 0.55 m) within 0.79–1.39 s.
All violated the permitted 10 degree / 30 mm pose deviations. A 2 ms integration
run for the initial controller also fell at 0.934 s, close to its 5 ms result at
0.930 s, so the observed failure is not resolved by this integration change.

`sweep_summary.json` records each outcome. Individual `trajectory.npz` files
record actual qpos, qvel, ctrl, tension, contact flags and ground normal force,
plus predicted/actual accelerations and deviations from the original reference.
No trajectory is certified for Teacher initialization or a feasible reference.
No model, reward, original reference, or production source was modified.
