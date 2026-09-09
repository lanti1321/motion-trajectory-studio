# OpenArm_O6

Bimanual OpenArm with Linker Hand O6 instead of the Damiao parallel jaws.

- Wrist: 20 mm adapter flange on `link7` +Z, palms toward the midline at `q7 = 0`.
- Hands: official Linker O6 URDF meshes (Apache-2.0). Coupled DIP/IP joints follow MCP/CMC via MuJoCo equalities.
- Entry: `scene.xml` (floor). `scene_table.xml` adds the table used in teleop sim.
- TCP bodies: `openarm_left_hand_tcp`, `openarm_right_hand_tcp`.
- Pose: double-click this model to get a rest-pose trajectory and the **部件模块** dock. Each hand exposes its 6 actuated joints (thumb CMC yaw/pitch and four finger MCPs). DIP/IP exist in the mesh but follow MCP/CMC through MuJoCo equalities and do not have sliders.

Arm channels `q0`–`q6` / `q8`–`q14` match `OpenArm_V1`. Gripper channels `q7` / `q15` are intentionally unmapped. Independent finger joints use their MuJoCo names.

Hardware playback in `hardware_controller/` still talks to Damiao grippers. This model is for preview and trajectory editing until an O6 CAN path is added.
