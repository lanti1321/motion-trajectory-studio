O6 visual assets for OpenArm MuJoCo.

- `adapter_flange_link7.stl`: from `~/Downloads/灵心巧手转接件测试专用.stp` (Creo, 2026-09-03). CAD +Y (thickness) rotated to OpenArm link7 +Z.
- `urdf_src/`: official LinkerHand O6 meshes/URDF from https://github.com/linker-bot/linkerhand-urdf (Apache-2.0).
- Default scene yaws each hand ±90° about tool +Z so palms face the midline at TABLE_READY without moving joint 7. Rebuild with `--cad-aligned` for official yaw.
