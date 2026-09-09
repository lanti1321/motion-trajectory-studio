# 模型库

仓库内置并跟踪 `OpenArm_V1/`（达妙夹爪）和 `OpenArm_O6/`（灵心 O6 灵巧手），用于保证全新克隆后可以立即加载完整示例模型。
其他放入本目录的本地模型默认仍由 `.gitignore` 排除，不会意外提交。

每个模型使用一个独立子文件夹，软件以子文件夹名称作为模型名称：

```text
model_library/
├── MyRobot/
│   ├── MyRobot.xml
│   └── meshes/
└── AnotherRobot/
    ├── robot.urdf
    └── meshes/
```

将整个模型目录复制进来后，软件会自动刷新并显示新模型。模型内的 mesh、纹理、MJCF include 等相对路径保持原目录结构即可。

入口文件自动选择顺序：

1. 与文件夹同名的 `.xml`、`.mjcf` 或 `.urdf`；
2. `model.xml`、`model.mjcf`、`model.urdf`；
3. `scene.xml`、`scene.mjcf`、`robot.urdf`；
4. 唯一模型文件或目录中的第一个有效模型文件。

如果目录中存在多个模型文件并且自动选择不正确，在该模型文件夹中添加 `model.json`：

```json
{
  "entry": "path/to/main_model.xml"
}
```

`entry` 必须是相对于模型文件夹的路径，不能指向文件夹外部。

`model.json`还可以携带模型预设：

- `adjustments`：加载后执行模型几何修正，例如两个关节中心的目标距离；
- `default_mapping`：轨迹通道到模型关节的默认映射；
- 映射可包含`scale`、`offset`、`minimum`、`maximum`，用于单位换算、方向反转和限幅。

`OpenArm_V1/model.json` 是包含 422mm 肩宽与夹爪标定的完整示例。`OpenArm_O6/model.json` 保留同样的手臂通道，并用每只手 6 个主动手指关节名替换夹爪通道；DIP/IP 由 MCP/CMC 耦合，不单独映射。
