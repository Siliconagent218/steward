# Steward

Steward 是基于 DeepSeek Harness 的具身智能机器人系统。设计见仓库根目录的 [steward.md](../steward.md)，当前实现状态见其中的「11.1 Phase 0 实现说明」。

## 快速开始

```sh
uv sync
.venv/bin/steward demo          # 构建厨房并运行「把杯子放到桌上」，按 World State 判定成败
./scripts/check.sh              # Python 测试、DSH 插件编译与测试、组合配置快照比对
```

## 与 DeepSeek Harness 一起运行

```sh
.venv/bin/steward up                      # WCA :7400，机器人端口 :7410，Ego API :7420
./scripts/setup-dsh-profiles.sh           # 构建插件，在 var/dsh 下创建 ego 与 world profile
export DEEPSEEK_API_KEY=...
DSH_HOME=var/dsh/robot-1 dsh/node_modules/.bin/dsh --profile ego "把杯子放到桌上"
DSH_HOME=var/dsh/world dsh/node_modules/.bin/dsh --profile world --port 3091
```

`ego` profile 是机器人的 Ego Harness，只能经 Ego API 访问自我态；`world` profile 是 World Harness，经世界控制接口构建资产、组合场景、控制 Episode。两者使用不同的 `DSH_HOME`。

## 目录

| 路径 | 内容 |
|---|---|
| `src/steward/scene` | WS 与 ES 共用的 SceneState、校验器、MJCF 转换 |
| `src/steward/assets`、`authoring` | 资产生成、校验、预览、资产库；场景与 Scenario |
| `src/steward/world` | WSim 与 Steward 主机（WCA、机器人端口） |
| `src/steward/ego` | Ego Runtime：估计器、ESim、技能、Ego API |
| `src/steward/observer` | ES 与 World State 的对比评估 |
| `dsh/` | DeepSeek Harness 插件与 bundle（锁定 `@deepseek-ai/dsh@0.2.1-rc.3`） |
| `scripts/` | profile 创建、检查、多种子回归 |
| `compat/snapshots` | 组合配置快照 |

运行数据写入 `var/`，不纳入版本管理。
