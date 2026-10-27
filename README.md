# pi-train

在 Pi 里采集人机共写的训练样本。打开采集后，每一句 assistant 定稿都会停下来等你放行或改写，原始回合写入 `~/.pi/agent/train/`，再用 `/train export` 切成 SFT / 偏好 / 轨迹 JSONL。

## 安装

```bash
pi install git:github.com/Skjz/pi-train
```

本地路径：

```bash
pi install /path/to/pi-train
```

重启 Pi，或新开一个 session。

## 试用

```text
/train start 把 README 里的安装步骤总结成三句话
```

然后照常发任务。每句回复结束后会弹出：

- 放行
- 改写正文（只改 text；定稿用助手口吻，不要写成对模型下的指令）
- 改工具参数（有 tool call 时才出现；JSON 编辑 `name` / `arguments`）
- 丢掉工具（有 tool call 时才出现）
- 中止本轮

改写正文后若仍有工具调用，会再问：保留、改参数、或丢掉。改过的参数会写进定稿，这一轮按改后的调用执行。

任务做完：

```text
/train success
/train export
```

导出目录默认是当前工作目录下的 `train-export/<episode_id>/`。

## 命令

| 命令 | 作用 |
|---|---|
| `/train start [任务]` | 开始采集 |
| `/train stop` | 结束采集，不打成功/失败 |
| `/train success [备注]` | 标记成功并结束 |
| `/train fail [备注]` | 标记失败并结束 |
| `/train status` | 看当前 episode |
| `/train export [目录]` | 写出三类 JSONL |

启动时加上 `--train` 会在 session 一开始就进入采集，任务名是 `cli --train`。

## 导出文件

- `sft.jsonl`：每一回合的 `messages`（上下文 + `human_final`）
- `pref.jsonl`：只含你改过的回合，`chosen` / `rejected`
- `traj.jsonl`：整段成功轨迹（最后一回合的上下文 + 最后一句 `human_final`）
- `manifest.json`：episode 元数据

原始混写日志不要直接拿去训练，用这三份。

## 限制

- 只在 interactive / RPC（有对话框）时拦截。print / json 模式会自动放行并打标。
- 改写 / 改参数 / 丢掉工具后会去掉本轮 thinking，下一轮只看到定稿，避免模型把改写当成用户介入。
- 嵌套工具调用（codemode 内部）不弹窗。
- Escape 视为放行，并在记录里标 `dismissed`。

## 许可

[MIT](LICENSE)
