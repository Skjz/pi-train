# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Private GitHub repository at `https://github.com/Skjz/pi-train`, installable with `pi install git:github.com/Skjz/pi-train`.
- Rewrite flow can edit tool-call `name` and `arguments` as JSON (`改工具参数`). Changed calls run this turn and are stored as `human_final`.

### Changed

- After editing assistant text, the follow-up tool prompt is 保留 / 改参数 / 丢掉, not only keep-or-drop.
- README local install example uses `/path/to/pi-train` instead of a machine-specific home path.

## [0.1.0] - 2026-10-26

### Added

- Pi extension that records a human-in-the-loop training episode while the agent works.
- `/train start [task]` begins capture; every assistant turn pauses for 放行 / 改写正文 / 丢掉工具 / 中止本轮.
- `/train success`, `/train fail`, and `/train stop` close the episode with a result label.
- `/train export [dir]` writes `sft.jsonl`, `pref.jsonl`, `traj.jsonl`, and `manifest.json`.
- `--train` starts capture as soon as the session opens.
- Raw turns are appended to `~/.pi/agent/train/<episode_id>.jsonl` immediately, outside the model context.
