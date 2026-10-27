/**
 * Train capture: pause after each assistant turn, let a human accept or
 * rewrite it, and export SFT / preference / trajectory JSONL.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type {
	AssistantMessage,
	TextContent,
	ToolCall,
	ToolResultMessage,
	UserMessage,
} from "@earendil-works/pi-ai";
import type {
	ExtensionAPI,
	ExtensionContext,
	SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { Box, Text } from "@earendil-works/pi-tui";

const CUSTOM_TYPE = "train-capture";

const ACCEPT = "放行";
const EDIT = "改写正文";
const EDIT_TOOLS = "改工具参数";
const DROP_TOOLS = "丢掉工具";
const ABORT = "中止本轮";
const KEEP_TOOLS = "保留工具调用";

type EditAction = "accept" | "rewrite" | "drop_tools" | "abort";

interface TrainToolCall {
	id: string;
	name: string;
	arguments: Record<string, unknown>;
}

interface TrainChatMessage {
	role: "user" | "assistant" | "tool";
	content: string;
	tool_calls?: TrainToolCall[];
	tool_call_id?: string;
	name?: string;
}

interface EditMeta {
	action: EditAction;
	changed: boolean;
	dismissed?: boolean;
	keep_tools?: boolean;
	tools_edited?: boolean;
}

interface EpisodeHeader {
	type: "episode";
	episode_id: string;
	task: string;
	session_id?: string;
	session_file?: string;
	cwd?: string;
	started_at: string;
}

interface TurnRecord {
	type: "turn";
	episode_id: string;
	turn: number;
	task: string;
	timestamp: string;
	context: TrainChatMessage[];
	agent_draft: TrainChatMessage;
	human_final: TrainChatMessage;
	edit: EditMeta;
}

interface EpisodeEnd {
	type: "episode_end";
	episode_id: string;
	success: boolean | null;
	note?: string;
	ended_at: string;
	turns: number;
	rewrites: number;
}

type DiskRecord = EpisodeHeader | TurnRecord | EpisodeEnd;

interface SlimEntry {
	kind: "episode_start" | "turn" | "episode_end";
	episode_id: string;
	task?: string;
	turn?: number;
	action?: EditAction;
	changed?: boolean;
	success?: boolean | null;
	note?: string;
}

interface Recording {
	episodeId: string;
	task: string;
	turn: number;
	rewrites: number;
	filePath: string;
}

function isAssistantMessage(message: AgentMessage): message is AssistantMessage {
	return message.role === "assistant" && Array.isArray(message.content);
}

function cloneAssistant(message: AssistantMessage): AssistantMessage {
	return structuredClone(message);
}

function textOf(message: AssistantMessage): string {
	return message.content
		.filter((block): block is TextContent => block.type === "text")
		.map((block) => block.text)
		.join("\n\n");
}

function toolCallsOf(message: AssistantMessage): ToolCall[] {
	return message.content.filter((block): block is ToolCall => block.type === "toolCall");
}

function stripThinking(message: AssistantMessage): AssistantMessage {
	const next = cloneAssistant(message);
	next.content = next.content.filter((block) => block.type !== "thinking");
	return next;
}

function setText(message: AssistantMessage, text: string): AssistantMessage {
	const next = cloneAssistant(message);
	const tools = next.content.filter((block): block is ToolCall => block.type === "toolCall");
	next.content = [{ type: "text", text }, ...tools];
	return next;
}

function dropToolCalls(message: AssistantMessage): AssistantMessage {
	return setToolCalls(message, []);
}

function toolsToJson(tools: ToolCall[]): string {
	return JSON.stringify(
		tools.map((call) => ({
			id: call.id,
			name: call.name,
			arguments: call.arguments ?? {},
		})),
		null,
		2,
	);
}

function parseToolsJson(raw: string): ToolCall[] | undefined {
	try {
		const parsed: unknown = JSON.parse(raw);
		if (!Array.isArray(parsed)) {
			return undefined;
		}
		const out: ToolCall[] = [];
		for (const item of parsed) {
			if (!item || typeof item !== "object" || Array.isArray(item)) {
				return undefined;
			}
			const rec = item as Record<string, unknown>;
			if (typeof rec.name !== "string" || rec.name.trim() === "") {
				return undefined;
			}
			const args = rec.arguments;
			if (args === undefined || args === null || typeof args !== "object" || Array.isArray(args)) {
				return undefined;
			}
			const id =
				typeof rec.id === "string" && rec.id.trim() !== ""
					? rec.id
					: `train-${Math.random().toString(16).slice(2, 10)}`;
			out.push({
				type: "toolCall",
				id,
				name: rec.name,
				arguments: args as Record<string, unknown>,
			});
		}
		return out;
	} catch {
		return undefined;
	}
}

function setToolCalls(message: AssistantMessage, tools: ToolCall[]): AssistantMessage {
	const next = cloneAssistant(message);
	const text = next.content.filter((block): block is TextContent => block.type === "text");
	next.content = [...text, ...tools];
	if (tools.length === 0) {
		if (next.stopReason === "toolUse") {
			next.stopReason = "stop";
		}
	} else {
		next.stopReason = "toolUse";
	}
	return next;
}

async function editToolCallsFromUi(
	ctx: ExtensionContext,
	message: AssistantMessage,
): Promise<{ message: AssistantMessage; edited: boolean; dismissed: boolean }> {
	const current = toolCallsOf(message);
	const edited = await ctx.ui.editor("改工具参数（JSON：id / name / arguments）", toolsToJson(current));
	if (edited === undefined) {
		return { message, edited: false, dismissed: true };
	}
	const parsed = parseToolsJson(edited);
	if (!parsed) {
		ctx.ui.notify("工具 JSON 无效，已保留原来的 tool call", "warning");
		return { message, edited: false, dismissed: false };
	}
	return { message: setToolCalls(message, parsed), edited: true, dismissed: false };
}

function userText(content: UserMessage["content"]): string {
	if (typeof content === "string") {
		return content;
	}
	return content
		.map((part) => (part.type === "text" ? part.text : "[image omitted]"))
		.join("\n");
}

function toolResultText(message: ToolResultMessage): string {
	return message.content
		.map((part) => (part.type === "text" ? part.text : "[image omitted]"))
		.join("\n");
}

function serializeAssistant(message: AssistantMessage): TrainChatMessage {
	const toolCalls = toolCallsOf(message).map((call) => ({
		id: call.id,
		name: call.name,
		arguments: call.arguments ?? {},
	}));
	const out: TrainChatMessage = {
		role: "assistant",
		content: textOf(message),
	};
	if (toolCalls.length > 0) {
		out.tool_calls = toolCalls;
	}
	return out;
}

function serializeAgentMessage(message: AgentMessage): TrainChatMessage | undefined {
	if (message.role === "user") {
		return { role: "user", content: userText(message.content) };
	}
	if (isAssistantMessage(message)) {
		return serializeAssistant(message);
	}
	if (message.role === "toolResult") {
		return {
			role: "tool",
			content: toolResultText(message),
			tool_call_id: message.toolCallId,
			name: message.toolName,
		};
	}
	return undefined;
}

function snapshotContext(branch: SessionEntry[]): TrainChatMessage[] {
	const out: TrainChatMessage[] = [];
	for (const entry of branch) {
		if (entry.type !== "message") {
			continue;
		}
		const serialized = serializeAgentMessage(entry.message);
		if (serialized) {
			out.push(serialized);
		}
	}
	return out;
}

function sameAssistant(a: TrainChatMessage, b: TrainChatMessage): boolean {
	return JSON.stringify(a) === JSON.stringify(b);
}

function trainDir(): string {
	return join(getAgentDir(), "train");
}

function newEpisodeId(): string {
	const now = new Date();
	const stamp = now.toISOString().replace(/[-:]/g, "").replace("T", "-").slice(0, 15);
	const rand = Math.random().toString(16).slice(2, 6);
	return `train-${stamp}-${rand}`;
}

function appendJsonl(filePath: string, record: DiskRecord): void {
	mkdirSync(dirname(filePath), { recursive: true });
	appendFileSync(filePath, `${JSON.stringify(record)}\n`, "utf8");
}

function readJsonl(filePath: string): DiskRecord[] {
	const text = readFileSync(filePath, "utf8");
	const records: DiskRecord[] = [];
	for (const line of text.split("\n")) {
		if (!line.trim()) {
			continue;
		}
		records.push(JSON.parse(line) as DiskRecord);
	}
	return records;
}

function usage(): string {
	return [
		"/train start [任务]  开始采集",
		"/train stop          结束采集",
		"/train success [备注] 标记成功",
		"/train fail [备注]    标记失败",
		"/train status        当前状态",
		"/train export [目录]  写出 JSONL",
	].join("\n");
}

export default function (pi: ExtensionAPI): void {
	let recording: Recording | undefined;
	let lastEnded: Recording | undefined;
	let blockTools = false;

	pi.registerFlag("train", {
		description: "Start the session in training capture mode",
		type: "boolean",
		default: false,
	});

	function updateUi(ctx: ExtensionContext): void {
		if (!ctx.hasUI) {
			return;
		}
		if (!recording) {
			ctx.ui.setStatus("train-capture", undefined);
			ctx.ui.setWidget("train-capture", undefined);
			return;
		}
		const label = `TRAIN t${recording.turn} · ${recording.rewrites}改`;
		ctx.ui.setStatus("train-capture", ctx.ui.theme.fg("warning", label));
		ctx.ui.setWidget("train-capture", [
			ctx.ui.theme.fg("warning", "TRAIN 采集中"),
			`任务: ${recording.task}`,
			`回合 ${recording.turn} · 改写 ${recording.rewrites}`,
			ctx.ui.theme.fg("dim", recording.episodeId),
		]);
	}

	function persistSlim(data: SlimEntry): void {
		pi.appendEntry(CUSTOM_TYPE, data);
	}

	function startRecording(ctx: ExtensionContext, task: string): Recording {
		const episodeId = newEpisodeId();
		const filePath = join(trainDir(), `${episodeId}.jsonl`);
		recording = {
			episodeId,
			task: task.trim() || "untitled",
			turn: 0,
			rewrites: 0,
			filePath,
		};
		appendJsonl(filePath, {
			type: "episode",
			episode_id: episodeId,
			task: recording.task,
			session_id: ctx.sessionManager.getSessionId(),
			session_file: ctx.sessionManager.getSessionFile(),
			cwd: ctx.cwd,
			started_at: new Date().toISOString(),
		});
		persistSlim({
			kind: "episode_start",
			episode_id: episodeId,
			task: recording.task,
		});
		pi.setSessionName(`train: ${recording.task}`.slice(0, 80));
		updateUi(ctx);
		return recording;
	}

	function endRecording(
		ctx: ExtensionContext,
		success: boolean | null,
		note?: string,
	): Recording | undefined {
		if (!recording) {
			return undefined;
		}
		const ended = recording;
		appendJsonl(ended.filePath, {
			type: "episode_end",
			episode_id: ended.episodeId,
			success,
			note,
			ended_at: new Date().toISOString(),
			turns: ended.turn,
			rewrites: ended.rewrites,
		});
		persistSlim({
			kind: "episode_end",
			episode_id: ended.episodeId,
			success,
			note,
			turn: ended.turn,
		});
		lastEnded = ended;
		recording = undefined;
		blockTools = false;
		updateUi(ctx);
		return ended;
	}

	function restoreFromSession(ctx: ExtensionContext): void {
		recording = undefined;
		blockTools = false;
		let restored: Recording | undefined;
		for (const entry of ctx.sessionManager.getBranch()) {
			if (entry.type !== "custom" || entry.customType !== CUSTOM_TYPE) {
				continue;
			}
			const data = entry.data as SlimEntry | undefined;
			if (!data) {
				continue;
			}
			if (data.kind === "episode_start" && data.episode_id) {
				restored = {
					episodeId: data.episode_id,
					task: data.task || "untitled",
					turn: 0,
					rewrites: 0,
					filePath: join(trainDir(), `${data.episode_id}.jsonl`),
				};
			} else if (data.kind === "turn" && restored && data.episode_id === restored.episodeId) {
				restored.turn = data.turn ?? restored.turn;
				if (data.changed) {
					restored.rewrites += 1;
				}
			} else if (data.kind === "episode_end" && restored && data.episode_id === restored.episodeId) {
				restored = undefined;
			}
		}
		recording = restored;
		updateUi(ctx);
	}

	function exportEpisode(filePath: string, outDir: string): { sft: number; pref: number } {
		const records = readJsonl(filePath);
		const header = records.find((row): row is EpisodeHeader => row.type === "episode");
		const end = [...records].reverse().find((row): row is EpisodeEnd => row.type === "episode_end");
		const turns = records.filter((row): row is TurnRecord => row.type === "turn");
		mkdirSync(outDir, { recursive: true });

		const sftLines: string[] = [];
		const prefLines: string[] = [];
		for (const turn of turns) {
			sftLines.push(
				JSON.stringify({
					episode_id: turn.episode_id,
					turn: turn.turn,
					task: turn.task,
					messages: [...turn.context, turn.human_final],
				}),
			);
			if (turn.edit.changed) {
				prefLines.push(
					JSON.stringify({
						episode_id: turn.episode_id,
						turn: turn.turn,
						task: turn.task,
						messages: turn.context,
						chosen: turn.human_final,
						rejected: turn.agent_draft,
					}),
				);
			}
		}

		const last = turns[turns.length - 1];
		const traj = last
			? [
					JSON.stringify({
						episode_id: last.episode_id,
						task: last.task,
						success: end?.success ?? null,
						messages: [...last.context, last.human_final],
					}),
				]
			: [];

		writeFileSync(join(outDir, "sft.jsonl"), sftLines.map((line) => `${line}\n`).join(""), "utf8");
		writeFileSync(join(outDir, "pref.jsonl"), prefLines.map((line) => `${line}\n`).join(""), "utf8");
		writeFileSync(join(outDir, "traj.jsonl"), traj.map((line) => `${line}\n`).join(""), "utf8");
		writeFileSync(
			join(outDir, "manifest.json"),
			`${JSON.stringify(
				{
					episode_id: header?.episode_id,
					task: header?.task,
					success: end?.success ?? null,
					turns: turns.length,
					rewrites: turns.filter((turn) => turn.edit.changed).length,
					source: filePath,
					exported_at: new Date().toISOString(),
				},
				null,
				2,
			)}\n`,
			"utf8",
		);
		return { sft: sftLines.length, pref: prefLines.length };
	}

	pi.registerEntryRenderer<SlimEntry>(CUSTOM_TYPE, (entry, _options, theme) => {
		const data = entry.data;
		if (!data) {
			return undefined;
		}
		let label = "TRAIN";
		if (data.kind === "episode_start") {
			label = `TRAIN 开始 · ${data.task ?? ""}`;
		} else if (data.kind === "turn") {
			const mark = data.changed ? data.action ?? "rewrite" : "accept";
			label = `TRAIN t${data.turn ?? "?"} · ${mark}`;
		} else if (data.kind === "episode_end") {
			const result = data.success === true ? "success" : data.success === false ? "fail" : "stop";
			label = `TRAIN 结束 · ${result}`;
		}
		const box = new Box(0, 0);
		box.addChild(new Text(theme.fg("dim", label), 0, 0));
		return box;
	});

	pi.on("session_start", (event, ctx) => {
		restoreFromSession(ctx);
		if (event.reason === "startup" && pi.getFlag("train") === true && !recording) {
			startRecording(ctx, "cli --train");
			if (ctx.hasUI) {
				ctx.ui.notify("已按 --train 开始采集。用 /train success 结束。", "info");
			}
		}
	});

	pi.on("session_shutdown", (_event, ctx) => {
		updateUi(ctx);
	});

	pi.on("turn_end", () => {
		blockTools = false;
	});

	pi.on("tool_call", (event) => {
		if (event.parentToolCallId) {
			return;
		}
		if (!recording || !blockTools) {
			return;
		}
		return {
			block: true,
			reason: "train-capture: trainer dropped or aborted this turn",
			terminate: true,
		};
	});

	pi.on("message_end", async (event, ctx) => {
		if (!recording) {
			return;
		}
		if (!isAssistantMessage(event.message)) {
			return;
		}
		if (event.message.stopReason === "error" || event.message.stopReason === "aborted") {
			return;
		}
		const hasText = textOf(event.message).trim().length > 0;
		const hasTools = toolCallsOf(event.message).length > 0;
		if (!hasText && !hasTools) {
			return;
		}

		const draft = cloneAssistant(event.message);
		const agentSnapshot = serializeAssistant(draft);
		let finalMessage = cloneAssistant(draft);
		let action: EditAction = "accept";
		let dismissed = false;
		let keepTools: boolean | undefined;
		let toolsEdited = false;

		if (!ctx.hasUI) {
			action = "accept";
		} else {
			const options = hasTools ? [ACCEPT, EDIT, EDIT_TOOLS, DROP_TOOLS, ABORT] : [ACCEPT, EDIT, ABORT];
			const preview = (textOf(draft).trim() || JSON.stringify(toolCallsOf(draft).map((call) => call.name))).slice(
				0,
				180,
			);
			const choice = await ctx.ui.select(`TRAIN t${recording.turn + 1}  ${preview}`, options);
			if (!choice) {
				action = "accept";
				dismissed = true;
			} else if (choice === EDIT || choice === EDIT_TOOLS) {
				if (choice === EDIT) {
					const edited = await ctx.ui.editor("改写 assistant 正文", textOf(draft));
					if (edited !== undefined) {
						finalMessage = setText(finalMessage, edited);
						action = "rewrite";
					} else {
						dismissed = true;
					}
				}
				if (toolCallsOf(finalMessage).length > 0) {
					const toolChoice =
						choice === EDIT_TOOLS
							? EDIT_TOOLS
							: await ctx.ui.select("工具调用", [KEEP_TOOLS, EDIT_TOOLS, DROP_TOOLS]);
					if (toolChoice === DROP_TOOLS) {
						finalMessage = dropToolCalls(finalMessage);
						keepTools = false;
						action = "rewrite";
					} else if (toolChoice === EDIT_TOOLS) {
						const result = await editToolCallsFromUi(ctx, finalMessage);
						finalMessage = result.message;
						if (result.edited) {
							toolsEdited = true;
							keepTools = toolCallsOf(finalMessage).length > 0;
							action = "rewrite";
							if (!keepTools) {
								action = "drop_tools";
							}
						} else if (result.dismissed && choice === EDIT_TOOLS) {
							dismissed = true;
						}
					} else {
						keepTools = true;
					}
				}
			} else if (choice === DROP_TOOLS) {
				finalMessage = dropToolCalls(finalMessage);
				action = "drop_tools";
				keepTools = false;
			} else if (choice === ABORT) {
				finalMessage = dropToolCalls(finalMessage);
				action = "abort";
				keepTools = false;
			} else {
				action = "accept";
			}
		}

		const humanSnapshot = serializeAssistant(finalMessage);
		const changed = !sameAssistant(agentSnapshot, humanSnapshot);
		recording.turn += 1;
		if (changed) {
			recording.rewrites += 1;
		}

		const turn: TurnRecord = {
			type: "turn",
			episode_id: recording.episodeId,
			turn: recording.turn,
			task: recording.task,
			timestamp: new Date().toISOString(),
			context: snapshotContext(ctx.sessionManager.getBranch()),
			agent_draft: agentSnapshot,
			human_final: humanSnapshot,
			edit: {
				action,
				changed,
				...(dismissed ? { dismissed: true } : {}),
				...(keepTools === undefined ? {} : { keep_tools: keepTools }),
				...(toolsEdited ? { tools_edited: true } : {}),
			},
		};
		appendJsonl(recording.filePath, turn);
		persistSlim({
			kind: "turn",
			episode_id: recording.episodeId,
			turn: recording.turn,
			action,
			changed,
		});
		updateUi(ctx);

		if (action === "drop_tools" || action === "abort") {
			blockTools = true;
		}
		if (action === "abort") {
			ctx.abort();
		}

		if (changed) {
			// Drop this turn's CoT so the next request sees gold as the model's own
			// assistant output. Keeping original thinking next to rewritten text/tools
			// makes reasoning models treat the gold as a user injection.
			return { message: stripThinking(finalMessage) };
		}
		return;
	});

	pi.registerCommand("train", {
		description: "采集人机共写训练样本（start / stop / success / fail / export）",
		getArgumentCompletions: (prefix) => {
			const commands = ["start", "stop", "success", "fail", "status", "export"];
			return commands
				.filter((name) => name.startsWith(prefix.trim()))
				.map((name) => ({ value: name, label: name }));
		},
		handler: async (args, ctx) => {
			const trimmed = args.trim();
			const space = trimmed.indexOf(" ");
			const sub = (space === -1 ? trimmed : trimmed.slice(0, space)).toLowerCase();
			const rest = space === -1 ? "" : trimmed.slice(space + 1).trim();

			if (!sub || sub === "status" || sub === "help") {
				if (!recording) {
					ctx.ui.notify(usage(), "info");
					return;
				}
				ctx.ui.notify(
					`采集中 ${recording.episodeId}\n任务: ${recording.task}\n回合 ${recording.turn} · 改写 ${recording.rewrites}\n${recording.filePath}`,
					"info",
				);
				return;
			}

			if (sub === "start") {
				if (recording) {
					ctx.ui.notify(`已经在采集 ${recording.episodeId}，先 /train stop`, "warning");
					return;
				}
				let task = rest;
				if (!task && ctx.hasUI) {
					task = (await ctx.ui.input("这次采集的任务是什么？"))?.trim() ?? "";
				}
				if (!task) {
					task = "untitled";
				}
				const started = startRecording(ctx, task);
				ctx.ui.notify(`开始采集 ${started.episodeId}`, "info");
				return;
			}

			if (sub === "stop" || sub === "success" || sub === "fail") {
				if (!recording) {
					ctx.ui.notify("当前没有采集中的 episode", "warning");
					return;
				}
				const success = sub === "success" ? true : sub === "fail" ? false : null;
				const ended = endRecording(ctx, success, rest || undefined);
				ctx.ui.notify(
					`已结束 ${ended?.episodeId}（${sub}，${ended?.turns ?? 0} 回合）\n接着 /train export`,
					"info",
				);
				return;
			}

			if (sub === "export") {
				const target = recording ?? lastEnded;
				const filePath = target?.filePath;
				const fallbackId = target?.episodeId;
				if (!filePath) {
					ctx.ui.notify("没有可导出的 episode。先 /train start。", "warning");
					return;
				}
				const outDir = resolve(ctx.cwd, rest || join("train-export", fallbackId ?? "episode"));
				try {
					const counts = exportEpisode(filePath, outDir);
					ctx.ui.notify(`已导出到 ${outDir}\nSFT ${counts.sft} 条 · 偏好 ${counts.pref} 条`, "info");
				} catch (error) {
					const message = error instanceof Error ? error.message : String(error);
					ctx.ui.notify(`导出失败: ${message}`, "error");
				}
				return;
			}

			ctx.ui.notify(usage(), "warning");
		},
	});
}
