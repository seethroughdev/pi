import { afterEach, describe, expect, it, vi } from "vitest";
import { stream } from "../src/api/openai-responses.ts";
import { getModel, normalizeContext } from "../src/compat.ts";
import type { AssistantMessage, Context, Model } from "../src/types.ts";

type InputItem = { type?: string; role?: string; content?: unknown; reasoning?: { effort?: string } };
type Payload = { model: string; input: InputItem[]; reasoning?: { effort?: string }; [key: string]: unknown };

const user = (content: string, timestamp: number) => ({ role: "user" as const, content, timestamp });
const completedSse = `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_resp","role":"assistant","status":"in_progress","content":[]}}

data: {"type":"response.output_text.delta","output_index":0,"item_id":"msg_resp","delta":"answer"}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_resp","role":"assistant","status":"completed","content":[{"type":"output_text","text":"answer","annotations":[]}]}}

data: {"type":"response.completed","response":{"id":"resp","status":"completed"}}

`;

function supportedModel(
	modelId: "gpt-6-sol" | "gpt-6-astra" | "gpt-6.1-sol" | "gpt-5.4" = "gpt-6-sol",
): Model<"openai-responses"> {
	return getModel("openai", modelId);
}

function response(model: Model<"openai-responses">, effort?: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "answer" }],
		api: model.api,
		provider: model.provider,
		model: model.id,
		...(effort === undefined ? {} : { providerThinkingLevel: effort }),
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 1,
	};
}

async function capture(
	model: Model<"openai-responses">,
	context: Context,
	reasoning: "low" | "high" | "off",
	mutate?: (payload: Payload) => void,
	apiKey = "sk-test-key",
	reasoningSummary?: "auto" | "detailed" | "concise" | null,
): Promise<{ payload: Payload; message: AssistantMessage }> {
	let payload: Payload | undefined;
	vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
		payload = JSON.parse(String(init?.body)) as Payload;
		return new Response(completedSse, {
			status: 200,
			headers: { "content-type": "text/event-stream" },
		});
	});
	const result = await stream(model, normalizeContext(context), {
		apiKey,
		reasoningEffort: reasoning === "off" ? undefined : reasoning,
		reasoningSummary,
		cacheRetention: "none",
		onPayload: (value) => {
			const captured = value as Payload;
			payload = captured;
			mutate?.(captured);
			return captured;
		},
	}).result();
	if (!payload) throw new Error("Expected request payload");
	return { payload, message: result };
}

function configurationUpdates(payload: Payload): string[] {
	return payload.input
		.filter((item) => item.type === "configuration_update")
		.map((item) => item.reasoning?.effort ?? "");
}

// Regression coverage for https://github.com/earendil-works/pi/issues/9335.
describe("OpenAI Responses mid-conversation effort", () => {
	afterEach(() => vi.restoreAllMocks());

	it("annotates only the selected public Responses models with their native mappings", () => {
		for (const id of ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"] as const) {
			expect(getModel("openai", id).compat?.supportsMidConvoEffort).toBe(true);
			expect(getModel("openai", id).thinkingLevelMap?.minimal).toBeNull();
		}
		expect(getModel("openai", "gpt-6-astra").thinkingLevelMap?.off).toBeNull();
		expect(getModel("openai", "gpt-6.1-sol").thinkingLevelMap?.off).toBeNull();
		expect(getModel("openai", "gpt-6-sol").thinkingLevelMap?.off).toBe("none");
		expect(getModel("openai", "gpt-6-luna").thinkingLevelMap?.off).toBe("none");
		expect(getModel("openai", "gpt-5.4").compat?.supportsMidConvoEffort).toBeUndefined();
		expect(getModel("openai-codex", "gpt-6-sol").compat?.supportsMidConvoEffort).toBeUndefined();
		expect(getModel("azure", "gpt-6-sol").compat?.supportsMidConvoEffort).toBeUndefined();
	});

	it("maps Pi off to native none only on models that accept it", async () => {
		const result = await capture(supportedModel(), { messages: [user("one", 1)] }, "off");
		expect(result.payload.reasoning).toEqual({ effort: "none" });
		expect(result.message.providerThinkingLevel).toBe("none");

		const astraModel = supportedModel("gpt-6-astra");
		const astra = await capture(astraModel, { messages: [user("one", 1)] }, "off");
		expect(astra.payload.reasoning).toBeUndefined();
		expect(astra.message.providerThinkingLevel).toBeUndefined();
		const astraLow = await capture(astraModel, { messages: [user("one", 1)] }, "low");
		const astraOff = await capture(
			astraModel,
			{ messages: [user("one", 1), astraLow.message, user("two", 2)] },
			"off",
		);
		expect(astraOff.payload.reasoning?.effort).toBe("low");
		expect(configurationUpdates(astraOff.payload)).toEqual([]);
		expect(astraOff.message.providerThinkingLevel).toBe("low");
	});

	it("rejects explicitly unsupported minimal effort before transport or metadata stamping", async () => {
		const fetch = vi.spyOn(globalThis, "fetch");
		const result = await stream(supportedModel(), normalizeContext({ messages: [user("one", 1)] }), {
			apiKey: "sk-test-key",
			reasoningEffort: "minimal",
			cacheRetention: "none",
		}).result();

		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("Unsupported reasoning effort minimal");
		expect(result.providerThinkingLevel).toBeUndefined();
		expect(fetch).not.toHaveBeenCalled();
	});

	it("maps summary-only requests to their documented default native effort", async () => {
		const result = await capture(
			supportedModel(),
			{ messages: [user("one", 1)] },
			"off",
			undefined,
			"sk-test-key",
			"detailed",
		);
		expect(result.payload.reasoning).toEqual({ effort: "medium", summary: "detailed" });
		expect(result.message.providerThinkingLevel).toBe("medium");
	});

	it("keeps reasoning and include fields stable across transitions to and from native none", async () => {
		const model = supportedModel();
		const off = await capture(model, { messages: [user("one", 1)] }, "off");
		const fromOff = await capture(model, { messages: [user("one", 1), off.message, user("two", 2)] }, "low");
		expect(fromOff.payload.reasoning).toEqual({ effort: "none" });
		expect(fromOff.payload.include).toBeUndefined();
		expect(configurationUpdates(fromOff.payload)).toEqual(["low"]);

		const low = await capture(model, { messages: [user("one", 1)] }, "low");
		const toOff = await capture(model, { messages: [user("one", 1), low.message, user("two", 2)] }, "off");
		expect(toOff.payload.reasoning).toEqual({ effort: "low", summary: "auto" });
		expect(toOff.payload.include).toEqual(["reasoning.encrypted_content"]);
		expect(configurationUpdates(toOff.payload)).toEqual(["none"]);
	});

	it("allows ChatGPT-plan OAuth on the public Responses endpoint", async () => {
		const result = await capture(supportedModel(), { messages: [user("one", 1)] }, "low", undefined, "oauth-token");
		expect(result.payload.model).toBe("gpt-6-sol");
		expect(result.message.providerThinkingLevel).toBe("low");
	});

	it("replays a captured context without mutating it", async () => {
		const context: Context = { messages: [user("one", 1), response(supportedModel(), "low"), user("two", 2)] };
		const original = structuredClone(context);
		const first = await capture(supportedModel(), context, "high");
		const second = await capture(supportedModel(), context, "high");
		expect(first.payload.input).toEqual(second.payload.input);
		expect(configurationUpdates(first.payload)).toEqual(["high"]);
		expect(context).toEqual(original);
	});

	it("replays the captured payload unchanged on retry", async () => {
		const bodies: string[] = [];
		vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
			bodies.push(String(init?.body));
			if (bodies.length === 1) {
				return new Response('{"error":{"message":"retry"}}', {
					status: 500,
					headers: { "content-type": "application/json", "x-should-retry": "true" },
				});
			}
			return new Response(completedSse, {
				status: 200,
				headers: { "content-type": "text/event-stream" },
			});
		});
		const result = await stream(supportedModel(), normalizeContext({ messages: [user("one", 1)] }), {
			apiKey: "sk-test-key",
			reasoningEffort: "low",
			cacheRetention: "none",
			maxRetries: 1,
			maxRetryDelayMs: 0,
		}).result();
		expect(bodies).toHaveLength(2);
		expect(bodies[1]).toBe(bodies[0]);
		expect(result.providerThinkingLevel).toBe("low");
	});

	it("keeps the initial effort and replays low → high → low before each user turn", async () => {
		const model = supportedModel();
		const first = await capture(model, { messages: [user("one", 1)] }, "low");
		const second = await capture(model, { messages: [user("one", 1), first.message, user("two", 2)] }, "high");
		const third = await capture(
			model,
			{ messages: [user("one", 1), first.message, user("two", 2), second.message, user("three", 3)] },
			"low",
		);

		expect(first.payload.reasoning?.effort).toBe("low");
		expect(first.message.providerThinkingLevel).toBe("low");
		expect(second.payload.reasoning?.effort).toBe("low");
		expect(configurationUpdates(second.payload)).toEqual(["high"]);
		expect(second.message.providerThinkingLevel).toBe("high");
		expect(third.payload.reasoning?.effort).toBe("low");
		expect(configurationUpdates(third.payload)).toEqual(["high", "low"]);
		expect(second.payload.input).toEqual([
			{ role: "user", content: [{ type: "input_text", text: "one" }] },
			{
				type: "message",
				role: "assistant",
				id: "msg_resp",
				status: "completed",
				content: [{ type: "output_text", text: "answer", annotations: [] }],
			},
			{ type: "configuration_update", reasoning: { effort: "high" } },
			{ role: "user", content: [{ type: "input_text", text: "two" }] },
		]);
		expect(third.payload.input).toEqual([
			{ role: "user", content: [{ type: "input_text", text: "one" }] },
			{
				type: "message",
				role: "assistant",
				id: "msg_resp",
				status: "completed",
				content: [{ type: "output_text", text: "answer", annotations: [] }],
			},
			{ type: "configuration_update", reasoning: { effort: "high" } },
			{ role: "user", content: [{ type: "input_text", text: "two" }] },
			{
				type: "message",
				role: "assistant",
				id: "msg_resp",
				status: "completed",
				content: [{ type: "output_text", text: "answer", annotations: [] }],
			},
			{ type: "configuration_update", reasoning: { effort: "low" } },
			{ role: "user", content: [{ type: "input_text", text: "three" }] },
		]);
		expect(third.message.providerThinkingLevel).toBe("low");
	});

	it("coalesces repeated selections and skips empty user inputs", async () => {
		const model = supportedModel();
		const first = response(model, "low");
		const second = response(model, "high");
		const result = await capture(
			model,
			{
				messages: [user("one", 1), first, user("  ", 2), user("two", 3), second, user("three", 4)],
			},
			"high",
		);
		expect(configurationUpdates(result.payload)).toEqual(["high"]);
	});

	it("keeps the current effective effort during tool continuations", async () => {
		const model = supportedModel();
		const previousTurn = response(model, "low");
		const toolTurn = response(model, "high");
		toolTurn.content = [{ type: "toolCall", id: "call_1|fc_1", name: "run", arguments: {} }];
		const result = await capture(
			model,
			{
				messages: [
					user("one", 1),
					previousTurn,
					user("run it", 2),
					toolTurn,
					{
						role: "toolResult",
						toolCallId: "call_1|fc_1",
						toolName: "run",
						content: [],
						isError: false,
						timestamp: 2,
					},
				],
			},
			"low",
		);
		expect(result.payload.reasoning?.effort).toBe("low");
		expect(configurationUpdates(result.payload)).toEqual(["high"]);
		expect(result.message.providerThinkingLevel).toBe("high");

		const nextUser = await capture(
			model,
			{
				messages: [
					user("one", 1),
					previousTurn,
					user("run it", 2),
					toolTurn,
					{
						role: "toolResult",
						toolCallId: "call_1|fc_1",
						toolName: "run",
						content: [],
						isError: false,
						timestamp: 2,
					},
					user("new user", 3),
				],
			},
			"low",
		);
		expect(configurationUpdates(nextUser.payload)).toEqual(["high", "low"]);
		expect(nextUser.payload.input.at(-2)).toEqual({ type: "configuration_update", reasoning: { effort: "low" } });
		expect(nextUser.payload.input.at(-1)).toEqual({
			role: "user",
			content: [{ type: "input_text", text: "new user" }],
		});
	});

	it("keeps effort updates aligned when transformation inserts tool results and removes failed turns", async () => {
		const model = supportedModel();
		const toolTurn = response(model, "low");
		toolTurn.content = [{ type: "toolCall", id: "call_1|fc_1", name: "run", arguments: {} }];
		toolTurn.stopReason = "toolUse";
		const failedTurn = { ...response(model, "high"), stopReason: "error" as const };
		const result = await capture(
			model,
			{
				systemPrompt: "Initial instructions",
				messages: [
					user("one", 1),
					toolTurn,
					{ role: "system", content: "Later instructions", timestamp: 2 },
					failedTurn,
					user("two", 3),
				],
			},
			"high",
		);
		expect(result.payload.reasoning?.effort).toBe("low");
		expect(configurationUpdates(result.payload)).toEqual(["high"]);
		expect(result.payload.input.slice(-4)).toEqual([
			{ type: "function_call_output", call_id: "call_1", output: "No result provided" },
			{ role: "developer", content: "Later instructions" },
			{ type: "configuration_update", reasoning: { effort: "high" } },
			{ role: "user", content: [{ type: "input_text", text: "two" }] },
		]);
		expect(result.message.providerThinkingLevel).toBe("high");
	});

	it("does not apply a new selection backward to a completed legacy turn", async () => {
		const model = supportedModel();
		const completedLegacyTurn = response(model);
		const result = await capture(
			model,
			{ messages: [user("one", 1), response(model, "low"), user("two", 2), completedLegacyTurn] },
			"high",
		);
		expect(result.payload.reasoning?.effort).toBe("low");
		expect(configurationUpdates(result.payload)).toEqual([]);
	});

	it.each(["error", "aborted", "pending"] as const)("ignores %s assistant effort metadata", async (stopReason) => {
		const failed = { ...response(supportedModel(), "high"), stopReason } as AssistantMessage;
		const result = await capture(supportedModel(), { messages: [user("one", 1), failed, user("two", 2)] }, "low");
		expect(result.payload.reasoning?.effort).toBe("low");
		expect(configurationUpdates(result.payload)).toEqual([]);
	});

	it("treats a model change as a fresh baseline and discards earlier updates", async () => {
		const model = supportedModel();
		const otherModel = { ...response(model, "high"), model: "gpt-6-luna" };
		const result = await capture(
			model,
			{ messages: [user("one", 1), response(model, "low"), user("other", 2), otherModel, user("new baseline", 3)] },
			"high",
		);
		expect(result.payload.reasoning?.effort).toBe("high");
		expect(configurationUpdates(result.payload)).toEqual([]);
	});

	it("starts a fresh baseline when history has no applicable effort metadata", async () => {
		const model = supportedModel();
		const legacy = response(model);
		const otherModel = { ...response(model, "low"), model: "gpt-6-luna" };
		const result = await capture(model, { messages: [user("one", 1), legacy, otherModel, user("two", 2)] }, "high");
		expect(result.payload.reasoning?.effort).toBe("high");
		expect(configurationUpdates(result.payload)).toEqual([]);
	});

	it("leaves unsupported model, endpoint, API, and explicit false-capability payloads unchanged", async () => {
		const unsupported = supportedModel("gpt-5.4");
		const customEndpoint: Model<"openai-responses"> = {
			...supportedModel(),
			baseUrl: "https://proxy.example/v1",
		};
		const disabled: Model<"openai-responses"> = {
			...supportedModel(),
			compat: { ...supportedModel().compat, supportsMidConvoEffort: false },
		};
		const wrongApi = {
			...supportedModel(),
			api: "azure-openai-responses",
		} as unknown as Model<"openai-responses">;
		for (const model of [unsupported, customEndpoint, disabled, wrongApi]) {
			const result = await capture(model, { messages: [user("one", 1)] }, "low", (payload) => {
				payload.temperature = 0.2;
			});
			expect(configurationUpdates(result.payload)).toEqual([]);
			expect(result.message.providerThinkingLevel).toBeUndefined();
			expect(result.payload.temperature).toBe(0.2);
		}

		const whitespace = await capture(unsupported, { messages: [user("  ", 1)] }, "low");
		expect(whitespace.payload.input).toEqual([{ role: "user", content: [{ type: "input_text", text: "  " }] }]);
	});

	it("does not record effort metadata for failed responses", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValue(
			new Response(
				'data: {"type":"response.failed","response":{"status":"failed","error":{"code":"failed","message":"failed"}}}\n\n',
				{
					status: 200,
					headers: { "content-type": "text/event-stream" },
				},
			),
		);
		const result = await stream(supportedModel(), normalizeContext({ messages: [user("one", 1)] }), {
			apiKey: "sk-test-key",
			reasoningEffort: "low",
			cacheRetention: "none",
		}).result();
		expect(result.stopReason).toBe("error");
		expect(result.providerThinkingLevel).toBeUndefined();
	});

	it("rejects sampling overrides and hooks that conflict with managed request state", async () => {
		const model: Model<"openai-responses"> = {
			...supportedModel(),
			samplingParams: { reasoning: { effort: "high" } },
		};
		const sampled = await capture(model, { messages: [user("one", 1)] }, "low");
		expect(sampled.message.stopReason).toBe("error");
	});

	it("rejects overrides that change the protected payload or enable automatic truncation", async () => {
		const model = supportedModel();
		for (const mutate of [
			(payload: Payload) => {
				payload.model = "gpt-5.4";
			},
			(payload: Payload) => {
				payload.input = [];
			},
			(payload: Payload) => {
				payload.reasoning = { effort: "high" };
			},
			(payload: Payload) => {
				payload.truncation = "auto";
			},
			(payload: Payload) => {
				payload.store = true;
			},
			(payload: Payload) => {
				payload.context_management = {};
			},
			(payload: Payload) => {
				payload.previous_response_id = "resp_remote";
			},
			(payload: Payload) => {
				payload.background = true;
			},
			(payload: Payload) => {
				payload.stream = false;
			},
		]) {
			const result = await capture(model, { messages: [user("one", 1)] }, "low", mutate);
			expect(result.message.stopReason).toBe("error");
			expect(result.message.providerThinkingLevel).toBeUndefined();
		}
	});
});
