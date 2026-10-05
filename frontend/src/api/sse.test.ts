import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("returns complete events and keeps the unfinished rest", () => {
    const parser = new SseParser();
    expect(parser.push('event: progress\ndata: {"node":"retrieve"}\n\nevent: rep')).toEqual([
      { event: "progress", data: '{"node":"retrieve"}' },
    ]);
    expect(parser.push('ort\ndata: {"a":1}\n\n')).toEqual([{ event: "report", data: '{"a":1}' }]);
  });

  it("handles an event that arrives one character at a time", () => {
    const parser = new SseParser();
    const text = 'event: answer\ndata: {"answer":"Total £58,000"}\n\n';
    const events = [...text].flatMap((character) => parser.push(character));
    expect(events).toEqual([{ event: "answer", data: '{"answer":"Total £58,000"}' }]);
  });

  it("ignores keep-alive comments and accepts Windows line endings", () => {
    const parser = new SseParser();
    expect(parser.push(": keep-alive\n\n")).toEqual([]);
    expect(parser.push("event: tool\r\ndata: {}\r\n\r\n")).toEqual([{ event: "tool", data: "{}" }]);
  });

  it("joins data that is spread over several lines", () => {
    const parser = new SseParser();
    expect(parser.push("data: first\ndata: second\n\n")).toEqual([{ event: "message", data: "first\nsecond" }]);
  });
});
