// Reading Server-Sent Events from a POST request.
//
// The browser's built-in EventSource can only send GET requests without custom
// headers, so it cannot carry our question or the sign-in token. Instead the
// answer of an ordinary fetch() is read piece by piece and split into events here.
//
// The format (see app/api/sse.py):   event: progress\n data: {...}\n \n
// A line that starts with ":" is a keep-alive comment and is ignored.

export interface RawEvent {
  event: string;
  data: string;
}

/** Splits text that arrives in arbitrary pieces into complete events. */
export class SseParser {
  private buffer = "";

  /** Add the next piece of the response. Returns the events it completed. */
  push(chunk: string): RawEvent[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const events: RawEvent[] = [];
    let end = this.buffer.indexOf("\n\n");
    while (end !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      const parsed = parseBlock(block);
      if (parsed) events.push(parsed);
      end = this.buffer.indexOf("\n\n");
    }
    return events;
  }
}

function parseBlock(block: string): RawEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // comment (keep-alive)
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

/** Read a streamed response to its end, calling `onEvent` for every event. */
export async function readEvents(response: Response, onEvent: (event: RawEvent) => void): Promise<void> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const event of parser.push(decoder.decode(value, { stream: true }))) onEvent(event);
  }
  for (const event of parser.push(decoder.decode() + "\n\n")) onEvent(event);
}
