/**
 * chatApi.js
 * -------------
 * Wraps the fetch() call to the FastAPI backend's POST /api/chat endpoint
 * in a single function, sendMessage(), so components never call fetch()
 * directly -- the same "isolate the external call" principle from the
 * backend's llm_client.py (Day 11), now applied on the frontend.
 *
 * WHY EXPLICIT ERROR HANDLING HERE MATTERS:
 * fetch() itself only rejects (throws) on a genuine NETWORK failure (the
 * server is unreachable, DNS fails, etc.) -- it does NOT reject just
 * because the server responded with an error status like 400 or 500.
 * A response with status 500 is still a "successful" fetch from
 * JavaScript's point of view; you have to check response.ok yourself.
 * On top of that, response.json() can ALSO throw if the body isn't
 * actually valid JSON (e.g. the server crashed and returned an HTML
 * error page instead) -- both failure modes are handled explicitly below
 * rather than left to crash the UI with an unhandled promise rejection.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export class ChatApiError extends Error {
  constructor(message, status, details) {
    super(message);
    this.name = "ChatApiError";
    this.status = status;
    this.details = details;
  }
}

/**
 * Shared response-handling logic, extracted so sendMessage(),
 * generateTitle(), and regenerateResponse() don't each duplicate the same
 * network-failure / malformed-JSON / non-OK-status handling three times.
 */
async function handleFetch(url, options) {
  let rawResponse;

  try {
    rawResponse = await fetch(url, options);
  } catch (networkError) {
    throw new ChatApiError(
      "Could not reach the server. Check your connection and that the backend is running.",
      0,
      networkError.message
    );
  }

  let body;
  try {
    body = await rawResponse.json();
  } catch (parseError) {
    throw new ChatApiError(
      "The server sent back a response that wasn't valid JSON.",
      rawResponse.status,
      parseError.message
    );
  }

  if (!rawResponse.ok) {
    const detail = body?.detail || body;
    throw new ChatApiError(
      detail?.message || `Request failed with status ${rawResponse.status}`,
      rawResponse.status,
      detail
    );
  }

  return body;
}

/**
 * Sends a message to the backend and returns the parsed response.
 *
 * @param {string} message - the user's message text
 * @param {string|null} sessionId - a session ID (Day 13: ALWAYS provided
 *   now, generated client-side via crypto.randomUUID() the moment "New
 *   Chat" is clicked -- the backend lazily creates the session on this
 *   first call if it doesn't recognize the ID yet, see server/main.py).
 * @returns {Promise<{session_id: string, response: string, message_count: number, latency_ms: number}>}
 * @throws {ChatApiError} on any network failure, non-OK HTTP status, or malformed JSON response
 */
export async function sendMessage(message, sessionId) {
  return handleFetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
}

/**
 * Day 13: asks the backend to generate a short (3-5 word) title
 * summarizing a session's first exchange. Called once, right after the
 * FIRST successful reply in a brand-new session.
 *
 * @param {string} sessionId
 * @returns {Promise<{session_id: string, title: string}>}
 */
export async function generateTitle(sessionId) {
  return handleFetch(`${API_BASE_URL}/api/sessions/${sessionId}/title`, {
    method: "POST",
  });
}

/**
 * Day 13: regenerates the model's response to the LAST user message in a
 * session -- the backend pops the previous exchange first (so the
 * session doesn't end up with a duplicated user message), then re-runs
 * generation. Returns the same shape as sendMessage()'s response.
 *
 * @param {string} sessionId
 * @returns {Promise<{session_id: string, response: string, message_count: number, latency_ms: number}>}
 */
export async function regenerateResponse(sessionId) {
  return handleFetch(`${API_BASE_URL}/api/sessions/${sessionId}/regenerate`, {
    method: "POST",
  });
}

/**
 * Sends a message and STREAMS the reply, calling onChunk(text) for each
 * piece of text as it arrives, and resolving with the final metadata
 * ({session_id, message_count, latency_ms}) once the stream completes.
 * Carried forward from the Day 12 improvement -- works identically here,
 * since the backend's /api/chat/stream endpoint uses the SAME Day 13
 * lazy-session-creation logic as the non-streaming /api/chat endpoint.
 *
 * @param {string} message
 * @param {string} sessionId - Day 13: always a real, client-generated UUID
 * @param {(text: string) => void} onChunk
 * @returns {Promise<{session_id: string, message_count: number, latency_ms: number}>}
 * @throws {ChatApiError}
 */
export async function streamMessage(message, sessionId, onChunk) {
  let rawResponse;
  try {
    rawResponse = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });
  } catch (networkError) {
    throw new ChatApiError(
      "Could not reach the server. Check your connection and that the backend is running.",
      0,
      networkError.message
    );
  }

  if (!rawResponse.ok) {
    let body = {};
    try {
      body = await rawResponse.json();
    } catch {
      // No JSON body available -- fall through with the generic message below.
    }
    const detail = body?.detail || body;
    throw new ChatApiError(
      detail?.message || `Request failed with status ${rawResponse.status}`,
      rawResponse.status,
      detail
    );
  }

  const reader = rawResponse.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneMetadata = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const eventTypeMatch = rawEvent.match(/^event: (.+)$/m);
      const dataMatch = rawEvent.match(/^data: (.+)$/m);
      if (!eventTypeMatch || !dataMatch) continue;

      const eventType = eventTypeMatch[1];
      const data = JSON.parse(dataMatch[1]);

      if (eventType === "chunk") {
        onChunk(data.text);
      } else if (eventType === "done") {
        doneMetadata = data;
      } else if (eventType === "error") {
        throw new ChatApiError(data.message || "Streaming failed.", 500, data);
      }
    }
  }

  if (!doneMetadata) {
    throw new ChatApiError("Stream ended without a completion event.", 0, null);
  }
  return doneMetadata;
}
