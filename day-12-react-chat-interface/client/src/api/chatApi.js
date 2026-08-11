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
 * Sends a message to the backend and returns the parsed response.
 *
 * @param {string} message - the user's message text
 * @param {string|null} sessionId - existing session ID, or null to start a new session
 * @returns {Promise<{session_id: string, response: string, message_count: number, latency_ms: number}>}
 * @throws {ChatApiError} on any network failure, non-OK HTTP status, or malformed JSON response
 */
export async function sendMessage(message, sessionId) {
  let rawResponse;

  try {
    rawResponse = await fetch(`${API_BASE_URL}/api/chat`, {
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
