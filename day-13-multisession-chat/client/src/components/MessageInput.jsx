/**
 * MessageInput.jsx
 * --------------------
 * A textarea for composing messages, with:
 *   - Enter to send, Shift+Enter for a newline (the standard chat-app
 *     convention -- NOT the browser's native form-submit behavior,
 *     which has to be explicitly intercepted)
 *   - A live character counter
 *   - A send button, disabled while a request is in flight OR the
 *     input is empty, wired to useState for the input's own value
 *
 * MAX_LENGTH is enforced both visually (the counter turns red/blocks
 * further typing) and would ALSO need to be validated server-side --
 * client-side limits are a UX convenience, never a security boundary,
 * exactly the "never trust client input alone" principle from Day 9's
 * JSON schema validation, applied here to a form field instead of an
 * LLM response.
 */

import { useState } from "react";

const MAX_LENGTH = 2000;

export default function MessageInput({ onSend, disabled }) {
  const [value, setValue] = useState("");

  const trimmedLength = value.length;
  const isOverLimit = trimmedLength > MAX_LENGTH;
  const isEmpty = value.trim().length === 0;
  const canSend = !isEmpty && !isOverLimit && !disabled;

  function handleSend() {
    if (!canSend) return;
    onSend(value.trim());
    setValue("");
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-3">
      <div className="flex items-end gap-3 max-w-3xl mx-auto">
        <div className="flex-1 relative">
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
            placeholder={disabled ? "Waiting for response..." : "Type a message... (Enter to send, Shift+Enter for new line)"}
            className={`w-full resize-none rounded-xl border px-4 py-2.5 pr-16 text-sm text-slate-800
                        placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500
                        disabled:bg-slate-50 disabled:cursor-not-allowed
                        ${isOverLimit ? "border-red-400 focus:ring-red-400" : "border-slate-300"}`}
            style={{ minHeight: "44px", maxHeight: "160px" }}
          />
          <span
            className={`absolute bottom-2 right-3 text-[11px] select-none ${
              isOverLimit ? "text-red-500 font-medium" : "text-slate-400"
            }`}
          >
            {trimmedLength}/{MAX_LENGTH}
          </span>
        </div>

        <button
          onClick={handleSend}
          disabled={!canSend}
          className={`flex-shrink-0 h-11 px-5 rounded-xl text-sm font-medium transition-colors
                      ${canSend
                        ? "bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800"
                        : "bg-slate-200 text-slate-400 cursor-not-allowed"}`}
        >
          Send
        </button>
      </div>
    </div>
  );
}
