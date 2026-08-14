/**
 * MessageInput.jsx
 * --------------------
 * A textarea for composing messages, with:
 *   - Enter to send, Shift+Enter for a newline
 *   - A live character counter
 *   - A send button, disabled while a request is in flight OR the
 *     input is empty, wired to useState for the input's own value
 *
 * VARIANT PROP (new): "bottom" (default) renders the standard fixed
 * bottom bar with a top border and full-width background, matching every
 * chat app's normal in-conversation input. "centered" renders JUST the
 * input+button row with no bar chrome, since App.jsx places it inside a
 * vertically-centered hero container for a brand-new, empty chat --
 * the same pattern ChatGPT/Claude.ai use: the input starts centered on
 * an empty conversation, then moves to the bottom the moment the first
 * message is sent (App.jsx switches which variant renders based on
 * whether the active session has any messages yet).
 *
 * MAX_LENGTH is enforced both visually (the counter turns red/blocks
 * further typing) and would ALSO need to be validated server-side --
 * client-side limits are a UX convenience, never a security boundary.
 */

import { useState } from "react";

const MAX_LENGTH = 2000;

export default function MessageInput({ onSend, disabled, variant = "bottom" }) {
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

  const inputRow = (
    <div className="flex items-end gap-3 max-w-2xl w-full mx-auto">
      <div className="flex-1 relative">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          autoFocus={variant === "centered"}
          placeholder={disabled ? "Waiting for a reply..." : "Message... (Enter to send, Shift+Enter for new line)"}
          className={`w-full resize-none rounded-xl border bg-white px-4 py-3 pr-16 text-[14px] font-sans text-ink
                      placeholder:text-muted focus:outline-none focus:border-accent/70 focus:ring-2 focus:ring-accent/15
                      disabled:opacity-50 disabled:cursor-not-allowed shadow-sm
                      ${isOverLimit ? "border-red-400" : "border-border"}`}
          style={{ minHeight: "50px", maxHeight: "160px" }}
        />
        <span
          className={`absolute bottom-3 right-4 font-mono text-[10px] select-none ${
            isOverLimit ? "text-red-500 font-medium" : "text-muted"
          }`}
        >
          {trimmedLength}/{MAX_LENGTH}
        </span>
      </div>

      <button
        onClick={handleSend}
        disabled={!canSend}
        className={`flex-shrink-0 h-12 px-5 rounded-xl font-sans text-sm font-medium transition-colors
                    ${canSend
                      ? "bg-accent text-accent-ink hover:bg-accent/90"
                      : "bg-surface text-muted cursor-not-allowed"}`}
      >
        Send
      </button>
    </div>
  );

  if (variant === "centered") {
    return inputRow;
  }

  return (
    <div className="border-t border-border bg-bg px-5 py-4">
      {inputRow}
    </div>
  );
}
