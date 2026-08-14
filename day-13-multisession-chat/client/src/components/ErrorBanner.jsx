/**
 * ErrorBanner.jsx
 * -------------------
 * Displays a dismissible error message when a request fails -- covers
 * BOTH failure modes chatApi.js can throw: a network failure (backend
 * unreachable) and a non-OK HTTP response (400/404/500 from the server
 * itself).
 */

export default function ErrorBanner({ error, onDismiss, onRetry }) {
  if (!error) return null;

  const isNetworkError = error.status === 0;

  return (
    <div className="mx-5 mt-3 mb-1 max-w-3xl md:mx-auto">
      <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3">
        <span className="text-red-500 font-mono mt-0.5" aria-hidden="true">
          !
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-sans text-sm font-medium text-red-800">
            {isNetworkError ? "Connection problem" : `Error ${error.status || ""}`}
          </p>
          <p className="font-sans text-sm text-red-700 mt-0.5">{error.message}</p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {onRetry && (
            <button
              onClick={onRetry}
              className="font-mono text-[10px] uppercase tracking-wide text-red-700 hover:text-red-900 underline"
            >
              Retry
            </button>
          )}
          <button
            onClick={onDismiss}
            className="text-red-400 hover:text-red-600 text-lg leading-none"
            aria-label="Dismiss error"
          >
            &times;
          </button>
        </div>
      </div>
    </div>
  );
}
