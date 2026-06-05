# HTTP Status Codes in Practice

HTTP status codes are a three-digit signal the server sends back with every response.
The first digit defines the class of the result; the remaining two carry the specific
meaning. Using them correctly makes APIs predictable for clients, caches, and humans
reading logs.

## The five classes

- **1xx Informational** — the request was received and the process is continuing. Rare
  in application code; `100 Continue` is the one most people encounter.
- **2xx Success** — the request succeeded. `200 OK` for general success, `201 Created`
  when a request created a resource (return its location), `204 No Content` when there
  is nothing to send back (a successful DELETE, for example).
- **3xx Redirection** — further action is needed. `301 Moved Permanently` and
  `308 Permanent Redirect` are cacheable; `302`/`307` are temporary.
- **4xx Client errors** — the request was malformed or not allowed. The client should
  not repeat it unchanged.
- **5xx Server errors** — the server failed to fulfill an apparently valid request.

## Choosing the right 4xx

The 4xx range is where most API design mistakes happen. A few distinctions worth
internalizing:

- `400 Bad Request` — the request itself is malformed (bad JSON, missing required
  field). Use it for validation failures.
- `401 Unauthorized` — authentication is missing or invalid. Despite the name, this is
  about *authentication*, not authorization. Send a `WWW-Authenticate` header.
- `403 Forbidden` — the caller is authenticated but not allowed to do this. No amount of
  re-authenticating will help.
- `404 Not Found` — the resource does not exist. Some APIs deliberately return 404
  instead of 403 to avoid revealing that a resource exists at all.
- `409 Conflict` — the request collides with current state (a duplicate create, an edit
  against a stale version).
- `422 Unprocessable Entity` — the syntax is fine but the semantics are wrong. Many APIs
  use this for validation errors instead of 400.
- `429 Too Many Requests` — rate limiting. Pair it with a `Retry-After` header.

## 5xx and what clients should do

`500 Internal Server Error` is the catch-all for an unhandled exception — it tells the
client nothing actionable, so log the details server-side. `502 Bad Gateway` and
`504 Gateway Timeout` come from proxies and load balancers when an upstream is broken or
slow; `503 Service Unavailable` signals temporary overload or maintenance and, like 429,
should carry `Retry-After`.

A good client retries 5xx and 429 responses with exponential backoff and jitter, but
never retries 4xx responses other than 429, because a malformed request will stay
malformed no matter how many times it is sent.

## Idempotency and caching

Status codes interact with two HTTP features that matter for reliability. Caches key off
codes: `200`, `301`, and `404` are cacheable by default, while `302` and most errors are
not. Idempotency keys let a client safely retry a `POST` after a network failure without
creating duplicate resources — the server returns the original `201` (or a `409`) instead
of acting twice. Designing with both in mind turns flaky networks into a non-issue.
