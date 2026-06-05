# Caching Strategies

Caching trades freshness for speed and cost. A cache stores the result of expensive work
— a database query, a rendered page, a remote API call — so that future requests can be
served from fast storage instead of redoing the work. The hard part is never the storing;
it is deciding when a cached value is no longer trustworthy.

## Where caches live

- **Client-side** — the browser cache, controlled by HTTP headers like `Cache-Control`
  and `ETag`. Free bandwidth savings, but you cannot purge it on demand.
- **CDN / edge** — geographically distributed caches close to users. Excellent for static
  assets and cacheable API responses.
- **Application** — an in-process cache (a dictionary, an LRU) or a shared cache like
  Redis or Memcached sitting between your app and the database.

## Read patterns

- **Cache-aside (lazy loading)** — the application checks the cache first; on a miss it
  reads the database, stores the result, and returns it. Simple and resilient: if the
  cache is down, the app still works (just slower). The cost is a cold-start penalty and
  the risk of serving stale data until the entry expires.
- **Read-through** — the cache itself knows how to load missing entries from the backing
  store. The application only ever talks to the cache. Cleaner code, but the cache becomes
  a hard dependency.

## Write patterns

- **Write-through** — every write goes to the cache and the database synchronously. Reads
  are always warm and consistent, at the cost of slower writes.
- **Write-back (write-behind)** — writes go to the cache immediately and are flushed to the
  database asynchronously. Fast writes, but a cache failure can lose data.
- **Write-around** — writes go straight to the database, bypassing the cache. Good when
  written data is rarely read soon after; avoids polluting the cache with cold entries.

## Invalidation and expiry

There are only a few honest ways to keep a cache from lying:

- **TTL (time to live)** — entries expire after a fixed duration. Dead simple and bounds
  staleness, but picks the same trade-off for hot and cold data alike.
- **Explicit invalidation** — the code that writes new data also deletes or updates the
  cached copy. Precise, but easy to forget a code path, which is how stale bugs are born.
- **Versioned keys** — instead of mutating an entry, write under a new key (e.g. include a
  version or content hash). Old readers keep working; new writes are atomic.

## Two failure modes to design against

The **thundering herd** (or cache stampede) happens when a popular entry expires and many
requests miss simultaneously, all hammering the database at once. Mitigate it with a short
lock so only one request recomputes, or by recomputing slightly before expiry.

**Cache penetration** is when requests repeatedly miss for keys that do not exist,
defeating the cache entirely. Cache the negative result (a "not found" marker) with a short
TTL, or front the cache with a Bloom filter. Both turn a stream of expensive misses back
into cheap hits.
