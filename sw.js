/* Service worker — static app shell cached for offline study.
   The AI endpoint (/api/chat, /.netlify/*) is NEVER intercepted or cached: it always goes to the network. */
const VERSION = 'mb-v2';
const SHELL = ['./', 'index.html', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png', 'icons/apple-touch-icon.png', 'icons/favicon-32.png', 'vendor/marked.min.js'];
const CACHEABLE_HOSTS = ['fonts.googleapis.com', 'fonts.gstatic.com'];

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(VERSION);
    await c.addAll(SHELL);
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== VERSION) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== 'GET') return;                                   // POSTs (AI chat) are never touched
  if (url.origin === location.origin && (url.pathname.startsWith('/api/') || url.pathname.startsWith('/.netlify/'))) return; // AI/API: network only
  if (url.origin !== location.origin && !CACHEABLE_HOSTS.includes(url.hostname)) return;

  // Page loads: network first (so new deploys appear), cached copy when offline
  if (req.mode === 'navigate') {
    e.respondWith((async () => {
      try {
        const r = await fetch(req);
        if (r.ok) { const c = await caches.open(VERSION); c.put('index.html', r.clone()); }
        return r;
      } catch (_) {
        return (await caches.match('index.html')) || (await caches.match('./')) || Response.error();
      }
    })());
    return;
  }

  // Everything else (icons, fonts, libraries): cache first, refresh in background
  e.respondWith((async () => {
    const c = await caches.open(VERSION);
    const hit = await c.match(req);
    const net = fetch(req).then((r) => { if (r && (r.ok || r.type === 'opaque')) c.put(req, r.clone()); return r; }).catch(() => null);
    return hit || (await net) || Response.error();
  })());
});
