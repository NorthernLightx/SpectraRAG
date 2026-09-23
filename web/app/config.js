/* Frontend → API base URL.

   Empty string means "same origin", the default for local dev and the combined
   (single-service) deploy, so api.js falls back to window.location.origin.

   When the frontend is served from its own host (decoupled deploy),
   `web-build/build.mjs --api-base` overwrites this file with the API service
   URL. The URL itself is supplied at build time and is not committed. */
window.SPECTRARAG_API_BASE = "";

/* True only on hosted deploys (baked in with the API base): hides the local
   Ollama provider option, which can't work from a public origin. */
window.SPECTRARAG_HOSTED = false;
