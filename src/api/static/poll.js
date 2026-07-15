/*
 * Minimal vanilla-JS polling-swap (Phase 5 dashboard auto-refresh).
 *
 * HTMX-equivalent for our use case: each [data-poll] element fetches its
 * data-poll URL every data-poll-interval seconds and replaces its innerHTML
 * with the response. No SSE, no build step, no external dependency.
 *
 * HTMX-ready: to swap in real htmx later, add the htmx <script> and replace
 * data-poll="/x?partial=1" data-poll-interval="5" with hx-get="/x?partial=1"
 * hx-trigger="every 5s" hx-target="this" hx-swap="innerHTML" — same server
 * contract (the ?partial=1 fragment).
 */
(function () {
  "use strict";
  function startPolling(el) {
    var url = el.getAttribute("data-poll");
    if (!url) return;
    var seconds = parseInt(el.getAttribute("data-poll-interval") || "5", 10);
    var ms = (isNaN(seconds) ? 5 : seconds) * 1000;
    setInterval(function () {
      fetch(url, { headers: { "X-Requested-With": "turing-poll" } })
        .then(function (r) { return r.ok ? r.text() : ""; })
        .then(function (html) { if (html) el.innerHTML = html; })
        .catch(function () { /* best-effort: a transient fetch error is ignored */ });
    }, ms);
  }
  function init() {
    document.querySelectorAll("[data-poll]").forEach(startPolling);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
