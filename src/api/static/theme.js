/* Theme toggle — flips light/dark and remembers the choice in localStorage.
   The initial theme (before first paint) is set by the inline <head> script in
   base.html so there is no flash; this only owns the click on the toggle. */
(function () {
  "use strict";
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", function () {
    var next = root.dataset.theme === "light" ? "dark" : "light";
    root.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch (e) {
      /* private-mode storage may throw — the in-memory flip still applies. */
    }
  });
})();
