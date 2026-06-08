---
name: chrome-extension
description: Build production-ready Chrome extensions with Manifest V3 — popup UI with React and Tailwind, content scripts with Shadow DOM, background service worker, type-safe storage and messaging, and minimum-permission manifest.
---

**When to Use**
- Building a new Chrome extension or browser plugin.
- Adding features to an existing Chrome extension.
- Migrating a Manifest V2 extension to V3.
- Debugging extension permissions, messaging, or content script issues.

**Core Principles**
1. **Manifest V3 Only**: Chrome Web Store requires V3 for new submissions.
2. **Minimum Permissions**: Never request `<all_urls>` or `tabs` when `activeTab` suffices.
3. **Type-Safe Storage**: Use `StorageSchema` interface wrapping `chrome.storage.sync`/`local`.
4. **Type-Safe Messaging**: Use `MessageMap` pattern for typed send/receive between contexts.
5. **Shadow DOM for Content Scripts**: Isolate injected styles from host page.
6. **Stateless Service Worker**: All state in chrome.storage — service workers can be killed anytime.
7. **TypeScript Strict Mode**: `strict: true` in tsconfig.json.
8. **Component Inventory**: Determine popup/content/background/options/side-panel needs upfront.

**Workflow**
1. Analyze requirements → determine component inventory (popup, content, background, options).
2. Audit minimum permissions needed.
3. Scaffold project with Vite + @crxjs/vite-plugin.
4. Implement type-safe storage and messaging layers first.
5. Build components (background handler, content script, popup, options).
6. Configure manifest.json with minimum permissions.
7. Build and verify (`tsc --noEmit`, `npm run build`, load unpacked).

**References**
- Load `reference.md` for Manifest V3 reference, chrome.* API patterns, and build configuration.
- Load `examples.md` for typed storage, messaging, and component implementations.

**Scripts**
- `scripts/generate_extension_scaffold.py`: Generate project directory structure.
