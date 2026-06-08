---
description: Chrome Extension Reference
---

## Manifest V3 Required Fields

```json
{
  "manifest_version": 3,
  "name": "Extension Name",
  "version": "1.0.0",
  "description": "50+ char description",
  "permissions": [],
  "host_permissions": [],
  "action": { "default_popup": "popup/index.html" },
  "background": { "service_worker": "background/index.ts", "type": "module" }
}
```

## Permission Catalog

| Permission | When to Use |
|---|---|
| `storage` | Always — settings persistence |
| `activeTab` | Access current tab on user action (prefer over `tabs`) |
| `scripting` | Programmatic content script injection |
| `contextMenus` | Right-click menu items |
| `alarms` | Scheduled periodic tasks |
| `notifications` | Desktop notifications |
| `sidePanel` | Persistent side panel |
| `host_permissions` | Access specific domains (never `<all_urls>`) |

## chrome.storage API

- `chrome.storage.sync`: Settings synced across devices (100KB limit, 512 items max).
- `chrome.storage.local`: Local-only data (unlimited with `unlimitedStorage` permission).
- `chrome.storage.session`: Cleared on browser close (Manifest V3+).
- Always use `StorageSchema` interface for type safety.
- Listen for changes: `chrome.storage.onChanged.addListener(callback)`.

## chrome.runtime Messaging

- `chrome.runtime.sendMessage(payload)`: One-off message to background/other contexts.
- `chrome.runtime.onMessage.addListener(handler)`: Receive messages.
- `chrome.runtime.connect()`: Port-based long-lived connections (for streaming).
- Use `MessageMap` type for request/response typing.
- Include a `type` discriminator field in all messages.

## Content Script Patterns

- Inject via manifest `content_scripts` or programmatically via `chrome.scripting.executeScript`.
- Always check for double-injection: `if (document.getElementById('my-ext-root')) return`.
- Use Shadow DOM: `element.attachShadow({ mode: 'open' })` to isolate styles.
- Clean up on unload: `window.addEventListener('unload', cleanup)`.
- Use MutationObserver for dynamic page content.

## Service Worker Patterns

- Event-driven: register listeners at top level (not inside async functions).
- Use `chrome.alarms` for periodic tasks (not `setInterval`).
- Keep stateless — read/write state from chrome.storage.
- Service workers can be terminated between events.

## Build Configuration (Vite)

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { crx } from '@crxjs/vite-plugin';
import manifest from './public/manifest.json';

export default defineConfig({
  plugins: [react(), crx({ manifest })],
  build: {
    rollupOptions: {
      input: {
        popup: 'popup/index.html',
        options: 'options/index.html',
      },
    },
  },
});
```

## Chrome Web Store Requirements

- Privacy policy required if extension collects data.
- Screenshots: 1280x800 or 640x400, at least 1 required.
- Review timeline: 1-3 business days typically.
- Version updates: increment version in manifest.json before submitting.
