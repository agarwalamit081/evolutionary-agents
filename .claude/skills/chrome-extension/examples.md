---
description: Chrome Extension Examples
---

**Example 1: Type-Safe Storage Wrapper (lib/storage.ts)**

```typescript
interface StorageSchema {
  theme: 'light' | 'dark';
  fontSize: number;
  recentSearches: string[];
  apiToken?: string; // local only, never sync
}

const storage = {
  async get<K extends keyof StorageSchema>(key: K): Promise<StorageSchema[K]> {
    const result = await chrome.storage.local.get(key);
    return result[key] as StorageSchema[K];
  },

  async set<K extends keyof StorageSchema>(key: K, value: StorageSchema[K]): Promise<void> {
    await chrome.storage.local.set({ [key]: value });
  },

  onChange<K extends keyof StorageSchema>(
    key: K,
    callback: (newVal: StorageSchema[K], oldVal: StorageSchema[K]) => void
  ) {
    chrome.storage.onChanged.addListener((changes, _area) => {
      if (changes[key]) {
        callback(changes[key].newValue, changes[key].oldValue);
      }
    });
  },
};
```

---

**Example 2: Type-Safe Messaging (lib/messaging.ts)**

```typescript
interface MessageMap {
  GET_SETTINGS: { request: void; response: { theme: string } };
  SAVE_SETTINGS: { request: { theme: string }; response: { success: boolean } };
  EXTRACT_PAGE_DATA: { request: { selector: string }; response: { text: string } };
}

type MessageAction = keyof MessageMap;

async function sendMessage<A extends MessageAction>(
  action: A,
  payload: MessageMap[A]['request']
): Promise<MessageMap[A]['response']> {
  return chrome.runtime.sendMessage({ action, payload });
}

function onMessage(handlers: {
  [A in MessageAction]?: (
    payload: MessageMap[A]['request']
  ) => Promise<MessageMap[A]['response']>;
}) {
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    const handler = handlers[message.action as MessageAction];
    if (handler) {
      handler(message.payload).then(sendResponse);
      return true; // Keep channel open for async response
    }
  });
}
```

---

**Example 3: Background Service Worker (background/index.ts)**

```typescript
chrome.runtime.onInstalled.addListener(() => {
  // Set default settings
  chrome.storage.local.set({ theme: 'light', fontSize: 14 });

  // Create context menu
  chrome.contextMenus.create({
    id: 'extract-text',
    title: 'Extract selected text',
    contexts: ['selection'],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'extract-text' && tab?.id) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'HIGHLIGHT_SELECTION',
      payload: { text: info.selectionText },
    });
  }
});

// Handle alarms for periodic tasks
chrome.alarms.create('sync-data', { periodInMinutes: 30 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'sync-data') {
    // Sync logic here
  }
});

// Handle messages from popup/content
onMessage({
  GET_SETTINGS: async () => {
    const theme = await storage.get('theme');
    return { theme };
  },
  SAVE_SETTINGS: async (payload) => {
    await storage.set('theme', payload.theme);
    return { success: true };
  },
});
```

---

**Example 4: Content Script with Shadow DOM (content/index.ts)**

```typescript
const EXTENSION_ID = 'my-ext-root';

function mount() {
  // Prevent double injection
  if (document.getElementById(EXTENSION_ID)) return;

  const host = document.createElement('div');
  host.id = EXTENSION_ID;
  const shadow = host.attachShadow({ mode: 'open' });

  // Inject styles into Shadow DOM (isolated from page)
  const style = document.createElement('style');
  style.textContent = `
    .panel { background: white; border: 1px solid #ccc; padding: 16px; border-radius: 8px; }
    .panel h3 { margin: 0 0 8px 0; }
  `;
  shadow.appendChild(style);

  // Create UI
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.innerHTML = '<h3>My Extension</h3><p>Active on this page</p>';
  shadow.appendChild(panel);

  document.body.appendChild(host);
}

// Listen for messages from background
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === 'HIGHLIGHT_SELECTION') {
    console.log('Selected text:', message.payload.text);
    sendResponse({ received: true });
  }
});

// Cleanup on unload
window.addEventListener('unload', () => {
  document.getElementById(EXTENSION_ID)?.remove();
});

mount();
```

---

**Example 5: Popup with React (popup/App.tsx)**

```tsx
import { useState, useEffect } from 'react';

export default function Popup() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    chrome.runtime.sendMessage(
      { action: 'GET_SETTINGS', payload: undefined },
      (response) => {
        setTheme(response.theme);
        setLoading(false);
      }
    );
  }, []);

  const handleSave = async () => {
    chrome.runtime.sendMessage(
      { action: 'SAVE_SETTINGS', payload: { theme } },
      (response) => {
        if (response.success) window.close();
      }
    );
  };

  if (loading) return <div className="p-4 text-center">Loading...</div>;

  return (
    <div className="w-80 p-4">
      <h1 className="text-lg font-bold mb-4">Settings</h1>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={theme === 'dark'} onChange={() => setTheme(theme === 'dark' ? 'light' : 'dark')} />
        Dark Mode
      </label>
      <button onClick={handleSave} className="mt-4 w-full bg-blue-500 text-white py-2 rounded">
        Save
      </button>
    </div>
  );
}
```

---

**Example 6: Options Page with Auto-Save (options/App.tsx)**

```tsx
import { useState, useEffect } from 'react';
import { storage } from '../lib/storage';

export default function Options() {
  const [fontSize, setFontSize] = useState(14);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    storage.get('fontSize').then(setFontSize);
  }, []);

  const handleSave = async () => {
    await storage.set('fontSize', fontSize);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-md mx-auto p-8">
      <h1 className="text-2xl font-bold mb-6">Extension Options</h1>
      <label className="block mb-4">
        Font Size: {fontSize}px
        <input type="range" min={12} max={24} value={fontSize}
          onChange={(e) => setFontSize(Number(e.target.value))}
          className="w-full mt-2" />
      </label>
      <button onClick={handleSave} className="bg-blue-500 text-white px-4 py-2 rounded">
        {saved ? '✓ Saved' : 'Save'}
      </button>
    </div>
  );
}
```

---

**Example 7: Vite Configuration (vite.config.ts)**

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { crx } from '@crxjs/vite-plugin';
import manifest from './public/manifest.json';

export default defineConfig({
  plugins: [react(), crx({ manifest })],
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        popup: 'src/popup/index.html',
        options: 'src/options/index.html',
      },
      output: {
        entryFileNames: '[name]/index.js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
});
```
