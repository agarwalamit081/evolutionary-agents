---
description: Frontend Architecture Examples
---

**Example 1: Accessible Form Input with Error State (Tailwind + React)**

```tsx
import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
  id: string;
}

export const AccessibleInput: React.FC<InputProps> = ({ label, error, id, className, ...props }) => {
  const errorId = `${id}-error`;
  const hasError = !!error;

  return (
    <div className="flex flex-col gap-1.5 w-full max-w-sm">
      <label htmlFor={id} className="text-sm font-medium text-gray-700 dark:text-gray-300">
        {label}
      </label>
      <input
        id={id}
        aria-invalid={hasError}
        aria-describedby={hasError ? errorId : undefined}
        className={`px-3 py-2 rounded-md border bg-white dark:bg-gray-800
          text-gray-900 dark:text-gray-100 transition-colors duration-200
          focus:outline-none focus:ring-2 focus:ring-offset-1 dark:focus:ring-offset-gray-900
          ${hasError
            ? 'border-red-500 focus:ring-red-500'
            : 'border-gray-300 dark:border-gray-600 focus:ring-blue-500 hover:border-gray-400'}
          ${className}`}
        {...props}
      />
      {hasError && (
        <p id={errorId} className="text-sm text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}
    </div>
  );
};
```

---

**Example 2: Responsive CSS Grid with Dark Mode and Design Tokens**

```css
:root {
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f3f4f6;
  --color-text-primary: #111827;
  --spacing-md: 1rem;
  --radius-md: 0.5rem;
}

@media (prefers-color-scheme: dark) {
  :root {
    --color-bg-primary: #111827;
    --color-bg-secondary: #1f2937;
    --color-text-primary: #f9fafb;
  }
}

.card-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
}

@media (min-width: 768px) {
  .card-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (min-width: 1024px) {
  .card-grid { grid-template-columns: repeat(3, 1fr); }
}
```

---

**Example 3: Enterprise Data Fetching Hook with AbortController**

```typescript
import { useState, useEffect, useCallback, useRef } from 'react';

interface FetchState<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
}

export function useFetch<T>(url: string, options?: RequestInit): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({ data: null, isLoading: true, error: null });
  const abortRef = useRef<AbortController | null>(null);

  const executeFetch = useCallback(async () => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setState({ data: null, isLoading: true, error: null });

    try {
      const response = await fetch(url, { ...options, signal: abortRef.current.signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: T = await response.json();
      setState({ data, isLoading: false, error: null });
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return;
      setState({ data: null, isLoading: false, error: error as Error });
    }
  }, [url, options]);

  useEffect(() => {
    executeFetch();
    return () => { abortRef.current?.abort(); };
  }, [executeFetch]);

  return state;
}
```

---

**Example 4: Compound Component Pattern (Tabs)**

```typescript
import React, { createContext, useContext, useState } from 'react';

interface TabsContextType { activeTab: string; setActiveTab: (id: string) => void; }
const TabsContext = createContext<TabsContextType | undefined>(undefined);

export const Tabs: React.FC<{ children: React.ReactNode; defaultTab?: string }> = ({
  children, defaultTab = 'tab1'
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab);
  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div className="tabs-container">{children}</div>
    </TabsContext.Provider>
  );
};

export const Tab: React.FC<{ id: string; children: React.ReactNode }> = ({ id, children }) => {
  const ctx = useContext(TabsContext);
  if (!ctx) throw new Error('Tab must be used within Tabs');
  const isActive = ctx.activeTab === id;
  return (
    <button role="tab" aria-selected={isActive} onClick={() => ctx.setActiveTab(id)}
      className={`px-4 py-2 ${isActive ? 'border-b-2 border-blue-500 font-bold' : 'text-gray-500'}`}>
      {children}
    </button>
  );
};

export const TabPanel: React.FC<{ id: string; children: React.ReactNode }> = ({ id, children }) => {
  const ctx = useContext(TabsContext);
  if (!ctx) throw new Error('TabPanel must be used within Tabs');
  return ctx.activeTab === id ? <div role="tabpanel" className="p-4">{children}</div> : null;
};
```

---

**Example 5: Robust Async Fetch with Retry and Exponential Backoff**

```typescript
async function fetchWithRetry(
  url: string, options: RequestInit = {}, retries = 3, backoff = 1000
): Promise<unknown> {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url, { ...options, signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, backoff * Math.pow(2, i)));
    }
  }
}
```

---

**Example 6: Custom useDebounce Hook**

```typescript
import { useState, useEffect } from 'react';

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

// Usage: const searchTerm = useDebounce(inputValue, 300);
```

---

**Example 7: Optimistic Update with React Query**

```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query';

function useUpdateTodo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (todo: Todo) => api.updateTodo(todo),
    onMutate: async (newTodo) => {
      await queryClient.cancelQueries({ queryKey: ['todos'] });
      const previous = queryClient.getQueryData(['todos']);
      queryClient.setQueryData(['todos'], (old: Todo[]) =>
        old.map(t => t.id === newTodo.id ? newTodo : t)
      );
      return { previous };
    },
    onError: (_err, _newTodo, context) => {
      queryClient.setQueryData(['todos'], context?.previous);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['todos'] });
    },
  });
}
```

---

**Example 8: Service Worker Registration**

```typescript
export function registerServiceWorker() {
  if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
    window.addEventListener('load', async () => {
      try {
        const registration = await navigator.serviceWorker.register('/sw.js');
        console.log('SW registered:', registration.scope);
      } catch (error) {
        console.error('SW registration failed:', error);
      }
    });
  }
}
```
