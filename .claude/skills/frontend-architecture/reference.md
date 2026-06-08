---
description: Frontend Architecture Reference
---

## Accessibility (a11y) Checklist

- **Focus Management**: All interactive elements must have visible `:focus-visible` outline. Never `outline: none` without replacement.
- **Color Contrast**: Text must have 4.5:1 ratio against background. Large text (18pt+) requires 3:1.
- **ARIA**: Use only when semantic HTML is insufficient (`aria-expanded` for dropdowns, `aria-live` for dynamic content).
- **Forms**: Every `<input>` must have an associated `<label>` (via `for`/`id` or wrapping).

## Responsive Breakpoints (Standard)

| Token | Width | Target |
|---|---|---|
| `sm` | 640px | Tablet portrait |
| `md` | 768px | Tablet landscape |
| `lg` | 1024px | Desktop |
| `xl` | 1280px | Large desktop |

## Tailwind CSS Best Practices

- Use `@apply` sparingly; prefer utility classes in HTML/JSX.
- Use `group` and `group-hover` for parent-triggered child styles.
- Use `dark:` prefix for dark mode.
- Use `sr-only` for screen-reader-only text.

## State Management Hierarchy

1. **Local State**: `useState` (form inputs, toggles).
2. **Derived State**: `useMemo` (filtered lists, computed values).
3. **Shared Local State**: Context API or Zustand (theme, auth user).
4. **Server State**: React Query, SWR, or RTK Query (NEVER put server state in Redux/Context manually).

## Custom Hooks Rules

- Must start with `use`.
- Only call at the top level (no conditions/loops).
- Return clean, predictable objects/arrays.

## JavaScript Error Handling

- Always wrap top-level async calls in try/catch.
- Use custom Error classes for domain-specific errors.
- Clear intervals/timeouts on unmount. Use `WeakMap`/`WeakSet` for object caching.

## Anti-Patterns

- `<div onClick={...}>` instead of `<button>`.
- Hardcoded hex values (`bg-[#123456]`) instead of theme variables.
- Fixed heights (`h-96`) on containers with dynamic text.
- Prop drilling more than 3 levels deep (use Context or Composition).
- Fetching data in `useEffect` without a data-fetching library.
- Inline object/array creation in JSX props (causes re-renders).
