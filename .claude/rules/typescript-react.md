# TypeScript and React Guidelines

## TypeScript Type Safety
- NEVER use `any`, `@ts-ignore`, or bypass TypeScript type checks. Fix the root cause of the type error.
- ALWAYS define proper interfaces or types for function parameters, return values, and state objects.
- ALWAYS use `unknown` instead of `any` when the type is truly not known at compile time, and narrow it with type guards.
- NEVER leave `// @ts-expect-error` or `// @ts-ignore` comments without a clear explanation and a plan to remove them.

## React State Management
- NEVER mutate React state directly. ALWAYS use immutable update patterns.
- ALWAYS use functional state updates when the new state depends on the previous state: `setState(prev => prev + 1)`.
- NEVER use `push()`, `splice()`, or other mutating array methods on state. Use spread syntax or array methods that return new arrays.
- Prefer `useReducer` for complex state logic with multiple sub-values or when the next state depends on the previous one.

## React useEffect Rules
- EVERY `useEffect` hook MUST have an explicit dependency array. Omitting it is a bug.
- Dependency arrays MUST be exhaustive. Include ALL values referenced inside the effect.
- If a `useEffect` attaches event listeners, opens WebSockets, starts intervals, or creates subscriptions, it MUST return a cleanup function.
- NEVER create cascading re-render cycles. If state update A triggers effect B which updates state A, you have an infinite loop.
- NEVER use `useEffect` for computations that can be derived during render. Use `useMemo` instead.

## React Component Best Practices
- Keep UI components, state management, type definitions, and business logic in SEPARATE files.
- NEVER shove all concerns into a single massive file. If a file exceeds 500 lines, refactor into smaller modules.
- ALWAYS include proper ARIA labels, alt text for images, semantic HTML tags, and keyboard navigation support.
- ALWAYS implement skeleton screens, spinners, or disabled button states during asynchronous operations.

## Async Operations in React
- ALWAYS `await` promises. NEVER create unhandled promise rejections.
- ALL async operations MUST have `try/catch` blocks with appropriate error handling.
- NEVER fire multiple concurrent state updates without considering race conditions.
- Use `AbortController` for fetch requests in `useEffect` cleanup to prevent memory leaks.

## Frontend Performance
- ALWAYS implement proper pagination for large datasets. Never render thousands of DOM nodes at once.
- Clean up event listeners in component unmount lifecycle methods.
- Use `React.memo` for components that receive the same props frequently and don't need re-rendering.
- Use `useCallback` and `useMemo` to prevent unnecessary re-renders of child components.
- ALL layouts MUST work on mobile, tablet, and desktop. NEVER create desktop-only layouts.

## CSS and Styling
- NEVER generate inline styles that conflict with existing global styles or component libraries.
- NEVER use Tailwind utility classes that override the design system without justification.
- ALWAYS check responsive breakpoints: mobile (< 640px), tablet (640-1024px), desktop (> 1024px).
