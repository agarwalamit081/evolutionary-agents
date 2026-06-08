---
name: frontend-architecture
description: Component-driven frontend architecture covering UI/UX design principles, state management, TypeScript patterns, and modern JavaScript/Node.js best practices.
---

**When to Use**
- Creating new UI components, hooks, or state management logic.
- Styling components or implementing responsive/accessible layouts.
- Optimizing render performance or refactoring prop-drilling.
- Writing CSS, SCSS, Tailwind utility classes, or JS/TS code.

**Core Principles**
1. **Composition over Inheritance**: Use children props and compound components for flexible UIs.
2. **Accessibility First (WCAG 2.1 AA)**: 4.5:1 contrast for normal text, visible focus states, semantic HTML.
3. **Mobile-First Responsive**: Start with base mobile styles, scale up with `min-width` queries.
4. **Design Tokens**: Never hardcode magic numbers. Use CSS variables or Tailwind theme values.
5. **State Colocation**: Keep state close to usage. Server state via React Query/SWR — never in global Context manually.
6. **Type Safety**: Strict TypeScript interfaces for props, state, and API responses. No `any`.
7. **Modern JS**: ES modules (`import`/`export`), `const`/`let` (no `var`), async/await, immutability.
8. **Performance**: Memoize expensive computations, minimize CLS, use AbortController for fetches.

**References**
- Load `reference.md` for accessibility checklists, state management hierarchy, Tailwind patterns, and JS anti-patterns.
- Load `examples.md` for component patterns, hooks, and styling examples.

**Scripts**
- `scripts/check_contrast.py`: Validate WCAG color contrast ratios.
- `scripts/generate_component.py`: Scaffold React component files with TypeScript interfaces.
- `scripts/analyze_js_deps.py`: Check package.json for common anti-patterns.
