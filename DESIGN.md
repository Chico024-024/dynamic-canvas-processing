# DCP Studio — Design System

## Design Contract

**Product:** DCP Studio — Autonomous Song C Composer
**Surface:** Web Studio (Desktop-first, Operate + Experience mode)
**Version:** 0.1.0-core
**Seed:** `seed: 7f3a2e1c9b4d` (concept-seed assigned direction)

## Thesis

DCP Studio is an **autonomous composition surface**, not a parameter-tuning tool. The UI must communicate: *DCP decides, you listen.* Every interaction reinforces autonomy — no strategy dropdowns, no duration sliders, no module toggles. The interface gets out of the way; the music speaks.

## Own-World

### Palette

| Role | Token | Hex | Usage |
|------|-------|-----|-------|
| **Void** | `--void` | `#06070a` | Page ground, deepest elevation |
| **Abyss** | `--abyss` | `#0a0b0e` | Primary surface, cards |
| **Abyss-Elevated** | `--abyss-elevated` | `#111218` | Hover/active cards |
| **Abyss-High** | `--abyss-high` | `#1a1b22` | Inputs, borders |
| **Cyan** | `--cyan` | `#00f5d4` | Deck A, primary action, Deck A accent |
| **Cyan-Soft** | `--cyan-soft` | `#00e5c4` | Hover states, Deck A glow |
| **Coral** | `--coral` | `#ff007f` | Deck B, danger, Deck B accent |
| **Coral-Soft** | `--coral-soft` | `#ff2d8f` | Hover states, Deck B glow |
| **Violet** | `--violet` | `#a78bfa` | Bridge, ghost, autonomy |
| **Violet-Soft** | `#c4b5fd` | Muted violet, secondary text |
| **Gold** | `#fbbf24` | Playhead, warnings, playhead |
| **Gold-Soft** | `#fde68a` | Hover gold |
| **Emerald** | `#10b981` | Success, ready states |
| **Emerald-Soft** | `#6ee7b7` | Muted success |
| **Text-Primary** | `--text-primary` | `#f1f5f9` | Primary text |
| **Text-Secondary** | `--text-secondary` | `#94a3b8` | Secondary text, labels |
| **Text-Muted** | `--text-muted` | `#64748b` | Muted, disabled, hints |
| **Text-Invisible** | `--text-invisible` | `#334155` | Disabled, invisible |

### Gradients

| Token | Definition | Usage |
|-------|------------|-------|
| `--grad-primary` | `linear-gradient(135deg, var(--cyan), var(--violet))` | Primary actions, logo |
| `--grad-deck-a` | `linear-gradient(135deg, var(--cyan), #00d4b8)` | Deck A accents |
| `--grad-deck-b` | `linear-gradient(135deg, var(--coral), var(--violet))` | Deck B accents |
| `--grad-bridge` | `linear-gradient(90deg, var(--cyan), var(--violet), var(--coral))` | Bridge zone, autonomy |
| `--grad-ghost` | `linear-gradient(90deg, var(--violet), transparent)` | Ghost resolution zone |
| `--grad-glass` | `linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))` | Glass cards |
| `--grad-glass-hover` | `linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.04))` | Hover glass |

### Elevation System

| Level | Token | Box Shadow | Usage |
|-------|-------|------------|-------|
| 0 | `--shadow-0` | `none` | Flat, inline |
| 1 | `--shadow-1` | `0 1px 2px rgba(0,0,0,0.3), 0 1px 1px rgba(0,0,0,0.2)` | Cards, resting |
| 2 | `--shadow-2` | `0 4px 12px rgba(0,0,0,0.4), 0 2px 4px rgba(0,0,0,0.3)` | Hover cards |
| 3 | `--shadow-3` | `0 8px 32px rgba(0,0,0,0.5), 0 4px 8px rgba(0,0,0,0.3)` | Modals, dropdowns |
| 4 | `--shadow-glow-cyan` | `0 0 24px rgba(0, 245, 212, 0.2)` | Cyan focus/glow |
| 5 | `--shadow-glow-violet` | `0 0 32px rgba(167, 139, 250, 0.25)` | Violet focus/glow |
| 6 | `--shadow-glow-coral` | `0 0 24px rgba(255, 0, 127, 0.2)` | Coral focus/glow |

### Glass System

| Token | Definition | Usage |
|-------|------------|-------|
| `--glass-1` | `rgba(255,255,255,0.03)` | Subtle glass, backgrounds |
| `--glass-2` | `rgba(255,255,255,0.06)` | Cards, panels |
| `--glass-3` | `rgba(255,255,255,0.1)` | Hover glass |
| `--glass-4` | `rgba(255,255,255,0.15)` | Active glass, overlays |
| `--glass-border` | `rgba(255,255,255,0.08)` | Glass borders |
| `--glass-border-hover` | `rgba(255,255,255,0.16)` | Hover glass borders |
| `--glass-border-accent` | `rgba(0,245,212,0.3)` | Accent glass borders |

### Border Radius

| Token | Value | Usage |
|-------|-------|-------|
| `--radius-1` | `4px` | Chips, badges, small controls |
| `--radius-2` | `8px` | Buttons, inputs, chips |
| `--radius-3` | `12px` | Cards, panels |
| `--radius-4` | `16px` | Panels, modals |
| `--radius-5` | `24px` | Hero areas, large modals |
| `--radius-full` | `9999px` | Pills, avatars, badges |

### Spacing Scale (8px base)

| Token | Value |
|-------|-------|
| `--space-1` | 4px |
| `--space-2` | 8px |
| `--space-3` | 12px |
| `--space-4` | 16px |
| `--space-5` | 20px |
| `--space-6` | 24px |
| `--space-7` | 32px |
| `--space-8` | 40px |
| `--space-9` | 48px |
| `--space-10` | 64px |

### Typography

| Role | Font | Size | Weight | Line Height | Letter Spacing |
|------|------|------|--------|-------------|----------------|
| Display | Space Grotesk | 3.5rem / 56px | 700 | 1.1 | -0.03em |
| H1 | Space Grotesk | 2.25rem / 36px | 700 | 1.2 | -0.02em |
| H2 | Space Grotesk | 1.5rem / 24px | 600 | 1.3 | -0.01em |
| H3 | Space Grotesk | 1.125rem / 18px | 600 | 1.4 | 0 |
| Body-Large | Space Grotesk | 1.125rem / 18px | 400 | 1.6 | 0 |
| Body | Space Grotesk | 1rem / 16px | 400 | 1.6 | 0 |
| Body-Small | Space Grotesk | 0.875rem / 14px | 400 | 1.5 | 0 |
| Caption | Space Grotesk | 0.75rem / 12px | 500 | 1.5 | 0.02em |
| Overline | Space Grotesk | 0.625rem / 10px | 600 | 1.5 | 0.1em |
| Mono-Large | JetBrains Mono | 1.125rem / 18px | 500 | 1.4 | 0 |
| Mono | JetBrains Mono | 1rem / 16px | 500 | 1.4 | 0 |
| Mono-Small | JetBrains Mono | 0.875rem / 14px | 500 | 1.4 | 0 |
| Mono-Tiny | JetBrains Mono | 0.75rem / 12px | 500 | 1.4 | 0 |

### Motion

| Token | Value | Usage |
|-------|-------|-------|
| `--motion-instant` | `0ms` | Immediate |
| `--motion-fast` | `120ms cubic-bezier(0.2, 0, 0, 1)` | Micro-interactions |
| `--motion-normal` | `200ms cubic-bezier(0.2, 0, 0, 1)` | Standard transitions |
| `--motion-slow` | `350ms cubic-bezier(0.2, 0, 0, 1)` | Panels, modals |
| `--motion-spring` | `400ms cubic-bezier(0.34, 1.56, 0.64, 1)` | Springy, delightful |
| `--motion-slow-spring` | `600ms cubic-bezier(0.2, 0.6, 0.2, 1)` | Panels, drawers |

### Z-Index

| Layer | Value |
|-------|-------|
| Base | 0 |
| Dropdown | 100 |
| Sticky Header | 200 |
| Sidebar | 300 |
| Modal Backdrop | 400 |
| Modal | 500 |
| Toast | 600 |
| Tooltip | 700 |
| Loading Overlay | 1000 |

### Breakpoints

| Name | Width |
|------|-------|
| Mobile | < 640px |
| Tablet | 640px - 1024px |
| Desktop | 1024px - 1440px |
| Wide | > 1440px |

## Component Specifications

### Buttons

#### Primary (Primary Action)
- **Background:** `linear-gradient(135deg, var(--cyan), var(--violet))`
- **Text:** `--bg-primary` (`#0a0b0e`)
- **Padding:** `14px 28px`
- **Radius:** `12px`
- **Font:** Space Grotesk, 0.875rem, 600
- **Shadow:** `0 4px 16px rgba(0, 245, 212, 0.3)`
- **Hover:** `translateY(-2px)`, shadow `0 8px 24px rgba(0, 245, 212, 0.4)`
- **Active:** `translateY(0)`
- **Disabled:** `opacity: 0.5`, `cursor: not-allowed`, no transform

#### Secondary
- **Background:** `var(--glass-2)`
- **Border:** `1px solid var(--glass-border)`
- **Text:** `--text-primary`
- **Padding:** `12px 24px`
- **Radius:** `12px`
- **Hover:** `--glass-3`, border `--glass-border-hover`

#### Ghost
- **Background:** `transparent`
- **Text:** `--text-secondary`
- **Hover:** `--glass-1`, text `--text-primary`
- **Padding:** `10px 16px`
- **Radius:** `8px`

#### Icon Button
- **Size:** `44px × 44px`
- **Background:** `var(--glass-1)`
- **Border:** `1px solid var(--glass-border)`
- **Radius:** `50%`
- **Hover:** `--glass-2`, border `--glass-border-hover`

#### Generate Button (Hero)
- **Size:** Full width on mobile, max-width 400px on desktop
- **Height:** `56px`
- **Font:** Space Grotesk, 1rem, 600
- **Gap:** `12px`
- **Loading state:** Spinner replaces icon, text "Composing Song C..."

### Inputs

#### File Drop Zone
- **Background:** `var(--glass-1)`
- **Border:** `2px dashed var(--glass-border)`
- **Radius:** `16px`
- **Padding:** `48px 24px`
- **Hover/Drag:** Border `--cyan`, background `var(--cyan-glow)`, shadow `--shadow-glow-cyan`
- **Drag Active:** `transform: scale(1.02)`
- **Has File:** Solid border `--cyan`, background `var(--cyan-glow)`, shows file card

#### Select
- **Background:** `var(--glass-2)`
- **Border:** `1px solid var(--glass-border)`
- **Radius:** `12px`
- **Padding:** `12px 16px`
- **Font:** Space Grotesk, 0.875rem, 400
- **Hover:** Border `--glass-border-hover`
- **Focus:** Border `--cyan`, box-shadow `0 0 0 3px var(--cyan-glow)`

#### Range Slider
- **Track:** Height `6px`, background `var(--glass-2)`, radius `3px`
- **Thumb:** `20×20px`, `linear-gradient(135deg, var(--cyan), var(--violet))`, radius `50%`
- **Hover:** Thumb `scale(1.15)`
- **Focus:** Ring `0 0 0 3px var(--cyan-glow)`

#### Toggle
- **Track:** `52×28px`, background `var(--glass-2)`, border `var(--glass-border)`, radius `9999px`
- **Thumb:** `24×24px`, white, shadow `0 2px 4px rgba(0,0,0,0.2)`
- **Checked:** Background `linear-gradient(135deg, var(--cyan), var(--violet))`, thumb `translateX(24px)`

### Cards & Panels

#### Card (Default)
- **Background:** `var(--glass-2)`
- **Border:** `1px solid var(--glass-border)`
- **Radius:** `16px`
- **Padding:** `24px`
- **Backdrop Filter:** `blur(16px)`
- **Hover:** Border `var(--glass-border-hover)`, shadow `var(--shadow-2)`

#### Card (Elevated)
- **Background:** `var(--abyss)`
- **Border:** `1px solid var(--glass-border)`
- **Radius:** `16px`
- **Shadow:** `var(--shadow-2)`

#### Glass Card
- **Background:** `linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))`
- **Border:** `1px solid var(--glass-border)`
- **Radius:** `16px`
- **Backdrop Filter:** `blur(16px)`

### Form Controls

#### Toggle (Switch)
- **Track:** `52×28px`, `var(--glass-2)`, border `var(--glass-border)`, radius `9999px`
- **Thumb:** `24×24px`, white, shadow `0 2px 4px rgba(0,0,0,0.2)`, radius `50%`
- **Checked:** Background `linear-gradient(135deg, var(--cyan), var(--violet))`, thumb `translateX(24px)`

#### Chip Toggle
- **Padding:** `8px 16px`
- **Radius:** `9999px`
- **Background:** `var(--glass-2)`
- **Border:** `1px solid var(--glass-border)`
- **Checked:** Background `linear-gradient(135deg, var(--cyan), var(--violet))`, text `#0a0b0e`, border transparent

### Visualizer

#### Canvas
- **Background:** `--void` (`#06070a`)
- **Aspect Ratio:** `16:5` (desktop), `4:3` (mobile)
- **Border:** `1px solid var(--glass-border)`
- **Radius:** `16px`

#### Waveform
- **Deck A (Cyan):** Stroke `#00f5d4`, glow `rgba(0,245,212,0.3)`, width `1.5px`
- **Deck B (Coral):** Stroke `#ff007f`, glow `rgba(255,0,127,0.3)`, width `1.5px`
- **Center Line:** Center of lane, scale `0.75 × laneHeight`

#### Phase Markers
- **Launchpad (Cyan):** Vertical line `#00f5d4`, glow, label "🚀 LAUNCHPAD"
- **Impact (Coral):** Vertical line `#ff007f`, glow, label "💥 IMPACT"
- **Breath (Violet):** Gradient zone, dashed borders, label "🫧 BREATH WINDOW"
- **Bridge Bars (Violet):** Dashed lines every bar, highlight Bar 4 & 8
- **Playhead:** Gold `#fbbf24`, dashed line, time label

### States

#### Focus Visible
- Outline: `2px solid var(--cyan)`, offset `2px`

#### Disabled
- Opacity: `0.5`
- Cursor: `not-allowed`
- No hover/active transforms

#### Loading
- Skeleton: `--glass-3` background, shimmer animation `1.5s infinite`
- Spinner: `24×24px`, border `2px solid var(--glass-border)`, border-top-color `var(--cyan)`, radius `50%`, spin `1s linear infinite`

#### Success
- Border: `1px solid var(--emerald)`
- Background: `rgba(16, 185, 129, 0.1)`
- Text: `var(--emerald)`

#### Error
- Border: `1px solid var(--coral)`
- Background: `rgba(255, 0, 127, 0.1)`
- Text: `var(--coral)`

#### Warning
- Border: `1px solid var(--gold)`
- Background: `rgba(251, 191, 36, 0.15)`
- Text: `var(--gold)`

### Layout

#### Grid
- **Container Max Width:** `1400px`
- **Main Padding:** `24px` (desktop), `16px` (mobile)
- **Grid Gap:** `24px` (desktop), `16px` (mobile)
- **Upload Grid:** `1fr auto 1fr` (desktop), `1fr` (mobile)
- **Controls Grid:** `repeat(auto-fit, minmax(200px, 1fr))`
- **Info Grid:** `repeat(auto-fit, minmax(200px, 1fr))`

#### Container
- **Max Width:** `1400px`
- **Margin:** `0 auto`
- **Padding:** `24px` (desktop), `16px` (mobile)

#### Header
- **Height:** `64px` (desktop), `56px` (mobile)
- **Background:** `var(--abyss)`
- **Border Bottom:** `1px solid var(--border-subtle)`
- **Position:** Sticky, top 0, z-index 200

#### Sidebar (Mobile)
- **Width:** `280px`
- **Background:** `var(--abyss)`
- **Border Right:** `1px solid var(--border-subtle)`
- **Overlay:** `rgba(0,0,0,0.6)`, z-index 299

#### Settings Panel (Desktop)
- **Width:** `320px`
- **Background:** `var(--abyss)`
- **Border Left:** `1px solid var(--border-subtle)`

### Accessibility

- **Contrast Ratio:** Minimum 4.5:1 for text, 3:1 for UI elements
- **Focus Visible:** Always visible, `2px solid var(--cyan)`, offset `2px`
- **Reduced Motion:** Respects `prefers-reduced-motion: reduce` — all animations `0.01ms`
- **Color Blind Safe:** Palette tested for deuteranopia/protanopia/tritanopia
- **Screen Reader:** Semantic HTML, ARIA labels, live regions for status
- **Keyboard:** Full keyboard navigation, visible focus, logical tab order
- **Touch Targets:** Minimum `44×44px` touch targets

### Dark Mode Only

This application is **dark-mode only**. No light mode support. The design system is built around the dark void aesthetic essential for audio production environments.

## Implementation Notes

### CSS Custom Properties
All tokens defined as CSS custom properties on `:root` for runtime theming and easy maintenance.

### Progressive Enhancement
- Core functionality works without JavaScript (file upload, basic playback)
- Animations enhance, never required
- Graceful degradation for unsupported features

### Performance
- CSS-only animations where possible (transform, opacity)
- GPU-accelerated transforms (`will-change: transform`)
- Lazy-loaded non-critical assets
- Efficient canvas rendering (requestAnimationFrame, dirty rects)

### Responsive Breakpoints
```css
@media (max-width: 640px) { /* Mobile */ }
@media (min-width: 641px) and (max-width: 1024px) { /* Tablet */ }
@media (min-width: 1025px) { /* Desktop */ }
@media (min-width: 1441px) { /* Wide */ }
```

### Print Styles
- Hide header, sidebar, controls, visualizer controls
- Show only visualizer canvas and telemetry
- Monochrome palette for printing