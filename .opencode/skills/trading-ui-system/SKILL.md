---
name: trading-ui-system
description: "Use ONLY when building or modifying frontend UI components, pages, or styles for the React betting dashboard. Covers the professional trading visual design system: dark theme, data-dense layout, monospace numbers, color semantics, and trading-specific components."
---

# Trading UI System — Agente Betting Engine

## Design Philosophy

Every component must look like a **professional trading terminal** (Bloomberg, TradingView, Kraken), not a generic dashboard. Rules:

1. **Dark-first**: Background `#0a0a0f`, cards `#111118`, never pure white
2. **Data-dense**: Compact padding, no excessive whitespace, scroll over pagination
3. **Monospace for ALL numbers**: Odds, probabilities, EV, percentages, dates/times
4. **Color semantics**: Green = profit/positive EV, Red = loss/negative EV, Amber = warning/caution, Blue = info/accent
5. **Flat design**: Subtle borders (`#1e1e2e`), NO box-shadows, NO gradients (except gauge fills)

## Design Tokens

Use Tailwind v4 CSS variables defined in `src/index.css`:

```css
/* Always reference these -- never hardcode colors */
bg-bg-primary      /* #0a0a0f — main background */
bg-bg-secondary    /* #111118 — cards, panels */
bg-bg-tertiary     /* #1a1a24 — hover states, input bg */
bg-bg-hover        /* #22222e — active/hover */
border-border      /* #1e1e2e — card borders */
border-border-light /* #2a2a3a — table row separators */
text-text-primary  /* #e1e1e6 — main text */
text-text-secondary /* #8b8b9e — muted labels */
text-text-muted    /* #5c5c6e — disabled, placeholders */
text-profit        /* #00c076 — positive EV, green */
text-loss          /* #ff5353 — negative EV, red */
text-warning       /* #f59e0b — caution */
text-accent        /* #3b82f6 — links, active nav */
bg-profit/10       /* green backgrounds */
bg-loss/10         /* red backgrounds */
font-mono          /* JetBrains Mono — all numbers */
font-sans          /* Inter — labels, headers */
```

## Layout System

```
┌─────────────────────────────────────────┐
│  Sidebar (w-16 md:w-56)                │
│  ┌───────────────────────────────────┐  │
│  │ Logo + Nav links                  │  │
│  │ • Dashboard                       │  │
│  │ • Partidos                        │  │
│  │ • Buscar                          │  │
│  │ • Jobs                            │  │
│  └───────────────────────────────────┘  │
│                                         │
├─────────────────────────────────────────┤
│  Main content (flex-1 overflow-y-auto)  │
│  max-w-6xl mx-auto                      │
│  p-4 md:p-6                             │
└─────────────────────────────────────────┘
```

- Sidebar: fixed width, full height, border-right
- Main: scrollable, max-width constrained
- Never use padding > 1.5rem (p-6)
- Grid system: `grid gap-3 sm:grid-cols-2 lg:grid-cols-3`

## Component Specifications

### 1. ProbabilityGauge (`src/components/ProbabilityGauge.tsx`)

Purpose: Display 1X2 model probabilities as a horizontal segmented bar (like order book depth).

```
[HOME ████████████ 45% │ DRAW ████ 25% │ AWAY ████████ 30%]
   45% HOME         25% Empate        30% AWAY
```

- 3 segments: home (accent blue), draw (warning amber), away (accent-secondary purple)
- Labels below with percentages in monospace
- Height: h-6, rounded-full
- Only show text inside segment if width > 5%

### 2. EVBar (`src/components/EVBar.tsx`)

Purpose: Horizontal bar showing EV range (optimistic / base / pessimistic).

```
EV: ████████████████████░ +8.3%
    [-2%] [base: +5%] [+12%]
```

- Green fill if EV > 0, red if EV < 0
- Width proportional to EV magnitude
- Show base value, min/max markers
- Font-mono for numbers

### 3. DataTable (`src/components/DataTable.tsx`)

Purpose: Compact data table for markets, predictions, comparisons.

- Compact rows (py-2)
- Monospace for all values
- Right-aligned numbers, left-aligned labels
- Sticky header with uppercase labels
- Row hover effect
- No pagination on small datasets (< 50 rows)

### 4. Badge (`src/components/Badge.tsx`)

Purpose: Status indicators for risk, tier, confidence, priority.

Variants:
- `profit` (green) — STRONG_BET, high, approved
- `loss` (red) — NO_BET, rejected, low
- `warning` (amber) — WATCH, medium, caution
- `info` (blue) — info, neutral
- `neutral` (gray) — default

Always use `rounded-full`, font-mono, 11px.

### 5. MetricCard (`src/components/MetricCard.tsx`)

Purpose: Single stat display for health, ROI, counts.

```
● Servicio
ok
```

- Dot indicator (green/amber/red) next to label
- Value in font-mono font-semibold
- Optional subtitle in text-muted

### 6. ComboCard (`src/components/ComboCard.tsx`)

Purpose: Safe combination display with legs list and joint probability.

```
┌──────────────────────────────┐
│ STRONG_BET                   │
│                              │
│ 1. Colombia gana @ 2.10      │
│ 2. Under 2.5 @ 1.85         │
│                              │
│ Prob conjunta: 45.2% ████░  │
│ EV: +8.3%                   │
│ Stake: 1.5%                 │
└──────────────────────────────┘
```

### 7. Ticker (`src/components/Ticker.tsx`)

Purpose: Horizontal scrolling ticker for key stats (optional).

## Page Specifications

### Dashboard (`/`) — Health + Quick Links

- 3 MetricCards: Service, DB, Schemas
- Quick links grid (4 items)
- Error state: red border card with error message
- Loading state: centered "Cargando..." text

### Matches (`/matches`) — Match Explorer (replaces /hoy)

- Grid of MatchCards
- Each card: team names, date/time, status badge (EN VIVO with pulse dot), prediction pills
- Loading: "Cargando partidos..."
- Empty: "No hay partidos próximos" with dashed border

### Match Detail (`/matches/:id`) — Full Analysis

This is the main trading view. Must include:

1. **Header**: Team names, date, round, status
2. **ProbabilityGauge**: 1X2 model probabilities
3. **Predictions Table**: Market, outcome, prob%, confidence tier
4. **Combinations Grid**: ComboCards with priority badges
5. **Sections** (future): EV breakdown, SHARP metrics, analysis tabs

### Search (`/search`) — Team/Match Search

- Search input with font-mono
- Results split: Teams (left) + Matches (right)
- Minimum 2 chars to search
- No results state

### Jobs (`/jobs`) — Job Trigger Panel

- Grid of JobCards
- Each card: label, "Ejecutar" button, loading/success/error state
- Success shows job_id in monospace
- Error shows message in red

## Color Semantics Reference

| Data Type   | Color     | Usage                          |
|-------------|-----------|--------------------------------|
| EV > 0      | `profit`  | Positive expected value        |
| EV < 0      | `loss`    | Negative expected value        |
| Confidence high | `profit` | High confidence predictions  |
| Confidence low  | `loss`   | Low confidence predictions   |
| WATCH       | `warning` | Watch list items               |
| NO_BET      | `loss`    | Blocked/rejected bets          |
| STRONG_BET  | `profit`  | Approved singles               |
| MODERATE_BET| `warning` | Medium conviction              |
| WEAK_BET    | `loss`    | Low conviction                 |
| Live match  | `loss`    | EN VIVO badge with pulse       |
| Scheduled   | `neutral` | Default status                 |

## When adding new features

1. Check if a matching component already exists in `src/components/`
2. Always use the design tokens from `index.css` — never hardcode colors
3. Every number must use `font-mono` class
4. Every card/panel must use `bg-bg-secondary border border-border rounded-lg p-4`
5. Every table must use `font-mono text-xs` with `text-right` for number columns
6. Badges for status/risk/priority always use `rounded-full`
7. Loading states: centered muted text or skeleton
8. Error states: red border card with error message
9. Empty states: dashed border with muted text centered
