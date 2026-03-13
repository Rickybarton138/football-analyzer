# DugoutIQ - UX/UI Research: Sports Coaching & Analytics App Design

> **Date**: 2026-03-04
> **Purpose**: Comprehensive UX/UI research to inform DugoutIQ's visual identity, interaction patterns, and overall design system.
> **Sources**: Dribbble, Behance, Muzli, SaaSFrame, Harvard Science Review, Sportsmith, Folio3, Design4Users, Lollypop Design, Eleken, Medium, professional sports platforms (Hudl, StatsBomb, Wyscout, Catapult), FA England Football Learning, and 2025-2026 design trend analyses.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Industry Leaders & Their Design Language](#2-industry-leaders--their-design-language)
3. [Color Palettes for Sports/Coaching](#3-color-palettes-for-sportscoaching)
4. [Typography](#4-typography)
5. [Data Visualization Patterns](#5-data-visualization-patterns)
6. [Navigation Patterns](#6-navigation-patterns)
7. [Dark Dashboard Trends 2025-2026](#7-dark-dashboard-trends-2025-2026)
8. [Video + Data Side-by-Side Patterns](#8-video--data-side-by-side-patterns)
9. [Onboarding & Empty States](#9-onboarding--empty-states)
10. [Mobile vs Desktop Considerations](#10-mobile-vs-desktop-considerations)
11. [FA & Coaching Platform Analysis](#11-fa--coaching-platform-analysis)
12. [Competitor UI Teardowns](#12-competitor-ui-teardowns)
13. [Recommended Design System for DugoutIQ](#13-recommended-design-system-for-dugoutiq)
14. [Key Takeaways & Action Items](#14-key-takeaways--action-items)

---

## 1. Executive Summary

After analyzing 50+ sports analytics dashboards, coaching platforms, and design trend reports, clear patterns emerge for what makes professional sports coaching software feel world-class:

**The winners share these traits:**
- Dark-first interfaces with strategic color accents (not "dark for dark's sake")
- Data presented as narrative, not spreadsheets -- every chart answers a question
- Video and data live side-by-side, not in separate silos
- Left sidebar navigation with collapsible states for complex multi-module apps
- Progressive disclosure -- show summary first, let users drill down
- Empty states that teach, not just fill space
- Mobile-optimized but desktop-designed (coaches plan on laptops, review on tablets/phones)

**The key insight for DugoutIQ**: The gap in the market is not more data -- it is data that tells coaches WHAT TO DO. The UI must reflect this by putting AI recommendations and actionable insights front-and-center, not buried behind charts.

---

## 2. Industry Leaders & Their Design Language

### Hudl + StatsBomb (Market Leader)
- **Visual identity**: Clean, professional, predominantly white/light backgrounds with dark accents
- **Key feature**: "Data insights side-by-side with quality footage" -- their biggest selling point is unified video + data
- **Radar charts**: StatsBomb radars are iconic -- multi-axis player comparison visualizations
- **Color**: Hudl uses orange (#F26522) as primary. StatsBomb uses a deeper professional blue palette
- **Layout**: Dense information architecture. Analyst-first design. Not pretty, but functional
- **Weakness**: Designed for professional analysts, intimidating for grassroots coaches. Not mobile-first

### Wyscout
- **Design approach**: Database-style interface. Searchable, filterable, table-heavy
- **Strength**: 170+ leagues, massive data
- **Weakness**: Looks like enterprise software, not coaching software. No warmth

### Catapult
- **Design approach**: Hardware-integrated (GPS, wearables). Dashboard-heavy
- **Key pattern**: Real-time biometric data visualization (heart rate, load, sprint counts)
- **Color**: Dark backgrounds with vibrant data colors (green, orange, red traffic light system)
- **Strength**: Excellent at workload/injury monitoring charts
- **Weakness**: Requires physical hardware. Price point ($50K+) excludes grassroots

### VEO
- **Design approach**: Consumer-friendly, modern SaaS feel
- **Key feature**: Auto-filmed matches with AI camera
- **Color**: Clean whites, modern blues, approachable
- **Strength**: Beautiful onboarding, simple upload flow
- **Weakness**: Limited analytics depth. No tactical AI

### Key Insight for DugoutIQ
The professional tools (Hudl, Wyscout, Catapult) are powerful but ugly and intimidating. The consumer tools (VEO) are pretty but shallow. DugoutIQ's design opportunity is to be **both beautiful AND deep** -- professional enough for academy coaches, approachable enough for grassroots.

---

## 3. Color Palettes for Sports/Coaching

### What the Research Shows

Sports app color palettes fall into distinct categories. The most effective for coaching/analytics platforms combine a dark foundation with strategic accent colors that convey energy, trust, and precision.

### Palette Archetypes That Work

#### A. "Electric Arena Rush" (High Energy, Tech-Forward)
- Deep Navy: `#002A5E`
- Electric Blue: `#0074FF`
- Neon Green: `#00FF85`
- Stadium Gold: `#FFDD00`
- Clean White: `#FFFFFF`
- **Use**: Esports, fast-paced analytics, youth engagement
- **Feels like**: A packed stadium under floodlights

#### B. "Championship Spotlight" (Premium, Heroic)
- Midnight Navy: `#001326`
- Royal Blue: `#0050B3`
- Trophy Gold: `#FFD700`
- Victory Red: `#FF4B4B`
- Soft Off-White: `#F5F7FA`
- **Use**: Title cards, achievement screens, premium dashboards
- **Feels like**: Trophy ceremony under spotlights

#### C. "Neon Tactics Grid" (Digital Coaching Board)
- Dark Indigo: `#0A0E27`
- Neon Green: `#00FF85`
- Cyan: `#00E5FF`
- Soft Gray: `#A0AEC0`
- White: `#FFFFFF`
- **Use**: Tactical boards, formation views, play animation
- **Feels like**: A digital whiteboard on the touchline

#### D. "Monochrome Match Day" (Minimal, Professional)
- Near Black: `#1A1A2E`
- Charcoal: `#16213E`
- Steel Gray: `#0F3460`
- Accent Blue: `#E94560`
- White: `#FFFFFF`
- **Use**: Serious analytics, data-heavy dashboards, professional reports
- **Feels like**: A scouting room at a professional club

### Current DugoutIQ Palette Assessment

Current: Emerald `#10B981` + Cyan `#06B6D4` on Slate-900

**Verdict**: Good foundation. The emerald/cyan combination is distinctive and avoids the overused blue-only sports trope. However, consider:

1. **Add a warm accent** -- Gold or amber for achievements, highlights, and CTAs. Pure cool tones can feel clinical.
2. **Define your gray scale** -- Need at least 5 gray tones for card elevation, borders, disabled states, secondary text.
3. **Error/Warning/Success** -- Don't use your brand green for "success" states -- they'll blend. Consider a distinct lime or teal for success.
4. **Consider adding Navy depth** -- A deeper navy (`#0F172A`) for sidebar and navigation areas creates hierarchy against the slate-900 content area.

### Recommended DugoutIQ Palette (Enhanced)

```
Primary Emerald:     #10B981  (brand, primary actions, key metrics)
Secondary Cyan:      #06B6D4  (links, secondary actions, data accents)
Accent Gold:         #F59E0B  (achievements, warnings, attention-grabbing CTAs)
Accent Rose:         #F43F5E  (errors, negative stats, alerts)

Background Dark:     #0F172A  (deepest layer - sidebar, nav)
Background Base:     #1E293B  (main content area - Slate 800)
Background Card:     #334155  (card surfaces - Slate 700)
Background Elevated: #475569  (hover states, elevated cards - Slate 600)

Text Primary:        #F1F5F9  (main text - Slate 100)
Text Secondary:      #94A3B8  (secondary text - Slate 400)
Text Muted:          #64748B  (disabled, hints - Slate 500)

Border Default:      #334155  (card borders - Slate 700)
Border Subtle:       #1E293B  (dividers - Slate 800)
```

---

## 4. Typography

### Key Findings from Research

1. **Dark mode requires heavier weights**: Thin fonts wash out on dark backgrounds. Use Medium (500) as your body weight, not Regular (400). Semi-bold (600) for headings, Bold (700) for hero metrics.

2. **Increase spacing in dark mode**: Line height should be 1.6-1.7 for body text (vs 1.5 in light mode). Letter spacing can be slightly increased for readability.

3. **Professional sports apps use these font families**:
   - **Inter** -- The dominant SaaS dashboard font. Clean, readable, excellent number rendering. Used by Linear, Vercel, and many analytics tools.
   - **Geist** -- Vercel's font. Modern, technical, excellent for code/data display.
   - **Plus Jakarta Sans** -- Geometric but warm. Good for approachable professional feel.
   - **DM Sans** -- Clean geometric sans. Popular in sports apps on Dribbble.
   - **Outfit** -- Modern, slightly rounded. Good for coaching apps that want to feel approachable.
   - **Space Grotesk** -- Technical, slightly futuristic. Good for analytics/data-heavy interfaces.
   - **JetBrains Mono** -- For monospaced data display (stats, numbers, scores).

4. **The typography stack pattern for sports dashboards**:
   - Hero metrics / big numbers: **Tabular figures** (monospaced numbers for alignment)
   - Headings: Geometric sans-serif, semi-bold
   - Body: Clean sans-serif, medium weight
   - Data labels: Small caps or reduced size of body font
   - Stats/numbers: Monospaced or tabular-lining figures

### Recommended for DugoutIQ

```
Primary Font:     Inter (already excellent for dashboards)
Heading Font:     Inter (or Space Grotesk for more character)
Mono Font:        JetBrains Mono (for stats, scores, xG values)

Scale:
  xs:    12px / 0.75rem   (labels, captions, timestamps)
  sm:    14px / 0.875rem  (secondary text, metadata)
  base:  16px / 1rem      (body text)
  lg:    18px / 1.125rem  (card titles, section headers)
  xl:    20px / 1.25rem   (page subtitles)
  2xl:   24px / 1.5rem    (page titles)
  3xl:   30px / 1.875rem  (hero metrics, big numbers)
  4xl:   36px / 2.25rem   (dashboard KPIs, match scores)

Weight:
  Regular:   400  (only for large display text)
  Medium:    500  (body text default in dark mode)
  Semibold:  600  (headings, interactive elements)
  Bold:      700  (hero numbers, emphasis)
```

---

## 5. Data Visualization Patterns

### What Works in Sports Analytics

Based on Harvard Science Review, Sportsmith, and Folio3 research, plus analysis of 20+ sports dashboards:

#### The Golden Rule
> "If a user has to hover over a data point to understand the basic gist of the chart, the visualization has failed." -- SaaSFrame 2026

#### Chart Types and When to Use Them

| Visualization | Best For | Sports Example | Avoid When |
|---------------|----------|----------------|------------|
| **Line Chart** | Trends over time | xG progression across a match, fitness loads over season | Comparing categories |
| **Bar Chart** | Comparing categories | Shot accuracy by player, passes per zone | Time-series data |
| **Donut/Ring** | Part-to-whole (max 3-4 segments) | Possession %, pass success rate | More than 4 segments |
| **Heatmap** | Spatial patterns | Player positioning, touch maps, shot locations | When exact values matter |
| **Radar Chart** | Multi-attribute comparison | Player comparison profiles (StatsBomb-style) | More than 7 axes |
| **Scatter Plot** | Correlation/clustering | Shot distance vs xG, players by metric pairs | Without trend lines |
| **Momentum Graph** | Flow over time | Match dominance, possession waves | Static single-point data |
| **Shot Map** | Spatial + outcome | xG map, shot locations with color-coded outcomes | Non-spatial data |
| **Pitch Map** | Tactical visualization | Passing networks, formation shapes, press zones | Non-football data |

#### Key Best Practices from Sportsmith (Vancouver Whitecaps)

1. **"We Don't Do Ugly"** -- Make the commitment that visualizations are always on-brand and aesthetically pleasing.
2. **"Efficiently Beautiful"** -- Balance beauty with production speed through templates, preset themes, and established color palettes.
3. **Know your purpose before choosing a chart** -- Is this exploratory (let user find insights) or directed (lead user to a specific finding)?
4. **Pre-attentive attributes** -- Use position, color, size, and shape to convey meaning BEFORE the user consciously processes the chart.
5. **The Datasaurus Dozen** -- Summary statistics can be identical for wildly different data distributions. Always visualize, never rely on numbers alone.

#### Humans Process Images 60,000x Faster Than Text
Liverpool FC's switch from spreadsheets to color-coded xG shot probability maps directly improved their analysis speed. This is the benchmark for DugoutIQ.

#### Sports-Specific Visualization Patterns

**Match Timeline / Momentum Graph**: Shows who dominated each phase of the match. Tennis uses "momentum waves." Football should show possession, territorial control, and scoring chances over 90 minutes. This is HIGH VALUE for coaches reviewing matches.

**Passing Networks**: Node-and-edge graphs showing who passes to whom, with line thickness indicating frequency. Powerful for understanding team shape and connectivity.

**Pressure Maps**: Where on the pitch a team applies pressing intensity. Color gradient overlays on a pitch outline.

**Formation Heatmaps**: Average positions of all players, showing actual vs intended formation.

**xG Timeline**: Cumulative expected goals over the match -- a signature modern football visualization.

### Card-Based Stats Presentation

The dominant pattern across all sports analytics dashboards is the **stat card grid**:

```
+------------------+  +------------------+  +------------------+
|  POSSESSION      |  |  SHOTS ON TARGET |  |  PASS ACCURACY   |
|  63%             |  |  7               |  |  87%             |
|  [=========-]    |  |  [=====----]     |  |  [=========-]    |
|  +5% vs avg      |  |  +2 vs avg      |  |  -3% vs avg      |
+------------------+  +------------------+  +------------------+
```

**Key elements of effective stat cards:**
- Metric name (secondary text, top)
- Hero number (large, bold, tabular figures)
- Visual indicator (mini bar, sparkline, or ring)
- Comparison/delta (vs average, vs last match, trend arrow)
- Color coding (green = above average, red = below)

---

## 6. Navigation Patterns

### The 2026 Consensus

Based on SaaSFrame analysis of hundreds of SaaS products and Reddit UX research:

**Left sidebar is the dominant pattern** for complex multi-module apps. It is universally preferred over top navigation for these reasons:
- More vertical scaling for deep feature sets
- Can be collapsed to icons-only for more content space
- Persistent visibility while scrolling content
- Natural information hierarchy (sections, sub-items)
- Better for multi-level navigation

### SaaS Navigation Best Practices

1. **Primary metrics above the fold** with clear hierarchy
2. **Secondary actions in top right** (profile, notifications, settings)
3. **Left sidebar navigation** -- almost universal in successful SaaS
4. **Tables default to 10-15 rows** (not infinite scroll)
5. **Filters are persistent, not hidden** -- users need to see active filter state
6. **Modified F-pattern** -- top-left quadrant is highest-value real estate for your "North Star Metric"

### Recommended Navigation Structure for DugoutIQ

```
SIDEBAR (left, collapsible):
  [Logo / Brand]
  ──────────────
  Dashboard        (home icon)
  Upload           (upload icon)
  ──────────────
  MATCHES          (section label)
    Recent Matches (list icon)
    [Match Name]   (expandable to sub-pages)
      Overview
      AI Coach
      Training
      Tactical
      Analytics
      Players
  ──────────────
  COACHING         (section label)
    Training Plans (clipboard icon)
    Session Builder(layers icon)
    Development    (trending-up icon)
  ──────────────
  Live Coaching    (radio icon)
  ──────────────
  Settings         (gear icon, bottom)
  Profile          (user icon, bottom)

TOP BAR (fixed):
  [Breadcrumb: Dashboard > Match: Arsenal vs Chelsea > AI Coach]
  [Search]                                    [Notifications] [Profile]
```

### Object-Oriented vs Workflow-Based

For DugoutIQ, use a **hybrid approach**:
- **Object-oriented** for matches (each match is an "object" with related data)
- **Workflow-based** for the upload/analysis pipeline (step-by-step: upload > processing > review)
- **Object-oriented** for players (each player is a hub of stats, clips, development)

---

## 7. Dark Dashboard Trends 2025-2026

### The State of Dark Mode in 2026

Dark mode has shifted from trend to standard expectation. 82.7% of consumers use dark mode on their devices. For "power user" tools like analytics dashboards, dark mode is the default, not the alternative.

### Key Principles (2026 Best Practices)

1. **Not just inverted colors**: Dark mode is NOT light mode with inverted colors. Pure black (#000000) backgrounds with pure white (#FFFFFF) text create extreme contrast that causes eye strain. Use near-black backgrounds and off-white text.

2. **Layered darkness**: Use 3-5 shades of dark gray to create depth, hierarchy, and visual separation between elements:
   - Deepest: `#0F172A` (sidebar, navigation shell)
   - Base: `#1E293B` (main content background)
   - Card: `#334155` (card surfaces, raised elements)
   - Elevated: `#475569` (modals, dropdowns, hover states)

3. **Glassmorphism (used sparingly)**: The frosted-glass effect (backdrop blur + semi-transparency) is still relevant in 2025-2026 but fading as a dominant trend. Use it selectively for:
   - Modal overlays
   - Floating action panels
   - Video overlay controls
   - NOT for primary navigation or data-dense areas

4. **Accent colors need adjustment**: Colors that look vibrant on white backgrounds can appear neon/oversaturated on dark backgrounds. Slightly desaturate accent colors for dark mode.

5. **Shadows don't work the same**: Drop shadows barely visible on dark backgrounds. Use subtle glow effects, thin light borders, or slight background color shifts instead.

6. **Font weight matters more**: Thin/light font weights that look elegant on white become unreadable on dark. Bump body text to Medium (500) weight.

7. **Icons need individual treatment**: Line icons that work on light backgrounds may get lost on dark. Use slightly thicker strokes or filled variants.

### What's Actually Trending (Not Just Hype)

From Lyssna's 2026 survey of UX designers:

- **Glassmorphism is fading** -- it was the most frequently mentioned "fading trend"
- **Functional minimalism is rising** -- clean interfaces that prioritize content over decoration
- **AI-integrated UI** -- interfaces that surface AI insights naturally within the workflow
- **Personalized/modular dashboards** -- drag-and-drop widget rearrangement, saved views
- **Micro-interactions** -- subtle animations that confirm actions and create delight
- **Accessible design as default** -- WCAG compliance is expected, not optional

### DugoutIQ Dark Mode Implementation Notes

- Use Tailwind's dark mode utilities (already in stack)
- Define CSS custom properties for the layered dark palette
- Ensure all brand colors (emerald, cyan) are tested against dark backgrounds
- Add gold/amber as a warm accent to prevent the interface feeling too cold
- Consider offering a light mode for printed reports / presentation mode

---

## 8. Video + Data Side-by-Side Patterns

### The Industry Standard

Hudl StatsBomb's biggest selling point: "Data insights side-by-side with quality footage -- validate analysis with additional context."

This is exactly what DugoutIQ needs to nail.

### Layout Patterns for Video + Data

#### A. Split Panel (50/50 or 60/40)
```
+---------------------------+---------------------------+
|                           |                           |
|    VIDEO PLAYER           |    DATA PANEL             |
|    (with timeline,        |    (stats, charts,        |
|     playback controls,    |     event markers,        |
|     drawing tools)        |     AI insights)          |
|                           |                           |
+---------------------------+---------------------------+
```
- **Best for**: Desktop analysis workflows
- **Used by**: Hudl Sportscode, coaching review sessions
- **Key feature**: Clicking a data point (event in timeline) jumps video to that moment

#### B. Video Primary with Overlay Panel
```
+-----------------------------------------------+
|                                                |
|    VIDEO PLAYER (full width)                   |
|                                                |
|    +------------------+                        |
|    | FLOATING PANEL   |                        |
|    | - Live stats     |                        |
|    | - AI alerts      |                        |
|    | - Event markers  |                        |
|    +------------------+                        |
|                                                |
+-----------------------------------------------+
|    TIMELINE with event markers                 |
+-----------------------------------------------+
```
- **Best for**: Live coaching, game review where video is primary
- **Used by**: VEO camera view, broadcast overlays
- **Key feature**: Panel can be toggled/dismissed, video stays dominant

#### C. Video Top, Data Bottom (Stacked)
```
+-----------------------------------------------+
|    VIDEO PLAYER                                |
|    (16:9 aspect, full width)                   |
+-----------------------------------------------+
|    TAB BAR: [Stats] [Events] [AI Coach] [Map]  |
+-----------------------------------------------+
|    DATA CONTENT (scrollable)                   |
|    - Selected tab content                      |
|    - Charts, tables, insights                  |
+-----------------------------------------------+
```
- **Best for**: Mobile / tablet, single-column layouts
- **Used by**: Most mobile sports apps
- **Key feature**: Video stays pinned at top while user scrolls through data

#### D. Video with Synchronized Timeline + Data Panels
```
+---------------------------+---------------------------+
|    VIDEO PLAYER           |    TACTICAL MAP           |
|                           |    (2D pitch view,        |
|                           |     player positions,     |
|                           |     synchronized to video)|
+---------------------------+---------------------------+
|    EVENT TIMELINE                                      |
|    [goal] [shot] [corner] [foul] [sub] [shot]         |
+-------------------------------------------------------+
|    AI INSIGHTS PANEL                                   |
|    "Your left side was exposed for 12 minutes..."     |
+-------------------------------------------------------+
```
- **Best for**: Full match analysis workflow
- **Used by**: Professional analysis suites
- **Key feature**: Video, tactical map, and timeline all synchronized

### Recommendation for DugoutIQ

Use **Pattern D** as the full desktop experience. This is the most powerful layout and aligns with DugoutIQ's differentiator (AI that tells you what to do). The synchronized view lets coaches see:
1. The actual footage
2. The tactical bird's-eye view
3. The timeline of key events
4. The AI's interpretation and recommendations

For mobile/tablet, fall back to **Pattern C** (stacked) with the video pinned at top.

### Critical Interaction: Event-Video Linking
When a user clicks an event (goal, shot, tactical moment) anywhere in the data panels, the video MUST jump to that timestamp. This is the single most important interaction in a video analytics tool.

---

## 9. Onboarding & Empty States

### Empty State Best Practices (2026)

From Eleken, SaaSFrame, and Flowjam research:

#### Three Types of Empty States

1. **Informational**: "It's empty, here's why" -- context and education
2. **Action-oriented**: "Let's get going" -- nudge toward first action
3. **Celebratory**: "All caught up!" -- reward completion

#### The Rules

1. **Never show "No data yet"** -- This is lazy and unhelpful. Instead:
   - Explain what will appear here when populated
   - Show a sample/demo dashboard with dummy data
   - Provide a clear CTA to populate the screen

2. **Match the message to the moment** -- A new user on an empty dashboard needs different messaging than someone who has processed all their matches

3. **Use illustrations/icons that match your brand** -- Don't use generic stock illustrations. Custom visuals build brand identity

4. **Maintain design consistency** -- Empty states should use the same color palette, typography, and spacing as populated states

5. **Include a single clear CTA** -- "Upload Your First Match" not a wall of options

#### DugoutIQ Empty State Examples

**Dashboard (no matches yet)**:
```
[Pitch illustration with dotted lines]

Welcome to DugoutIQ

Upload your first match to unlock AI-powered
tactical analysis and coaching insights.

[Upload Match Video]  (primary button)

Or try a demo match to see what's possible  (text link)
```

**AI Coach (no analysis yet)**:
```
[Brain/lightbulb illustration]

Your AI Coach is Ready

Once you upload and process a match video,
your AI coach will provide:
- Tactical recommendations
- Player performance insights
- Training session suggestions

[Upload a Match]  (primary button)
```

**Player Profile (no player data)**:
```
[Jersey illustration]

No Player Data Yet

Player profiles are automatically created when
you process match videos. Each player will get:
- Performance metrics across matches
- Development tracking over time
- AI-generated improvement areas

[Process Your First Match]  (primary button)
```

### Onboarding Flow

The best SaaS onboarding in 2026 follows this pattern:

1. **Welcome modal** (1 screen): "Welcome, Coach! Let's set up your team."
2. **Progressive profile** (2-3 steps): Team name, age group, league level
3. **First action prompt**: "Upload your first match or try our demo"
4. **Checklist widget**: Persistent sidebar or card showing setup progress
5. **Demo/sample data**: Option to explore with pre-loaded match data

#### Onboarding Checklist Pattern
```
Getting Started (2/5 complete)
  [x] Create your account
  [x] Set up your team
  [ ] Upload your first match
  [ ] Review AI analysis
  [ ] Create a training session
```

---

## 10. Mobile vs Desktop Considerations

### The Coach's Reality

Based on sports app research and coaching workflow analysis:

| Activity | Primary Device | Context |
|----------|---------------|---------|
| Match filming | Phone/tablet (touchline) | Standing, one-handed |
| Live coaching notes | Tablet (dugout) | Quick glances, cold/wet |
| Post-match review | Laptop/desktop (home/office) | Sitting, focused analysis |
| Training session prep | Laptop/tablet (home) | Detailed planning |
| Quick stat check | Phone (anywhere) | On-the-go, 30-second task |
| Team communication | Phone | Messaging, notifications |
| Pre-match briefing | Tablet/laptop | Presenting to team |

### Design Strategy: Desktop-First, Mobile-Aware

DugoutIQ is primarily a **desktop/tablet analysis tool** that needs to be **usable on mobile** for quick tasks.

**Desktop (primary)**:
- Full dashboard with side-by-side panels
- Video analysis with synchronized views
- Detailed statistical tables
- Training session builder
- Report generation

**Tablet (secondary)**:
- Touchline mode for live match notes
- Simplified dashboard with key metrics
- Video review (stacked layout)
- Training session viewer

**Mobile (tertiary)**:
- Quick stat summaries
- Match notifications / alerts
- AI coach text-based insights
- Share clips and reports
- Team communication

### Responsive Breakpoints

```
Desktop:   >= 1280px  (full experience)
Tablet:    768-1279px (adapted layout, stacked panels)
Mobile:    < 768px    (simplified, task-focused)
```

### Mobile Navigation Pattern

For mobile, replace the left sidebar with:
- **Bottom tab bar** (5 items max): Dashboard, Matches, Upload, Coaching, Profile
- **Hamburger menu** for secondary items: Settings, Help, Feedback

---

## 11. FA & Coaching Platform Analysis

### England Football Learning (learn.englandfootball.com)

**Design language**:
- Clean, content-first design
- White/light backgrounds
- FA blue as primary color
- Card-based content browsing (sessions, articles, courses)
- Progressive disclosure: overview cards > detailed content
- Category-based navigation (Sessions, Courses, Articles, Community)
- Mobile-responsive

**Content structure**:
- Sessions organized by skill (passing, pressing, tackling, movement)
- Articles for coaching theory and methodology
- Courses for formal qualifications
- Community forum for peer support
- "Learning Dashboard" for personal progress tracking

**What DugoutIQ can learn**:
- The FA uses approachable, non-intimidating language
- Content is organized by coaching TOPIC, not by data type
- Sessions include clear objectives, age groups, and skill focus
- Personalization: "Tell us about yourself" survey customizes experience

### The FA Boot Room (thefa.com/bootroom)

**Design language**:
- More traditional FA branding (navy, red, white)
- Content-heavy, article-based
- England DNA framework prominently featured
- Coaching fundamentals, qualifications, resources
- Less modern design than England Football Learning

**Key takeaway**: The FA is moving toward digital-first coaching education. DugoutIQ's training focus feature (generating FA-style drills from match analysis) directly connects to this ecosystem.

### CoachMentor.co.uk (Not Found)

No active coaching education platform at coachmentor.co.uk. The domain does not appear to host a sports coaching platform. This is relevant as Ricky's CoachMentor project can potentially occupy this space.

### UK Coaching (ukcoaching.org)

- Multi-sport coaching organization
- Course-based platform
- Club membership model
- Resources and CPD focus
- Not data/analytics focused -- purely education

---

## 12. Competitor UI Teardowns

### Summary Grid

| Product | Primary Color | Background | Nav Pattern | Strength | Weakness |
|---------|--------------|------------|-------------|----------|----------|
| Hudl StatsBomb | Orange/Blue | Light | Top nav + sidebar | Data + video integration | Complex, analyst-only |
| Wyscout | Dark blue | Light/Dark | Sidebar | Massive database | Enterprise feel, no coaching |
| Catapult | Green/Orange | Dark | Dashboard | Real-time biometrics | Hardware-dependent, expensive |
| VEO | Blue | Light | Top nav | Beautiful UX, easy upload | Shallow analytics |
| StatsBomb | Blue/Red | Light | Tab-based | Iconic radar charts | Data-only, no video (pre-Hudl) |
| XPS Network | Blue | Light | Sidebar | Multi-sport, session planning | Generic, not football-specific |
| Onform | Blue | Light | Mobile-first | Video annotation tools | Individual sport focus |
| SportsViz | Custom | Dark | Dashboard | Real-time match dashboards | Niche, rugby-focused |

### Design Patterns Across All Competitors

1. **All use card-based layouts** for stats and metrics
2. **Most use blue as primary** -- DugoutIQ's emerald/cyan is distinctively different
3. **Professional tools default to light mode** -- DugoutIQ's dark-first is a bold differentiator
4. **None prominently feature AI coaching recommendations** -- DugoutIQ's key advantage
5. **Video and data are typically separate modules** -- Integration is the premium feature
6. **Mobile is an afterthought for most** -- Usually just a companion app

---

## 13. Recommended Design System for DugoutIQ

### Brand Identity

```
Name:           DugoutIQ
Tagline:        "AI that coaches, not just counts"
Personality:    Professional, intelligent, warm, approachable
Voice:          Expert but not intimidating. Like a knowledgeable assistant coach.
```

### Color System

```css
/* Brand */
--brand-primary:      #10B981;  /* Emerald - primary actions, positive metrics */
--brand-secondary:    #06B6D4;  /* Cyan - links, secondary elements, data */
--brand-accent:       #F59E0B;  /* Amber - achievements, warnings, CTAs */

/* Semantic */
--color-success:      #22C55E;  /* Distinct from brand emerald */
--color-warning:      #F59E0B;  /* Amber */
--color-error:        #EF4444;  /* Red */
--color-info:         #3B82F6;  /* Blue */

/* Surfaces (dark mode) */
--surface-deepest:    #0F172A;  /* Sidebar, nav shell */
--surface-base:       #1E293B;  /* Main content background */
--surface-card:       #334155;  /* Card backgrounds */
--surface-elevated:   #475569;  /* Modals, dropdowns */
--surface-overlay:    rgba(0, 0, 0, 0.6);  /* Modal backdrops */

/* Text */
--text-primary:       #F1F5F9;  /* Main text */
--text-secondary:     #94A3B8;  /* Secondary text */
--text-muted:         #64748B;  /* Disabled, hints */
--text-inverse:       #0F172A;  /* Text on light backgrounds */

/* Borders */
--border-default:     #334155;
--border-subtle:      #1E293B;
--border-focus:       #10B981;  /* Focus rings */

/* Data Visualization (6-color scale) */
--data-1:             #10B981;  /* Emerald */
--data-2:             #06B6D4;  /* Cyan */
--data-3:             #8B5CF6;  /* Violet */
--data-4:             #F59E0B;  /* Amber */
--data-5:             #EF4444;  /* Red */
--data-6:             #EC4899;  /* Pink */
```

### Typography System

```css
--font-sans:     'Inter', system-ui, sans-serif;
--font-mono:     'JetBrains Mono', 'Fira Code', monospace;
--font-display:  'Space Grotesk', 'Inter', sans-serif;  /* Optional: for hero numbers */
```

### Spacing Scale

```
4px   (0.25rem) - tight spacing (icon padding)
8px   (0.5rem)  - compact spacing (within cards)
12px  (0.75rem) - default gap
16px  (1rem)    - card padding, section gaps
24px  (1.5rem)  - section padding
32px  (2rem)    - large section gaps
48px  (3rem)    - page section separation
```

### Border Radius

```
--radius-sm:    4px   (buttons, inputs)
--radius-md:    8px   (cards)
--radius-lg:    12px  (modals, large containers)
--radius-xl:    16px  (hero cards)
--radius-full:  9999px (pills, avatars)
```

### Shadow System (Dark Mode)

```css
/* Use glow/border instead of traditional shadows */
--shadow-sm:    0 0 0 1px rgba(51, 65, 85, 0.5);
--shadow-md:    0 0 0 1px rgba(51, 65, 85, 0.5), 0 4px 12px rgba(0, 0, 0, 0.3);
--shadow-lg:    0 0 0 1px rgba(51, 65, 85, 0.5), 0 8px 24px rgba(0, 0, 0, 0.4);
--shadow-glow:  0 0 12px rgba(16, 185, 129, 0.15);  /* Emerald glow for focus */
```

### Component Patterns

**Stat Card**:
- 8px border radius
- 1px border (border-default)
- 16px padding
- Metric name: text-secondary, xs size
- Hero number: text-primary, 3xl size, font-mono, tabular-nums
- Delta indicator: color-coded arrow + percentage
- Optional sparkline or mini bar

**Data Table**:
- Striped rows using surface-card / surface-base alternation
- Fixed header
- Sortable columns with clear indicators
- 10-15 rows default with pagination
- Row hover state using surface-elevated

**Pitch Visualization**:
- Dark green pitch base (#1a472a) or simplified flat gray pitch
- White lines for pitch markings
- Brand colors for team data
- Semi-transparent heat map overlays
- Player dots with jersey numbers

---

## 14. Key Takeaways & Action Items

### The 10 Commandments for DugoutIQ's Design

1. **AI recommendations front-and-center** -- No competitor does this. Make "what to do" the hero, not the data itself.

2. **Video + data must be synchronized** -- Clicking any data point should jump to the relevant video moment. This is table stakes for professional tools.

3. **Dark mode is the default** -- Use layered darkness (4-5 shades), not flat black. Increase font weights. Desaturate accent colors slightly.

4. **Left sidebar navigation** -- Collapsible, with clear section grouping (Matches, Coaching, Settings). Bottom tab bar on mobile.

5. **Card-based stat presentation** -- Hero number + context (vs average, trend). Never show a number without context.

6. **Progressive disclosure** -- Summary first, details on click/expand. Don't overwhelm coaches with every metric at once.

7. **Empty states that teach** -- Every empty screen should explain what will appear, why it matters, and how to populate it. Include a demo/sample option.

8. **The emerald/cyan palette is distinctive** -- Don't abandon it. Add amber for warmth, use navy for depth. You already stand out from the sea of blue competitors.

9. **Mobile is for quick checks, desktop is for analysis** -- Design the full experience for desktop/tablet first, then create a focused mobile companion.

10. **Accessibility is required, not optional** -- WCAG AA contrast ratios minimum. Tabular figures for number alignment. Color should never be the ONLY indicator (add icons/text).

### Immediate Design Actions

- [ ] Create a Figma design system with the recommended color tokens, typography scale, and component library
- [ ] Design the empty state illustrations (pitch-themed, on-brand)
- [ ] Build the stat card component with all variants (positive, negative, neutral, loading)
- [ ] Design the video + data split-panel layout for desktop
- [ ] Create the sidebar navigation with collapsible states
- [ ] Design the onboarding flow (3-step setup + checklist)
- [ ] Build a data visualization color scale that works on dark backgrounds
- [ ] Design the mobile bottom tab navigation
- [ ] Create loading/skeleton states for all data-heavy components
- [ ] Test all colors for WCAG AA compliance

### Inspiration Links

- [Dribbble: Sports Analytics](https://dribbble.com/tags/sports-analytics)
- [Dribbble: Sports Dashboard](https://dribbble.com/tags/sports-dashboard)
- [Dribbble: Football App](https://dribbble.com/tags/football_app)
- [Dribbble: Coaching App](https://dribbble.com/tags/coaching-app)
- [Behance: Sports Analytics](https://www.behance.net/search/projects/sports%20analytics)
- [Behance: Football UI](https://www.behance.net/search/projects/football%20ui)
- [SaaSFrame: Dashboard Examples](https://www.saasframe.io/categories/dashboard)
- [SaaSFrame: Empty States](https://www.saasframe.io/patterns/empty-state)
- [SaaSFrame: Onboarding](https://www.saasframe.io/categories/user-onboarding)
- [Muzli: Dashboard Design 2026](https://muz.li/blog/best-dashboard-design-examples-inspirations-for-2026/)
- [Figma: Dashboard Templates](https://www.figma.com/templates/dashboard-designs/)
- [UI8: Mabol Sports Dashboard Kit](https://ui8.net/caraka/products/mabol---sports-live-dashboard-ui-kit)
- [Hudl StatsBomb Platform](https://www.hudl.com/products/statsbomb/platform)
- [FA England Football Learning](https://learn.englandfootball.com/)
- [Sportsmith: 10-Step Data Viz Guide](https://www.sportsmith.co/articles/10-step-data-viz-guide/)

---

*Research compiled for DugoutIQ (football-analyzer) project. Ricky Barton, March 2026.*
