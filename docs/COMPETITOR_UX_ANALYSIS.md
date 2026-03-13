# Competitor & UX Deep Analysis — Manager Mentor Rebrand

## Executive Summary

After analyzing 10+ competitors (VEO, HUDL, Metrica, Pixellot, Trace, Catapult, Nacsport, Wyscout, InStat, Coach Logic) and current UX trends, the clear strategic direction is:

**Make this an extension of Coach Mentor** — unify the brand, share the design system, and position video analysis as the "Match Day" module within the Coach Mentor ecosystem.

### Why Unify with Coach Mentor?
1. **Coach Logic proves it**: They're pivoting from video analysis toward coach development (SAM). You already HAVE coach development (Coach Mentor). Adding video analysis to it is the natural evolution.
2. **No competitor does both**: Nobody combines coaching education + AI session planning + video analysis + training recommendations. This is your moat.
3. **Brand cohesion**: Two separate brands (DugoutIQ + CoachMentor) splits mindshare. One brand is stronger.
4. **Shared user base**: The coach who uses Coach Mentor to plan sessions is the SAME coach who films matches and wants analysis.

---

## Competitor Landscape Summary

### Tier 1: Enterprise (Not competing with us)
| Platform | Primary Color | Feel | Price | Gap |
|----------|--------------|------|-------|-----|
| **Catapult** | Red #FF4655 on navy | Elite, institutional | Custom ($$$$) | Too expensive, no coaching education |
| **Pixellot** | Orange on black | Broadcast infrastructure | Enterprise | Sells to federations, not coaches |
| **Wyscout** | Orange (Hudl) on dark | Data authority | Custom | Scouting only, no coaching tools |
| **InStat** | Navy + light blue | Data precision | Custom | Data-first, no coaching development |

### Tier 2: Direct Competitors
| Platform | Primary Color | Feel | Price | Gap |
|----------|--------------|------|-------|-----|
| **VEO** | Green #3DD64F on black | Accessible, sporty | €67-300/mo + €1,200 camera | Hardware required, basic analytics, no coaching |
| **HUDL** | Orange #FF6600 on white/dark | Professional standard | $400-1,600/yr | Priced out grassroots, no AI, confusing products |
| **Metrica** | Cyan/Blue on dark | Premium SaaS | €10-150/mo | No coaching education, analyst-focused |
| **Nacsport** | Green #4CAF50 on white | Workmanlike | Tiered desktop | Desktop-only, dated UX |

### Tier 3: Adjacent/Niche
| Platform | Primary Color | Feel | Price | Gap |
|----------|--------------|------|-------|-----|
| **Trace** | Green on white | Consumer, youthful | Camera + sub | Parents only, no coaching tools |
| **Coach Logic** | Teal #00BCD4 on white | Educational | Unknown | No tactical analysis, basic video |

### The Open Lane
**Nobody combines: Coaching Education + AI Session Planning + Video Analysis + AI Tactical Recommendations + Training Focus**

That's Coach Mentor + Video Analysis = the full coaching lifecycle.

---

## Brand & Design Analysis

### What Works in Sports Tech Branding

**Color Psychology:**
- **Green** (VEO, Trace, Nacsport) → Growth, the pitch, accessibility, grassroots
- **Orange** (Hudl, Pixellot) → Energy, action, confidence, standing out
- **Blue/Cyan** (Metrica, InStat) → Trust, technology, data, intelligence
- **Red** (Catapult) → Power, elite, intensity
- **Teal** (Coach Logic) → Balance, education, thoughtfulness

**Coach Mentor's Current Palette:**
- Primary: Pitch Green `#43A047`
- Deep Green: `#2E7D32`
- Gold: `#E9C46A`
- Fonts: DM Sans + Playfair Display
- Theme: Light (white backgrounds)

**DugoutIQ's Current Palette (what you don't like):**
- Primary: Emerald `#10B981`
- Accent: Cyan `#06B6D4`
- Background: Slate-900 `#0f172a`
- Theme: Dark

### The Problem with DugoutIQ's Current Design
1. **Emerald + Cyan feels generic** — could be any SaaS dashboard, nothing says "football" or "coaching"
2. **No brand connection to Coach Mentor** — completely different identity
3. **Dark-only feels cold** — Coach Mentor is warm, supportive, educational. DugoutIQ feels like a different company
4. **"DugoutIQ" as a name** doesn't connect to the Coach Mentor ecosystem

---

## Recommended Brand Strategy

### Option: Extend Coach Mentor

**Name:** "CoachMentor Match Analysis" or "CoachMentor Pro"
- Keep "CoachMentor" as the master brand
- Video analysis becomes a module/feature within the platform
- Navigation: Home | Create | Analyse | Library | Mentor | **Match Day** | Review

**Color Palette (aligned with Coach Mentor):**
```
Primary:     #43A047 (Pitch Green — inherited from Coach Mentor)
Deep:        #2E7D32 (Deep Green — inherited)
Gold:        #E9C46A (Highlights, achievements — inherited)
Data Blue:   #42A5F5 (Analytics, AI insights — Coach Mentor's existing Sky Blue)
Alert:       #E76F51 (Warnings, urgent — inherited Warm Coral)

Background:  #FAFAFA (Light mode — Coach Mentor default)
Surface:     #FFFFFF (Cards)
Dark BG:     #1A1A1A (Video analysis mode — dark when viewing video)
Text:        #1A1A1A (Charcoal — inherited)
```

**Why Light Mode (with dark video mode)?**
- Coach Mentor is light. Brand consistency matters.
- VEO, Metrica, Catapult are all dark → we differentiate by being light
- Trace uses light and feels approachable — grassroots coaches prefer it
- **Video analysis screens CAN be dark** (switch to dark when viewing video, like how YouTube switches)
- Light mode is more accessible (easier to read in outdoor/sideline conditions)

**Typography (inherited from Coach Mentor):**
- Primary: DM Sans (body, UI)
- Display: Playfair Display (headings, special moments)
- Data: Tabular figures for statistics

---

## UX Patterns to Adopt

### From VEO (what works):
- **Video player dominates the screen** — center stage
- **Event timeline** along the bottom with color-coded markers
- **Shot map on pitch diagram** — SVG top-down pitch view
- **Analytics Studio** for cross-match comparison
- **Coach Assist** AI panel alongside video
- **"Follow-cam" + "Interactive" mode** toggle

### From HUDL (what works):
- **Playlist-based navigation** — clips organized in sidebar
- **Drawing tools overlay** on video
- **Uniform design system** — consistent, documented components
- **Sport-specific landing pages** for marketing
- **Orange used sparingly** (accent only) — we should use Gold sparingly

### From Metrica (what works):
- **Transparent pricing** — essential for grassroots market
- **"No hardware required"** messaging
- **Tiered products** with low entry point (€10/mo)
- **TV-quality telestration** output
- **Dark mode for video analysis** workspace

### From Trace (what works):
- **Social media UX patterns** for highlights/sharing
- **Player profiles** for recruitment
- **"Set and forget"** messaging (simplicity)
- **Consumer-friendly, joyful brand energy**

### From Coach Logic (what works):
- **Collaborative analysis** — players watch and comment
- **SAM-style coach observation** (we have AI Coach already)
- **Educational brand positioning** (matches Coach Mentor perfectly)
- **Mobile-first engagement** (push notifications)

### From UX Research (best practices):
- **Stat cards** as hero elements (big number + delta + sparkline)
- **Amber/Gold accent** for achievements and CTAs
- **4-shade surface scale** for depth without flatness
- **Inter or DM Sans** for dashboard text (500 weight in dark mode)
- **Tabular figures** for statistical data alignment
- **Video + data side-by-side** with synchronized timestamps
- **Never show empty "No data yet"** — always show what will appear + demo option
- **Onboarding checklist** (2/5 complete style) to drive activation
- **F-pattern scanning** — primary metrics in top-left quadrant

### Key Layout Pattern for Match Analysis:
```
┌─────────────────────────────────────────────┐
│  Header: Team A vs Team B | Score | Date    │
├─────────────┬───────────────────────────────┤
│             │                               │
│  AI Coach   │     Video Player              │
│  Panel      │     (dark background)         │
│             │                               │
│  Insights   ├───────────────────────────────┤
│  Training   │  Event Timeline               │
│  Drills     │  (color-coded markers)         │
│             ├───────────────────────────────┤
│  Suggested  │  Stats / Tactical View        │
│  Questions  │  (pitch diagram / charts)     │
│             │                               │
└─────────────┴───────────────────────────────┘
```

---

## What Makes Us Different (No Competitor Has This)

1. **AI that tells coaches WHAT TO DO** — not just what happened
2. **Training drills generated from match weaknesses** — VEO's Coach Assist suggests generic drills; ours are specific to detected weaknesses
3. **FA-aligned session plans** — nobody else connects match analysis to coaching education frameworks
4. **Coaching education + video analysis in one platform** — Coach Logic is closest but has no tactical AI
5. **Software-only, no hardware** — unlike VEO ($1,200 camera) or Catapult (wearables)
6. **Affordable for grassroots** — unlike HUDL ($400+ minimum)
7. **Coach development tracking** — improve as a coach over time, not just analyze matches

---

## Recommended Next Steps

1. **Rebrand from DugoutIQ to CoachMentor Pro / Match Analysis**
2. **Adopt Coach Mentor's design system** (Pitch Green, DM Sans, light theme)
3. **Add dark video mode** — switch to dark background only when viewing video
4. **Redesign the frontend** with Coach Mentor's component library
5. **Integrate navigation** — video analysis as a tab within the Coach Mentor app
6. **Prioritize the AI coaching insights** — make them the hero, not buried in tabs
7. **Add collaborative features** — players can view clips, comment (Coach Logic-style)
8. **Transparent pricing page** — Free → £9.99 → £29.99 tiers

---

## Competitor Design Reference

### VEO Brand Colors
- Active Green: `#3DD64F`
- Calm Green: `#093E14`
- Black: `#000000`
- Secondary pairs: Blue, Purple, Pink, Red, Orange, Yellow (active + calm duos)

### HUDL Brand Colors
- Primary Orange: ~`#FF6600`
- Ink/Dark: Black
- Design system: "Uniform" (uniform.hudl.com)
- Font: FF Daxline Pro Bold
- Logo: Three people in a huddle (from above)

### Coach Mentor Brand Colors (Current)
- Pitch Green: `#43A047`
- Deep Green: `#2E7D32`
- Forest Green: `#1B5E20`
- Gold: `#E9C46A`
- Sky Blue: `#42A5F5`
- Warm Coral: `#E76F51`
- Charcoal: `#1A1A1A`
- Off White: `#FAFAFA`
- Fonts: DM Sans + Playfair Display
