# Manager Mentor v2

AI-powered football video analysis for grassroots coaches.
**Owner**: Ricky (rickybarton138@btinternet.com)

## Tech Stack
- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: Python FastAPI
- **Database**: Supabase (project: telomind / `wtuzuagspfezcqbjvdwl`)
- **Video AI**: TwelveLabs (index: `manager-mentor-matches` / `69c70b5374e8033fe643609e`)
- **Video Hosting**: Mux (org: Rickai / env: Manager Mentor)
- **AI Coach**: Claude (Anthropic API)
- **Payments**: Stripe
- **Deployment**: Netlify (frontend) + Railway or Render (backend)

## Architecture
Three-layer stack:
1. **Coach Interface** (React) — Upload, Dashboard, Highlights, Insights
2. **Analysis Engine** — TwelveLabs semantic + OpenCV CV + Claude interpretation
3. **Video Delivery** — Mux hosting, clipping, streaming, thumbnails

Pipeline: Upload -> Mux host -> TwelveLabs index -> Analyse -> Clip -> Display

## Agent Behaviour Rules
1. **Plan Mode Default** — enter plan mode for 3+ step tasks; stop and re-plan on failure
2. **Subagent Strategy** — offload research/exploration to subagents; one task per subagent
3. **Self-Improvement Loop** — after any user correction, update `tasks/lessons.md`
4. **Verification Before Done** — prove it works before marking complete
5. **Demand Elegance (Balanced)** — challenge hacky solutions on non-trivial changes
6. **Autonomous Bug Fixing** — just fix bugs using logs/errors/tests

## Task Management
- Plans and progress: `tasks/todo.md`
- Lessons learned: `tasks/lessons.md`

## Core Principles
- **Simplicity First** — minimise code impact, let TwelveLabs/Mux do the heavy lifting
- **No Laziness** — find root causes, senior-level standards
- **Grassroots Focus** — build for volunteer coaches with phone footage, not pro teams
