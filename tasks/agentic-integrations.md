# Agentic AI API Integrations — football-analyzer / Manager Mentor

Priority: **P3 — Secondary**
Master plan: `~/agentic-ai-apis/INTEGRATION_PLAN.md`

## Prerequisite
Apify MCP must be connected. See master plan.

## Agents to wire in

| Agent | Integration point | Purpose |
|---|---|---|
| [Sports Intelligence Autopilot](https://apify.com/actor_researcher.48/sports-intelligence-autopilot) | Match data ingestion worker | 38 leagues including Soccer, real-time stats — feeds cross-match tactical memory (pairs with RuVector) |
| [Academic Paper Scraper](https://apify.com/labrat011/academic-paper-scraper) | Sports-science research pipeline | Semantic Scholar + arXiv papers on tactical periodisation, game model theory |
| [Enterprise MCP Gateway](https://apify.com/nexgendata/enterprise-mcp-gateway) | Unified data layer | Sports data in one MCP endpoint |

## Integration with existing RuVector
Sports Intelligence Autopilot output → vectorise with existing RuVector (port 6333) → query during match analysis for cross-match patterns.

## Environment vars
```
APIFY_TOKEN=
APIFY_NEXGEN_GATEWAY_TOKEN=
RUVECTOR_URL=http://localhost:6333    # existing
```

## Next action
Low priority — this is a P3. Wire in after primehaul-leads, astra-removals, transport-manager, and rick-ai are live. When ready: ingest one Premier League match via Sports Intelligence Autopilot, vectorise the play-by-play, validate retrieval from RuVector.
