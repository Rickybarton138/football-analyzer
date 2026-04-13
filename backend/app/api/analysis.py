"""Analysis endpoints — tactical analysis, highlights, coaching advice, chat, auto-annotation."""

import json
import logging
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import AnalysisRequest, AnalysisType
from app.services.gemini_service import GeminiService
from app.services.coach_ai import CoachAI
from app.services.supabase_service import SupabaseService
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()
gemini = GeminiService()
coach = CoachAI()
db = SupabaseService()


@router.post("/run")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Trigger analysis on a match. Returns immediately, processes in background.

    If existing Gemini analysis exists in the DB, uses Claude to reinterpret it
    for the requested analysis type (much faster than re-analysing video).
    Requires existing Gemini analysis in the DB — uploads always populate this.
    """
    match = await db.select_one("matches", request.match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    # Check for existing raw analysis in DB
    existing = await db.select("analyses", f"match_id=eq.{request.match_id}&status=eq.complete")
    existing_raw = ""
    for a in existing:
        if a.get("tactical_raw"):
            existing_raw += a["tactical_raw"] + "\n\n---\n\n"

    if not existing_raw:
        raise HTTPException(400, "No analysis data available for this match yet. Wait for processing to finish.")

    analysis_id = str(uuid.uuid4())
    await db.insert("analyses", {
        "id": analysis_id,
        "match_id": request.match_id,
        "analysis_type": request.analysis_type,
        "status": "processing",
        "prompt": request.prompt,
    })

    # Build team identity context from match record
    our_color = match.get("team_color", "")
    opp_color = match.get("opponent_color", "")
    our_name = match["title"].split(" vs ")[0].strip() if " vs " in match.get("title", "") else "Our Team"
    opp_name = match.get("opponent") or "Opposition"

    team_identity = ""
    if our_color or opp_color:
        team_identity = (
            f"\n*** CRITICAL — TEAM IDENTIFICATION ***\n"
            f"The {our_color.upper()} team in the video is {our_name} — this is OUR team (the coached team).\n"
            f"The {opp_color.upper()} team in the video is {opp_name} — this is the OPPOSITION.\n"
            f"Whenever the raw data mentions '{our_color.capitalize()} Team' or '{our_color}' kit, that is {our_name} (US).\n"
            f"Whenever the raw data mentions '{opp_color.capitalize()} Team' or '{opp_color}' kit, that is {opp_name} (THEM).\n"
            f"DO NOT confuse the two. This mapping is definitive.\n"
            f"***\n"
        )

    background_tasks.add_task(
        reinterpret_analysis,
        analysis_id,
        existing_raw,
        request.analysis_type,
        request.prompt or "",
        team_identity,
        our_name,
        our_color,
        opp_name,
        opp_color,
    )

    return {"analysis_id": analysis_id, "status": "processing"}


async def reinterpret_analysis(
    analysis_id: str, raw_analysis: str, analysis_type: str, coach_context: str,
    team_identity: str = "",
    our_name: str = "Our Team", our_color: str = "", opp_name: str = "Opposition", opp_color: str = "",
):
    """Fast path: use Claude to reinterpret existing Gemini analysis for a specific focus."""
    try:
        update_data = {"status": "complete"}
        context_line = f"\nCoach's additional context: {coach_context}" if coach_context else ""

        prompts = {
            AnalysisType.full: (
                "Provide a comprehensive match analysis covering tactical setup, key moments, "
                "strengths, weaknesses, and actionable coaching advice for both teams."
            ),
            AnalysisType.our_team: (
                f"Focus ONLY on {our_name} (the {our_color.upper()} team in the video). "
                f"This is the coached team. Provide:\n"
                f"1. How {our_name} played — formation, shape in/out of possession\n"
                f"2. Build-up play and attacking patterns\n"
                f"3. Defensive organisation and pressing\n"
                f"4. Top 3 strengths shown\n"
                f"5. Top 3 areas for improvement\n"
                f"6. Key individual performances (good and bad)\n"
                f"7. Actionable coaching points for the next training session\n"
                f"Do NOT analyse {opp_name} ({opp_color} team) — keep this entirely about {our_name}'s performance."
            ),
            AnalysisType.opposition: (
                f"Focus ONLY on {opp_name} (the {opp_color.upper()} team in the video). "
                f"This is the opposition. Create a scouting report:\n"
                f"1. Formation and tactical setup\n"
                f"2. How they build up — short/long, which side\n"
                f"3. Attacking patterns and threats\n"
                f"4. Defensive shape and vulnerabilities\n"
                f"5. Key players and danger men\n"
                f"6. Set piece routines (attacking and defending)\n"
                f"7. How to exploit their weaknesses next time\n"
                f"Do NOT analyse {our_name} ({our_color} team) — keep this entirely about scouting {opp_name}."
            ),
            AnalysisType.highlights: (
                "Extract KEY TACTICAL MOMENTS from this analysis — NOT specific goals, "
                "scores, cards, or disciplinary events (the AI video analysis fabricates these).\n"
                "Focus ONLY on:\n"
                "- Good pressing sequences and where they won the ball back\n"
                "- Defensive transitions — how quickly teams recovered shape\n"
                "- Attacking overloads and positional rotations\n"
                "- Set piece delivery and movement patterns\n"
                "- Moments where shape broke down or gaps appeared\n"
                "- Individual tactical contributions (dribbles, switches of play, through balls)\n"
                "- Turning points in momentum (who dominated which periods)\n"
                "Include approximate timestamps where available. "
                "Do NOT report scorelines, goals, cards, or sendings off."
            ),
            AnalysisType.tactical: (
                "Provide a detailed tactical breakdown: "
                "formations, pressing triggers, build-up patterns, defensive shape, "
                "transitions, set piece organisation, and width/depth management for both teams."
            ),
            AnalysisType.player_spotlight: (
                "Identify every player mentioned in the analysis. "
                "For each, describe their position, key moments, notable actions (good and bad), "
                "and an overall performance rating with development recommendations."
            ),
        }

        focus = prompts.get(analysis_type, "Analyse this football match.")
        system_prompt = (
            "You are Manager Mentor, an elite grassroots football coaching AI. "
            "You have raw video analysis data from all quarters of a match. "
            "Synthesise the quarters into a single coherent report.\n"
            f"{team_identity}\n"
            "IMPORTANT: The raw data comes from AI video analysis which frequently "
            "halluccinates disciplinary events. DO NOT mention yellow cards, red cards, "
            "bookings, sendings off, or dismissals — the AI invents these. "
            "Also do not report exact scorelines as fact. "
            "Focus on tactical patterns, formations, positioning, and style of play.\n\n"
            f"TASK: {focus}{context_line}"
        )

        # Replace ALL color-based team labels with actual names so Claude
        # can't get confused. Simple case-insensitive string replacement.
        import re
        labelled_analysis = raw_analysis
        if our_color and our_name:
            # "Yellow Team" -> "VTFC", "yellow team" -> "VTFC"
            labelled_analysis = re.sub(
                rf'{re.escape(our_color)}\s+Team', our_name, labelled_analysis, flags=re.IGNORECASE
            )
        if opp_color and opp_name:
            labelled_analysis = re.sub(
                rf'{re.escape(opp_color)}\s+Team', opp_name, labelled_analysis, flags=re.IGNORECASE
            )
        # Strip lines mentioning cards/sendings off — Gemini hallucinates these
        cleaned_lines = []
        for line in labelled_analysis.split("\n"):
            low = line.lower()
            if any(phrase in low for phrase in [
                "yellow card", "red card", "sent off", "sending off",
                "second yellow", "two yellow", "dismissal", "dismissed",
                "straight red", "booked",
            ]):
                continue  # drop this line entirely
            cleaned_lines.append(line)
        labelled_analysis = "\n".join(cleaned_lines)

        # Prepend a clear mapping header to the data itself
        if our_color and opp_color:
            color_header = (
                f"=== TEAM KEY ===\n"
                f"{our_color.upper()} KIT = {our_name} (this is OUR team, the coached team)\n"
                f"{opp_color.upper()} KIT = {opp_name} (this is the OPPOSITION)\n"
                f"================\n\n"
            )
            labelled_analysis = color_header + labelled_analysis

        logger.info("System prompt:\n%s", system_prompt[:500])
        logger.info("First 300 chars of labelled data:\n%s", labelled_analysis[:300])

        response = await coach.client.messages.create(
            model=coach.model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": f"MATCH ANALYSIS DATA:\n\n{labelled_analysis}"}],
        )
        result_text = response.content[0].text

        if analysis_type == AnalysisType.player_spotlight:
            update_data["player_analysis_raw"] = result_text
        elif analysis_type == AnalysisType.highlights:
            update_data["highlights_raw"] = result_text
        elif analysis_type == AnalysisType.our_team:
            update_data["coaching_advice"] = result_text
        elif analysis_type == AnalysisType.opposition:
            update_data["highlights_raw"] = result_text  # reuse field for opposition report
        elif analysis_type in (AnalysisType.tactical, AnalysisType.full):
            update_data["tactical_raw"] = result_text
            update_data["coaching_advice"] = result_text

        await db.update("analyses", analysis_id, update_data)
        logger.info("Reinterpretation complete for %s (%s)", analysis_id, analysis_type)

    except Exception as e:
        logger.exception("Reinterpretation failed for %s", analysis_id)
        await db.update("analyses", analysis_id, {
            "status": "failed",
            "error_message": str(e),
        })


@router.get("/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis results."""
    analysis = await db.select_one("analyses", analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")
    return analysis


@router.get("/match/{match_id}")
async def get_match_analyses(match_id: str):
    """List all analyses for a match."""
    return await db.select("analyses", f"match_id=eq.{match_id}")


@router.post("/ask")
async def ask_about_match(match_id: str, question: str):
    """Ask a natural language question about a match."""
    analyses = await db.select("analyses", f"match_id=eq.{match_id}&status=eq.complete", limit=1)
    if not analyses:
        raise HTTPException(400, "No completed analysis found for this match. Run analysis first.")

    context = ""
    a = analyses[0]
    if a.get("tactical_raw"):
        context += f"Tactical Analysis:\n{a['tactical_raw']}\n\n"
    if a.get("coaching_advice"):
        context += f"Coaching Advice:\n{a['coaching_advice']}\n\n"
    if a.get("highlights_raw"):
        context += f"Highlights:\n{a['highlights_raw']}\n\n"

    answer = await coach.answer_question(question, context)
    return {"question": question, "answer": answer}


@router.post("/session-plan")
async def generate_session_plan(match_id: str, available_minutes: int = 90):
    """Generate a training session plan based on match analysis."""
    analyses = await db.select("analyses", f"match_id=eq.{match_id}&status=eq.complete", limit=1)
    if not analyses:
        raise HTTPException(400, "No completed analysis found.")

    coaching_advice = analyses[0].get("coaching_advice", "")
    if not coaching_advice:
        raise HTTPException(400, "No coaching advice available. Run a tactical or full analysis first.")

    plan = await coach.generate_session_plan(coaching_advice, available_minutes)
    return {"session_plan": plan, "based_on_match": match_id}


# --- AI Auto-Annotation (Frame-based Gemini Vision) ---

COLOR_HEX = {
    "red": "#ef4444", "blue": "#3b82f6", "yellow": "#eab308", "green": "#10b981",
    "white": "#ffffff", "black": "#18181b", "orange": "#f97316", "purple": "#a855f7",
    "pink": "#ec4899", "grey": "#71717a",
}


class AutoAnnotateRequest(BaseModel):
    match_id: str
    timestamp_sec: float = 0
    frame_base64: str  # data:image/png;base64,... or raw base64


@router.post("/auto-annotate")
async def auto_annotate(request: AutoAnnotateRequest):
    """Generate ONE focused coaching annotation for the current video frame.

    Sends the actual paused frame image to Gemini 2.5 Flash along with
    tactical context. Gemini SEES the frame and returns spatially-grounded
    annotations using its native bounding-box detection (0-1000 range).

    The coaching principle: one coaching point per frame, answer
    "what should change here?", not just "what happened".
    """
    import asyncio
    import base64
    from google.genai import types as genai_types

    match = await db.select_one("matches", request.match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    # Load tactical context for this timestamp
    analyses = await db.select("analyses", f"match_id=eq.{request.match_id}&status=eq.complete")
    corpus = "\n\n---\n\n".join(a.get("tactical_raw", "") for a in analyses if a.get("tactical_raw"))

    # Team identity
    our_color = match.get("team_color") or ""
    opp_color = match.get("opponent_color") or ""
    our_name = match["title"].split(" vs ")[0].strip() if " vs " in match.get("title", "") else "Our Team"
    opp_name = match.get("opponent") or "Opposition"

    team_block = ""
    if our_color and opp_color:
        team_block = (
            f"\nTEAM IDENTIFICATION:\n"
            f"- {our_color.upper()} kit = {our_name} (the coached team — OUR team)\n"
            f"- {opp_color.upper()} kit = {opp_name} (the opposition)\n"
        )

    mins = int(request.timestamp_sec // 60)
    secs = int(request.timestamp_sec % 60)

    # Extract raw base64 (strip data URL prefix if present)
    frame_b64 = request.frame_base64
    if "base64," in frame_b64:
        frame_b64 = frame_b64.split("base64,", 1)[1]
    frame_bytes = base64.b64decode(frame_b64)

    # Tactical context excerpt: find text near the requested timestamp
    context_excerpt = corpus[:8000] if corpus else "No tactical analysis available."

    prompt = f"""You are an elite football coaching analyst looking at a paused frame from a match video at approximately {mins}:{secs:02d}.

TASK: Identify ONE clear coaching point from this frame — what is the most important tactical observation, and what should change?

{team_block}

INSTRUCTIONS:
1. Look at the frame carefully. Identify visible players by their kit colour and shirt number (if readable).
2. Identify the single most important tactical coaching point visible in this moment.
3. Create 2-4 annotation elements to illustrate your point (arrows showing where a player SHOULD move, zones highlighting dangerous space, text explaining the coaching point).
4. Use box_2d format [ymin, xmin, ymax, xmax] normalized to 0-1000 for all positions, where (0,0) is the top-left corner of the frame.

COACHING CONTEXT from match analysis near this timestamp:
{context_excerpt[:4000]}

OUTPUT FORMAT (strict JSON):
{{
  "coaching_point": "One sentence — what the coach should tell the team about this moment",
  "detail": "2-3 sentences expanding on why this matters and what should change",
  "players_visible": [
    {{"box_2d": [ymin, xmin, ymax, xmax], "team": "team"|"opposition", "number": "shirt number or ?", "role": "brief role description"}}
  ],
  "annotations": [
    {{
      "type": "arrow"|"curved_arrow"|"zone"|"text"|"circle"|"spotlight",
      "box_2d": [ymin, xmin, ymax, xmax],
      "end_box_2d": [ymin, xmin, ymax, xmax],
      "label": "short label for the annotation",
      "team": "team"|"opposition"|"neutral"
    }}
  ]
}}

RULES:
- ONE coaching point only. Don't try to annotate everything — just the most important thing.
- Annotations should show what SHOULD happen (the correct option), not just describe what is happening.
- Arrow annotations: box_2d = start position, end_box_2d = where the arrow points TO.
- Zone annotations: box_2d defines the highlighted area.
- Text annotations: box_2d defines where the text label is placed.
- Maximum 4 annotation elements. Less is more — if a player can't grasp it in 3 seconds, it's noise.
- Do NOT mention yellow/red cards, scorelines, or disciplinary events."""

    try:
        # Send frame + prompt to Gemini 2.5 Flash (multimodal)
        response = await gemini.client.aio.models.generate_content(
            model=gemini.model,
            contents=[
                genai_types.Part.from_bytes(data=frame_bytes, mime_type="image/png"),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        result = json.loads(raw)

        # Map team tokens to hex colours and normalize coordinates from 0-1000 to 0-1
        team_hex = COLOR_HEX.get(our_color, "#10b981")
        opp_hex = COLOR_HEX.get(opp_color, "#ef4444")
        color_map = {"team": team_hex, "opposition": opp_hex, "neutral": "#ffffff"}

        def norm(box: list) -> dict:
            """Convert [ymin, xmin, ymax, xmax] in 0-1000 to {x, y, x2, y2} in 0-1."""
            if not box or len(box) < 4:
                return {"x": 0.5, "y": 0.5, "x2": 0.5, "y2": 0.5}
            return {
                "x": box[1] / 1000,
                "y": box[0] / 1000,
                "x2": box[3] / 1000,
                "y2": box[2] / 1000,
            }

        # Transform annotations into the frontend-compatible format
        elements = []
        for ann in result.get("annotations", []):
            pos = norm(ann.get("box_2d", []))
            end = norm(ann.get("end_box_2d", ann.get("box_2d", [])))
            hex_color = color_map.get(ann.get("team", "neutral"), "#ffffff")

            ann_type = ann.get("type", "text")
            if ann_type in ("arrow", "curved_arrow"):
                elements.append({
                    "shape": ann_type, "x": pos["x"], "y": pos["y"],
                    "x2": end["x"], "y2": end["y"],
                    "label": ann.get("label", ""), "hex_color": hex_color,
                })
            elif ann_type in ("zone", "circle", "spotlight"):
                elements.append({
                    "shape": "rect" if ann_type == "zone" else ann_type,
                    "x": pos["x"], "y": pos["y"],
                    "x2": pos["x2"], "y2": pos["y2"],
                    "label": ann.get("label", ""), "hex_color": hex_color,
                })
            elif ann_type == "text":
                elements.append({
                    "shape": "text", "x": pos["x"], "y": pos["y"],
                    "label": ann.get("label", ""), "hex_color": hex_color,
                })

        # Player markers
        players = []
        for p in result.get("players_visible", []):
            ppos = norm(p.get("box_2d", []))
            players.append({
                "shape": "marker",
                "x": (ppos["x"] + ppos["x2"]) / 2,
                "y": (ppos["y"] + ppos["y2"]) / 2,
                "label": p.get("number", "?"),
                "hex_color": color_map.get(p.get("team", "neutral"), "#ffffff"),
            })

        return {
            "coaching_point": result.get("coaching_point", ""),
            "detail": result.get("detail", ""),
            "annotations": [{
                "timestamp_sec": request.timestamp_sec,
                "type": "coaching_point",
                "description": result.get("coaching_point", ""),
                "elements": elements + players,
            }],
            "match_id": request.match_id,
            "timestamp_sec": request.timestamp_sec,
        }

    except json.JSONDecodeError as e:
        logger.error("Auto-annotate JSON parse failed: %s\nRaw: %.500s", e, raw)
        raise HTTPException(500, f"Gemini returned invalid annotation data: {e}")
    except Exception as e:
        logger.exception("Auto-annotate failed for match %s", request.match_id)
        raise HTTPException(500, str(e))


# --- Conversational AI Coach ---

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    match_id: str
    messages: list[ChatMessage]


@router.post("/chat")
async def chat_with_coach(request: ChatRequest):
    """Conversational AI coaching — agentic loop with tool use.

    Send the full conversation history. The coach can search video,
    analyse moments, generate clips, and design drills during the conversation.
    """
    match = await db.select_one("matches", request.match_id)
    if not match:
        raise HTTPException(404, "Match not found")

    # Convert to API format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Team identity — so the coach can map color-based Gemini output to real names
    our_name = match["title"].split(" vs ")[0].strip() if " vs " in match.get("title", "") else "Our Team"
    opp_name = match.get("opponent") or "Opposition"

    result = await coach.chat(
        messages=messages,
        match_id=request.match_id,
        playback_id=match.get("mux_playback_id"),
        our_name=our_name,
        opp_name=opp_name,
        our_color=match.get("team_color") or "",
        opp_color=match.get("opponent_color") or "",
    )

    return result
