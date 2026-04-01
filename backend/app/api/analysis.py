"""Analysis endpoints — tactical analysis, highlights, coaching advice, chat."""

import logging
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import AnalysisRequest, AnalysisType
from app.services.twelvelabs_service import TwelveLabsService
from app.services.gemini_service import GeminiService
from app.services.coach_ai import CoachAI
from app.services.supabase_service import SupabaseService
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()
twelvelabs = TwelveLabsService()
gemini = GeminiService()
coach = CoachAI()
db = SupabaseService()


@router.post("/run")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Trigger analysis on a match. Returns immediately, processes in background."""
    match = await db.select_one("matches", request.match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    if not match.get("twelvelabs_video_id"):
        raise HTTPException(400, "Match not yet indexed. Wait for indexing to complete.")

    analysis_id = str(uuid.uuid4())
    await db.insert("analyses", {
        "id": analysis_id,
        "match_id": request.match_id,
        "analysis_type": request.analysis_type,
        "status": "processing",
        "prompt": request.prompt,
    })

    background_tasks.add_task(
        run_analysis_pipeline,
        analysis_id,
        match["twelvelabs_video_id"],
        match.get("mux_playback_id"),
        request.analysis_type,
        request.prompt or "",
    )

    return {"analysis_id": analysis_id, "status": "processing"}


async def run_analysis_pipeline(
    analysis_id: str, video_id: str, mux_playback_id: str | None,
    analysis_type: str, coach_context: str,
):
    """Background: run the analysis pipeline.

    Uses Gemini 2.5 Flash for video analysis (no rate limits, cheap),
    then Claude interprets the output into coaching advice.
    Falls back to TwelveLabs if no Mux playback ID available.
    """
    try:
        update_data = {"status": "complete"}
        coach_line = f"\nCoach's additional context: {coach_context}" if coach_context else ""

        # Build prompt — keep concise for Gemini (long prompts cause refusals)
        if analysis_type == AnalysisType.full:
            prompt = (
                "You are an expert football analyst. Analyse this match footage. "
                "Identify key moments with timestamps, describe the tactical setup of both teams, "
                "and compare their strengths and weaknesses." + coach_line
            )
        elif analysis_type == AnalysisType.highlights:
            prompt = (
                "You are an expert football analyst. Identify every key moment in this match "
                "with timestamps. Include goals, shots, saves, tackles, fouls, counter-attacks, "
                "set pieces, and defensive errors. State which team was involved and why it matters." + coach_line
            )
        elif analysis_type == AnalysisType.tactical:
            prompt = (
                "You are an expert football analyst. Provide a detailed tactical analysis of this match. "
                "Cover formations, build-up play, pressing, attacking patterns, defensive shape, "
                "and set pieces for both teams." + coach_line
            )
        elif analysis_type == AnalysisType.player_spotlight:
            prompt = (
                "You are an expert football analyst. Identify each player visible in this match. "
                "Describe their jersey colour and number if visible, position, key moments, "
                "notable actions good and bad, and overall performance." + coach_line
            )
        else:
            prompt = f"Analyse this football match.{coach_line}"

        # Use Gemini for video analysis (no rate limits)
        # Check for existing Gemini file URI, fall back to TwelveLabs
        gemini_uri = gemini.get_active_file_uri()
        if gemini_uri:
            logger.info("Using Gemini (found active file: %s)", gemini_uri[:60])
            raw_analysis = await gemini.analyse_with_file_uri(gemini_uri, prompt)
        elif video_id:
            logger.info("Falling back to TwelveLabs (no Gemini file)")
            raw_analysis = await twelvelabs.analyse_video(video_id, prompt)
        else:
            raise Exception("No video source available for analysis")

        if analysis_type == AnalysisType.player_spotlight:
            update_data["player_analysis_raw"] = raw_analysis
        elif analysis_type == AnalysisType.highlights:
            update_data["highlights_raw"] = raw_analysis
        elif analysis_type in (AnalysisType.tactical, AnalysisType.full):
            update_data["tactical_raw"] = raw_analysis
            if analysis_type == AnalysisType.full:
                update_data["highlights_raw"] = raw_analysis

            # Claude interprets the raw analysis into coaching advice
            coaching = await coach.interpret_analysis(raw_analysis, coach_context)
            update_data["coaching_advice"] = coaching["coaching_advice"]

        await db.update("analyses", analysis_id, update_data)

    except Exception as e:
        logger.exception("Analysis pipeline failed for %s", analysis_id)
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

    result = await coach.chat(
        messages=messages,
        match_id=request.match_id,
        video_id=match.get("twelvelabs_video_id"),
        playback_id=match.get("mux_playback_id"),
    )

    return result
