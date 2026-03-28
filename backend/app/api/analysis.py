"""Analysis endpoints — tactical analysis, highlights, coaching advice."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import AnalysisRequest, AnalysisType
from app.services.twelvelabs_service import TwelveLabsService
from app.services.coach_ai import CoachAI
from app.services.supabase_service import SupabaseService
import uuid

router = APIRouter()
twelvelabs = TwelveLabsService()
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
        request.analysis_type,
        request.prompt or "",
    )

    return {"analysis_id": analysis_id, "status": "processing"}


async def run_analysis_pipeline(
    analysis_id: str, video_id: str, analysis_type: str, coach_context: str
):
    """Background: run the full analysis pipeline."""
    try:
        update_data = {"status": "complete"}

        if analysis_type == AnalysisType.highlights:
            highlights = await twelvelabs.get_highlights(video_id)
            update_data["highlights_raw"] = highlights[0]["raw_analysis"] if highlights else ""

        elif analysis_type == AnalysisType.tactical:
            tactical = await twelvelabs.get_tactical_analysis(video_id, coach_context)
            coaching = await coach.interpret_analysis(tactical, coach_context)
            update_data["tactical_raw"] = tactical
            update_data["coaching_advice"] = coaching["coaching_advice"]

        elif analysis_type == AnalysisType.full:
            # Run both highlights and tactical
            highlights = await twelvelabs.get_highlights(video_id)
            tactical = await twelvelabs.get_tactical_analysis(video_id, coach_context)
            coaching = await coach.interpret_analysis(tactical, coach_context)
            update_data["highlights_raw"] = highlights[0]["raw_analysis"] if highlights else ""
            update_data["tactical_raw"] = tactical
            update_data["coaching_advice"] = coaching["coaching_advice"]

        elif analysis_type == AnalysisType.player_spotlight:
            coach_line = f"\nCoach's notes: {coach_context}" if coach_context else ""
            prompt = (
                "Identify each individual player visible in this football match. "
                "For each player, describe: their position, key moments, "
                "notable actions (good and bad), and an overall assessment."
                + coach_line
            )
            analysis = await twelvelabs.analyse_video(video_id, prompt)
            update_data["player_analysis_raw"] = analysis

        await db.update("analyses", analysis_id, update_data)

    except Exception as e:
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
