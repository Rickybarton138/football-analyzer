"""Claude AI coaching service — interprets analysis and generates advice."""

import anthropic
from app.core.config import get_settings


class CoachAI:
    def __init__(self):
        s = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)
        self.model = s.ai_model

    async def interpret_analysis(self, tactical_analysis: str, coach_context: str = "") -> dict:
        """Take TwelveLabs analysis and generate coaching advice."""
        system_prompt = (
            "You are Manager Mentor, an AI assistant for grassroots football coaches. "
            "You've just received tactical analysis of a match. Your job is to:\n"
            "1. Summarise the key findings in plain English (no jargon overload)\n"
            "2. Identify the top 3 things the team did well\n"
            "3. Identify the top 3 areas for improvement\n"
            "4. Suggest 2-3 specific training drills that address the weaknesses\n"
            "5. Give an overall match rating out of 10 with justification\n\n"
            "Be encouraging but honest. These are volunteer coaches with limited training time. "
            "Focus on actionable, practical advice they can implement in a 90-minute session."
        )

        user_message = f"Here is the tactical analysis of the match:\n\n{tactical_analysis}"
        if coach_context:
            user_message += f"\n\nCoach's notes: {coach_context}"

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return {
            "coaching_advice": response.content[0].text,
            "model": self.model,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }

    async def answer_question(self, question: str, match_context: str) -> str:
        """Let coaches ask questions about their match."""
        system_prompt = (
            "You are Manager Mentor, an AI football coaching assistant. "
            "A grassroots coach is asking about their recent match. "
            "Answer based on the match analysis provided. Be specific, practical, "
            "and reference specific moments where possible. "
            "If you don't have enough information, say so honestly."
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Match analysis:\n{match_context}\n\nCoach's question: {question}"},
            ],
        )
        return response.content[0].text

    async def generate_session_plan(self, weaknesses: str, available_time: int = 90) -> str:
        """Generate a training session plan based on identified weaknesses."""
        system_prompt = (
            "You are Manager Mentor, designing a training session for a grassroots football team. "
            "Create a structured session plan with warm-up, main drills, and cool-down. "
            "Each drill should have: name, duration, setup, instructions, and coaching points. "
            "Keep it practical — these teams train on public pitches with basic equipment."
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Based on match analysis, these are the key areas to work on:\n{weaknesses}\n\n"
                        f"Available training time: {available_time} minutes.\n"
                        "Design a focused session plan."
                    ),
                },
            ],
        )
        return response.content[0].text
