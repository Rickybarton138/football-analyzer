"""Claude AI coaching service — agentic loop with tool use.

Manager Mentor: a conversational AI assistant manager that can search match
video, look up player stats, generate clips, and design training sessions —
all while maintaining a natural coaching conversation.
"""

import json
import logging
import anthropic
from app.core.config import get_settings
from app.services.twelvelabs_service import TwelveLabsService
from app.services.supabase_service import SupabaseService
from app.services.mux_service import MuxService

logger = logging.getLogger(__name__)

COACHING_PERSONA = """You are Manager Mentor — a seasoned grassroots football coach and AI assistant.

BACKGROUND:
You've spent 20 years in grassroots football. You hold a UEFA B licence. You've managed teams from U9s to adult non-league. You know what it's like coaching with 14 players, no goalkeeper coach, and a pitch with a slope. You've done your badges, run Saturday morning sessions in the rain, and dealt with parents who think their kid should play striker.

COMMUNICATION STYLE:
- Conversational and natural — talk like a real coach at the touchline, not a textbook
- Encouraging but honest — "good intent but the execution let you down" not "that was bad"
- Use real football language naturally: "low block", "caught in transition", "pockets between the lines", "overloaded the right side"
- Keep it practical — these coaches have 90 minutes to train, not a full academy setup
- When referencing match moments, be specific: "around the 23rd minute when your right-back pushed too high..."
- Ask follow-up questions to understand the coach's situation and priorities

COACHING KNOWLEDGE:
- FA Coaching Pathway (Level 1, Level 2, UEFA B, A, Pro)
- Tactical periodisation (Vitor Frade) — morphocycles, specificity principle
- FA Four Corner Model: technical, tactical, physical, psychological/social
- Playing principles hierarchy: game model → phases → principles → sub-principles → key actions
- STEP Framework (Space, Task, Equipment, People) for adapting drills

RULES:
- Never invent observations you can't back up with the analysis or video search results
- If you don't have enough information, say so: "I'd need to see that passage of play again to be sure"
- When you find relevant video moments using the search tool, reference them with timestamps so the coach can watch
- Always make advice actionable — every observation should lead to "and here's what you can do about it"
- Proactively suggest training drills that address weaknesses you identify
- Remember this is grassroots — resources are limited, keep suggestions realistic"""

TOOLS = [
    {
        "name": "search_match_video",
        "description": "Search for specific moments in the match video using natural language. Returns timestamped clips. Use this when the coach asks about specific events, or when you want to reference a particular moment to support your advice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query, e.g. 'counter-attack leading to goal', 'goalkeeper distribution', 'defensive shape at set pieces'"
                },
                "video_id": {
                    "type": "string",
                    "description": "The TwelveLabs video ID to search within"
                }
            },
            "required": ["query", "video_id"]
        }
    },
    {
        "name": "get_match_analysis",
        "description": "Retrieve the full analysis data for a match including tactical analysis, coaching advice, highlights, and player analysis. Use this to ground your responses in actual match data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "match_id": {
                    "type": "string",
                    "description": "The match ID to get analysis for"
                }
            },
            "required": ["match_id"]
        }
    },
    {
        "name": "get_match_info",
        "description": "Get basic match information — title, opponent, formation, duration, status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "match_id": {
                    "type": "string",
                    "description": "The match ID"
                }
            },
            "required": ["match_id"]
        }
    },
    {
        "name": "analyse_video_moment",
        "description": "Ask TwelveLabs to analyse a specific aspect of the video with a custom prompt. Use this for deep-dive questions the pre-built analysis doesn't cover.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The TwelveLabs video ID"
                },
                "prompt": {
                    "type": "string",
                    "description": "Specific analysis prompt, e.g. 'How does the defensive line hold its shape when the opposition play long balls?'"
                }
            },
            "required": ["video_id", "prompt"]
        }
    },
    {
        "name": "create_video_clip",
        "description": "Generate a clip URL for a specific time range in the match video. Use this when you want to show the coach a particular moment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "playback_id": {
                    "type": "string",
                    "description": "The Mux playback ID"
                },
                "start": {
                    "type": "number",
                    "description": "Start time in seconds"
                },
                "end": {
                    "type": "number",
                    "description": "End time in seconds"
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what happens in this clip"
                }
            },
            "required": ["playback_id", "start", "end"]
        }
    },
    {
        "name": "get_player_list",
        "description": "Get the squad list with player names, positions, and squad numbers.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "design_training_drill",
        "description": "Design a specific training drill to address an identified weakness. Returns a structured drill with setup, instructions, coaching points, and progressions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "The tactical/technical area to work on, e.g. 'pressing triggers in a 4-4-2', 'switching play under pressure'"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "How many minutes for this drill",
                    "default": 15
                },
                "num_players": {
                    "type": "integer",
                    "description": "Number of players available",
                    "default": 16
                }
            },
            "required": ["focus"]
        }
    }
]


class CoachAI:
    def __init__(self):
        s = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)
        self.model = s.ai_model
        self.twelvelabs = TwelveLabsService()
        self.db = SupabaseService()
        self.mux = MuxService()

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as a string."""
        try:
            if tool_name == "search_match_video":
                results = await self.twelvelabs.search_moments(
                    tool_input["query"],
                    tool_input.get("video_id"),
                )
                if not results:
                    return "No matching moments found for that query."
                return json.dumps(results, default=str)

            elif tool_name == "get_match_analysis":
                analyses = await self.db.select(
                    "analyses",
                    f"match_id=eq.{tool_input['match_id']}&status=eq.complete",
                )
                if not analyses:
                    return "No completed analysis found for this match."
                return json.dumps(analyses, default=str)

            elif tool_name == "get_match_info":
                match = await self.db.select_one("matches", tool_input["match_id"])
                if not match:
                    return "Match not found."
                return json.dumps(match, default=str)

            elif tool_name == "analyse_video_moment":
                result = await self.twelvelabs.analyse_video(
                    tool_input["video_id"],
                    tool_input["prompt"],
                )
                return result

            elif tool_name == "create_video_clip":
                clip_url = self.mux.get_clip_url(
                    tool_input["playback_id"],
                    tool_input["start"],
                    tool_input["end"],
                )
                thumbnail = self.mux.get_thumbnail_at(
                    tool_input["playback_id"],
                    tool_input["start"],
                )
                return json.dumps({
                    "clip_url": clip_url,
                    "thumbnail_url": thumbnail,
                    "start": tool_input["start"],
                    "end": tool_input["end"],
                    "description": tool_input.get("description", ""),
                })

            elif tool_name == "get_player_list":
                players = await self.db.select("players")
                if not players:
                    return "No players in the squad yet."
                return json.dumps(players, default=str)

            elif tool_name == "design_training_drill":
                # Use Claude itself to design the drill
                resp = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=(
                        "You are designing a football training drill for a grassroots team. "
                        "Return a structured drill with: Name, Duration, Setup (pitch size, "
                        "equipment, player positions), Instructions (step by step), "
                        "Coaching Points (3-4 key things to watch for), and Progressions "
                        "(2 ways to make it harder). Use the STEP framework. "
                        "Keep it realistic for a public pitch with cones and bibs."
                    ),
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Design a {tool_input.get('duration_minutes', 15)}-minute drill "
                            f"for {tool_input.get('num_players', 16)} players.\n"
                            f"Focus area: {tool_input['focus']}"
                        ),
                    }],
                )
                return resp.content[0].text

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return f"Tool error: {str(e)}"

    async def chat(
        self,
        messages: list[dict],
        match_id: str | None = None,
        video_id: str | None = None,
        playback_id: str | None = None,
    ) -> dict:
        """Run the agentic coaching conversation loop.

        Args:
            messages: Conversation history [{role, content}, ...]
            match_id: Current match ID (injected into system context)
            video_id: TwelveLabs video ID for the current match
            playback_id: Mux playback ID for clip generation

        Returns:
            {response: str, clips: [...], tools_used: [...]}
        """
        # Build system prompt with match context
        system = COACHING_PERSONA
        if match_id:
            system += f"\n\nCURRENT MATCH CONTEXT:\n- Match ID: {match_id}"
        if video_id:
            system += f"\n- TwelveLabs Video ID: {video_id}"
        if playback_id:
            system += f"\n- Mux Playback ID: {playback_id}"
        if video_id:
            system += (
                "\n\nYou have access to this match's video. Use the search and analysis "
                "tools to find specific moments when the coach asks about them. "
                "When you find relevant clips, use create_video_clip to generate "
                "watchable links the coach can click."
            )

        clips = []
        tools_used = []

        # Agentic loop — keep going until Claude stops using tools
        api_messages = list(messages)
        max_iterations = 8

        for _ in range(max_iterations):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system,
                tools=TOOLS,
                messages=api_messages,
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Process all tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info("Coach calling tool: %s", block.name)
                        tools_used.append(block.name)
                        result = await self._execute_tool(block.name, block.input)

                        # Collect clips from create_video_clip results
                        if block.name == "create_video_clip":
                            try:
                                clip_data = json.loads(result)
                                clips.append(clip_data)
                            except (json.JSONDecodeError, TypeError):
                                pass

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                # Add Claude's response + tool results to conversation
                api_messages.append({"role": "assistant", "content": response.content})
                api_messages.append({"role": "user", "content": tool_results})

            else:
                # Claude is done — extract final text
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                return {
                    "response": text,
                    "clips": clips,
                    "tools_used": tools_used,
                }

        # Fallback if max iterations reached
        return {
            "response": "I've been thinking about this quite deeply — let me give you what I have so far. Could you ask a more specific question so I can focus my analysis?",
            "clips": clips,
            "tools_used": tools_used,
        }

    # --- Legacy methods (kept for backward compatibility with analysis pipeline) ---

    async def interpret_analysis(self, tactical_analysis: str, coach_context: str = "") -> dict:
        """Take TwelveLabs analysis and generate coaching advice."""
        user_message = f"Here is the tactical analysis of the match:\n\n{tactical_analysis}"
        if coach_context:
            user_message += f"\n\nCoach's notes: {coach_context}"

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=COACHING_PERSONA + "\n\nProvide:\n1. Key findings summary\n2. Top 3 strengths\n3. Top 3 areas for improvement\n4. 2-3 training drill suggestions\n5. Overall match rating /10",
            messages=[{"role": "user", "content": user_message}],
        )
        return {
            "coaching_advice": response.content[0].text,
            "model": self.model,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }

    async def answer_question(self, question: str, match_context: str) -> str:
        """Simple question answering (legacy — use chat() for full agentic flow)."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=COACHING_PERSONA,
            messages=[
                {"role": "user", "content": f"Match analysis:\n{match_context}\n\nCoach's question: {question}"},
            ],
        )
        return response.content[0].text

    async def generate_session_plan(self, weaknesses: str, available_time: int = 90) -> str:
        """Generate a training session plan."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=COACHING_PERSONA + "\n\nDesign a structured training session with warm-up, main drills, and cool-down. Each drill needs: name, duration, setup, instructions, coaching points. Keep it realistic for grassroots.",
            messages=[{
                "role": "user",
                "content": f"Key areas from match analysis:\n{weaknesses}\n\nAvailable time: {available_time} minutes.\nDesign a focused session plan.",
            }],
        )
        return response.content[0].text
