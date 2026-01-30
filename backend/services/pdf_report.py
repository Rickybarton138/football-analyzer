"""
PDF Report Generator

Generates comprehensive PDF match analysis reports.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import os


@dataclass
class ReportSection:
    """A section of the PDF report."""
    title: str
    content: str
    data: Optional[Dict] = None


class PDFReportGenerator:
    """
    Generates PDF match analysis reports.

    Uses basic HTML-to-PDF approach for compatibility without heavy dependencies.
    """

    def __init__(self):
        self.sections: List[ReportSection] = []

    def reset(self):
        """Reset for new report."""
        self.sections = []

    def add_section(self, title: str, content: str, data: Optional[Dict] = None):
        """Add a section to the report."""
        self.sections.append(ReportSection(title=title, content=content, data=data))

    def generate_html_report(self,
                             match_info: Dict,
                             pass_stats: Dict,
                             formation_stats: Dict,
                             tactical_events: Dict,
                             coaching_insights: Dict,
                             output_path: str) -> str:
        """
        Generate an HTML report that can be converted to PDF.

        Args:
            match_info: Basic match information
            pass_stats: Pass analysis data
            formation_stats: Formation data
            tactical_events: Tactical events
            coaching_insights: AI coaching analysis
            output_path: Path to save the HTML file

        Returns:
            Path to the generated HTML file
        """
        html_content = self._build_html(
            match_info, pass_stats, formation_stats,
            tactical_events, coaching_insights
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _build_html(self,
                    match_info: Dict,
                    pass_stats: Dict,
                    formation_stats: Dict,
                    tactical_events: Dict,
                    coaching_insights: Dict) -> str:
        """Build the HTML content for the report."""

        summary = coaching_insights.get('summary', {}) or {}
        insights = coaching_insights.get('insights', [])
        critical_insights = coaching_insights.get('critical_insights', [])

        home_pass = pass_stats.get('home', {})
        away_pass = pass_stats.get('away', {})
        home_form = formation_stats.get('home', {})
        away_form = formation_stats.get('away', {})
        events = tactical_events.get('events', [])

        # Count events by type
        pressing_events = len([e for e in events if e.get('event_type') == 'pressing_trigger'])
        dangerous_attacks = len([e for e in events if e.get('event_type') == 'dangerous_attack'])
        counter_attacks = len([e for e in events if e.get('event_type') == 'counter_attack'])

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: white;
        }}
        .header {{
            background: linear-gradient(135deg, #0891b2, #6366f1);
            color: white;
            padding: 30px;
            margin: -20px -20px 20px -20px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .rating {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 15px;
            font-size: 18px;
            font-weight: bold;
        }}
        .section {{
            margin-bottom: 25px;
            padding: 20px;
            background: #f9fafb;
            border-radius: 10px;
            border-left: 4px solid #0891b2;
        }}
        .section h2 {{
            color: #0891b2;
            font-size: 18px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section h2::before {{
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #0891b2;
            border-radius: 50%;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #0891b2;
        }}
        .stat-card .label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .stat-row:last-child {{
            border-bottom: none;
        }}
        .stat-row .label {{
            color: #666;
        }}
        .stat-row .values {{
            display: flex;
            gap: 30px;
        }}
        .stat-row .home {{
            color: #ef4444;
            font-weight: 600;
        }}
        .stat-row .away {{
            color: #64748b;
            font-weight: 600;
        }}
        .insight {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid;
        }}
        .insight.critical {{
            border-color: #ef4444;
            background: #fef2f2;
        }}
        .insight.high {{
            border-color: #f97316;
            background: #fff7ed;
        }}
        .insight.medium {{
            border-color: #eab308;
            background: #fefce8;
        }}
        .insight .title {{
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .insight .message {{
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }}
        .insight .recommendation {{
            font-size: 13px;
            color: #0891b2;
            background: #f0fdfa;
            padding: 10px;
            border-radius: 5px;
        }}
        .strength-weakness {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .strengths {{
            background: #f0fdf4;
            border-color: #22c55e;
        }}
        .weaknesses {{
            background: #fff7ed;
            border-color: #f97316;
        }}
        .list-item {{
            padding: 8px 0;
            padding-left: 20px;
            position: relative;
            font-size: 14px;
        }}
        .list-item::before {{
            content: '•';
            position: absolute;
            left: 5px;
            color: inherit;
        }}
        .team-talk {{
            background: linear-gradient(135deg, #f0fdfa, #eff6ff);
            padding: 20px;
            border-radius: 10px;
            margin-top: 10px;
        }}
        .team-talk h3 {{
            color: #0891b2;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 12px;
            border-top: 1px solid #eee;
            margin-top: 30px;
        }}
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                padding: 0;
            }}
            .section {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Match Analysis Report</h1>
            <div class="subtitle">
                {match_info.get('video_path', 'Match Analysis')} |
                {match_info.get('duration_minutes', 0):.0f} minutes analyzed
            </div>
            <div class="rating">
                Performance: {summary.get('overall_rating', 'N/A')}
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <p>{summary.get('tactical_summary', 'Analysis summary not available.')}</p>

            <div class="strength-weakness" style="margin-top: 15px;">
                <div class="section strengths" style="margin-bottom: 0;">
                    <h3 style="color: #22c55e; font-size: 14px; margin-bottom: 10px;">Key Strengths</h3>
                    {"".join(f'<div class="list-item" style="color: #166534;">{s}</div>' for s in summary.get('key_strengths', []))}
                </div>
                <div class="section weaknesses" style="margin-bottom: 0;">
                    <h3 style="color: #f97316; font-size: 14px; margin-bottom: 10px;">Areas to Improve</h3>
                    {"".join(f'<div class="list-item" style="color: #c2410c;">{w}</div>' for w in summary.get('areas_to_improve', []))}
                </div>
            </div>
        </div>

        <!-- Key Statistics -->
        <div class="section">
            <h2>Key Statistics</h2>
            <div class="grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="stat-card">
                    <div class="value">{home_pass.get('possession_percent', 50):.0f}%</div>
                    <div class="label">Possession (Home)</div>
                </div>
                <div class="stat-card">
                    <div class="value">{home_pass.get('pass_accuracy', 0):.0f}%</div>
                    <div class="label">Pass Accuracy (Home)</div>
                </div>
                <div class="stat-card">
                    <div class="value">{pressing_events}</div>
                    <div class="label">Pressing Triggers</div>
                </div>
                <div class="stat-card">
                    <div class="value">{dangerous_attacks}</div>
                    <div class="label">Dangerous Attacks</div>
                </div>
            </div>
        </div>

        <!-- Passing Analysis -->
        <div class="section">
            <h2>Passing Analysis</h2>
            <div class="stat-row">
                <span class="label">Total Passes</span>
                <div class="values">
                    <span class="home">{home_pass.get('total_passes', 0)}</span>
                    <span class="away">{away_pass.get('total_passes', 0)}</span>
                </div>
            </div>
            <div class="stat-row">
                <span class="label">Successful Passes</span>
                <div class="values">
                    <span class="home">{home_pass.get('successful_passes', 0)}</span>
                    <span class="away">{away_pass.get('successful_passes', 0)}</span>
                </div>
            </div>
            <div class="stat-row">
                <span class="label">Pass Accuracy</span>
                <div class="values">
                    <span class="home">{home_pass.get('pass_accuracy', 0):.1f}%</span>
                    <span class="away">{away_pass.get('pass_accuracy', 0):.1f}%</span>
                </div>
            </div>
            <div class="stat-row">
                <span class="label">Forward Pass Ratio</span>
                <div class="values">
                    <span class="home">{home_pass.get('forward_pass_ratio', 0):.1f}%</span>
                    <span class="away">{away_pass.get('forward_pass_ratio', 0):.1f}%</span>
                </div>
            </div>
            <div class="stat-row">
                <span class="label">Possession</span>
                <div class="values">
                    <span class="home">{home_pass.get('possession_percent', 50):.1f}%</span>
                    <span class="away">{away_pass.get('possession_percent', 50):.1f}%</span>
                </div>
            </div>
        </div>

        <!-- Formation Analysis -->
        <div class="section">
            <h2>Formation Analysis</h2>
            <div class="grid">
                <div class="stat-card" style="text-align: left; padding: 20px;">
                    <div style="color: #ef4444; font-weight: 600; margin-bottom: 10px;">Home Team</div>
                    <div style="font-size: 24px; font-weight: bold; color: #0891b2; margin-bottom: 5px;">
                        {home_form.get('primary_formation', 'Unknown')}
                    </div>
                    <div style="font-size: 13px; color: #666;">
                        Defensive Line: {home_form.get('avg_defensive_line', 0):.1f}%<br>
                        Compactness: {home_form.get('avg_compactness', 0):.0f}<br>
                        Formation Changes: {home_form.get('formation_changes', 0)}
                    </div>
                </div>
                <div class="stat-card" style="text-align: left; padding: 20px;">
                    <div style="color: #64748b; font-weight: 600; margin-bottom: 10px;">Away Team</div>
                    <div style="font-size: 24px; font-weight: bold; color: #0891b2; margin-bottom: 5px;">
                        {away_form.get('primary_formation', 'Unknown')}
                    </div>
                    <div style="font-size: 13px; color: #666;">
                        Defensive Line: {away_form.get('avg_defensive_line', 0):.1f}%<br>
                        Compactness: {away_form.get('avg_compactness', 0):.0f}<br>
                        Formation Changes: {away_form.get('formation_changes', 0)}
                    </div>
                </div>
            </div>
        </div>

        <!-- Tactical Events -->
        <div class="section">
            <h2>Tactical Events Summary</h2>
            <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
                <div class="stat-card">
                    <div class="value">{pressing_events}</div>
                    <div class="label">Pressing Triggers</div>
                </div>
                <div class="stat-card">
                    <div class="value">{dangerous_attacks}</div>
                    <div class="label">Dangerous Attacks</div>
                </div>
                <div class="stat-card">
                    <div class="value">{counter_attacks}</div>
                    <div class="label">Counter Attacks</div>
                </div>
            </div>
        </div>

        <!-- AI Coaching Insights -->
        <div class="section">
            <h2>AI Coaching Insights</h2>
            <p style="color: #666; margin-bottom: 15px; font-size: 14px;">
                {len(insights)} tactical insights identified ({len(critical_insights)} require immediate attention)
            </p>

            {"".join(self._format_insight_html(insight) for insight in insights[:10])}
        </div>

        <!-- Team Talks -->
        <div class="section">
            <h2>Coaching Messages</h2>

            <div class="team-talk">
                <h3>Half-Time Talk</h3>
                <p>{summary.get('half_time_message', 'No message available.')}</p>
            </div>

            <div class="team-talk" style="margin-top: 15px;">
                <h3>Full-Time Review</h3>
                <p>{summary.get('full_time_message', 'No message available.')}</p>
            </div>
        </div>

        <div class="footer">
            <p>Generated by Football Analyzer AI | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p>This report was generated using computer vision analysis and AI coaching algorithms.</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _format_insight_html(self, insight: Dict) -> str:
        """Format a single insight as HTML."""
        priority = insight.get('priority', 'medium')
        return f"""
        <div class="insight {priority}">
            <div class="title">{insight.get('title', 'Insight')}</div>
            <div class="message">{insight.get('message', '')}</div>
            <div class="recommendation">
                <strong>Recommendation:</strong> {insight.get('recommendation', '')}
            </div>
        </div>
        """

    def generate_report_data(self,
                             match_info: Dict,
                             pass_stats: Dict,
                             formation_stats: Dict,
                             tactical_events: Dict,
                             coaching_insights: Dict) -> Dict:
        """
        Generate report data as a structured dictionary.

        This can be used for API responses or further processing.
        """
        summary = coaching_insights.get('summary', {}) or {}
        insights = coaching_insights.get('insights', [])

        return {
            'generated_at': datetime.now().isoformat(),
            'match_info': match_info,
            'summary': {
                'overall_rating': summary.get('overall_rating', 'N/A'),
                'tactical_summary': summary.get('tactical_summary', ''),
                'key_strengths': summary.get('key_strengths', []),
                'areas_to_improve': summary.get('areas_to_improve', []),
                'half_time_message': summary.get('half_time_message', ''),
                'full_time_message': summary.get('full_time_message', ''),
            },
            'statistics': {
                'passing': {
                    'home': pass_stats.get('home', {}),
                    'away': pass_stats.get('away', {}),
                },
                'formations': {
                    'home': formation_stats.get('home', {}),
                    'away': formation_stats.get('away', {}),
                },
                'tactical_events': tactical_events,
            },
            'coaching_insights': insights,
            'insights_count': {
                'total': len(insights),
                'critical': len([i for i in insights if i.get('priority') == 'critical']),
                'high': len([i for i in insights if i.get('priority') == 'high']),
                'medium': len([i for i in insights if i.get('priority') == 'medium']),
            }
        }


# Singleton instance
pdf_report_generator = PDFReportGenerator()
