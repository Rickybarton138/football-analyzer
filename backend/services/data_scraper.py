"""
Football Data Scraper Service

Collects match data, player stats, and events from various public sources
to build training datasets for ML models.

IMPORTANT: This scraper is designed to work with publicly available data sources
that allow data collection. Always respect robots.txt and rate limits.
"""
import asyncio
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import httpx
from bs4 import BeautifulSoup
import re

from services.training_data import (
    training_data_service,
    MatchAnnotation,
    PlayerAnnotation,
    EventAnnotation
)


@dataclass
class ScrapedMatch:
    """Match data from scraping."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    date: str
    competition: str
    venue: Optional[str] = None
    stats: Optional[Dict] = None
    events: Optional[List[Dict]] = None
    players: Optional[List[Dict]] = None


class FootballDataScraper:
    """
    Scraper for collecting football match data from public sources.

    Supported Sources:
    - FBref (free comprehensive stats)
    - Understat (xG data)
    - Transfermarkt (market values, basic stats)
    - Football-Data.co.uk (historical results)
    - API-Football (requires free API key)

    Note: Commercial sources like Opta, StatsBomb require paid licenses.
    """

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self.rate_limit_delay = 2.0  # seconds between requests

    async def close(self):
        await self.client.aclose()

    # ==================== FBref (free, comprehensive) ====================

    async def scrape_fbref_match(self, match_url: str) -> Optional[ScrapedMatch]:
        """
        Scrape match data from FBref.

        Example URL: https://fbref.com/en/matches/abc123/Team1-Team2-Premier-League
        """
        try:
            await asyncio.sleep(self.rate_limit_delay)
            response = await self.client.get(match_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract basic match info
            scorebox = soup.find('div', class_='scorebox')
            if not scorebox:
                return None

            teams = scorebox.find_all('strong')
            scores = scorebox.find_all('div', class_='score')

            if len(teams) < 2 or len(scores) < 2:
                return None

            home_team = teams[0].text.strip()
            away_team = teams[1].text.strip()
            home_score = int(scores[0].text.strip())
            away_score = int(scores[1].text.strip())

            # Extract date
            date_elem = soup.find('span', class_='venuetime')
            date_str = date_elem.get('data-venue-date', '') if date_elem else ''

            # Extract competition
            comp_elem = soup.find('a', href=re.compile(r'/en/comps/'))
            competition = comp_elem.text.strip() if comp_elem else ''

            # Extract team stats
            stats = await self._extract_fbref_stats(soup)

            # Extract player stats
            players = await self._extract_fbref_players(soup, home_team, away_team)

            return ScrapedMatch(
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                date=date_str,
                competition=competition,
                stats=stats,
                players=players
            )

        except Exception as e:
            print(f"Error scraping FBref: {e}")
            return None

    async def _extract_fbref_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract team statistics from FBref match page."""
        stats = {}

        # Find team stats tables
        stats_table = soup.find('div', id='team_stats')
        if stats_table:
            rows = stats_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    stat_name = cells[1].text.strip().lower().replace(' ', '_')
                    home_val = cells[0].text.strip()
                    away_val = cells[2].text.strip()

                    # Parse percentages
                    if '%' in home_val:
                        home_val = float(home_val.replace('%', ''))
                        away_val = float(away_val.replace('%', ''))
                    else:
                        try:
                            home_val = int(home_val)
                            away_val = int(away_val)
                        except ValueError:
                            continue

                    stats[f"home_{stat_name}"] = home_val
                    stats[f"away_{stat_name}"] = away_val

        return stats

    async def _extract_fbref_players(
        self,
        soup: BeautifulSoup,
        home_team: str,
        away_team: str
    ) -> List[Dict]:
        """Extract player statistics from FBref match page."""
        players = []

        # Find player stats tables
        for team, team_name in [('home', home_team), ('away', away_team)]:
            table = soup.find('table', id=re.compile(f'stats.*{team}.*summary'))
            if not table:
                continue

            rows = table.find('tbody').find_all('tr') if table.find('tbody') else []

            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) < 5:
                    continue

                player_name = cells[0].text.strip()
                if not player_name or player_name == 'Squad Total':
                    continue

                player_data = {
                    'name': player_name,
                    'team': team,
                    'minutes_played': self._parse_int(cells[3].text) if len(cells) > 3 else 0,
                    'goals': self._parse_int(cells[4].text) if len(cells) > 4 else 0,
                    'assists': self._parse_int(cells[5].text) if len(cells) > 5 else 0,
                }

                players.append(player_data)

        return players

    # ==================== Understat (xG data) ====================

    async def scrape_understat_match(self, match_id: str) -> Optional[Dict]:
        """
        Scrape xG data from Understat.

        Example: match_id = "12345"
        """
        try:
            await asyncio.sleep(self.rate_limit_delay)
            url = f"https://understat.com/match/{match_id}"
            response = await self.client.get(url)
            response.raise_for_status()

            # Understat embeds data in JavaScript
            match = re.search(r"var shotsData\s*=\s*JSON\.parse\('(.+?)'\)", response.text)
            if not match:
                return None

            # Decode the JSON data
            json_str = match.group(1).encode().decode('unicode_escape')
            shots_data = json.loads(json_str)

            # Calculate xG totals
            home_xg = sum(float(s.get('xG', 0)) for s in shots_data.get('h', []))
            away_xg = sum(float(s.get('xG', 0)) for s in shots_data.get('a', []))

            return {
                'home_xg': round(home_xg, 2),
                'away_xg': round(away_xg, 2),
                'home_shots': len(shots_data.get('h', [])),
                'away_shots': len(shots_data.get('a', [])),
                'shots': shots_data
            }

        except Exception as e:
            print(f"Error scraping Understat: {e}")
            return None

    async def scrape_understat_league(
        self,
        league: str,
        season: str
    ) -> List[Dict]:
        """
        Scrape all matches from an Understat league/season.

        Leagues: EPL, La_Liga, Bundesliga, Serie_A, Ligue_1, RFPL
        Season format: "2023" for 2023/24
        """
        try:
            await asyncio.sleep(self.rate_limit_delay)
            url = f"https://understat.com/league/{league}/{season}"
            response = await self.client.get(url)
            response.raise_for_status()

            # Extract dates data
            match = re.search(r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)", response.text)
            if not match:
                return []

            json_str = match.group(1).encode().decode('unicode_escape')
            dates_data = json.loads(json_str)

            matches = []
            for match_data in dates_data:
                matches.append({
                    'match_id': match_data.get('id'),
                    'home_team': match_data.get('h', {}).get('title'),
                    'away_team': match_data.get('a', {}).get('title'),
                    'home_score': int(match_data.get('goals', {}).get('h', 0)),
                    'away_score': int(match_data.get('goals', {}).get('a', 0)),
                    'home_xg': float(match_data.get('xG', {}).get('h', 0)),
                    'away_xg': float(match_data.get('xG', {}).get('a', 0)),
                    'date': match_data.get('datetime', '').split(' ')[0],
                })

            return matches

        except Exception as e:
            print(f"Error scraping Understat league: {e}")
            return []

    # ==================== Football-Data.co.uk (historical CSV) ====================

    async def scrape_football_data_csv(
        self,
        league: str,
        season: str
    ) -> List[Dict]:
        """
        Scrape historical match data from Football-Data.co.uk.

        Leagues: E0 (EPL), E1 (Championship), SP1 (La Liga), D1 (Bundesliga), etc.
        Season format: "2324" for 2023/24
        """
        try:
            await asyncio.sleep(self.rate_limit_delay)
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            response = await self.client.get(url)
            response.raise_for_status()

            import csv
            import io

            reader = csv.DictReader(io.StringIO(response.text))
            matches = []

            for row in reader:
                try:
                    matches.append({
                        'home_team': row.get('HomeTeam', ''),
                        'away_team': row.get('AwayTeam', ''),
                        'home_score': int(row.get('FTHG', 0)),
                        'away_score': int(row.get('FTAG', 0)),
                        'date': row.get('Date', ''),
                        'home_shots': int(row.get('HS', 0)) if row.get('HS') else None,
                        'away_shots': int(row.get('AS', 0)) if row.get('AS') else None,
                        'home_shots_on_target': int(row.get('HST', 0)) if row.get('HST') else None,
                        'away_shots_on_target': int(row.get('AST', 0)) if row.get('AST') else None,
                        'home_corners': int(row.get('HC', 0)) if row.get('HC') else None,
                        'away_corners': int(row.get('AC', 0)) if row.get('AC') else None,
                        'home_fouls': int(row.get('HF', 0)) if row.get('HF') else None,
                        'away_fouls': int(row.get('AF', 0)) if row.get('AF') else None,
                        'home_yellow': int(row.get('HY', 0)) if row.get('HY') else None,
                        'away_yellow': int(row.get('AY', 0)) if row.get('AY') else None,
                        'home_red': int(row.get('HR', 0)) if row.get('HR') else None,
                        'away_red': int(row.get('AR', 0)) if row.get('AR') else None,
                    })
                except (ValueError, TypeError):
                    continue

            return matches

        except Exception as e:
            print(f"Error scraping Football-Data: {e}")
            return []

    # ==================== API-Football (requires API key) ====================

    async def fetch_api_football(
        self,
        endpoint: str,
        api_key: str,
        params: Dict = None
    ) -> Optional[Dict]:
        """
        Fetch data from API-Football.

        Get free API key at: https://www.api-football.com/
        Free tier: 100 requests/day
        """
        try:
            await asyncio.sleep(self.rate_limit_delay)

            headers = {
                "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
                "x-rapidapi-key": api_key
            }

            url = f"https://api-football-v1.p.rapidapi.com/v3/{endpoint}"
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print(f"Error fetching API-Football: {e}")
            return None

    async def fetch_league_fixtures(
        self,
        api_key: str,
        league_id: int,
        season: int
    ) -> List[Dict]:
        """
        Fetch all fixtures for a league season from API-Football.

        League IDs: 39 (EPL), 140 (La Liga), 78 (Bundesliga), etc.
        """
        data = await self.fetch_api_football(
            "fixtures",
            api_key,
            {"league": league_id, "season": season}
        )

        if not data or 'response' not in data:
            return []

        matches = []
        for fixture in data['response']:
            teams = fixture.get('teams', {})
            goals = fixture.get('goals', {})

            matches.append({
                'fixture_id': fixture.get('fixture', {}).get('id'),
                'home_team': teams.get('home', {}).get('name'),
                'away_team': teams.get('away', {}).get('name'),
                'home_score': goals.get('home'),
                'away_score': goals.get('away'),
                'date': fixture.get('fixture', {}).get('date', '').split('T')[0],
                'venue': fixture.get('fixture', {}).get('venue', {}).get('name'),
            })

        return matches

    # ==================== Batch Import to Training Data ====================

    async def import_to_training_data(
        self,
        matches: List[Dict],
        competition: str = ""
    ) -> int:
        """Import scraped matches to training dataset."""
        count = 0

        for match_data in matches:
            try:
                match = MatchAnnotation(
                    match_id="",  # Will be auto-generated
                    video_path="",
                    home_team=match_data.get('home_team', ''),
                    away_team=match_data.get('away_team', ''),
                    final_score={
                        'home': match_data.get('home_score', 0),
                        'away': match_data.get('away_score', 0)
                    },
                    date=match_data.get('date', ''),
                    competition=competition or match_data.get('competition', ''),
                    venue=match_data.get('venue'),
                    home_possession=match_data.get('home_possession'),
                    away_possession=match_data.get('away_possession'),
                    home_shots=match_data.get('home_shots'),
                    away_shots=match_data.get('away_shots'),
                    home_shots_on_target=match_data.get('home_shots_on_target'),
                    away_shots_on_target=match_data.get('away_shots_on_target'),
                    home_corners=match_data.get('home_corners'),
                    away_corners=match_data.get('away_corners'),
                    home_fouls=match_data.get('home_fouls'),
                    away_fouls=match_data.get('away_fouls'),
                    home_xg=match_data.get('home_xg'),
                    away_xg=match_data.get('away_xg')
                )

                await training_data_service.add_match(match)
                count += 1

            except Exception as e:
                print(f"Error importing match: {e}")
                continue

        return count

    # ==================== Helper Methods ====================

    def _parse_int(self, value: str) -> int:
        """Safely parse integer from string."""
        try:
            return int(value.strip())
        except (ValueError, AttributeError):
            return 0

    def _parse_float(self, value: str) -> float:
        """Safely parse float from string."""
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return 0.0


# Global scraper instance
scraper = FootballDataScraper()
