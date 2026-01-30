import { useMemo } from 'react';
import type { DetectedPlayer, DetectedBall, TacticalAlert } from '../types';

interface TacticalBoardProps {
  players: DetectedPlayer[];
  ball?: DetectedBall;
  alerts: TacticalAlert[];
}

// Standard pitch dimensions (105m x 68m)
const PITCH_LENGTH = 105;
const PITCH_WIDTH = 68;
const SCALE = 6; // Scale factor for display

export function TacticalBoard({ players, ball, alerts }: TacticalBoardProps) {
  const boardWidth = PITCH_LENGTH * SCALE;
  const boardHeight = PITCH_WIDTH * SCALE;

  // Convert pitch coordinates to SVG coordinates
  const toSvg = (x: number, y: number) => ({
    x: x * SCALE,
    y: (PITCH_WIDTH - y) * SCALE, // Flip Y axis
  });

  // Group players by team
  const { homePlayers, awayPlayers, unknownPlayers } = useMemo(() => {
    const home: DetectedPlayer[] = [];
    const away: DetectedPlayer[] = [];
    const unknown: DetectedPlayer[] = [];

    players.forEach((p) => {
      if (p.team === 'home') home.push(p);
      else if (p.team === 'away') away.push(p);
      else unknown.push(p);
    });

    return { homePlayers: home, awayPlayers: away, unknownPlayers: unknown };
  }, [players]);

  return (
    <div className="w-full h-full flex items-center justify-center">
      <svg
        viewBox={`0 0 ${boardWidth} ${boardHeight}`}
        className="w-full h-full max-h-[400px]"
        style={{ backgroundColor: '#2d5a27' }}
      >
        {/* Pitch markings */}
        <PitchMarkings width={boardWidth} height={boardHeight} />

        {/* Alert zones */}
        {alerts.map((alert) => {
          if (!alert.position) return null;
          const pos = toSvg(alert.position.x, alert.position.y);
          return (
            <circle
              key={alert.alert_id}
              cx={pos.x}
              cy={pos.y}
              r={30}
              fill="rgba(239, 68, 68, 0.3)"
              stroke="#ef4444"
              strokeWidth={2}
              className="animate-pulse"
            />
          );
        })}

        {/* Home players (blue) */}
        {homePlayers.map((player) => {
          if (!player.pitch_position) return null;
          const pos = toSvg(player.pitch_position.x, player.pitch_position.y);
          return (
            <PlayerMarker
              key={player.track_id}
              x={pos.x}
              y={pos.y}
              id={player.track_id}
              color="#3b82f6"
              isGoalkeeper={player.is_goalkeeper}
            />
          );
        })}

        {/* Away players (red) */}
        {awayPlayers.map((player) => {
          if (!player.pitch_position) return null;
          const pos = toSvg(player.pitch_position.x, player.pitch_position.y);
          return (
            <PlayerMarker
              key={player.track_id}
              x={pos.x}
              y={pos.y}
              id={player.track_id}
              color="#ef4444"
              isGoalkeeper={player.is_goalkeeper}
            />
          );
        })}

        {/* Unknown players (gray) */}
        {unknownPlayers.map((player) => {
          if (!player.pitch_position) return null;
          const pos = toSvg(player.pitch_position.x, player.pitch_position.y);
          return (
            <PlayerMarker
              key={player.track_id}
              x={pos.x}
              y={pos.y}
              id={player.track_id}
              color="#6b7280"
              isGoalkeeper={player.is_goalkeeper}
            />
          );
        })}

        {/* Ball */}
        {ball?.pitch_position && (
          <BallMarker
            x={toSvg(ball.pitch_position.x, ball.pitch_position.y).x}
            y={toSvg(ball.pitch_position.x, ball.pitch_position.y).y}
            velocity={ball.velocity}
          />
        )}
      </svg>
    </div>
  );
}

function PitchMarkings({ width, height }: { width: number; height: number }) {
  const strokeColor = 'rgba(255, 255, 255, 0.8)';
  const strokeWidth = 2;

  // Scaled dimensions
  const penaltyAreaWidth = 40.32 * SCALE;
  const penaltyAreaDepth = 16.5 * SCALE;
  const goalAreaWidth = 18.32 * SCALE;
  const goalAreaDepth = 5.5 * SCALE;
  const centerCircleRadius = 9.15 * SCALE;
  const penaltySpotDistance = 11 * SCALE;

  return (
    <g stroke={strokeColor} strokeWidth={strokeWidth} fill="none">
      {/* Outer boundary */}
      <rect x={0} y={0} width={width} height={height} />

      {/* Center line */}
      <line x1={width / 2} y1={0} x2={width / 2} y2={height} />

      {/* Center circle */}
      <circle cx={width / 2} cy={height / 2} r={centerCircleRadius} />

      {/* Center spot */}
      <circle cx={width / 2} cy={height / 2} r={3} fill={strokeColor} />

      {/* Left penalty area */}
      <rect
        x={0}
        y={(height - penaltyAreaWidth) / 2}
        width={penaltyAreaDepth}
        height={penaltyAreaWidth}
      />

      {/* Right penalty area */}
      <rect
        x={width - penaltyAreaDepth}
        y={(height - penaltyAreaWidth) / 2}
        width={penaltyAreaDepth}
        height={penaltyAreaWidth}
      />

      {/* Left goal area */}
      <rect
        x={0}
        y={(height - goalAreaWidth) / 2}
        width={goalAreaDepth}
        height={goalAreaWidth}
      />

      {/* Right goal area */}
      <rect
        x={width - goalAreaDepth}
        y={(height - goalAreaWidth) / 2}
        width={goalAreaDepth}
        height={goalAreaWidth}
      />

      {/* Left penalty spot */}
      <circle
        cx={penaltySpotDistance}
        cy={height / 2}
        r={3}
        fill={strokeColor}
      />

      {/* Right penalty spot */}
      <circle
        cx={width - penaltySpotDistance}
        cy={height / 2}
        r={3}
        fill={strokeColor}
      />

      {/* Left goal */}
      <rect
        x={-5}
        y={(height - 7.32 * SCALE) / 2}
        width={5}
        height={7.32 * SCALE}
        fill="rgba(255, 255, 255, 0.3)"
      />

      {/* Right goal */}
      <rect
        x={width}
        y={(height - 7.32 * SCALE) / 2}
        width={5}
        height={7.32 * SCALE}
        fill="rgba(255, 255, 255, 0.3)"
      />
    </g>
  );
}

interface PlayerMarkerProps {
  x: number;
  y: number;
  id: number;
  color: string;
  isGoalkeeper: boolean;
}

function PlayerMarker({ x, y, id, color, isGoalkeeper }: PlayerMarkerProps) {
  return (
    <g className="player-marker">
      {/* Player circle */}
      <circle
        cx={x}
        cy={y}
        r={isGoalkeeper ? 12 : 10}
        fill={color}
        stroke="white"
        strokeWidth={isGoalkeeper ? 3 : 2}
      />
      {/* Player ID */}
      <text
        x={x}
        y={y + 4}
        textAnchor="middle"
        fill="white"
        fontSize={10}
        fontWeight="bold"
      >
        {id}
      </text>
    </g>
  );
}

interface BallMarkerProps {
  x: number;
  y: number;
  velocity?: { vx: number; vy: number; speed_kmh: number };
}

function BallMarker({ x, y, velocity }: BallMarkerProps) {
  return (
    <g>
      {/* Velocity vector */}
      {velocity && velocity.speed_kmh > 5 && (
        <line
          x1={x}
          y1={y}
          x2={x + velocity.vx * 5}
          y2={y - velocity.vy * 5}
          stroke="#f472b6"
          strokeWidth={2}
          markerEnd="url(#arrowhead)"
        />
      )}
      {/* Ball */}
      <circle
        cx={x}
        cy={y}
        r={6}
        fill="#fbbf24"
        stroke="white"
        strokeWidth={2}
      />
      {/* Arrow marker definition */}
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill="#f472b6" />
        </marker>
      </defs>
    </g>
  );
}
