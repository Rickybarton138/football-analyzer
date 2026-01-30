import { useState, useEffect, useRef, useCallback } from 'react';
import type { FrameDetection, TacticalAlert, MatchState } from '../types';

interface WebSocketCallbacks {
  onFrame?: (frame: FrameDetection) => void;
  onAlert?: (alert: TacticalAlert) => void;
  onMatchState?: (state: MatchState) => void;
  onError?: (error: Error) => void;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  sendMessage: (data: unknown) => void;
  lastMessage: unknown | null;
}

export function useWebSocket(
  url: string | null,
  callbacks: WebSocketCallbacks = {}
): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<unknown | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const heartbeatIntervalRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (!url) return;

    const wsUrl = url.startsWith('ws') ? url : `ws://${window.location.host}${url}`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);

        // Start heartbeat
        heartbeatIntervalRef.current = window.setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);

          // Route message to appropriate callback
          switch (data.type) {
            case 'frame':
              callbacks.onFrame?.(data.data as FrameDetection);
              break;
            case 'alert':
              callbacks.onAlert?.(data.data as TacticalAlert);
              break;
            case 'match_state':
              callbacks.onMatchState?.(data.data as MatchState);
              break;
            case 'pong':
              // Heartbeat response, ignore
              break;
            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        wsRef.current = null;

        // Clear heartbeat
        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
        }

        // Attempt reconnect after 3 seconds
        reconnectTimeoutRef.current = window.setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        callbacks.onError?.(new Error('WebSocket error'));
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      callbacks.onError?.(e as Error);
    }
  }, [url, callbacks]);

  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
    };
  }, [connect]);

  const sendMessage = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  return {
    isConnected,
    sendMessage,
    lastMessage,
  };
}

// Hook for sending frames to live processing
export function useLiveStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const startStream = useCallback(() => {
    const ws = new WebSocket(`ws://${window.location.host}/ws/live`);

    ws.onopen = () => {
      console.log('Live stream connected');
      setIsStreaming(true);
    };

    ws.onclose = () => {
      console.log('Live stream disconnected');
      setIsStreaming(false);
    };

    wsRef.current = ws;
  }, []);

  const stopStream = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const sendFrame = useCallback((frameData: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(frameData);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    isStreaming,
    startStream,
    stopStream,
    sendFrame,
  };
}
