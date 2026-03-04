import { create } from 'zustand';
import { api } from '../lib/api';
import type { AICoachData, ChatMessage, TrainingFocusData, TacticalAlert } from '../types/coaching';

interface CoachingState {
  // AI Coach
  coachData: AICoachData | null;
  coachLoading: boolean;
  coachError: string | null;

  // Chat
  chatMessages: ChatMessage[];
  chatLoading: boolean;

  // Training Focus
  trainingFocus: TrainingFocusData | null;
  trainingLoading: boolean;
  trainingError: string | null;

  // Tactical Alerts
  tacticalAlerts: TacticalAlert[];
  alertsLoading: boolean;
  alertsError: string | null;

  // Actions
  loadCoachingData: () => Promise<void>;
  loadTrainingFocus: () => Promise<void>;
  loadTacticalAlerts: () => Promise<void>;
  sendChat: (question: string) => Promise<void>;
  resetChat: () => void;
}

const INITIAL_CHAT: ChatMessage = {
  id: '1',
  role: 'assistant',
  content: "Hi! I'm your AI coaching assistant. Ask me anything about the match — possession, formations, player performance, what to work on, or any tactical questions!",
  timestamp: new Date(),
};

export const useCoachingStore = create<CoachingState>((set) => ({
  coachData: null,
  coachLoading: false,
  coachError: null,

  chatMessages: [INITIAL_CHAT],
  chatLoading: false,

  trainingFocus: null,
  trainingLoading: false,
  trainingError: null,

  tacticalAlerts: [],
  alertsLoading: false,
  alertsError: null,

  loadCoachingData: async () => {
    set({ coachLoading: true, coachError: null });
    try {
      const data = await api.get<{ coaching: AICoachData }>('/api/ai-coach/analysis');
      set({ coachData: data.coaching, coachLoading: false });
    } catch {
      set({ coachError: 'Could not load AI coaching analysis', coachLoading: false });
    }
  },

  loadTrainingFocus: async () => {
    set({ trainingLoading: true, trainingError: null });
    try {
      const data = await api.get<TrainingFocusData>('/api/ai-coach/training-focus');
      set({ trainingFocus: data, trainingLoading: false });
    } catch {
      set({ trainingError: 'Could not load training focus data', trainingLoading: false });
    }
  },

  loadTacticalAlerts: async () => {
    set({ alertsLoading: true, alertsError: null });
    try {
      const data = await api.get<{ alerts: TacticalAlert[] }>('/api/ai-coach/tactical-alerts');
      set({ tacticalAlerts: data.alerts || [], alertsLoading: false });
    } catch {
      set({ alertsError: 'Could not load tactical alerts', alertsLoading: false });
    }
  },

  sendChat: async (question: string) => {
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };
    set(s => ({ chatMessages: [...s.chatMessages, userMsg], chatLoading: true }));

    try {
      const data = await api.post<{ answer: string; confidence?: string; related_insights?: string[] }>(
        '/api/ai-coach/chat',
        { question }
      );
      const assistantMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        confidence: data.confidence,
        relatedInsights: data.related_insights,
      };
      set(s => ({ chatMessages: [...s.chatMessages, assistantMsg], chatLoading: false }));
    } catch {
      const errMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "Sorry, I couldn't process your question. Please try again.",
        timestamp: new Date(),
      };
      set(s => ({ chatMessages: [...s.chatMessages, errMsg], chatLoading: false }));
    }
  },

  resetChat: () => set({ chatMessages: [INITIAL_CHAT] }),
}));
