import { useState, useRef, useEffect } from 'react';
import { api } from '../lib/api';
import { Send, Loader2, MessageSquare, X, Play, Bot } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  clips?: Clip[];
  tools_used?: string[];
}

interface Clip {
  clip_url: string;
  thumbnail_url: string;
  start: number;
  end: number;
  description?: string;
}

interface CoachChatProps {
  matchId: string;
  isOpen: boolean;
  onClose: () => void;
  onSeekTo?: (time: number) => void;
  onPlayClip?: (start: number, end: number) => void;
}

export default function CoachChat({ matchId, isOpen, onClose, onSeekTo, onPlayClip }: CoachChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-open with a greeting on first open
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([{
        role: 'assistant',
        content: "Alright, I've had a good look at this match. What do you want to talk about? I can break down the tactics, look at individual performances, find specific moments, or help you plan your next training session. Fire away.",
      }]);
    }
  }, [isOpen]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    try {
      const chatMessages = newMessages
        .filter(m => !(m.role === 'assistant' && m === messages[0] && messages.length > 1 && newMessages.indexOf(m) === 0))
        .map(m => ({ role: m.role, content: m.content }));

      const result = await api.chat({
        match_id: matchId,
        messages: chatMessages,
      });

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: result.response,
        clips: result.clips,
        tools_used: result.tools_used,
      }]);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Sorry, something went wrong there. ${err.message || 'Try again.'}`,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-full sm:w-[420px] bg-zinc-950 border-l border-zinc-800 flex flex-col z-50 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-emerald-400" />
          <h2 className="font-semibold text-zinc-100">Manager Mentor</h2>
        </div>
        <button onClick={onClose} className="text-zinc-400 hover:text-zinc-200">
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] ${
              msg.role === 'user'
                ? 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-50'
                : 'bg-zinc-900 border border-zinc-800 text-zinc-200'
            } rounded-xl px-4 py-3`}>
              <div className="text-sm whitespace-pre-wrap leading-relaxed">
                {msg.content}
              </div>

              {/* Clips */}
              {msg.clips && msg.clips.length > 0 && (
                <div className="mt-3 space-y-2">
                  {msg.clips.map((clip, j) => (
                    <button
                      key={j}
                      onClick={() => onPlayClip ? onPlayClip(clip.start, clip.end) : onSeekTo?.(clip.start)}
                      className="w-full flex items-center gap-3 bg-zinc-800/50 hover:bg-zinc-800 border border-zinc-700 rounded-lg p-2 transition-colors text-left"
                    >
                      <div className="w-16 h-10 bg-zinc-700 rounded overflow-hidden flex-shrink-0">
                        {clip.thumbnail_url && (
                          <img src={clip.thumbnail_url} alt="" className="w-full h-full object-cover" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-zinc-300 truncate">{clip.description || 'Match moment'}</p>
                        <p className="text-xs text-zinc-500">{formatTime(clip.start)} - {formatTime(clip.end)}</p>
                      </div>
                      <Play className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                    </button>
                  ))}
                </div>
              )}

              {/* Tools used indicator */}
              {msg.tools_used && msg.tools_used.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {[...new Set(msg.tools_used)].map((tool, j) => (
                    <span key={j} className="text-[10px] bg-zinc-800 text-zinc-500 px-1.5 py-0.5 rounded">
                      {tool.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3">
              <div className="flex items-center gap-2 text-sm text-zinc-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                Analysing...
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-zinc-800 p-3">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about the match..."
            disabled={isLoading}
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 text-white px-3 rounded-lg transition-colors"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
        <p className="text-[10px] text-zinc-600 mt-1.5 px-1">
          Manager Mentor can search video, analyse moments, and design training drills
        </p>
      </div>
    </div>
  );
}
