import { useEffect, useState, useRef } from 'react';
import { useCoachingStore } from '../../stores/coachingStore';
import { Card, CardTitle } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { Modal } from '../../components/ui/Modal';
import { Spinner } from '../../components/ui/Spinner';
import { EmptyState } from '../../components/ui/EmptyState';
import { cn } from '../../lib/utils';
import { RATING_COLORS } from '../../lib/constants';
import {
  Brain, AlertTriangle, MessageSquare, Send, Megaphone, ClipboardList, X
} from 'lucide-react';

export default function CoachingPage() {
  const {
    coachData, coachLoading, coachError, loadCoachingData,
    chatMessages, chatLoading, sendChat,
  } = useCoachingStore();

  const [selectedCategory, setSelectedCategory] = useState('all');
  const [showTeamTalk, setShowTeamTalk] = useState<'half_time' | 'full_time' | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { loadCoachingData(); }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  const handleSend = () => {
    if (!chatInput.trim() || chatLoading) return;
    sendChat(chatInput.trim());
    setChatInput('');
  };

  if (coachLoading) return <Spinner label="AI Coach is analyzing the match..." className="py-20" />;
  if (coachError || !coachData) {
    return (
      <EmptyState
        icon={<Brain className="w-8 h-8 text-slate-500" />}
        title="AI Coach Unavailable"
        description={coachError || 'Could not load coaching analysis'}
        action={<Button variant="secondary" size="sm" onClick={loadCoachingData}>Retry</Button>}
      />
    );
  }

  const categories = ['all', 'tactical', 'pressing', 'possession', 'defensive', 'attacking', 'formation'];
  const filtered = selectedCategory === 'all' ? coachData.insights : coachData.insights.filter(i => i.category === selectedCategory);

  const suggestedQuestions = [
    "How was our possession?", "Who was our best player?", "What should we work on?",
    "How did our pressing look?", "Summarize the match",
  ];

  return (
    <div className="space-y-6">
      {/* Summary */}
      {coachData.summary && (
        <div className="bg-gradient-to-r from-emerald-500/10 via-cyan-500/10 to-emerald-500/10 rounded-2xl p-6 border border-emerald-500/20">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-14 h-14 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">AI Coaching Expert</h2>
                <p className="text-slate-400 text-sm">Tactical analysis and recommendations</p>
              </div>
            </div>
            <div className="text-right">
              <span className="text-slate-400 text-xs block mb-1">Overall Rating</span>
              <span className={cn('text-2xl font-bold', RATING_COLORS[coachData.summary.overall_rating] || 'text-white')}>
                {coachData.summary.overall_rating}
              </span>
            </div>
          </div>

          <p className="text-slate-300 mb-4">{coachData.summary.tactical_summary}</p>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-emerald-500/10 rounded-xl p-4 border border-emerald-500/20">
              <h4 className="text-emerald-400 font-semibold text-sm mb-2">Key Strengths</h4>
              <ul className="space-y-1">
                {coachData.summary.key_strengths.map((s, i) => (
                  <li key={i} className="text-slate-300 text-sm flex items-start gap-2">
                    <span className="text-emerald-500">&#x2022;</span> {s}
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-amber-500/10 rounded-xl p-4 border border-amber-500/20">
              <h4 className="text-amber-400 font-semibold text-sm mb-2">Areas to Improve</h4>
              <ul className="space-y-1">
                {coachData.summary.areas_to_improve.map((a, i) => (
                  <li key={i} className="text-slate-300 text-sm flex items-start gap-2">
                    <span className="text-amber-500">&#x2022;</span> {a}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="flex gap-3 mt-4">
            <Button variant="secondary" className="flex-1" onClick={() => setShowTeamTalk('half_time')}>
              <Megaphone className="w-4 h-4" /> Half-Time Talk
            </Button>
            <Button variant="secondary" className="flex-1" onClick={() => setShowTeamTalk('full_time')}>
              <ClipboardList className="w-4 h-4" /> Full-Time Review
            </Button>
          </div>
        </div>
      )}

      {/* Team Talk Modal */}
      <Modal open={!!showTeamTalk} onClose={() => setShowTeamTalk(null)}
        title={showTeamTalk === 'half_time' ? 'Half-Time Team Talk' : 'Full-Time Review'}>
        {coachData.summary && (
          <>
            <div className="bg-slate-800/50 rounded-xl p-5 mb-4">
              <p className="text-white text-lg leading-relaxed">
                {showTeamTalk === 'half_time' ? coachData.summary.half_time_message : coachData.summary.full_time_message}
              </p>
            </div>
            <div className="space-y-3">
              <h4 className="text-slate-400 font-semibold text-sm">Key Points:</h4>
              {coachData.critical_insights.slice(0, 3).map((insight, i) => (
                <div key={i} className="rounded-lg p-3 border bg-slate-800/50 border-slate-700/30">
                  <div className="font-semibold text-white text-sm mb-1">{insight.title}</div>
                  <div className="text-sm text-slate-400">{insight.recommendation}</div>
                </div>
              ))}
            </div>
            <Button className="w-full mt-4" onClick={() => setShowTeamTalk(null)}>Got It</Button>
          </>
        )}
      </Modal>

      {/* Critical Alerts */}
      {coachData.critical_insights.length > 0 && (
        <Card className="border-red-500/30 bg-red-500/5">
          <CardTitle className="text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" /> Urgent Attention ({coachData.critical_insights.length})
          </CardTitle>
          <div className="grid gap-3 mt-3">
            {coachData.critical_insights.slice(0, 3).map((insight, i) => (
              <div key={i} className="bg-red-500/10 rounded-xl p-4 border border-red-500/20">
                <h4 className="text-white font-semibold">{insight.title}</h4>
                <p className="text-slate-300 text-sm mt-1">{insight.message}</p>
                <div className="mt-2 p-2 bg-slate-800/50 rounded-lg">
                  <span className="text-emerald-400 text-sm font-medium">Recommendation: </span>
                  <span className="text-slate-300 text-sm">{insight.recommendation}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Category Filter */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-slate-400 text-sm mr-2">Filter:</span>
        {categories.map(cat => (
          <button key={cat} onClick={() => setSelectedCategory(cat)}
            className={cn('px-3 py-1.5 rounded-lg text-sm capitalize transition-all',
              selectedCategory === cat ? 'bg-emerald-500 text-white' : 'bg-slate-800 text-slate-400 hover:text-white'
            )}>
            {cat}
          </button>
        ))}
        <span className="text-slate-500 text-xs ml-2">({filtered.length} insights)</span>
      </div>

      {/* Insights Grid */}
      <div className="grid gap-4">
        {filtered.map((insight, i) => {
          return (
            <Card key={i}>
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Badge variant={insight.priority as any}>{insight.priority}</Badge>
                  <Badge variant="default">{insight.category}</Badge>
                </div>
              </div>
              <h4 className="text-white font-semibold mb-1">{insight.title}</h4>
              <p className="text-slate-300 text-sm mb-3">{insight.message}</p>
              <div className="bg-slate-800/50 rounded-lg p-3">
                <span className="text-emerald-400 text-sm font-medium">Recommendation: </span>
                <span className="text-slate-300 text-sm">{insight.recommendation}</span>
              </div>
            </Card>
          );
        })}
        {filtered.length === 0 && (
          <EmptyState title="No Insights Found" description="Try selecting a different category filter" />
        )}
      </div>

      {/* Stats Summary */}
      <Card>
        <CardTitle>Analysis Summary</CardTitle>
        <div className="grid grid-cols-5 gap-4 mt-3">
          {[
            { label: 'Total', value: coachData.total_insights, color: '' },
            { label: 'Critical', value: coachData.insights.filter(i => i.priority === 'critical').length, color: 'text-red-400 bg-red-500/10 border-red-500/20' },
            { label: 'High', value: coachData.insights.filter(i => i.priority === 'high').length, color: 'text-orange-400 bg-orange-500/10 border-orange-500/20' },
            { label: 'Medium', value: coachData.insights.filter(i => i.priority === 'medium').length, color: 'text-amber-400 bg-amber-500/10 border-amber-500/20' },
            { label: 'Low/Info', value: coachData.insights.filter(i => i.priority === 'low' || i.priority === 'info').length, color: 'text-slate-400 bg-slate-500/10 border-slate-500/20' },
          ].map((s, i) => (
            <div key={i} className={cn('text-center p-3 rounded-xl border', s.color || 'bg-slate-800/50 border-slate-700/30')}>
              <div className={cn('text-2xl font-bold', s.color ? '' : 'text-white')}>{s.value}</div>
              <div className="text-slate-400 text-xs">{s.label}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Floating Chat Button */}
      <button onClick={() => setShowChat(true)}
        className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full shadow-lg hover:shadow-emerald-500/30 transition-all hover:scale-110 flex items-center justify-center z-40">
        <MessageSquare className="w-6 h-6 text-white" />
      </button>

      {/* Chat Panel */}
      {showChat && (
        <div className="fixed bottom-6 right-6 w-96 h-[600px] bg-background rounded-2xl shadow-2xl border border-emerald-500/30 flex flex-col z-50 overflow-hidden">
          <div className="bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 p-4 border-b border-slate-700/50 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="text-white font-semibold">AI Coach Chat</h3>
                <p className="text-slate-400 text-xs">Ask about the match</p>
              </div>
            </div>
            <button onClick={() => setShowChat(false)} className="text-slate-400 hover:text-white p-1">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {chatMessages.map(msg => (
              <div key={msg.id} className={cn('flex', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
                <div className={cn('max-w-[85%] rounded-2xl px-4 py-3',
                  msg.role === 'user'
                    ? 'bg-emerald-500 text-white rounded-br-md'
                    : 'bg-slate-800 text-slate-200 rounded-bl-md border border-slate-700/50'
                )}>
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  {msg.confidence && (
                    <span className={cn('text-xs mt-2 block',
                      msg.confidence === 'high' ? 'text-emerald-400' : msg.confidence === 'medium' ? 'text-amber-400' : 'text-slate-400'
                    )}>Confidence: {msg.confidence}</span>
                  )}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-800 rounded-2xl rounded-bl-md px-4 py-3 border border-slate-700/50">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {chatMessages.length <= 1 && (
            <div className="px-4 pb-2">
              <p className="text-slate-500 text-xs mb-2">Suggested:</p>
              <div className="flex flex-wrap gap-2">
                {suggestedQuestions.map((q, i) => (
                  <button key={i} onClick={() => setChatInput(q)}
                    className="text-xs px-3 py-1.5 bg-slate-800 text-emerald-400 rounded-full hover:bg-slate-700 border border-slate-700/50">
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="p-4 border-t border-slate-700/50 bg-slate-900/50">
            <div className="flex gap-2">
              <input type="text" value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSend()}
                placeholder="Ask about the match..."
                className="flex-1 bg-slate-800 text-white placeholder-slate-500 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50 border border-slate-700/50" />
              <button onClick={handleSend} disabled={!chatInput.trim() || chatLoading}
                className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center disabled:opacity-50">
                <Send className="w-5 h-5 text-white" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
