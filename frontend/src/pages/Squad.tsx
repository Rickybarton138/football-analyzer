import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api';
import { UserPlus, Users, Loader2 } from 'lucide-react';

export default function Squad() {
  const queryClient = useQueryClient();
  const [name, setName] = useState('');
  const [number, setNumber] = useState('');
  const [position, setPosition] = useState('');

  const { data: players, isLoading } = useQuery({
    queryKey: ['players'],
    queryFn: api.listPlayers,
  });

  const addPlayer = useMutation({
    mutationFn: () =>
      api.createPlayer({
        name,
        squad_number: number ? parseInt(number) : undefined,
        position: position || undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['players'] });
      setName('');
      setNumber('');
      setPosition('');
    },
  });

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">Squad</h1>
      <p className="text-zinc-400 mb-8">Register players for AI recognition in match footage</p>

      {/* Add Player Form */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 mb-8">
        <h2 className="font-semibold text-zinc-200 mb-4 flex items-center gap-2">
          <UserPlus className="w-5 h-5" />
          Add Player
        </h2>
        <div className="grid grid-cols-3 gap-3">
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Player name"
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
          />
          <input
            type="number"
            value={number}
            onChange={(e) => setNumber(e.target.value)}
            placeholder="Squad #"
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
          />
          <input
            type="text"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            placeholder="Position (e.g. CB, CM)"
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <button
          onClick={() => name && addPlayer.mutate()}
          disabled={!name || addPlayer.isPending}
          className="mt-3 bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
        >
          {addPlayer.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Add Player'}
        </button>
      </div>

      {/* Player List */}
      {isLoading ? (
        <Loader2 className="w-6 h-6 text-emerald-400 animate-spin mx-auto" />
      ) : !(players as any[])?.length ? (
        <div className="text-center py-12 border border-dashed border-zinc-700 rounded-xl">
          <Users className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
          <p className="text-zinc-500">No players registered yet</p>
        </div>
      ) : (
        <div className="space-y-2">
          {(players as any[]).map((player) => (
            <div
              key={player.id}
              className="bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-3 flex items-center justify-between"
            >
              <div className="flex items-center gap-4">
                <span className="w-8 h-8 rounded-full bg-emerald-500/10 text-emerald-400 flex items-center justify-center text-sm font-bold">
                  {player.squad_number || '?'}
                </span>
                <div>
                  <p className="font-medium text-zinc-100">{player.name}</p>
                  {player.position && <p className="text-xs text-zinc-500">{player.position}</p>}
                </div>
              </div>
              <span className={`text-xs px-2 py-1 rounded ${
                player.twelvelabs_entity_id
                  ? 'bg-emerald-500/10 text-emerald-400'
                  : 'bg-zinc-800 text-zinc-500'
              }`}>
                {player.twelvelabs_entity_id ? 'AI Ready' : 'No face data'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
