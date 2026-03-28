import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Loader2, CheckCircle } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002/api';

export default function MatchUpload() {
  const navigate = useNavigate();
  const [step, setStep] = useState<'details' | 'uploading' | 'processing'>('details');
  const [title, setTitle] = useState('');
  const [opponent, setOpponent] = useState('');
  const [formation, setFormation] = useState('');
  const [notes, setNotes] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');

  const handleUpload = useCallback(async () => {
    if (!file || !title) return;
    setError('');

    try {
      setStep('uploading');

      const formData = new FormData();
      formData.append('video', file);
      formData.append('title', title);
      formData.append('opponent', opponent);
      formData.append('formation', formation);
      formData.append('notes', notes);

      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 100));
      };

      const match: any = await new Promise((resolve, reject) => {
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            let msg = 'Upload failed';
            try { msg = JSON.parse(xhr.responseText).detail || msg; } catch {}
            reject(new Error(msg));
          }
        };
        xhr.onerror = () => reject(new Error('Upload failed — check your connection'));
        xhr.open('POST', `${API_BASE}/matches/upload`);
        xhr.send(formData);
      });

      setStep('processing');
      setTimeout(() => navigate(`/match/${match.id}`), 2000);

    } catch (err: any) {
      setError(err.message || 'Something went wrong');
      setStep('details');
    }
  }, [file, title, opponent, formation, notes, navigate]);

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Upload Match</h1>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      {step === 'details' && (
        <div className="space-y-6">
          <label className="block border-2 border-dashed border-zinc-700 rounded-xl p-8 text-center cursor-pointer hover:border-emerald-500/50 transition-colors">
            <input
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            {file ? (
              <div>
                <CheckCircle className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                <p className="font-medium text-emerald-400">{file.name}</p>
                <p className="text-sm text-zinc-500 mt-1">{(file.size / 1024 / 1024).toFixed(0)} MB</p>
              </div>
            ) : (
              <div>
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
                <p className="text-zinc-400">Drop match video here or click to browse</p>
                <p className="text-sm text-zinc-600 mt-1">MP4, MOV, WebM up to 2GB</p>
              </div>
            )}
          </label>

          <div className="grid grid-cols-2 gap-4">
            <div className="col-span-2">
              <label className="block text-sm font-medium text-zinc-400 mb-1">Match Title *</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. vs Parkstone FC - League Match"
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-400 mb-1">Opponent</label>
              <input
                type="text"
                value={opponent}
                onChange={(e) => setOpponent(e.target.value)}
                placeholder="e.g. Parkstone FC"
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-zinc-400 mb-1">Formation</label>
              <input
                type="text"
                value={formation}
                onChange={(e) => setFormation(e.target.value)}
                placeholder="e.g. 4-1-3-2"
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
              />
            </div>
            <div className="col-span-2">
              <label className="block text-sm font-medium text-zinc-400 mb-1">Coach Notes</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
                placeholder="Anything you want the AI to focus on..."
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none resize-none"
              />
            </div>
          </div>

          <button
            onClick={handleUpload}
            disabled={!file || !title}
            className="w-full bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 disabled:text-zinc-500 text-white py-3 rounded-lg font-medium transition-colors"
          >
            Upload & Analyse
          </button>
        </div>
      )}

      {step === 'uploading' && (
        <div className="text-center py-12">
          <Loader2 className="w-12 h-12 text-emerald-400 animate-spin mx-auto mb-4" />
          <h3 className="text-xl font-medium">Uploading video...</h3>
          <div className="mt-4 w-full bg-zinc-800 rounded-full h-2">
            <div
              className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-zinc-400 mt-2">{progress}%</p>
        </div>
      )}

      {step === 'processing' && (
        <div className="text-center py-12">
          <Loader2 className="w-12 h-12 text-blue-400 animate-spin mx-auto mb-4" />
          <h3 className="text-xl font-medium">Processing & Indexing...</h3>
          <p className="text-zinc-400 mt-2">TwelveLabs is analysing your footage. This may take a few minutes.</p>
        </div>
      )}
    </div>
  );
}
