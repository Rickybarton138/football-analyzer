import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import { Upload, Loader2, CheckCircle, Link } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002/api';

type Mode = 'file' | 'url';

export default function MatchUpload() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>('url');
  const [step, setStep] = useState<'details' | 'uploading' | 'processing'>('details');
  const [title, setTitle] = useState('');
  const [opponent, setOpponent] = useState('');
  const [formation, setFormation] = useState('');
  const [notes, setNotes] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');

  const handleUrlUpload = useCallback(async () => {
    if (!videoUrl || !title) return;
    setError('');

    // Ensure URL has protocol
    let url = videoUrl.trim();
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'https://' + url;
    }
    if (!url.includes('.') || url.length < 10) {
      setError('Please enter a valid video URL');
      return;
    }

    try {
      setStep('processing');
      const match: any = await api.uploadFromUrl({
        video_url: url, title, opponent, formation, notes,
      });
      setTimeout(() => navigate(`/match/${match.id}`), 1500);
    } catch (err: any) {
      setError(err.message || 'Something went wrong');
      setStep('details');
    }
  }, [videoUrl, title, opponent, formation, notes, navigate]);

  const handleFileUpload = useCallback(async () => {
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
          {/* Mode toggle */}
          <div className="flex bg-zinc-900 border border-zinc-800 rounded-lg p-1">
            <button
              onClick={() => setMode('url')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'url' ? 'bg-emerald-500/20 text-emerald-400' : 'text-zinc-400 hover:text-zinc-200'}`}
            >
              <Link className="w-4 h-4" /> VEO / Video URL
            </button>
            <button
              onClick={() => setMode('file')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'file' ? 'bg-emerald-500/20 text-emerald-400' : 'text-zinc-400 hover:text-zinc-200'}`}
            >
              <Upload className="w-4 h-4" /> File Upload
            </button>
          </div>

          {/* URL input */}
          {mode === 'url' && (
            <div>
              <label className="block text-sm font-medium text-zinc-400 mb-1">Video Download URL *</label>
              <input
                type="url"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                placeholder="Paste VEO download link or direct video URL..."
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
              />
              <p className="text-xs text-zinc-600 mt-1">VEO: Share → Download → copy the download URL. Fastest option — no upload needed.</p>
            </div>
          )}

          {/* File input */}
          {mode === 'file' && (
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
          )}

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
            onClick={mode === 'url' ? handleUrlUpload : handleFileUpload}
            disabled={mode === 'url' ? (!videoUrl || !title) : (!file || !title)}
            className="w-full bg-emerald-500 hover:bg-emerald-600 disabled:bg-zinc-700 disabled:text-zinc-500 text-white py-3 rounded-lg font-medium transition-colors"
          >
            {mode === 'url' ? 'Import & Analyse' : 'Upload & Analyse'}
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
