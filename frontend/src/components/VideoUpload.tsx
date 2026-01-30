import { useState, useCallback } from 'react';
import { Upload, FileVideo, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import type { VideoUploadResponse } from '../types';

interface VideoUploadProps {
  onUploaded: (response: VideoUploadResponse) => void;
}

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export function VideoUpload({ onUploaded }: VideoUploadProps) {
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploadResponse, setUploadResponse] = useState<VideoUploadResponse | null>(null);

  const handleFileSelect = useCallback(async (file: File) => {
    // Validate file type
    const validTypes = ['.mp4', '.avi', '.mov', '.mkv'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!validTypes.includes(ext)) {
      setError(`Invalid file type. Supported: ${validTypes.join(', ')}`);
      setStatus('error');
      return;
    }

    // Validate file size (5GB max)
    const maxSize = 5 * 1024 * 1024 * 1024;
    if (file.size > maxSize) {
      setError('File too large. Maximum size is 5GB.');
      setStatus('error');
      return;
    }

    setFileName(file.name);
    setStatus('uploading');
    setError(null);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const xhr = new XMLHttpRequest();

      // Track upload progress
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const pct = Math.round((e.loaded / e.total) * 100);
          setProgress(pct);
        }
      };

      xhr.onload = () => {
        if (xhr.status === 200) {
          const response: VideoUploadResponse = JSON.parse(xhr.responseText);
          setUploadResponse(response);
          setStatus('success');
          onUploaded(response);
        } else {
          const errorData = JSON.parse(xhr.responseText);
          setError(errorData.detail || 'Upload failed');
          setStatus('error');
        }
      };

      xhr.onerror = () => {
        setError('Network error during upload');
        setStatus('error');
      };

      xhr.open('POST', '/api/video/upload');
      xhr.send(formData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setStatus('error');
    }
  }, [onUploaded]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center
          transition-colors cursor-pointer
          ${status === 'idle' ? 'border-slate-300 hover:border-blue-400 hover:bg-blue-50' : ''}
          ${status === 'uploading' ? 'border-blue-400 bg-blue-50' : ''}
          ${status === 'success' ? 'border-green-400 bg-green-50' : ''}
          ${status === 'error' ? 'border-red-400 bg-red-50' : ''}
        `}
      >
        <input
          type="file"
          accept=".mp4,.avi,.mov,.mkv"
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={status === 'uploading'}
        />

        {status === 'idle' && (
          <div className="space-y-2">
            <Upload className="w-12 h-12 mx-auto text-slate-400" />
            <p className="text-slate-600">
              Drag and drop your match video here
            </p>
            <p className="text-sm text-slate-400">
              or click to browse (MP4, AVI, MOV, MKV)
            </p>
          </div>
        )}

        {status === 'uploading' && (
          <div className="space-y-2">
            <Loader2 className="w-12 h-12 mx-auto text-blue-500 animate-spin" />
            <p className="text-blue-600 font-medium">{fileName}</p>
            <div className="w-full max-w-xs mx-auto bg-blue-100 rounded-full h-2">
              <div
                className="bg-blue-500 rounded-full h-2 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-blue-500">{progress}% uploaded</p>
          </div>
        )}

        {status === 'success' && uploadResponse && (
          <div className="space-y-2">
            <CheckCircle className="w-12 h-12 mx-auto text-green-500" />
            <p className="text-green-600 font-medium">{fileName}</p>
            <div className="text-sm text-slate-600 space-y-1">
              <p>Duration: {formatDuration(uploadResponse.duration_ms)}</p>
              <p>Resolution: {uploadResponse.resolution.join('x')}</p>
              <p>FPS: {uploadResponse.fps.toFixed(1)}</p>
            </div>
          </div>
        )}

        {status === 'error' && (
          <div className="space-y-2">
            <AlertCircle className="w-12 h-12 mx-auto text-red-500" />
            <p className="text-red-600">{error}</p>
            <button
              onClick={() => {
                setStatus('idle');
                setError(null);
              }}
              className="text-sm text-blue-500 hover:underline"
            >
              Try again
            </button>
          </div>
        )}
      </div>

      {/* File info after successful upload */}
      {status === 'success' && uploadResponse && (
        <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
          <FileVideo className="w-8 h-8 text-blue-500" />
          <div className="flex-1 min-w-0">
            <p className="font-medium text-slate-800 truncate">{fileName}</p>
            <p className="text-xs text-slate-500">
              Ready for processing • {formatDuration(uploadResponse.duration_ms)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}
