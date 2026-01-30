import { useState, useEffect } from 'react';

interface ProcessingJob {
  video_id: string;
  status: string;
  progress_pct: number;
  current_frame: number;
  total_frames: number;
  error_message?: string;
}

export function ProcessingStatus() {
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchStatus = async () => {
    try {
      // Fetch status for known video ID (in production, you'd track this)
      const videoId = '69f39a66-ebb7-4ad6-9a78-7619ff8bd888';
      const response = await fetch(`/api/video/${videoId}/status`);
      if (response.ok) {
        const data = await response.json();
        setJobs([data]);
      } else {
        setJobs([]);
      }
    } catch (error) {
      console.error('Failed to fetch processing status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000); // Poll every 3 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="text-slate-500 text-sm">Loading processing status...</div>
    );
  }

  if (jobs.length === 0) {
    return (
      <div className="text-slate-500 text-sm">No videos currently processing</div>
    );
  }

  return (
    <div className="space-y-4">
      {jobs.map((job) => (
        <div key={job.video_id} className="border border-slate-200 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="font-medium text-slate-700 text-sm truncate max-w-[200px]">
              {job.video_id.substring(0, 8)}...
            </span>
            <span className={`text-xs px-2 py-1 rounded-full ${
              job.status === 'completed' ? 'bg-green-100 text-green-700' :
              job.status === 'failed' ? 'bg-red-100 text-red-700' :
              job.status === 'processing' ? 'bg-blue-100 text-blue-700' :
              'bg-slate-100 text-slate-700'
            }`}>
              {job.status.toUpperCase()}
            </span>
          </div>

          {job.status === 'processing' && (
            <>
              <div className="w-full bg-slate-200 rounded-full h-3 mb-2">
                <div
                  className="bg-blue-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(job.progress_pct, 100)}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-slate-500">
                <span>{job.progress_pct.toFixed(2)}%</span>
                <span>{job.current_frame.toLocaleString()} / {job.total_frames.toLocaleString()} frames</span>
              </div>
            </>
          )}

          {job.status === 'completed' && (
            <div className="text-green-600 text-sm">
              Analysis complete! {job.total_frames.toLocaleString()} frames processed.
            </div>
          )}

          {job.status === 'failed' && job.error_message && (
            <div className="text-red-600 text-sm">
              Error: {job.error_message}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
