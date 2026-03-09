const BASE_URL = import.meta.env.VITE_API_URL || '';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error');
    throw new ApiError(res.status, text);
  }

  return res.json();
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: 'PUT', body: body ? JSON.stringify(body) : undefined }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),

  // Raw fetch for file uploads (no JSON headers)
  upload: (path: string, formData: FormData, onProgress?: (pct: number) => void): Promise<{ video_id: string }> => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      if (onProgress) {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
        });
      }
      xhr.onload = () => {
        if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
        else reject(new ApiError(xhr.status, xhr.responseText));
      };
      xhr.onerror = () => reject(new Error('Upload failed'));
      xhr.open('POST', `${BASE_URL}${path}`);
      xhr.send(formData);
    });
  },

  baseUrl: BASE_URL,

  /**
   * Chunked resumable upload with retry, resume, and per-chunk progress.
   */
  uploadChunked: async (
    file: File,
    metadata: object,
    matchHalf: 'full' | 'first' | 'second',
    matchId: string | null,
    onProgress: (info: {
      pct: number;
      chunksDone: number;
      totalChunks: number;
      speedMBps: number;
      etaSeconds: number;
    }) => void,
    abortSignal?: { aborted: boolean },
  ): Promise<{ video_id: string; upload_id: string }> => {
    // 1. Init session
    const initRes = await request<{ upload_id: string; chunk_size: number; total_chunks: number }>(
      '/api/video/upload/init',
      {
        method: 'POST',
        body: JSON.stringify({
          filename: file.name,
          total_size: file.size,
          match_half: matchHalf,
          match_id: matchId,
          metadata,
        }),
      },
    );

    const { upload_id, chunk_size, total_chunks } = initRes;

    // Save to localStorage for resume
    localStorage.setItem(`upload_${upload_id}`, JSON.stringify({ filename: file.name, matchHalf, matchId }));

    // 2. Check which chunks already uploaded (resume case)
    const statusRes = await request<{ received_chunks: number[] }>(
      `/api/video/upload/${upload_id}/status`,
    );
    const alreadyReceived = new Set(statusRes.received_chunks);

    // 3. Upload chunks
    let chunksDone = alreadyReceived.size;
    const startTime = Date.now();

    for (let i = 0; i < total_chunks; i++) {
      if (abortSignal?.aborted) {
        return { video_id: '', upload_id }; // Paused — return upload_id for resume
      }

      if (alreadyReceived.has(i)) continue; // Skip already-uploaded chunks

      const start = i * chunk_size;
      const end = Math.min(start + chunk_size, file.size);
      const chunk = file.slice(start, end);

      // Retry up to 3 times with backoff
      let lastError: Error | null = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const res = await fetch(`${BASE_URL}/api/video/upload/${upload_id}/chunk/${i}`, {
            method: 'PUT',
            body: chunk,
            headers: { 'Content-Type': 'application/octet-stream' },
          });
          if (!res.ok) throw new ApiError(res.status, await res.text());
          lastError = null;
          break;
        } catch (err) {
          lastError = err as Error;
          if (attempt < 2) await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
        }
      }
      if (lastError) throw lastError;

      chunksDone++;
      const elapsed = (Date.now() - startTime) / 1000;
      const bytesUploaded = chunksDone * chunk_size;
      const speedMBps = elapsed > 0 ? bytesUploaded / (1024 * 1024) / elapsed : 0;
      const remainingChunks = total_chunks - chunksDone;
      const etaSeconds = speedMBps > 0 ? (remainingChunks * chunk_size) / (speedMBps * 1024 * 1024) : 0;

      onProgress({
        pct: Math.round((chunksDone / total_chunks) * 100),
        chunksDone,
        totalChunks: total_chunks,
        speedMBps: Math.round(speedMBps * 10) / 10,
        etaSeconds: Math.round(etaSeconds),
      });
    }

    // 4. Complete — concatenate on server
    const completeRes = await request<{ video_id: string }>(
      `/api/video/upload/${upload_id}/complete`,
      { method: 'POST' },
    );

    // Clean up localStorage
    localStorage.removeItem(`upload_${upload_id}`);

    return { video_id: completeRes.video_id, upload_id };
  },

  /**
   * Check for a resumable upload session. Returns upload_id or null.
   */
  getResumableUpload: (): string | null => {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('upload_')) {
        return key.replace('upload_', '');
      }
    }
    return null;
  },

  clearResumableUpload: (uploadId: string) => {
    localStorage.removeItem(`upload_${uploadId}`);
  },

  // ── URL Import ──
  importFromUrl: (url: string, metadata?: object, matchHalf?: string, matchId?: string | null) =>
    request<{ import_id: string; status: string }>('/api/video/import', {
      method: 'POST',
      body: JSON.stringify({ url, metadata, match_half: matchHalf || 'full', match_id: matchId }),
    }),

  getImportStatus: (importId: string) =>
    request<{ status: string; progress_pct: number; video_id: string | null; error: string | null }>(
      `/api/video/import/${importId}/status`,
    ),

  // ── VEO API ──
  veoConnect: (token: string) =>
    request<{ connected: boolean; user_info: Record<string, unknown> }>('/api/veo/connect', {
      method: 'POST',
      body: JSON.stringify({ token }),
    }),

  veoStatus: () =>
    request<{ connected: boolean }>('/api/veo/status'),

  veoRecordings: (page = 1, perPage = 20) =>
    request<{ recordings: VeoRecording[]; page: number; per_page: number }>(
      `/api/veo/recordings?page=${page}&per_page=${perPage}`,
    ),

  veoImport: (recordingId: string) =>
    request<{ import_id: string; status: string }>(`/api/veo/import/${recordingId}`, {
      method: 'POST',
    }),
};

export interface VeoRecording {
  id: string;
  title: string;
  date: string;
  duration: number;
  thumbnail_url: string;
}
