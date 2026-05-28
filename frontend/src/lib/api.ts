/**
 * api.ts — Client for the trekomend FastAPI backend.
 *
 * All calls go to /api/* on the same origin (no CORS issues).
 * Recommendation endpoints return a job_id (202 Accepted),
 * then the caller polls /api/jobs/{id} until completed.
 */

const API = '/api';

// ---- Types ----

export interface MovieResult {
  rank?: number;
  id?: number;
  tmdb_id?: number;
  title: string;
  primary_genre?: string;
  genres?: string;
  year?: number | null;
  overview?: string | null;
  vote_average?: number | null;
  score?: number | null;
  poster_path?: string | null;
  backdrop_path?: string | null;
  [key: string]: unknown;
}

/** Get the TMDB ID from a movie result, regardless of which field the API used. */
export function movieId(m: MovieResult): number {
  return m.tmdb_id ?? m.id ?? 0;
}

/** Get the poster URL for a movie. Uses real TMDB poster if available, otherwise a placeholder. */
export function posterUrl(m: MovieResult, size: string = 'w342'): string {
  if (m.poster_path) {
    return `https://image.tmdb.org/t/p/${size}${m.poster_path}`;
  }
  // Fallback: picsum with movie ID as seed for consistency
  const mid = movieId(m);
  return `https://picsum.photos/seed/${mid}/342/513`;
}

export interface JobResponse {
  job_id: string | null;
  status: string;
  type: string;
  message: string;
  mode: 'queue' | 'sync';
  elapsed_seconds?: number;
  results?: MovieResult[];
}

export interface JobStatus {
  id: string;
  type: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  payload?: Record<string, unknown>;
  results?: MovieResult[];
  error?: string;
  created_at?: number;
  started_at?: number;
  finished_at?: number;
}

interface ApiError {
  detail: string;
}

// ---- Helpers ----

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body: ApiError = await res.json();
      detail = body.detail || detail;
    } catch {}
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

// ---- Read endpoints (synchronous) ----

export async function searchMovies(query: string, limit = 20): Promise<MovieResult[]> {
  return fetchJson(`/movies/search?q=${encodeURIComponent(query)}&limit=${limit}`);
}

export async function getMovie(tmdbId: number): Promise<MovieResult> {
  return fetchJson(`/movies/${tmdbId}`);
}

export async function getGenres(): Promise<string[]> {
  return fetchJson('/movies/genres');
}

export async function getStats(): Promise<Record<string, unknown>> {
  return fetchJson('/stats');
}

// ---- Recommend endpoints (job queue or sync) ----

export async function recommendSimilar(title: string, limit = 12): Promise<JobResponse> {
  return fetchJson('/recommend/similar', {
    method: 'POST',
    body: JSON.stringify({ title, limit }),
  });
}

export async function recommendQuery(query: string, limit = 12): Promise<JobResponse> {
  return fetchJson('/recommend/query', {
    method: 'POST',
    body: JSON.stringify({ query, limit }),
  });
}

export async function recommendProfile(params: {
  liked: string[];
  dislike?: string[];
  mood?: string;
  mood_weight?: number;
  limit?: number;
}): Promise<JobResponse> {
  return fetchJson('/recommend/profile', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function recommendExplore(liked: string[], limit = 12): Promise<JobResponse> {
  return fetchJson('/recommend/explore', {
    method: 'POST',
    body: JSON.stringify({ liked, limit }),
  });
}

// ---- Job polling ----

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return fetchJson(`/jobs/${jobId}`);
}

/**
 * Poll a job until it completes or fails.
 * Returns results on success, throws on failure/timeout.
 */
export async function pollJob(
  jobId: string,
  onStatus?: (status: JobStatus) => void,
  timeoutMs = 20000,
  intervalMs = 600,
): Promise<MovieResult[]> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const job = await getJobStatus(jobId);
    onStatus?.(job);

    if (job.status === 'completed') {
      return job.results || [];
    }
    if (job.status === 'failed') {
      throw new Error(job.error || 'Recommendation failed');
    }

    await new Promise((r) => setTimeout(r, intervalMs));
  }

  throw new Error('Timed out waiting for results (20s)');
}
