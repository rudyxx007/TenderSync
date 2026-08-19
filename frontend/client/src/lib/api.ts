/** Signal Room design system: API methods preserve authenticated, direct evidence access. */
import { supabase } from '@/lib/supabase';
export class ApiError extends Error { constructor(public status: number, message: string) { super(message); this.name = 'ApiError'; } }
const API_BASE = ((import.meta.env.VITE_API_URL as string | undefined) || 'https://wrote-replacement-surveys-jan.trycloudflare.com').replace(/\/+$/, '');
async function token() { const { data: { session } } = await supabase.auth.getSession(); if (!session) throw new ApiError(401, 'Your session has expired. Please sign in again.'); return session.access_token; }
export async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const accessToken = await token();
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}`, ...options.headers } });
  if (!response.ok) { const err = await response.json().catch(() => ({ detail: response.statusText })); throw new ApiError(response.status, err.detail || 'The request could not be completed.'); }
  return response.json() as Promise<T>;
}
export async function apiUpload<T>(path: string, formData: FormData): Promise<T> {
  const accessToken = await token();
  const response = await fetch(`${API_BASE}${path}`, { method: 'POST', headers: { Authorization: `Bearer ${accessToken}` }, body: formData });
  if (!response.ok) { const err = await response.json().catch(() => ({ detail: response.statusText })); throw new ApiError(response.status, err.detail || 'The file upload failed.'); }
  return response.json() as Promise<T>;
}
export async function apiDownload(path: string, filename: string, method: 'GET' | 'POST' = 'GET') {
  const accessToken = await token(); const response = await fetch(`${API_BASE}${path}`, { method, headers: { Authorization: `Bearer ${accessToken}` } });
  if (!response.ok) throw new ApiError(response.status, 'The requested export could not be generated.');
  const url = URL.createObjectURL(await response.blob()); const link = document.createElement('a'); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(url);
}
