/** Signal Room design system: authentication is quiet infrastructure behind the decision workspace. */
import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const apiKey = (import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY) as string | undefined;
export const isSupabaseConfigured = Boolean(url && apiKey);
export const supabase = createClient(url || 'https://placeholder.supabase.co', apiKey || 'placeholder-publishable-key', {
  auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
});
