/** Signal Room design system: high-end registration layout with Particles, RetroGrid, ShimmerButton, and invite code handling. */
import { Link, useNavigate } from '@tanstack/react-router';
import { ArrowRight, LockKeyhole, Mail, UserRound, Sparkles } from 'lucide-react';
import { FormEvent, useMemo, useState } from 'react';
import { toast } from 'sonner';
import { AuthFrame, Field } from './Login';
import { isSupabaseConfigured, supabase } from '@/lib/supabase';
import { ShimmerButton } from '@/components/ui/ShimmerButton';

export default function Signup() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const invite = useMemo(() => new URLSearchParams(window.location.search).get('invite') || '', []);

  const handle = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!isSupabaseConfigured) {
      return toast.error('Add VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY to enable sign-up.');
    }
    setLoading(true);
    const data = new FormData(e.currentTarget);
    const { data: created, error } = await supabase.auth.signUp({
      email: String(data.get('email')),
      password: String(data.get('password')),
      options: {
        data: {
          name: String(data.get('name')),
        },
      },
    });
    setLoading(false);
    if (error) return toast.error(error.message);
    if (invite) sessionStorage.setItem('tendersync_invite', invite);
    if (created.session) {
      toast.success('Account created. Let’s set up your workspace.');
      navigate({ to: '/onboarding' });
    } else {
      toast.success('Check your email to verify your account, then sign in.');
    }
  };

  return (
    <AuthFrame>
      <div className="text-left">
        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 font-mono text-[10px] font-bold text-emerald-300">
          <Sparkles size={11} className="text-emerald-400" />
          START 14-DAY TRIAL
        </div>
        <h1 className="mt-4 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          Create your workspace
        </h1>
        <p className="mt-2 text-sm text-zinc-400">
          Join leading capture teams putting evidence behind every bid decision.
        </p>

        {invite && (
          <div className="mt-4 flex items-center gap-2 rounded-lg border border-violet-400/30 bg-violet-400/10 px-3.5 py-2 text-xs text-violet-200">
            <span className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-pulse" />
            <span>Organization invite detected: <strong className="font-mono">{invite}</strong></span>
          </div>
        )}

        <form onSubmit={handle} className="mt-7 space-y-4">
          <Field
            label="Full name"
            name="name"
            type="text"
            placeholder="Jane Doe"
            icon={<UserRound size={16} />}
            autoComplete="name"
          />
          <Field
            label="Work email"
            name="email"
            type="email"
            placeholder="name@company.com"
            icon={<Mail size={16} />}
            autoComplete="email"
          />
          <Field
            label="Password"
            name="password"
            type="password"
            placeholder="Create a strong password"
            icon={<LockKeyhole size={16} />}
            autoComplete="new-password"
          />

          <div className="pt-2">
            <ShimmerButton disabled={loading} className="w-full !py-3 font-bold text-sm">
              {loading ? 'Creating account…' : (
                <>
                  Create Account <ArrowRight size={16} className="ml-2" />
                </>
              )}
            </ShimmerButton>
          </div>
        </form>

        <div className="mt-6 border-t border-white/[0.08] pt-4 text-center text-xs text-zinc-400">
          Already have a workspace?{' '}
          <Link to="/login" className="font-bold text-emerald-400 hover:text-emerald-300 transition">
            Sign in
          </Link>
        </div>
      </div>
    </AuthFrame>
  );
}
