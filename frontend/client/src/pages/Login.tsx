/** Signal Room design system: high-end authentication layout with particles, animated cards, and responsive decision intelligence preview. */
import { Link, useNavigate } from '@tanstack/react-router';
import { ArrowRight, LockKeyhole, Mail, ShieldCheck, Sparkles, CheckCircle2, FileText, Bot } from 'lucide-react';
import { FormEvent, useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { Brand } from '@/components/Brand';
import { isSupabaseConfigured, supabase } from '@/lib/supabase';
import { Particles } from '@/components/ui/Particles';
import { RetroGrid } from '@/components/ui/RetroGrid';
import { ShimmerButton } from '@/components/ui/ShimmerButton';
import { MovingBorderCard } from '@/components/ui/MovingBorderCard';
import { CardSpotlight } from '@/components/ui/CardSpotlight';
import { PwinGauge, DecisionBadge } from '@/components/PwinGauge';

export default function Login() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const handle = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!isSupabaseConfigured) {
      return toast.error('Add VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY to connect sign-in.');
    }
    setLoading(true);
    const data = new FormData(e.currentTarget);
    const { error } = await supabase.auth.signInWithPassword({
      email: String(data.get('email')),
      password: String(data.get('password')),
    });
    setLoading(false);
    if (error) return toast.error(error.message);
    toast.success('Authenticated. Opening your workspace.');
    navigate({ to: '/dashboard' });
  };

  return (
    <AuthFrame>
      <div className="text-left">
        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 font-mono text-[10px] font-bold text-emerald-300">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
          SECURE WORKSPACE ACCESS
        </div>
        <h1 className="mt-4 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          Welcome back
        </h1>
        <p className="mt-2 text-sm text-zinc-400">
          Sign in to access your tender dossiers, radar fits, and proposal agents.
        </p>

        <form onSubmit={handle} className="mt-8 space-y-4">
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
            placeholder="••••••••••••"
            icon={<LockKeyhole size={16} />}
            autoComplete="current-password"
          />
          
          <div className="pt-2">
            <ShimmerButton disabled={loading} className="w-full !py-3 font-bold text-sm">
              {loading ? 'Authenticating…' : (
                <>
                  Sign in to Workspace <ArrowRight size={16} className="ml-2" />
                </>
              )}
            </ShimmerButton>
          </div>
        </form>

        <div className="mt-6 border-t border-white/[0.08] pt-4 text-center text-xs text-zinc-400">
          Don't have a workspace?{' '}
          <Link to="/signup" className="font-bold text-emerald-400 hover:text-emerald-300 transition">
            Create an account
          </Link>
        </div>
      </div>
    </AuthFrame>
  );
}

export function Field({
  label,
  name,
  type,
  icon,
  placeholder,
  autoComplete,
  required = true,
  defaultValue,
}: {
  label: string;
  name: string;
  type: string;
  icon?: React.ReactNode;
  placeholder?: string;
  autoComplete?: string;
  required?: boolean;
  defaultValue?: string | number;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs font-bold text-zinc-300">{label}</span>
      <div className="relative">
        <span className="absolute inset-y-0 left-3.5 flex items-center text-zinc-500">{icon}</span>
        <input
          required={required}
          name={name}
          type={type}
          placeholder={placeholder}
          defaultValue={defaultValue}
          autoComplete={autoComplete}
          className="h-11 w-full rounded-lg border border-white/10 bg-black/30 pl-10 pr-3.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none transition focus:border-emerald-400/50 focus:ring-2 focus:ring-emerald-400/20"
        />
      </div>
    </label>
  );
}

export function AuthFrame({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#0B0F17] px-4 py-12 selection:bg-emerald-400 selection:text-emerald-950">
      {/* Background Ambience */}
      <Particles quantity={30} color="#34d399" className="opacity-30" />
      <RetroGrid className="opacity-15" />
      <div className="pointer-events-none absolute -top-40 left-1/4 h-[500px] w-[500px] rounded-full bg-emerald-500/10 blur-[130px]" />
      <div className="pointer-events-none absolute bottom-0 right-10 h-[400px] w-[400px] rounded-full bg-violet-500/10 blur-[120px]" />

      {/* Top Header */}
      <header className="absolute top-6 left-6 z-20">
        <Brand />
      </header>

      {/* Main Container */}
      <div className="relative z-10 mx-auto grid w-full max-w-6xl items-center gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:px-6">
        {/* Left Side: Animated Decision Intelligence Showcase */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="hidden space-y-8 lg:block"
        >
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3.5 py-1 font-mono text-[10px] font-bold tracking-wide text-emerald-300">
              <Sparkles size={12} className="text-emerald-400" />
              EVALUATE • QUALIFY • WIN
            </div>
            <h2 className="mt-4 text-4xl font-extrabold tracking-tight text-white">
              AI Decision Intelligence for High-Stakes Pursuits.
            </h2>
            <p className="mt-3 text-base leading-7 text-zinc-400">
              Stop burning capture bandwidth on losing bids. TenderSync calculates mathematical Probability of Win (PWin) scores and drafts compliant responses in minutes.
            </p>
          </div>

          {/* Mini Live Preview inside MovingBorderCard */}
          <MovingBorderCard duration={5000} className="p-6">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
              <div>
                <p className="eyebrow text-emerald-400">Sample Pursuit Audit</p>
                <p className="mt-0.5 text-sm font-bold text-white">Department of Transportation RFP</p>
              </div>
              <DecisionBadge decision="BID" />
            </div>

            <div className="mt-5 grid grid-cols-[130px_1fr] items-center gap-5">
              <PwinGauge score={88} decision="BID" size={130} />
              <div className="space-y-2">
                <p className="text-xs font-bold text-zinc-200">Recommendation: Clear Bid Pursuit</p>
                <p className="text-xs leading-5 text-zinc-400">
                  All 7 hard compliance gates passed. High capability fit (20/20) and ISO 27001 verified.
                </p>
                <div className="flex gap-2 font-mono text-[9px] text-emerald-300">
                  <span className="rounded bg-emerald-400/10 px-2 py-0.5 border border-emerald-400/20">7/7 GATES</span>
                  <span className="rounded bg-emerald-400/10 px-2 py-0.5 border border-emerald-400/20">96% CONF.</span>
                </div>
              </div>
            </div>
          </MovingBorderCard>

          {/* Key Trust Pillars */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-start gap-3 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3.5">
              <ShieldCheck size={18} className="mt-0.5 text-emerald-400 shrink-0" />
              <div>
                <p className="text-xs font-bold text-white">Human-in-the-Loop</p>
                <p className="mt-0.5 text-[11px] text-zinc-400">Low confidence clauses flagged for manual review.</p>
              </div>
            </div>
            <div className="flex items-start gap-3 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3.5">
              <Bot size={18} className="mt-0.5 text-violet-400 shrink-0" />
              <div>
                <p className="text-xs font-bold text-white">LangGraph Agents</p>
                <p className="mt-0.5 text-[11px] text-zinc-400">5 proposal agents generate board-ready drafts.</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Right Side: Form Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="w-full max-w-md justify-self-center lg:justify-self-end"
        >
          <div className="relative overflow-hidden rounded-2xl border border-white/[0.1] bg-[#121722]/85 p-8 shadow-2xl backdrop-blur-2xl sm:p-10">
            <div className="pointer-events-none absolute -top-24 -right-24 h-48 w-48 rounded-full bg-emerald-500/10 blur-[60px]" />
            {children}
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <footer className="absolute bottom-4 z-20 text-center font-mono text-[10px] tracking-widest text-zinc-600">
        TENDERSYNC • ENTERPRISE CAPTURE INTELLIGENCE
      </footer>
    </div>
  );
}
