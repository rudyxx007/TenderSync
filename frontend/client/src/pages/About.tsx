import { Link } from '@tanstack/react-router';
import { motion } from 'framer-motion';
import { 
  AlertTriangle, 
  ArrowRight, 
  CheckCircle2, 
  Clock, 
  Flame, 
  HeartHandshake, 
  Lightbulb, 
  ShieldCheck, 
  Target, 
  TrendingUp, 
  Users, 
  Zap 
} from 'lucide-react';
import { Brand } from '@/components/Brand';

const problems = [
  {
    icon: Clock,
    title: 'Wasted Proposal Bandwidth',
    desc: 'Capture teams spend 40–80 hours drafting responses for tenders they had virtually zero chance of winning due to hidden compliance deal-killers.'
  },
  {
    icon: AlertTriangle,
    title: 'Subjective Guesswork',
    desc: 'Bid decisions are often made on gut feeling, rushed executive meetings, or FOMO rather than cold, objective mathematical qualification.'
  },
  {
    icon: Flame,
    title: 'Complex 100+ Page RFPs',
    desc: 'Crucial requirements, mandatory ISO criteria, and punitive SLAs get buried deep inside annexures and scanned appendix tables.'
  }
];

const principles = [
  {
    icon: Target,
    title: 'Evidence-First Decision Making',
    desc: 'Every verdict (BID, CONDITIONAL, or NO-BID) is backed by explicit citations from the RFP and matched against your exact organizational profile.'
  },
  {
    icon: ShieldCheck,
    title: 'Human-in-the-Loop Transparency',
    desc: 'We never hide AI uncertainty. When extraction confidence falls below 85%, we explicitly flag the exact clauses for human validation.'
  },
  {
    icon: Zap,
    title: 'Speed Without Sacrificing Rigor',
    desc: 'What used to take 3 days of cross-departmental committee review is compressed into a 90-second deterministic and RAG evaluation.'
  }
];

export default function About() {
  return (
    <div className="min-h-screen bg-[#0B0F17] text-zinc-100 selection:bg-emerald-400 selection:text-emerald-950">
      {/* Header Navigation */}
      <header className="sticky top-0 z-40 border-b border-white/[.08] bg-[#0B0F17]/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 lg:px-10">
          <Brand />
          <nav className="hidden items-center gap-8 text-xs font-semibold text-zinc-400 md:flex">
            <Link to="/" className="transition hover:text-white">Home</Link>
            <Link to="/how-it-works" className="transition hover:text-white">How It Works</Link>
            <Link to="/about" className="text-emerald-400 transition hover:text-emerald-300">About</Link>
          </nav>
          <div className="flex items-center gap-3">
            <Link to="/login" className="px-3.5 py-2 text-xs font-bold text-zinc-300 transition hover:text-white">
              Sign in
            </Link>
            <Link to="/signup" className="button-signal !px-4 !py-2 text-xs">
              Start Free <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden border-b border-white/[.08] py-20 lg:py-28">
        <div className="absolute inset-0 bg-gradient-to-b from-emerald-500/10 via-transparent to-transparent" />
        <div className="relative mx-auto max-w-5xl px-5 text-center lg:px-10">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3.5 py-1.5 font-mono text-[11px] font-semibold text-emerald-300"
          >
            <Lightbulb size={13} className="text-emerald-400" />
            OUR MISSION & STORY
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 }}
            className="mt-6 text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl"
          >
            Why We Built TenderSync
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.16 }}
            className="mx-auto mt-6 max-w-3xl text-base leading-8 text-zinc-400 sm:text-lg"
          >
            Enterprise procurement shouldn't be a game of chance. TenderSync was built to eliminate RFP guesswork, stop proposal burnout, and empower bid teams to pursue only the contracts they can win.
          </motion.p>
        </div>
      </section>

      {/* The Problem We Solve */}
      <section className="border-b border-white/[.08] bg-[#0E131F] py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-5 lg:px-10">
          <div className="max-w-2xl">
            <p className="eyebrow text-rose-400">The Problem</p>
            <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              The High Cost of Bad Bid Decisions
            </h2>
            <p className="mt-4 text-sm leading-7 text-zinc-400">
              In government and enterprise contracting, pursuing the wrong RFP isn't just an inconvenience—it burns hundreds of hours of elite engineering and capture bandwidth.
            </p>
          </div>

          <div className="mt-12 grid gap-6 md:grid-cols-3">
            {problems.map((prob) => (
              <div key={prob.title} className="panel p-6 border-rose-500/10 hover:border-rose-500/30 transition duration-300">
                <span className="grid h-10 w-10 place-items-center rounded-lg bg-rose-500/10 text-rose-400">
                  <prob.icon size={20} />
                </span>
                <h3 className="mt-6 text-lg font-bold text-white">{prob.title}</h3>
                <p className="mt-3 text-xs leading-6 text-zinc-400">{prob.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The TenderSync Solution & Vision */}
      <section className="border-b border-white/[.08] bg-[#0B0F17] py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-5 lg:px-10">
          <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
            <div>
              <p className="eyebrow text-emerald-400">Our Vision</p>
              <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
                Turning RFP Complexity Into Calculated Advantage
              </h2>
              <p className="mt-4 text-sm leading-7 text-zinc-400">
                TenderSync was engineered from the ground up to act as your team's autonomous Capture Director. By fusing deep document parsing (Docling), multi-lingual vector indexing (BGE-M3), and high-speed multi-agent intelligence (Groq + LangGraph), we give your team an unassailable edge.
              </p>
              <div className="mt-8 space-y-4">
                <div className="flex gap-3">
                  <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-emerald-400" />
                  <p className="text-xs leading-6 text-zinc-300">
                    <strong>Stop low-probability pursuits:</strong> Immediately disqualify tenders with non-negotiable hard gate deal killers.
                  </p>
                </div>
                <div className="flex gap-3">
                  <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-emerald-400" />
                  <p className="text-xs leading-6 text-zinc-300">
                    <strong>Focus on high-ROI bids:</strong> Double down on opportunities with PWin scores ≥ 75% and clear capability alignment.
                  </p>
                </div>
                <div className="flex gap-3">
                  <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-emerald-400" />
                  <p className="text-xs leading-6 text-zinc-300">
                    <strong>Draft faster with multi-agents:</strong> Go from decision to full proposal draft in minutes, ready for final human polish.
                  </p>
                </div>
              </div>
            </div>

            <div className="panel p-8 bg-gradient-to-br from-emerald-500/5 via-transparent to-transparent">
              <p className="eyebrow text-emerald-400">Our Core Principles</p>
              <div className="mt-6 space-y-6">
                {principles.map((pr) => (
                  <div key={pr.title} className="flex gap-4">
                    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-emerald-400/10 text-emerald-300">
                      <pr.icon size={16} />
                    </span>
                    <div>
                      <h4 className="text-sm font-bold text-white">{pr.title}</h4>
                      <p className="mt-1 text-xs leading-5 text-zinc-400">{pr.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="bg-[#0E131F] py-20 text-center">
        <div className="mx-auto max-w-3xl px-5">
          <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Join the Next Generation of Bid Intelligence
          </h2>
          <p className="mt-4 text-sm leading-7 text-zinc-400">
            Start evaluating tenders with mathematical rigor today.
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Link to="/signup" className="button-signal !px-6 !py-3">
              Get Started Free <ArrowRight size={16} />
            </Link>
            <Link to="/how-it-works" className="button-quiet !px-6 !py-3">
              Explore the Technical Pipeline
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[.08] bg-[#070A0F] py-8 text-center text-xs text-zinc-500">
        <p>© 2026 TenderSync. Engineered for serious pursuit teams.</p>
      </footer>
    </div>
  );
}
