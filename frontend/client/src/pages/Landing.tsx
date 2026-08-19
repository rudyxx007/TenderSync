import { Link } from '@tanstack/react-router';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowRight, 
  ArrowUpRight, 
  Bot, 
  CheckCircle2, 
  Cpu, 
  Database, 
  FileText, 
  Gauge, 
  Layers3, 
  Scale, 
  ShieldCheck, 
  Sparkles, 
  Table2, 
  Workflow, 
  Zap 
} from 'lucide-react';
import { Brand } from '@/components/Brand';
import { PwinGauge, DecisionBadge } from '@/components/PwinGauge';

const rotatingWords = [
  'Government RFPs',
  'Defense Pursuits',
  'Commercial Tenders',
  'Complex Proposals',
  'High-Stakes Bids'
];

const features = [
  {
    icon: FileText,
    title: 'Document Intelligence',
    text: 'Docling + RapidOCR converts dense 100+ page RFPs, tables, and scanned appendixes into structured evidence in seconds.',
    tag: 'DOCLING + OCR',
    color: '#34d399'
  },
  {
    icon: Scale,
    title: '4-Phase Decision Engine',
    text: 'Deterministic hard gates filter out non-negotiable failures, followed by 7 weighted scoring dimensions for mathematical rigor.',
    tag: '7 DIMENSIONS',
    color: '#fbbf24'
  },
  {
    icon: Bot,
    title: 'LangGraph Proposal Agents',
    text: 'Autonomous multi-agent team (Executive, Technical, Compliance, Pricing, Timeline) drafts tailor-fit response sections.',
    tag: '5 AI AGENTS',
    color: '#a78bfa'
  },
  {
    icon: Table2,
    title: 'Multi-Tender Comparison',
    text: 'Evaluate competing opportunities side-by-side on a unified radar matrix to prioritize highest-ROI pursuits.',
    tag: 'COMPARE MATRIX',
    color: '#38bdf8'
  },
  {
    icon: Layers3,
    title: 'Organization Workspaces',
    text: 'Multi-tenant collaboration with 8-character invite codes, role-based permissions, and shared executive dossiers.',
    tag: 'TEAM SYNC',
    color: '#ec4899'
  },
  {
    icon: ShieldCheck,
    title: 'Human-in-the-Loop Safeguards',
    text: 'When extraction confidence drops below 85%, the system immediately flags exact clauses for human validation.',
    tag: '85% THRESHOLD',
    color: '#34d399'
  }
];

export default function Landing() {
  const [wordIndex, setWordIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setWordIndex((prev) => (prev + 1) % rotatingWords.length);
    }, 2800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#0B0F17] text-zinc-100 selection:bg-emerald-400 selection:text-emerald-950">
      {/* Navigation Header */}
      <header className="sticky top-0 z-40 border-b border-white/[.08] bg-[#0B0F17]/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 lg:px-10">
          <Brand />
          <nav className="hidden items-center gap-8 text-xs font-semibold text-zinc-400 md:flex">
            <Link to="/" className="text-emerald-400 transition hover:text-emerald-300">Home</Link>
            <Link to="/how-it-works" className="transition hover:text-white">How It Works</Link>
            <Link to="/about" className="transition hover:text-white">About</Link>
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

      {/* Hero Section */}
      <section className="relative overflow-hidden border-b border-white/[.08] py-20 lg:py-32">
        {/* Ambient background glows */}
        <div className="pointer-events-none absolute -top-40 left-1/2 -translate-x-1/2 h-[500px] w-[800px] rounded-full bg-emerald-500/10 blur-[120px]" />
        <div className="pointer-events-none absolute top-1/3 right-10 h-[350px] w-[350px] rounded-full bg-violet-500/10 blur-[100px]" />

        <div className="relative mx-auto max-w-7xl px-5 lg:px-10">
          <div className="grid gap-14 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            {/* Left: Headline & Copy */}
            <div>
              <motion.div 
                initial={{ opacity: 0, y: 12 }} 
                animate={{ opacity: 1, y: 0 }} 
                className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3.5 py-1.5 font-mono text-[11px] font-semibold text-emerald-300"
              >
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                ENTERPRISE RFP DECISION INTELLIGENCE
              </motion.div>

              <motion.h1 
                initial={{ opacity: 0, y: 18 }} 
                animate={{ opacity: 1, y: 0 }} 
                transition={{ delay: 0.08, duration: 0.5 }} 
                className="mt-6 text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl"
              >
                Evaluate RFPs.<br />
                Win More{' '}
                <span className="relative inline-block text-emerald-400">
                  <AnimatePresence mode="wait">
                    <motion.span
                      key={wordIndex}
                      initial={{ y: 20, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      exit={{ y: -20, opacity: 0 }}
                      transition={{ duration: 0.35, ease: 'easeOut' }}
                      className="inline-block"
                    >
                      {rotatingWords[wordIndex]}
                    </motion.span>
                  </AnimatePresence>
                </span>
              </motion.h1>

              <motion.p 
                initial={{ opacity: 0, y: 18 }} 
                animate={{ opacity: 1, y: 0 }} 
                transition={{ delay: 0.16, duration: 0.5 }} 
                className="mt-6 max-w-xl text-base leading-8 text-zinc-400 sm:text-lg"
              >
                TenderSync turns complex 100-page tender PDFs into calculated, evidence-backed 
                <strong className="text-zinc-200"> BID</strong>, 
                <strong className="text-zinc-200"> CONDITIONAL</strong>, or 
                <strong className="text-zinc-200"> NO-BID</strong> decisions—and autonomous AI proposal drafts in minutes.
              </motion.p>

              <motion.div 
                initial={{ opacity: 0, y: 18 }} 
                animate={{ opacity: 1, y: 0 }} 
                transition={{ delay: 0.24, duration: 0.5 }} 
                className="mt-9 flex flex-wrap items-center gap-4"
              >
                <Link to="/signup" className="button-signal !px-6 !py-3.5 text-sm font-bold shadow-lg shadow-emerald-500/20">
                  Start Evaluating Free <ArrowRight size={16} />
                </Link>
                <Link to="/how-it-works" className="button-quiet !px-6 !py-3.5 text-sm font-bold">
                  Explore The Pipeline <ArrowUpRight size={16} />
                </Link>
              </motion.div>

              <div className="mt-12 flex flex-wrap items-center gap-6 border-t border-white/[.08] pt-6 text-xs text-zinc-400">
                <span className="flex items-center gap-2">
                  <CheckCircle2 size={15} className="text-emerald-400" />
                  Deterministic Hard Gates
                </span>
                <span className="flex items-center gap-2">
                  <CheckCircle2 size={15} className="text-emerald-400" />
                  7 Weighted Dimensions
                </span>
                <span className="flex items-center gap-2">
                  <CheckCircle2 size={15} className="text-emerald-400" />
                  LangGraph Proposal Writer
                </span>
              </div>
            </div>

            {/* Right: Live Interactive Sample Dossier Preview */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }} 
              animate={{ opacity: 1, scale: 1, y: 0 }} 
              transition={{ delay: 0.28, duration: 0.6 }}
              className="panel relative overflow-hidden p-6 shadow-2xl backdrop-blur-2xl"
            >
              <div className="flex items-center justify-between border-b border-white/[.08] pb-4">
                <div>
                  <p className="eyebrow text-emerald-400">Live Sample Evaluation</p>
                  <p className="mt-1 text-sm font-bold text-white">Federal Cloud Migration RFP</p>
                </div>
                <DecisionBadge decision="BID" />
              </div>

              <div className="mt-6 grid grid-cols-[1fr_1.2fr] items-center gap-6">
                <div className="grid place-items-center border-r border-white/[.08] pr-4">
                  <PwinGauge score={84} decision="BID" size={150} />
                </div>
                <div className="space-y-3">
                  <div>
                    <p className="eyebrow">Strategic Verdict</p>
                    <p className="mt-1 text-sm font-extrabold text-white">Strong Pursuit Signal (84% PWin)</p>
                    <p className="mt-1 text-xs leading-5 text-zinc-400">
                      Capability fit (20/20) and compliance ISO 27001 are fully verified.
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-2 font-mono text-[10px]">
                    <span className="rounded border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-1.5 text-emerald-300">
                      7/7 GATES PASSED
                    </span>
                    <span className="rounded border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-1.5 text-emerald-300">
                      94% CONFIDENCE
                    </span>
                  </div>
                </div>
              </div>

              <div className="mt-6 border-t border-white/[.08] pt-4 flex items-center justify-between text-xs">
                <span className="text-zinc-500">Autonomous Proposal Draft:</span>
                <span className="font-mono text-emerald-400">Ready in 5 sections</span>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Bento Grid */}
      <section className="border-b border-white/[.08] bg-[#0E131F] py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-5 lg:px-10">
          <div className="max-w-2xl">
            <p className="eyebrow text-emerald-400">Decision Intelligence Platform</p>
            <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              Everything Needed to Win Strategic Tenders
            </h2>
            <p className="mt-4 text-sm leading-7 text-zinc-400">
              A structured intelligence layer from first document drop to finalized executive proposals.
            </p>
          </div>

          <div className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {features.map((f) => (
              <div 
                key={f.title}
                className="panel group p-6 transition duration-300 hover:border-white/20 hover:-translate-y-1"
              >
                <div className="flex items-center justify-between">
                  <span 
                    className="grid h-10 w-10 place-items-center rounded-lg bg-white/[.06]"
                    style={{ color: f.color }}
                  >
                    <f.icon size={20} />
                  </span>
                  <span className="font-mono text-[10px] font-bold text-zinc-500">{f.tag}</span>
                </div>
                <h3 className="mt-6 text-base font-bold text-white">{f.title}</h3>
                <p className="mt-2 text-xs leading-6 text-zinc-400">{f.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Banner */}
      <section className="bg-[#0B0F17] py-20 text-center">
        <div className="mx-auto max-w-3xl px-5">
          <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Stop Guessing on RFPs. Start Winning.
          </h2>
          <p className="mt-4 text-sm leading-7 text-zinc-400">
            Join enterprise bid teams evaluating tenders with mathematical precision.
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Link to="/signup" className="button-signal !px-6 !py-3.5 font-bold">
              Start Free Evaluation <ArrowRight size={16} />
            </Link>
            <Link to="/how-it-works" className="button-quiet !px-6 !py-3.5 font-bold">
              See How It Works
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[.08] bg-[#070A0F] py-8 text-xs text-zinc-500">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-5 sm:flex-row lg:px-10">
          <Brand compact />
          <div className="flex gap-6 font-semibold">
            <Link to="/how-it-works" className="hover:text-white">How It Works</Link>
            <Link to="/about" className="hover:text-white">About</Link>
            <Link to="/login" className="hover:text-white">Sign In</Link>
          </div>
          <p>© 2026 TenderSync. Built for serious capture teams.</p>
        </div>
      </footer>
    </div>
  );
}
