import { Link } from '@tanstack/react-router';
import { motion } from 'framer-motion';
import { 
  ArrowRight, 
  Bot, 
  Brain, 
  CheckCircle2, 
  Cpu, 
  Database, 
  FileCheck, 
  FileText, 
  Gauge, 
  Layers, 
  Scale, 
  Search, 
  ShieldCheck, 
  Sparkles, 
  Zap 
} from 'lucide-react';
import { Brand } from '@/components/Brand';
import { PwinGauge, DecisionBadge } from '@/components/PwinGauge';

const pipelineStages = [
  {
    step: '01',
    name: 'Document Ingestion & OCR',
    tech: 'Docling + RapidOCR',
    desc: 'Dense RFP PDFs, complex tabular annexures, and scanned procurement files are parsed into pristine, machine-readable structured markdown.',
    icon: FileText,
    color: '#34d399',
    badge: 'STAGE 1'
  },
  {
    step: '02',
    name: 'Semantic Vector Embeddings',
    tech: 'BAAI/bge-m3 + pgvector',
    desc: 'Extracted text is split into semantic chunks and embedded into 1024-dimensional multi-lingual vectors stored directly in Supabase PostgreSQL.',
    icon: Database,
    color: '#a78bfa',
    badge: 'STAGE 2'
  },
  {
    step: '03',
    name: 'Extraction & Hard Gates',
    tech: 'Groq GPT-OSS-120B',
    desc: 'Deterministic rules check non-negotiable pass/fail gates (certifications, minimum revenue, insurance, submission deadlines) before scoring.',
    icon: ShieldCheck,
    color: '#fbbf24',
    badge: 'STAGE 3'
  },
  {
    step: '04',
    name: '7-Dimension Weighted Scoring',
    tech: 'Hybrid Reasoning Engine',
    desc: 'Context-grounded RAG retrieves relevant company past performance to compute a mathematically defensible Probability of Win (PWin: 0–100%).',
    icon: Scale,
    color: '#38bdf8',
    badge: 'STAGE 4'
  },
  {
    step: '05',
    name: 'Multi-Agent Proposal Writer',
    tech: 'LangGraph State Machine',
    desc: 'Autonomous agent team (Executive, Technical, Compliance, Commercial, Timeline) drafts tailor-fit response sections ready for Word export.',
    icon: Bot,
    color: '#ec4899',
    badge: 'STAGE 5'
  }
];

const scoringDimensions = [
  { name: 'Capability Fit', weight: 20, desc: 'Technical skills, scope coverage, architecture match, and tooling alignment.' },
  { name: 'Compliance Readiness', weight: 15, desc: 'Mandatory ISO certs (ISO 27001, 9001, SOC2), security frameworks, and legal criteria.' },
  { name: 'Past Performance', weight: 15, desc: 'Demonstrated case studies, domain experience, and sector-specific project track record.' },
  { name: 'Commercial Viability', weight: 15, desc: 'Budget thresholds, pricing margins, contract terms, and financial risk profiles.' },
  { name: 'Delivery Feasibility', weight: 15, desc: 'Staff capacity, resource availability, lead-time adequacy, and execution timeline.' },
  { name: 'Strategic Alignment', weight: 10, desc: 'Fit with core organization priorities, market positioning, and target sector expansion.' },
  { name: 'Competitive Landscape', weight: 10, desc: 'Incumbent presence, relationship strength, and differentiation advantage.' }
];

const proposalAgents = [
  { role: 'Executive Lead', focus: 'Value Proposition & Narrative', desc: 'Crafts the high-impact executive summary and competitive win themes.' },
  { role: 'Technical Architect', focus: 'Solution Design & Tech Stack', desc: 'Translates technical scope into detailed architectural deliverables.' },
  { role: 'Compliance Auditor', focus: 'Requirements & ISO Matrix', desc: 'Cross-verifies every mandatory RFP clause with auditable evidence.' },
  { role: 'Commercial Lead', focus: 'Costing & SLA Breakdown', desc: 'Structures pricing, milestones, and commercial risk mitigation.' },
  { role: 'Delivery Lead', focus: 'Timeline, RACI & Staffing', desc: 'Formulates realistic Gantt schedules, resource allocation, and SLAs.' }
];

export default function HowItWorks() {
  return (
    <div className="min-h-screen bg-[#0B0F17] text-zinc-100 selection:bg-emerald-400 selection:text-emerald-950">
      {/* Header Navigation */}
      <header className="sticky top-0 z-40 border-b border-white/[.08] bg-[#0B0F17]/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 lg:px-10">
          <Brand />
          <nav className="hidden items-center gap-8 text-xs font-semibold text-zinc-400 md:flex">
            <Link to="/" className="transition hover:text-white">Home</Link>
            <Link to="/how-it-works" className="text-emerald-400 transition hover:text-emerald-300">How It Works</Link>
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

      {/* Hero Banner */}
      <section className="relative overflow-hidden border-b border-white/[.08] py-20 lg:py-28">
        <div className="absolute inset-0 bg-gradient-to-b from-emerald-500/10 via-transparent to-transparent" />
        <div className="relative mx-auto max-w-5xl px-5 text-center lg:px-10">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3.5 py-1.5 font-mono text-[11px] font-semibold text-emerald-300"
          >
            <Zap size={13} className="text-emerald-400" />
            ENGINEERING & DECISION ARCHITECTURE
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 }}
            className="mt-6 text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl"
          >
            How TenderSync Works
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.16 }}
            className="mx-auto mt-6 max-w-3xl text-base leading-8 text-zinc-400 sm:text-lg"
          >
            From unstructured 100-page tender PDFs to mathematical Bid/No-Bid verdicts and autonomous multi-agent proposal generation. Discover the deep tech powering our pipeline.
          </motion.p>
        </div>
      </section>

      {/* Section 1: The 5-Stage Ingestion & Decision Pipeline */}
      <section className="border-b border-white/[.08] bg-[#0E131F] py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-5 lg:px-10">
          <div className="max-w-2xl">
            <p className="eyebrow text-emerald-400">End-to-End Pipeline</p>
            <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              5 Steps from Upload to Decisive Action
            </h2>
            <p className="mt-4 text-sm leading-7 text-zinc-400">
              Every tender undergoes strict document parsing, dense embedding vectorization, deterministic rule gating, and RAG-grounded LLM inference.
            </p>
          </div>

          <div className="mt-14 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {pipelineStages.map((stage, idx) => (
              <div 
                key={stage.step}
                className="panel group relative flex flex-col justify-between overflow-hidden p-6 transition duration-300 hover:border-white/20 hover:-translate-y-1"
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span 
                      className="grid h-10 w-10 place-items-center rounded-lg bg-white/[.06]"
                      style={{ color: stage.color }}
                    >
                      <stage.icon size={20} />
                    </span>
                    <span className="font-mono text-[10px] font-bold text-zinc-500">{stage.badge}</span>
                  </div>
                  <h3 className="mt-6 text-lg font-bold text-white">{stage.name}</h3>
                  <p className="mt-1 font-mono text-[11px] font-semibold" style={{ color: stage.color }}>{stage.tech}</p>
                  <p className="mt-3 text-xs leading-6 text-zinc-400">{stage.desc}</p>
                </div>
                <div className="mt-6 flex items-center gap-2 border-t border-white/[.08] pt-4">
                  <span className="font-mono text-[10px] font-bold text-zinc-500">STAGE {stage.step}</span>
                  <span className="h-1 flex-1 rounded-full bg-white/[.06] overflow-hidden">
                    <span className="block h-full bg-emerald-400" style={{ width: `${(idx + 1) * 20}%` }} />
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 2: 7-Dimension Weighted Scoring Matrix */}
      <section className="border-b border-white/[.08] bg-[#0B0F17] py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-5 lg:px-10">
          <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <div>
              <p className="eyebrow text-emerald-400">Scoring Engine</p>
              <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
                The 7 Weighted Scoring Dimensions
              </h2>
              <p className="mt-4 text-sm leading-7 text-zinc-400">
                Instead of a superficial LLM summary, TenderSync scores your company's profile against the RFP across 7 verified dimensions summing to 100%.
              </p>

              <div className="mt-8 space-y-3">
                {scoringDimensions.map((dim) => (
                  <div key={dim.name} className="panel flex items-center justify-between p-4">
                    <div className="pr-4">
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-bold text-white">{dim.name}</span>
                        <span className="rounded bg-emerald-400/10 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-300">
                          {dim.weight}% WEIGHT
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-zinc-400">{dim.desc}</p>
                    </div>
                    <div className="hidden sm:block text-right font-mono text-xs font-bold text-emerald-400">
                      {dim.weight} pts
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel p-8">
              <div className="text-center">
                <p className="eyebrow">Calculated Verdict Thresholds</p>
                <div className="mt-6 flex justify-center">
                  <PwinGauge score={85} decision="BID" size={180} />
                </div>
                <div className="mt-6 space-y-3 text-left">
                  <div className="rounded-lg border border-emerald-400/20 bg-emerald-400/10 p-3.5">
                    <div className="flex items-center justify-between font-mono text-xs font-bold text-emerald-300">
                      <span>PWin ≥ 75%</span>
                      <DecisionBadge decision="BID" />
                    </div>
                    <p className="mt-1 text-xs text-zinc-400">High win probability, solid capability coverage, all hard gates passed.</p>
                  </div>
                  <div className="rounded-lg border border-amber-400/20 bg-amber-400/10 p-3.5">
                    <div className="flex items-center justify-between font-mono text-xs font-bold text-amber-300">
                      <span>65% ≤ PWin &lt; 75%</span>
                      <DecisionBadge decision="CONDITIONAL" />
                    </div>
                    <p className="mt-1 text-xs text-zinc-400">Viable opportunity with identifiable risks; specific mitigations provided.</p>
                  </div>
                  <div className="rounded-lg border border-rose-400/20 bg-rose-400/10 p-3.5">
                    <div className="flex items-center justify-between font-mono text-xs font-bold text-rose-300">
                      <span>PWin &lt; 65% or Hard Gate Failed</span>
                      <DecisionBadge decision="NO-BID" />
                    </div>
                    <p className="mt-1 text-xs text-zinc-400">Critical deficiency or low ROI. Protects team bandwidth from bad bids.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: LangGraph Multi-Agent Proposal Writer */}
      <section className="border-b border-white/[.08] bg-[#0E131F] py-20 lg:py-28">
        <div className="mx-auto max-w-7xl px-5 lg:px-10">
          <div className="max-w-2xl">
            <p className="eyebrow text-violet-400">LangGraph Multi-Agent Engine</p>
            <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              5 Specialized Proposal Agents
            </h2>
            <p className="mt-4 text-sm leading-7 text-zinc-400">
              When you decide to bid, TenderSync deploys a specialized LangGraph state machine where 5 agents collaborate to draft a coherent, board-ready proposal.
            </p>
          </div>

          <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {proposalAgents.map((agent, i) => (
              <div key={agent.role} className="panel p-6">
                <div className="flex items-center justify-between">
                  <span className="grid h-8 w-8 place-items-center rounded-md bg-violet-400/15 font-mono text-xs font-bold text-violet-300">
                    0{i + 1}
                  </span>
                  <span className="font-mono text-[10px] text-zinc-500">STATE AGENT</span>
                </div>
                <h3 className="mt-4 text-base font-bold text-white">{agent.role}</h3>
                <p className="mt-1 font-mono text-xs text-violet-300">{agent.focus}</p>
                <p className="mt-3 text-xs leading-6 text-zinc-400">{agent.desc}</p>
              </div>
            ))}
            <div className="panel flex flex-col justify-center border-dashed border-emerald-400/30 bg-emerald-400/[.03] p-6 text-center">
              <CheckCircle2 size={28} className="mx-auto text-emerald-400" />
              <h3 className="mt-3 text-base font-bold text-white">Export to Word & PDF</h3>
              <p className="mt-2 text-xs leading-6 text-zinc-400">
                1-click download as formatted Microsoft Word (.docx) or executive PDF dossier.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Footer */}
      <section className="bg-[#0B0F17] py-20 text-center">
        <div className="mx-auto max-w-3xl px-5">
          <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Ready to evaluate your next RFP?
          </h2>
          <p className="mt-4 text-sm leading-7 text-zinc-400">
            Upload your document and get an evidence-backed evaluation in under 2 minutes.
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Link to="/signup" className="button-signal !px-6 !py-3">
              Start Free Evaluation <ArrowRight size={16} />
            </Link>
            <Link to="/about" className="button-quiet !px-6 !py-3">
              Why We Built TenderSync
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[.08] bg-[#070A0F] py-8 text-center text-xs text-zinc-500">
        <p>© 2026 TenderSync. Built for serious capture and bid teams.</p>
      </footer>
    </div>
  );
}
