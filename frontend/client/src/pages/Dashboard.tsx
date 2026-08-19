/** Signal Room design system: the command center foregrounds a single decisive upload and an auditable live pipeline. */
import { Link, useNavigate } from '@tanstack/react-router';
import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, 
  FileText, 
  FolderUp, 
  Sparkles, 
  Trophy, 
  Upload, 
  X, 
  CheckCircle2, 
  ArrowRight,
  ShieldAlert
} from 'lucide-react';
import { toast } from 'sonner';
import { apiFetch, apiUpload } from '@/lib/api';
import type { ProcessTenderResponse, TenderAnalysisSummary, ProposalResponse } from '@/types/api';
import { DecisionBadge, PwinGauge } from '@/components/PwinGauge';
import { ErrorBlock, LoadingBlock, PageHeader, formatDate } from '@/components/Workspace';
import { CardSpotlight } from '@/components/ui/CardSpotlight';
import { MovingBorderCard } from '@/components/ui/MovingBorderCard';
import { ShimmerButton } from '@/components/ui/ShimmerButton';

const pipeline = [
  ['01', 'Parsing document structure & tables', 'Docling + RapidOCR'],
  ['02', 'Generating semantic embeddings', 'BGE-M3 (1024-dim)'],
  ['03', 'Querying vector evidence', 'Supabase pgvector'],
  ['04', 'Extracting clauses & metadata', 'Groq GPT-OSS-120B'],
  ['05', 'Executing bid / no-bid engine', '7-dimension hybrid evaluation']
];

export default function Dashboard() {
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [files, setFiles] = useState<File[]>([]);
  const [stage, setStage] = useState(-1);
  const [result, setResult] = useState<ProcessTenderResponse | null>(null);
  const [tenders, setTenders] = useState<TenderAnalysisSummary[]>([]);
  const [proposals, setProposals] = useState<ProposalResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const [t, p] = await Promise.all([
        apiFetch<TenderAnalysisSummary[]>('/api/tenders'),
        apiFetch<ProposalResponse[]>('/api/proposals')
      ]);
      setTenders(t);
      setProposals(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load workspace data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (stage < 0 || stage >= 4) return;
    const t = window.setTimeout(() => setStage(stage + 1), 900);
    return () => window.clearTimeout(t);
  }, [stage]);

  const selected = (next: FileList | null) => {
    if (!next) return;
    const added = Array.from(next);
    if (mode === 'single') setFiles(added.slice(0, 1));
    else setFiles(added.slice(0, 10));
    setResult(null);
    setError('');
  };

  const run = async () => {
    if (!files.length) return;
    setStage(0);
    setError('');
    try {
      if (mode === 'single') {
        const form = new FormData();
        form.append('file', files[0]);
        const data = await apiUpload<ProcessTenderResponse>('/api/process-tender', form);
        setResult(data);
        setStage(5);
        toast.success('Tender evaluation complete.');
        refresh();
      } else {
        const form = new FormData();
        files.forEach((f) => form.append('files', f));
        await apiUpload('/api/tenders/batch', form);
        setStage(5);
        toast.success('Batch accepted for processing.');
        setFiles([]);
        navigate({ to: '/batches' });
      }
    } catch (e) {
      setStage(-1);
      setError(e instanceof Error ? e.message : 'The tender could not be processed.');
    }
  };

  const metrics = useMemo(() => {
    const scored = tenders.filter((t) => t.win_probability_score !== null);
    return [
      {
        label: 'Tenders evaluated',
        value: tenders.length,
        icon: FileText,
        color: 'text-zinc-200',
        glow: 'rgba(255,255,255,0.06)'
      },
      {
        label: 'Average PWin',
        value: scored.length ? `${Math.round(scored.reduce((a, t) => a + (t.win_probability_score || 0), 0) / scored.length)}%` : '—',
        icon: BarChart3,
        color: 'text-emerald-300',
        glow: 'rgba(52,211,153,0.15)'
      },
      {
        label: 'Active Pursuits',
        value: tenders.filter((t) => t.decision === 'BID' || t.decision === 'CONDITIONAL').length,
        icon: Sparkles,
        color: 'text-amber-300',
        glow: 'rgba(251,191,36,0.15)'
      },
      {
        label: 'Won proposals',
        value: proposals.filter((p) => p.status === 'won').length,
        icon: Trophy,
        color: 'text-violet-300',
        glow: 'rgba(167,139,250,0.15)'
      }
    ];
  }, [tenders, proposals]);

  return (
    <div className="space-y-8 pb-12">
      <PageHeader
        eyebrow="Command center"
        title="Make the next pursuit a better decision."
        text="Upload an RFP to establish the evidence, position, and most valuable next action."
        action={
          <Link to="/tenders" className="button-quiet">
            View all dossiers
          </Link>
        }
      />

      {loading ? (
        <LoadingBlock />
      ) : (
        <>
          {error && (
            <div className="mb-5">
              <ErrorBlock text={error} />
            </div>
          )}

          {/* Interactive Metric Cards with Spotlights */}
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {metrics.map(({ label, value, icon: Icon, color, glow }) => (
              <CardSpotlight
                key={label}
                color={glow}
                className="p-5"
              >
                <div className="flex items-center justify-between">
                  <p className="eyebrow">{label}</p>
                  <Icon size={18} className="text-zinc-500" />
                </div>
                <p className={`mt-4 text-3xl font-extrabold tracking-tight ${color}`}>{value}</p>
              </CardSpotlight>
            ))}
          </div>

          {/* Upload Dropzone & Pipeline Trace */}
          <section className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            {/* Upload Area */}
            <div className="panel overflow-hidden">
              <div className="flex items-center justify-between border-b border-white/[0.08] px-6 py-4">
                <div>
                  <p className="text-sm font-extrabold text-white">Evaluate tender material</p>
                  <p className="mt-0.5 text-xs text-zinc-400">PDF and image uploads · Single file up to 100MB</p>
                </div>
                <div className="rounded-lg bg-black/40 p-1 border border-white/[0.06]">
                  <button
                    onClick={() => {
                      setMode('single');
                      setFiles([]);
                    }}
                    className={`rounded-md px-3 py-1 text-xs font-bold transition ${
                      mode === 'single' ? 'bg-emerald-400/20 text-emerald-300 border border-emerald-400/30' : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    Single
                  </button>
                  <button
                    onClick={() => {
                      setMode('batch');
                      setFiles([]);
                    }}
                    className={`rounded-md px-3 py-1 text-xs font-bold transition ${
                      mode === 'batch' ? 'bg-emerald-400/20 text-emerald-300 border border-emerald-400/30' : 'text-zinc-400 hover:text-white'
                    }`}
                  >
                    Batch
                  </button>
                </div>
              </div>

              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  selected(e.dataTransfer.files);
                }}
                onClick={() => fileRef.current?.click()}
                className={`m-6 grid min-h-[220px] cursor-pointer place-items-center rounded-xl border-2 border-dashed transition duration-300 ${
                  isDragging
                    ? 'border-emerald-400 bg-emerald-400/10 scale-[1.01]'
                    : 'border-emerald-400/30 bg-emerald-400/[0.02] hover:border-emerald-400/60 hover:bg-emerald-400/[0.05]'
                } p-6 text-center`}
              >
                <input
                  ref={fileRef}
                  onChange={(e) => selected(e.target.files)}
                  accept=".pdf,image/*"
                  multiple={mode === 'batch'}
                  type="file"
                  className="hidden"
                />
                <div>
                  <span className="mx-auto grid h-14 w-14 place-items-center rounded-xl border border-emerald-400/20 bg-emerald-400/10 text-emerald-300 shadow-lg shadow-emerald-500/10">
                    <FolderUp size={24} />
                  </span>
                  <p className="mt-4 text-sm font-bold text-white">
                    Drop {mode === 'single' ? 'an RFP PDF' : 'up to 10 RFPs'} here
                  </p>
                  <p className="mt-1 text-xs text-zinc-400">or click to browse from your computer</p>
                </div>
              </div>

              {files.length > 0 && (
                <div className="mx-6 mb-6 flex flex-wrap items-center gap-3">
                  {files.map((file) => (
                    <span
                      key={file.name}
                      className="inline-flex max-w-full items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3.5 py-2 text-xs text-zinc-300"
                    >
                      <FileText size={14} className="shrink-0 text-emerald-400" />
                      <span className="max-w-[200px] truncate font-medium">{file.name}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setFiles(files.filter((x) => x !== file));
                        }}
                        className="text-zinc-500 hover:text-rose-400 transition"
                      >
                        <X size={14} />
                      </button>
                    </span>
                  ))}
                  <ShimmerButton onClick={run} className="!py-2.5 !px-5 text-xs">
                    {mode === 'single' ? 'Evaluate RFP' : 'Start Batch Process'}
                    <Upload size={14} className="ml-2" />
                  </ShimmerButton>
                </div>
              )}
            </div>

            {/* Pipeline Trace Card */}
            <div className="panel p-6">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div>
                  <p className="text-sm font-extrabold text-white">Pipeline Execution</p>
                  <p className="mt-0.5 text-xs text-zinc-400">Deterministic & RAG Trace</p>
                </div>
                <span className="font-mono text-[10px] font-bold text-emerald-400 bg-emerald-400/10 px-2.5 py-1 rounded border border-emerald-400/20">
                  {stage >= 0 ? `${Math.min(stage + 1, 5)} / 5 COMPLETED` : 'STANDBY READY'}
                </span>
              </div>

              <ol className="mt-6 space-y-4">
                {pipeline.map(([num, title, engine], i) => (
                  <li key={num} className="flex items-start gap-3.5">
                    <span
                      className={`grid h-7 w-7 shrink-0 place-items-center rounded-full border font-mono text-[10px] font-bold transition duration-300 ${
                        stage > i
                          ? 'border-emerald-400/40 bg-emerald-400/20 text-emerald-300'
                          : stage === i
                          ? 'border-emerald-400 bg-emerald-400/20 text-emerald-200 shadow-[0_0_16px_rgba(52,211,153,0.3)] animate-pulse'
                          : 'border-white/10 text-zinc-600 bg-white/[0.02]'
                      }`}
                    >
                      {stage > i ? '✓' : num}
                    </span>
                    <div className="pt-0.5">
                      <p className={`text-xs font-bold transition ${stage >= i ? 'text-zinc-200' : 'text-zinc-500'}`}>
                        {title}
                      </p>
                      <p className="mt-0.5 font-mono text-[10px] text-zinc-500">{engine}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </section>

          {/* Evaluation Result Banner */}
          {result?.status === 'success' && (
            <MovingBorderCard duration={5000} className="p-6">
              <div className="flex flex-col gap-6 sm:flex-row sm:items-center">
                <PwinGauge
                  score={result.data.evaluation.win_probability_score}
                  decision={result.data.evaluation.decision}
                />
                <div className="flex-1">
                  <p className="eyebrow text-emerald-400">Evaluation Finished</p>
                  <div className="mt-2 flex flex-wrap items-center gap-3">
                    <h2 className="text-xl font-extrabold tracking-tight text-white">
                      {result.data.issuing_authority || result.data.tender_id || 'Tender evaluation'}
                    </h2>
                    <DecisionBadge decision={result.data.evaluation.decision} />
                  </div>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-300">
                    {result.data.evaluation.recommendation_summary}
                  </p>
                  {result.data.confidence_score < 0.85 && (
                    <div className="mt-3 flex items-center gap-2 rounded-lg bg-amber-400/10 border border-amber-400/20 px-3 py-2 text-xs text-amber-200">
                      <ShieldAlert size={15} className="shrink-0 text-amber-400" />
                      <span>Extraction confidence is below 85%. Please review source clauses.</span>
                    </div>
                  )}
                </div>
                <div className="flex flex-wrap gap-3">
                  <Link
                    to="/tenders/$id"
                    params={{ id: result.data.analysis_id }}
                    className="button-signal"
                  >
                    View Dossier <ArrowRight size={14} className="ml-1" />
                  </Link>
                  <Link
                    to="/proposals"
                    search={{ tender: result.data.analysis_id } as any}
                    className="button-quiet"
                  >
                    Draft Proposal
                  </Link>
                </div>
              </div>
            </MovingBorderCard>
          )}

          {/* Recent Evaluations Table */}
          <section className="mt-8">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="eyebrow text-emerald-400">Recent Dossiers</p>
                <h2 className="mt-1 text-lg font-extrabold tracking-tight text-white">
                  Your Latest Evaluations
                </h2>
              </div>
              <Link to="/tenders" className="text-xs font-bold text-emerald-400 hover:text-emerald-300">
                View all dossiers →
              </Link>
            </div>

            {tenders.length === 0 ? (
              <div className="rounded-xl border border-white/[0.08] bg-[#121722]/40 p-8 text-center text-sm text-zinc-500">
                No tender evaluations yet. Upload an RFP above to create your first evidence-backed dossier.
              </div>
            ) : (
              <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-[#121722]/50 backdrop-blur-xl">
                <table className="w-full min-w-[650px] text-left">
                  <thead className="bg-white/[0.02] border-b border-white/[0.06] font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-400">
                    <tr>
                      <th className="px-6 py-3.5">Tender Document</th>
                      <th className="px-6 py-3.5">Verdict</th>
                      <th className="px-6 py-3.5">PWin Score</th>
                      <th className="px-6 py-3.5">Evaluated On</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.06]">
                    {tenders.slice(0, 5).map((t) => (
                      <tr
                        key={t.id}
                        className="text-sm transition duration-150 hover:bg-white/[0.03]"
                      >
                        <td className="px-6 py-4">
                          <Link
                            to="/tenders/$id"
                            params={{ id: t.id }}
                            className="font-bold text-zinc-200 hover:text-emerald-400 transition"
                          >
                            {t.filename}
                          </Link>
                          <p className="mt-0.5 text-xs text-zinc-400">
                            {t.issuing_authority || 'Issuing authority not extracted'}
                          </p>
                        </td>
                        <td className="px-6 py-4">
                          <DecisionBadge decision={t.decision} />
                        </td>
                        <td className="px-6 py-4 font-mono font-bold text-emerald-400">
                          {t.win_probability_score ?? '—'}
                          {t.win_probability_score !== null ? '%' : ''}
                        </td>
                        <td className="px-6 py-4 text-xs text-zinc-400">
                          {formatDate(t.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
