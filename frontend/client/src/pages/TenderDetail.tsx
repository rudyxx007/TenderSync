/** Signal Room design system: the dossier stacks decision evidence in an ordered, exportable executive narrative. */
import { Link, useNavigate, useParams } from '@tanstack/react-router';
import { useEffect, useMemo, useState } from 'react';
import { 
  CalendarDays, 
  CheckCircle2, 
  ChevronDown, 
  Download, 
  FileDown, 
  FileText, 
  ShieldAlert, 
  Sparkles, 
  XCircle,
  ArrowLeft,
  Scale,
  Radar as RadarIcon
} from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { toast } from 'sonner';
import { apiDownload, apiFetch } from '@/lib/api';
import type { TenderAnalysisDetail } from '@/types/api';
import { DecisionBadge, PwinGauge } from '@/components/PwinGauge';
import { ErrorBlock, LoadingBlock, PageHeader, formatDate } from '@/components/Workspace';
import { CardSpotlight } from '@/components/ui/CardSpotlight';
import { MovingBorderCard } from '@/components/ui/MovingBorderCard';
import { ShimmerButton } from '@/components/ui/ShimmerButton';

const pretty = (key: string) => key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());

export default function TenderDetail() {
  const { id } = useParams({ from: '/_authenticated/tenders/$id' });
  const navigate = useNavigate();
  const [data, setData] = useState<TenderAnalysisDetail | null>(null);
  const [error, setError] = useState('');
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    apiFetch<TenderAnalysisDetail>(`/api/tenders/${id}`)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [id]);

  const ev = data?.evaluation_data;
  const extracted = data?.extracted_data;

  const radar = useMemo(
    () =>
      ev
        ? Object.entries(ev.factor_scores?.scores || {}).map(([key, value]) => ({
            name: pretty(key),
            score: value * 20,
            raw: value,
            weight: ev.factor_scores.weights?.[key],
          }))
        : [],
    [ev]
  );

  const calendar = () => {
    if (!extracted?.submission_deadline) return toast.error('No submission deadline was extracted.');
    apiDownload(
      `/api/generate-calendar?deadline_string=${encodeURIComponent(extracted.submission_deadline)}`,
      `${data?.filename || 'tender'}-deadline.ics`,
      'POST'
    ).catch((e) => toast.error(e.message));
  };

  if (error) return <ErrorBlock text={error} />;
  if (!data || !ev || !extracted) return <LoadingBlock label="Opening executive dossier" />;

  return (
    <div className="space-y-8 pb-16">
      <PageHeader
        backTo="/tenders"
        eyebrow="Executive dossier"
        title={data.filename}
        text={`${extracted.issuing_authority || 'Issuing authority not extracted'} · ${extracted.tender_id || 'No tender reference extracted'}`}
        action={
          <div className="flex flex-wrap items-center gap-3">
            <button onClick={calendar} className="button-quiet">
              <CalendarDays size={15} /> Calendar .ICS
            </button>
            <ShimmerButton
              onClick={() =>
                apiDownload(
                  `/api/tenders/${id}/export-pdf`,
                  `${data.filename.replace(/\.[^.]+$/, '')}-dossier.pdf`
                ).catch((e) => toast.error(e.message))
              }
              className="!py-2.5 !px-4 text-xs font-bold"
            >
              <FileDown size={15} className="mr-1.5" /> Executive PDF
            </ShimmerButton>
          </div>
        }
      />

      {/* Top Banner: Verdict Gauge & Strategic Recommendation */}
      <MovingBorderCard duration={6000} className="p-8">
        <div className="grid gap-8 lg:grid-cols-[220px_1fr] lg:items-center">
          <div className="grid place-items-center border-b border-white/[0.08] pb-6 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-8">
            <PwinGauge score={ev.win_probability_score} decision={ev.decision} size={180} />
            <DecisionBadge decision={ev.decision} className="mt-4" />
          </div>
          <div>
            <p className="eyebrow text-emerald-400">Strategic Verdict</p>
            <h2 className="mt-2 text-2xl font-extrabold tracking-tight text-white sm:text-3xl">
              {ev.recommendation_summary}
            </h2>
            <p className="mt-3 text-sm leading-7 text-zinc-300">{ev.rationale}</p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <Link
                to="/proposals"
                search={{ tender: id } as any}
                className="button-signal !py-2.5 !px-5"
              >
                <Sparkles size={15} className="mr-1.5" /> Draft Proposal with AI
              </Link>
              <span className="inline-flex items-center rounded-lg border border-white/10 bg-white/[0.03] px-3.5 py-2 font-mono text-[11px] text-zinc-400">
                EVALUATED {formatDate(data.created_at)}
              </span>
            </div>
          </div>
        </div>
      </MovingBorderCard>

      {/* Low-confidence Warning */}
      {extracted.confidence_score < 0.85 && (
        <div className="flex gap-3.5 rounded-xl border border-amber-400/30 bg-amber-400/10 p-5 text-sm text-amber-100 backdrop-blur-md">
          <ShieldAlert size={20} className="shrink-0 text-amber-400 mt-0.5" />
          <div>
            <strong className="font-bold text-amber-300">Human validation required:</strong> Extraction confidence is{' '}
            {Math.round(extracted.confidence_score * 100)}%, below the 85% safety threshold. Please verify mandatory clauses and numerical deliverables before capture commitment.
          </div>
        </div>
      )}

      {/* Middle Row: Hard Gates & 7-Dimension Radar Fit */}
      <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        {/* Phase 1: Hard Gates */}
        <div className="panel overflow-hidden">
          <div className="border-b border-white/[0.08] p-6">
            <p className="eyebrow text-emerald-400">Phase One / Gatekeeper</p>
            <h2 className="mt-1 text-lg font-extrabold tracking-tight text-white">
              Non-Negotiable Hard Gates
            </h2>
          </div>
          <div className="divide-y divide-white/[0.06]">
            {(ev.hard_gates || []).length === 0 ? (
              <p className="p-6 text-sm text-zinc-500">No hard gate findings returned for this dossier.</p>
            ) : (
              ev.hard_gates.map((gate, i) => (
                <div key={`${gate.gate_name}-${i}`} className="flex gap-4 p-5 transition hover:bg-white/[0.02]">
                  <span
                    className={`mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full border ${
                      gate.passed
                        ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300'
                        : 'border-rose-400/30 bg-rose-400/10 text-rose-300'
                    }`}
                  >
                    {gate.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
                  </span>
                  <div>
                    <div className="flex flex-wrap items-center gap-2.5">
                      <p className="text-sm font-bold text-zinc-100">{gate.gate_name}</p>
                      {gate.severity && (
                        <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-400 bg-white/[0.04] px-2 py-0.5 rounded border border-white/[0.06]">
                          {gate.severity}
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs leading-5 text-zinc-400">{gate.detail}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* 7-Dimension Fit Radar Chart */}
        <div className="panel p-6">
          <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
            <div>
              <p className="eyebrow text-emerald-400">Phase Two / Fit Profile</p>
              <h2 className="mt-1 text-lg font-extrabold tracking-tight text-white">7 Scoring Dimensions</h2>
            </div>
            <span className="font-mono text-[10px] font-bold text-emerald-400 bg-emerald-400/10 px-2.5 py-1 rounded border border-emerald-400/20">
              RADAR FIT
            </span>
          </div>
          <div className="mt-4 h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radar} outerRadius="70%">
                <PolarGrid stroke="rgba(255,255,255,0.12)" />
                <PolarAngleAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 600 }} />
                <Radar dataKey="score" stroke="#34d399" fill="#34d399" fillOpacity={0.25} />
                <Tooltip
                  contentStyle={{
                    background: '#0E131F',
                    border: '1px solid rgba(255,255,255,0.12)',
                    borderRadius: 8,
                    fontSize: 12,
                    color: '#fff'
                  }}
                  formatter={(value: any) => [`${value}%`, 'Dimension Score']}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Bottom Row: Dimension Accordions & Source Evidence */}
      <section className="grid gap-6 xl:grid-cols-2">
        {/* Dimension Assessment Accordions */}
        <div className="panel overflow-hidden">
          <div className="border-b border-white/[0.08] p-6">
            <p className="eyebrow text-emerald-400">Scoring Evidence</p>
            <h2 className="mt-1 text-lg font-extrabold tracking-tight text-white">
              Dimension-by-Dimension Breakdown
            </h2>
          </div>
          <div className="divide-y divide-white/[0.06]">
            {Object.entries(ev.factor_scores?.scores || {}).map(([key, score]) => (
              <div key={key} className="transition hover:bg-white/[0.01]">
                <button
                  onClick={() => setOpen(open === key ? null : key)}
                  className="flex w-full items-center justify-between p-5 text-left"
                >
                  <div className="flex items-center gap-3.5">
                    <span className="font-mono text-xs font-bold text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
                      {score}/5
                    </span>
                    <span className="text-sm font-bold text-zinc-200">{pretty(key)}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-[10px] font-semibold text-zinc-500">
                      {ev.factor_scores.weights?.[key] || 0}% WT
                    </span>
                    <ChevronDown
                      size={16}
                      className={`text-zinc-500 transition duration-200 ${open === key ? 'rotate-180 text-emerald-400' : ''}`}
                    />
                  </div>
                </button>
                {open === key && (
                  <p className="border-t border-white/[0.06] bg-black/20 p-5 text-xs leading-6 text-zinc-300">
                    {ev.factor_scores.details?.[key] || 'No additional reasoning returned for this dimension.'}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Source Material Summary & Mitigations */}
        <div className="space-y-6">
          {ev.decision === 'CONDITIONAL' && (
            <div className="panel border-amber-400/30 bg-amber-400/[0.03] p-6">
              <p className="eyebrow text-amber-300">Mitigation Roadmap</p>
              <h2 className="mt-1 text-lg font-extrabold tracking-tight text-white">
                Actions Required Before Bid Submission
              </h2>
              <ul className="mt-4 space-y-3">
                {(ev.mitigations || []).map((m, i) => (
                  <li key={i} className="flex items-start gap-3 text-xs leading-6 text-zinc-300">
                    <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400 shadow-[0_0_8px_#fbbf24]" />
                    <span>{m}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="panel p-6">
            <p className="eyebrow text-emerald-400">Document Extraction</p>
            <h2 className="mt-1 text-lg font-extrabold tracking-tight text-white">Source Material Summary</h2>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <DataCell label="Submission Deadline" value={extracted.submission_deadline} />
              <DataCell label="Estimated Value" value={extracted.estimated_value_or_budget} />
            </div>
            <div className="mt-5 space-y-4">
              <List label="Key Deliverables" items={extracted.key_deliverables} />
              <List label="Mandatory Compliance Criteria" items={extracted.mandatory_compliance_criteria} />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function DataCell({ label, value }: { label: string; value?: string }) {
  return (
    <div className="rounded-lg border border-white/[0.08] bg-black/25 p-3.5">
      <p className="eyebrow text-zinc-500">{label}</p>
      <p className="mt-1.5 text-sm font-bold text-zinc-200">{value || 'Not specified'}</p>
    </div>
  );
}

function List({ label, items }: { label: string; items?: string[] }) {
  return (
    <div>
      <p className="eyebrow text-zinc-500">{label}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {items?.length ? (
          items.map((x, i) => (
            <span
              key={`${x}-${i}`}
              className="rounded-md border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-xs text-zinc-300"
            >
              {typeof x === 'string' ? x : JSON.stringify(x)}
            </span>
          ))
        ) : (
          <span className="text-xs text-zinc-500">No items extracted.</span>
        )}
      </div>
    </div>
  );
}
