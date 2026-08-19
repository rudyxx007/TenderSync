/** Signal Room design system: Signal Arc is the dominant, animated decision marker. */
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { DecisionType } from '@/types/api';
const color = (decision?: DecisionType) => decision === 'NO-BID' ? '#fb7185' : decision === 'CONDITIONAL' ? '#fbbf24' : '#34d399';
export function PwinGauge({ score, decision, size = 132, label = 'PWin' }: { score: number; decision?: DecisionType; size?: number; label?: string }) {
  const stroke = 10; const r = (size - stroke) / 2; const c = 2 * Math.PI * r; const safeScore = Math.max(0, Math.min(100, score));
  return <div className="relative grid place-items-center" style={{ width: size, height: size }}>
    <svg width={size} height={size} className="-rotate-90" aria-label={`${safeScore}% probability of win`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeWidth={stroke} className="text-white/[.07]" />
      <motion.circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color(decision)} strokeWidth={stroke} strokeLinecap="round" strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: c * (1 - safeScore / 100) }} transition={{ duration: 1.25, ease: [0.23, 1, 0.32, 1] }} style={{ filter: `drop-shadow(0 0 7px ${color(decision)}80)` }} />
    </svg>
    <div className="absolute inset-0 grid place-items-center text-center"><div><strong className="block text-2xl font-extrabold tracking-[-0.08em] text-white">{safeScore}</strong><span className="font-mono text-[9px] uppercase tracking-[.16em] text-zinc-500">{label}</span></div></div>
  </div>;
}
export function DecisionBadge({ decision, className }: { decision: DecisionType | null | undefined; className?: string }) { if (!decision) return null; const styles = decision === 'BID' ? 'border-emerald-400/25 bg-emerald-400/10 text-emerald-300' : decision === 'CONDITIONAL' ? 'border-amber-400/25 bg-amber-400/10 text-amber-200' : 'border-rose-400/25 bg-rose-400/10 text-rose-300'; return <span className={cn('inline-flex items-center gap-1.5 rounded-sm border px-2 py-1 font-mono text-[10px] font-medium tracking-[.12em]', styles, className)}><i className="h-1.5 w-1.5 rounded-full bg-current" />{decision}</span>; }
