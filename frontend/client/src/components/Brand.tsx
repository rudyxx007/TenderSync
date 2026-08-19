/** Signal Room design system: the official TenderSync wordmark and brand anchor. */
import { Link } from '@tanstack/react-router';
import { cn } from '@/lib/utils';

export function Brand({ className, compact = false }: { className?: string; compact?: boolean }) {
  return (
    <Link 
      to="/" 
      className={cn('inline-flex items-center gap-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400', className)} 
      aria-label="TenderSync home"
    >
      <img 
        src="/tendersync_logo.png" 
        alt="TenderSync Logo" 
        className={cn('w-auto shrink-0 object-contain rounded-md', compact ? 'h-8' : 'h-10')} 
      />
      <div className="flex flex-col">
        <span className={cn('font-extrabold tracking-tight text-white leading-none', compact ? 'text-sm' : 'text-base')}>
          Tender<span className="text-emerald-400">Sync</span>
        </span>
        {!compact && (
          <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-zinc-500 mt-0.5">
            Decision Intel
          </span>
        )}
      </div>
    </Link>
  );
}
