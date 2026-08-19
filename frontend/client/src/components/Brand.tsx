/** Signal Room design system: the official TenderSync logo brand anchor. */
import { Link } from '@tanstack/react-router';
import { cn } from '@/lib/utils';

export function Brand({ className, compact = false }: { className?: string; compact?: boolean }) {
  return (
    <Link 
      to="/" 
      className={cn('inline-flex flex-col items-start focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400', className)} 
      aria-label="TenderSync home"
    >
      <img 
        src="/tendersync_logo.png" 
        alt="TenderSync" 
        className={cn('w-auto shrink-0 object-contain object-left', compact ? 'h-9' : 'h-12 sm:h-14')} 
      />
      {!compact && (
        <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-zinc-400 -mt-1 pl-0.5">
          DECISION INTEL
        </span>
      )}
    </Link>
  );
}
