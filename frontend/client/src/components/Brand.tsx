/** Signal Room design system: the official TenderSync wordmark is the sole brand anchor across public and workspace surfaces. */
import { Link } from '@tanstack/react-router';
import { cn } from '@/lib/utils';
export function Brand({ className, compact = false }: { className?: string; compact?: boolean }) {
  return <Link to="/" className={cn('inline-flex items-center gap-2.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400', className)} aria-label="TenderSync home">
    <img src="/manus-storage/tendersync-official-logo_20a2fe21.png" alt="TenderSync" className={cn('w-auto shrink-0 object-contain object-left', compact ? 'h-9' : 'h-12')} />
  </Link>;
}
