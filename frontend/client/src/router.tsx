/** Signal Room design system: an intentional route map separates public clarity from protected decision work. */
import { createRootRoute, createRoute, createRouter, Outlet, redirect } from '@tanstack/react-router';
import { supabase, isSupabaseConfigured } from '@/lib/supabase';
import { apiFetch } from '@/lib/api';
import type { ProfileStatusResponse } from '@/types/api';
import Landing from '@/pages/Landing';
import HowItWorks from '@/pages/HowItWorks';
import About from '@/pages/About';
import Login from '@/pages/Login';
import Signup from '@/pages/Signup';
import Join from '@/pages/Join';
import Onboarding from '@/pages/Onboarding';
import { AppShell } from '@/components/AppShell';
import Dashboard from '@/pages/Dashboard';
import Tenders from '@/pages/Tenders';
import TenderDetail from '@/pages/TenderDetail';
import Compare from '@/pages/Compare';
import Proposals from '@/pages/Proposals';
import ProposalEditor from '@/pages/ProposalEditor';
import Discovery from '@/pages/Discovery';
import Batches from '@/pages/Batches';
import ProfileSettings from '@/pages/ProfileSettings';
import OrganizationSettings from '@/pages/OrganizationSettings';
import NotFound from '@/pages/NotFound';

const rootRoute = createRootRoute({ component: () => <Outlet /> });

// Public Routes
const indexRoute = createRoute({ getParentRoute: () => rootRoute, path: '/', component: Landing });
const howItWorksRoute = createRoute({ getParentRoute: () => rootRoute, path: '/how-it-works', component: HowItWorks });
const aboutRoute = createRoute({ getParentRoute: () => rootRoute, path: '/about', component: About });
const loginRoute = createRoute({ getParentRoute: () => rootRoute, path: '/login', component: Login });
const signupRoute = createRoute({ getParentRoute: () => rootRoute, path: '/signup', component: Signup });
const joinRoute = createRoute({ getParentRoute: () => rootRoute, path: '/join', component: Join });
const onboardingRoute = createRoute({ getParentRoute: () => rootRoute, path: '/onboarding', component: Onboarding });

// Authenticated Routes
const protectedRoute = createRoute({ 
  getParentRoute: () => rootRoute, 
  id: '_authenticated', 
  beforeLoad: async ({ location }) => { 
    if (!isSupabaseConfigured) return { profile: null }; 
    const { data: { session } } = await supabase.auth.getSession(); 
    if (!session) throw redirect({ to: '/login', search: { redirect: location.href } }); 
    try { 
      const status = await apiFetch<ProfileStatusResponse>('/api/profile/status'); 
      if (!status.can_use_app && location.pathname !== '/onboarding') throw redirect({ to: '/onboarding' }); 
      return { profile: status.profile }; 
    } catch (error) { 
      if (error instanceof Response) throw error; 
      throw redirect({ to: '/onboarding' }); 
    } 
  }, 
  component: AppShell 
});

const dashboardRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/dashboard', component: Dashboard });
const tendersRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/tenders', component: Tenders });
const tenderDetailRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/tenders/$id', component: TenderDetail });
const compareRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/tenders/compare', component: Compare });
const proposalsRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/proposals', component: Proposals });
const proposalEditorRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/proposals/$id', component: ProposalEditor });
const discoveryRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/discovery', component: Discovery });
const batchesRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/batches', component: Batches });
const profileRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/settings/profile', component: ProfileSettings });
const organizationRoute = createRoute({ getParentRoute: () => protectedRoute, path: '/settings/organization', component: OrganizationSettings });
const notFoundRoute = createRoute({ getParentRoute: () => rootRoute, path: '$', component: NotFound });

const routeTree = rootRoute.addChildren([
  indexRoute,
  howItWorksRoute,
  aboutRoute,
  loginRoute,
  signupRoute,
  joinRoute,
  onboardingRoute,
  protectedRoute.addChildren([
    dashboardRoute,
    tendersRoute,
    tenderDetailRoute,
    compareRoute,
    proposalsRoute,
    proposalEditorRoute,
    discoveryRoute,
    batchesRoute,
    profileRoute,
    organizationRoute
  ]),
  notFoundRoute
]);

export const router = createRouter({ routeTree, defaultPreload: 'intent', defaultNotFoundComponent: NotFound });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
