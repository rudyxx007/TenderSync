/** Signal Room design system: a calm public threshold gives way to the focused application rail. */
import { RouterProvider } from '@tanstack/react-router';
import { Toaster } from '@/components/ui/sonner';
import { TooltipProvider } from '@/components/ui/tooltip';
import ErrorBoundary from './components/ErrorBoundary';
import { ThemeProvider } from './contexts/ThemeContext';
import { router } from './router';
export default function App() { return <ErrorBoundary><ThemeProvider defaultTheme="dark" switchable><TooltipProvider><Toaster theme="dark" richColors position="top-right" /><RouterProvider router={router} /></TooltipProvider></ThemeProvider></ErrorBoundary>; }
