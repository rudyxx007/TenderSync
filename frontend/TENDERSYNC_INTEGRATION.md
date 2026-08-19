# TenderSync Frontend Integration

The frontend is prepared for the FastAPI backend in the connected TenderSync repository and Supabase authentication.

| Project setting | Purpose |
| --- | --- |
| `VITE_API_URL` | Public base URL of the deployed TenderSync FastAPI service, without a trailing slash. |
| `VITE_SUPABASE_URL` | Supabase project URL used for browser authentication. |
| `VITE_SUPABASE_ANON_KEY` | Supabase browser-safe anonymous key. |

The frontend is currently configured to use `https://wrote-replacement-surveys-jan.trycloudflare.com` as its fallback API base URL. A supplied `VITE_API_URL` value takes precedence, which allows the backend address to be changed without modifying the API client. The FastAPI service must allow the frontend's published origin through its CORS policy. The browser forwards the Supabase access token as a bearer token on each TenderSync API request.

> This is a `trycloudflare.com` quick tunnel. If its address changes when the tunnel restarts, update `VITE_API_URL` in the project configuration or replace the fallback URL in `client/src/lib/api.ts` before rebuilding the frontend.
