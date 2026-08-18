<div align="center">

  [![Stars][stars-shield]][stars-url]
  [![Forks][forks-shield]][forks-url]
  [![Issues][issues-shield]][issues-url]
  [![LinkedIn][linkedin-shield]][linkedin-url]

  <br />

  <a href="https://github.com/rudyxx007/TenderSync">
    <img src="assets/tendersync_logo.png" alt="TenderSync" width="250" height="250" />
  </a>

  <h1>TenderSync</h1>

  <h3><strong>Autonomous Tender Intelligence & AI Proposal Engine</strong></h3>
  <h3><sub>The engine behind your next government contract.</sub></h3>

  <br />

  <a href="#-deployment"><kbd>&nbsp;&nbsp;🌐 Live Demo&nbsp;&nbsp;</kbd></a>&ensp;
  <a href="https://github.com/rudyxx007/TenderSync/issues/new?labels=bug&title=Bug%3A+"><kbd>&nbsp;&nbsp;🐛 Report Bug&nbsp;&nbsp;</kbd></a>&ensp;
  <a href="https://github.com/rudyxx007/TenderSync/issues/new?labels=enhancement&title=Feature%3A+"><kbd>&nbsp;&nbsp;💡 Request Feature&nbsp;&nbsp;</kbd></a>

  <br /><br />

  <img alt="Status" src="https://img.shields.io/badge/Status-Under_Active_Development-10B981?style=for-the-badge&labelColor=0D1117" />

</div>

<br />

---

<details open>
<summary><h2>📋 Table of Contents</h2></summary>

- [About](#-about)
- [The Problem](#-the-problem)
- [How It Works](#%EF%B8%8F-how-it-works)
- [Key Features](#-key-features)
- [Screenshots](#-screenshots)
- [Built With](#%EF%B8%8F-built-with)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Roadmap](#%EF%B8%8F-roadmap)
- [Known Limitations](#%EF%B8%8F-known-limitations)
- [License](#-license)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)

</details>

<br />

---

## 💡 About

> [!NOTE]
> 🖼️ App screenshots will be added after the frontend is deployed.

**TenderSync** is an enterprise-grade B2B SaaS platform that helps companies evaluate government and corporate RFP (Request for Proposal) tenders using AI. It combines a **Retrieval-Augmented Generation (RAG) pipeline** with an industry-aligned **Bid/No-Bid decision engine** to transform days of manual tender review into a **2-minute, AI-powered decision**, personalized to each company's unique capabilities.

> _Think of it as a **Bloomberg Terminal for Tenders**: one upload, one score, one decision._

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 🎯 The Problem

Government and enterprise bidding is notoriously high-friction, error-prone, and slow. B2B companies lose **thousands of hours and millions in revenue** chasing tenders they don't qualify for, missing critical deadlines, and drafting 50-page proposals from scratch under pressure.

| Lifecycle Stage | Traditional Pain Point | Business Impact |
| :--- | :--- | :--- |
| **🔍 Discovery** | Manually searching fragmented public portals (CPPP, GeM) | Missed deadlines & high opportunity scouting overhead |
| **📄 Document Ingestion** | Reading 200+ page legalistic RFP PDFs manually | **3–5 business days** spent per evaluation |
| **❌ Compliance Gating** | Overlooking hidden mandatory certs or eligibility criteria | Instant disqualification & wasted bid prep costs |
| **🎲 Bid Qualification** | Bidding based on executive gut feeling rather than data | Low win rates & exhausted capture budgets |
| **⚖️ Trade-off Analysis** | Inability to objectively compare concurrent tender opportunities | Misallocated team resources on low-yield bids |
| **✍️ Proposal Drafting** | Writing technical responses & compliance matrices from scratch | Grueling 40-hour writing cycles & rushed submissions |
| **🏢 Team Collaboration** | Disconnected bids with no shared institutional memory | Siloed teams & repeated historical mistakes |

<br />

**TenderSync eliminates every bottleneck across the RFP lifecycle.**  
📡 **Auto-discover** live market tenders ➔ 🧠 **Extract** 200+ page PDFs in seconds ➔ 🚦 **Screen** hard-gate deal-breakers ➔ 📊 **Score** win probability across 7 dimensions ➔ 🤖 **Draft** tailored proposals with multi-agent AI.

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## ⚙️ How It Works

```mermaid
graph TD
    A["📄 Upload RFP PDF"] --> B["🔍 Docling Extraction"]
    B --> C["✂️ Intelligent Chunking"]
    C --> D["🧠 Hugging Face BGE-M3 Embeddings"]
    D --> E["💾 Supabase pgvector"]
    E --> F["🤖 Groq · GPT-OSS-120B<br/>Structured Extraction"]
    F --> G["⚖️ Hybrid Bid/No-Bid Engine"]
    G --> H{"Decision"}
    H -->|PWin ≥ 65| I["✅ BID"]
    H -->|40 ≤ PWin < 65| J["⚠️ CONDITIONAL"]
    H -->|PWin < 40| K["❌ NO-BID"]

    style I fill:#10B981,stroke:#059669,color:#fff
    style J fill:#F59E0B,stroke:#D97706,color:#fff
    style K fill:#EF4444,stroke:#DC2626,color:#fff
```

### The 4-Phase Evaluation Pipeline

|    Phase    | Description                                                                      | Method               |
| :---------: | :------------------------------------------------------------------------------- | :------------------- |
| **A** | **Hard Gate Checks**: deal killers like value mismatch or missing certs    | Deterministic rules  |
| **B** | **Numeric/Keyword Scoring**: capability overlap, cert matches, budget fit  | Algorithmic scoring  |
| **C** | **LLM Subjective Scoring**: competitive landscape, strategic alignment     | Groq GPT-OSS-120B    |
| **D** | **Weighted PWin Calculation**: 7 dimensions aggregated into a 0–100 score | Weighted aggregation |

### The 7 Scoring Dimensions

| Dimension             | Weight | What It Measures                                      |
| :-------------------- | :----: | :---------------------------------------------------- |
| Capability Fit        |  20%  | How well core capabilities match the RFP requirements |
| Compliance Readiness  |  10%  | Certification and regulatory coverage                 |
| Commercial Viability  |  15%  | Budget alignment and contract value thresholds        |
| Past Performance      |  15%  | Relevant sector experience                            |
| Competitive Landscape |  10%  | Estimated competition intensity                       |
| Strategic Alignment   |  15%  | Alignment with company's strategic focus areas        |
| Delivery Feasibility  |  15%  | Timeline, team capacity, and geographic coverage      |

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## ✨ Key Features

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🧠 AI-Powered Extraction</h3>
      <p>Leverages <b>Groq GPT-OSS-120B</b> and <b>Hugging Face BGE-M3</b> embeddings within a full RAG pipeline to extract tender IDs, issuing authorities, deadlines, budgets, deliverables, and compliance criteria from raw PDF documents.</p>
    </td>
    <td width="50%" valign="top">
      <h3>🚦 Hybrid Bid/No-Bid Engine</h3>
      <p>A 4-phase engine combining <b>deterministic hard-gate checks</b> with <b>LLM-scored subjective dimensions</b> across 7 weighted factors, producing a mathematically grounded recommendation with full factor breakdowns.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🏢 Multi-Tenant Profile Gating</h3>
      <p>Each user represents one company. Mandatory onboarding enforces completion of company name, certifications, and capabilities <b>before</b> any analysis is allowed, ensuring every evaluation is personalized.</p>
    </td>
    <td width="50%" valign="top">
      <h3>📊 Decision Dashboard</h3>
      <p>A dark-themed dashboard with a circular <b>PWin gauge</b>, color-coded decision indicators (emerald / amber / red), radar charts for factor breakdown, and hard-gate status pills.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🌍 Market Tenders Discovery</h3>
      <p>Automated scraping of public portals (e.g., CPPP) to bring relevant market tenders directly into your dashboard. Evaluate public tenders with one click.</p>
    </td>
    <td width="50%" valign="top">
      <h3>⚖️ Side-by-Side Comparison</h3>
      <p>Compare two analyzed tenders side-by-side using Groq LLM to evaluate budget, deadlines, compliance, and deliverable differences.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🤖 AI Proposal Writer (LangGraph)</h3>
      <p>A multi-agent orchestrator (Analyst, Researcher, Writer, Reviewer) that automatically generates a complete proposal draft based on the tender analysis and your company capabilities.</p>
    </td>
    <td width="50%" valign="top">
      <h3>📅 Calendar Export & PDF Reports</h3>
      <p>Generate downloadable <code>.ics</code> calendar files for deadlines, and export beautiful PDF reports of your tender evaluations using WeasyPrint and Plotly.</p>
    </td>
  </tr>
</table>

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 📸 Screenshots

> [!NOTE]
> Screenshots will be added after the frontend is deployed on Vercel.

| Page              | Description                                          |
| :---------------- | :--------------------------------------------------- |
| Landing Page      | Animated hero with gradient mesh, feature bento grid |
| Login / Signup    | Glassmorphic auth cards over gradient background     |
| Onboarding Wizard | 3-step company profile setup flow                    |
| Dashboard         | PDF upload zone, PWin result cards, recent analyses  |
| Tender Detail     | Full evaluation breakdown with gauge, gates, factors |
| Tender History    | Sortable analysis history table                      |

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 🛠️ Built With

<div align="center">
<br />
<table>
  <tr>
    <td align="center" width="96">
      <a href="https://nextjs.org/"><img src="https://skillicons.dev/icons?i=nextjs" width="48" height="48" alt="Next.js" /></a>
      <br /><sub><b>Next.js 15</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://www.typescriptlang.org/"><img src="https://skillicons.dev/icons?i=ts" width="48" height="48" alt="TypeScript" /></a>
      <br /><sub><b>TypeScript</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://tailwindcss.com/"><img src="https://skillicons.dev/icons?i=tailwind" width="48" height="48" alt="Tailwind" /></a>
      <br /><sub><b>Tailwind CSS</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://www.python.org/"><img src="https://skillicons.dev/icons?i=python" width="48" height="48" alt="Python" /></a>
      <br /><sub><b>Python 3.10+</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://fastapi.tiangolo.com/"><img src="https://skillicons.dev/icons?i=fastapi" width="48" height="48" alt="FastAPI" /></a>
      <br /><sub><b>FastAPI</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://supabase.com/"><img src="https://skillicons.dev/icons?i=supabase" width="48" height="48" alt="Supabase" /></a>
      <br /><sub><b>Supabase</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="96">
      <a href="https://vercel.com/"><img src="https://skillicons.dev/icons?i=vercel" width="48" height="48" alt="Vercel" /></a>
      <br /><sub><b>Vercel</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://www.postgresql.org/"><img src="https://skillicons.dev/icons?i=postgres" width="48" height="48" alt="PostgreSQL" /></a>
      <br /><sub><b>PostgreSQL</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://ui.shadcn.com/"><img src="https://cdn.simpleicons.org/shadcnui/ffffff" width="48" height="48" alt="shadcn/ui" /></a>
      <br /><sub><b>shadcn/ui</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://docs.pydantic.dev/"><img src="https://cdn.simpleicons.org/pydantic/E92063" width="48" height="48" alt="Pydantic" /></a>
      <br /><sub><b>Pydantic</b></sub>
    </td>
    <td align="center" width="96">
      <a href="https://groq.com/"><img src="https://cdn.simpleicons.org/groq/F55036" width="48" height="48" alt="Groq GPT-OSS-120B" /></a>
      <br /><sub><b>GPT-OSS-120B</b><br/>(via Groq)</sub>
    </td>
    <td align="center" width="96">
      <a href="https://github.com/docling-project/docling"><img src="https://cdn.simpleicons.org/readthedocs/8CA1AF" width="48" height="48" alt="Docling" /></a>
      <br /><sub><b>Docling</b></sub>
    </td>
  </tr>
</table>
<br />
</div>

| Layer                | Technologies                                                                             |
| :------------------- | :--------------------------------------------------------------------------------------- |
| **Frontend**   | Next.js 15 (App Router) · TypeScript · Tailwind CSS · shadcn/ui                       |
| **Backend**    | FastAPI · Python 3.10+ · Pydantic · Uvicorn                                           |
| **AI / ML**    | Groq GPT-OSS-120B (Extraction) · Hugging Face BGE-M3 (Embeddings) · Docling (PDF Parsing) |
| **Database**   | Supabase PostgreSQL + pgvector + Row-Level Security                                      |
| **Auth**       | Supabase Auth (JWT) ·`@supabase/ssr`                                                  |
| **Deployment** | Vercel (Frontend) · Local GPU Server (Backend)                                          |

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 🌐 Architecture

<div align="center">
  <img src="assets/architecture_diagram.png" alt="TenderSync System Architecture" width="800" />
</div>

<br />

> [!IMPORTANT]
> The backend currently runs on a **local development machine** (see [Known Limitations](#%EF%B8%8F-known-limitations)). Once deployed to production, the frontend on Vercel will communicate with the backend via HTTPS.

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 📂 Project Structure

```
TenderSync/
│
├── backend/
│   ├── main.py                 # FastAPI app entry point and router aggregator
│   ├── db.py                   # Centralized Supabase/Groq clients and user dependency injection
│   ├── document_extractor.py   # Unified document extraction logic (RapidOCR, PyPDFium2, Docling)
│   ├── embedding_service.py    # Hugging Face BAAI/bge-m3 1024-dim dense embedding service
│   ├── pipeline_service.py     # RAG pipeline orchestrator (chunking, embedding, semantic retrieval)
│   ├── bid_engine.py           # 4-phase hybrid Bid/No-Bid evaluation engine (7 scoring dimensions)
│   ├── proposal_engine.py      # LangGraph multi-agent proposal generator (Plan, Draft, Review)
│   ├── comparison_engine.py    # Multi-tender side-by-side comparative analysis (Groq LLM)
│   ├── tender_discovery.py     # Public tender scrapers (CPPP, GeM) & discovery engine
│   ├── report_generator.py     # WeasyPrint PDF evaluation report generator with Plotly charts
│   ├── profile_service.py      # Multi-tenant company profile CRUD & completeness validation
│   ├── schemas.py              # Pydantic v2 models for requests, responses, and validation
│   ├── auth.py                 # Supabase JWT authentication, user/org resolution, RBAC
│   ├── requirements.txt        # Production Python dependencies
│   ├── routers/                # Domain-driven FastAPI routers
│   │   ├── auth.py
│   │   ├── batches.py
│   │   ├── exports.py
│   │   ├── market.py
│   │   ├── orgs.py
│   │   ├── profiles.py
│   │   ├── proposals.py
│   │   └── tenders.py
│   ├── templates/
│   │   └── evaluation_report.html # Jinja2 HTML template for downloadable PDF reports
│   ├── supabase/
│   │   ├── schema.sql          # PostgreSQL + pgvector schema & RLS policies
│   │   └── migrations/         # Versioned SQL migrations (001 to 006)
│   └── tests/                  # 68 automated unit tests (100% pass rate)
│
├── docs/
│   └── APPLICATION_DESIGN.md   # Full architecture + design specification
│
├── assets/
│   ├── tendersync_logo.png      # Project logo
│   └── architecture_diagram.png # System architecture diagram
│
├── .gitignore
├── LICENSE
└── README.md                   # ← You are here
```

> [!NOTE]
> The Next.js frontend directory will be added once the frontend is built and integrated.

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 🚀 Getting Started

### Prerequisites

| Requirement                     | Version | Purpose                     |
| :------------------------------ | :------ | :-------------------------- |
| **Python**                | 3.10+   | Backend runtime             |
| **Node.js**               | 18+     | Frontend runtime            |
| **npm** or **pnpm**       | Latest  | Package management          |
| **Supabase Account**      | –      | Auth, PostgreSQL, pgvector  |
| **Groq API Key**          | –      | Fast LLM inference          |
| **Hugging Face Token**    | –      | BGE-M3 Embedding generation |

### 1. Clone & Backend Setup

```bash
# Clone the repository
git clone https://github.com/rudyxx007/TenderSync.git
cd TenderSync/backend

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Supabase
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SECRET_KEY=your-supabase-service-role-key

# AI Services
HF_TOKEN=your-huggingface-read-token
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b

# Development Mode (set to false in production)
ALLOW_DEV_BYPASS=true
DEVELOPMENT_USER_ID=your-test-user-uuid
```

| Variable                | Required | Description                                               |
| :---------------------- | :------: | :-------------------------------------------------------- |
| `SUPABASE_URL`        |    ✅    | Your Supabase project URL                                 |
| `SUPABASE_SECRET_KEY` |    ✅    | Service role key (never expose to frontend)               |
| `HF_TOKEN`            |    ✅    | Hugging Face API token for BGE-M3 embeddings              |
| `GROQ_API_KEY`        |    ✅    | API key from [groq.com](https://groq.com/)                |
| `GROQ_MODEL`          |    ❌    | LLM model identifier (default: `openai/gpt-oss-120b`)     |
| `ALLOW_DEV_BYPASS`    |    ❌    | Enables auth bypass for local testing (default: `false`)   |
| `DEVELOPMENT_USER_ID` |    ❌    | UUID used when bypass is enabled                          |

### 3. Run the Backend

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be live at `http://127.0.0.1:8000` with interactive Swagger docs at [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

### 4. Run Unit Tests

```bash
cd backend
python -m pytest tests/
```

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 🌐 Deployment

> [!IMPORTANT]
> 🚧 **The application is under active development. Deployment links will be updated here once live.**
>
> ```
> 🔗 Live App:  https://[COMING_SOON].vercel.app
> ```

| Layer              | Service                          |     Status     |
| :----------------- | :------------------------------- | :------------: |
| **Frontend** | Vercel (Next.js 15)              | 🔜 Coming Soon |
| **Backend**  | Local GPU Server / Render (FastAPI) |   ✅ Running   |
| **Database** | Supabase (PostgreSQL + pgvector) |   ✅ Active   |
| **Auth**     | Supabase Auth (JWT + RLS)        |   ✅ Active   |

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 📡 API Reference

All authenticated endpoints require `Authorization: Bearer <supabase_jwt>`.

<details>
<summary><kbd> 🔐 Authentication & User</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `GET` | `/api/me` | Returns current user ID and email |
| `GET` | `/api/auth/me` | Returns authenticated user details |

</details>

<details>
<summary><kbd> 🏢 Organization & Multi-Tenancy</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `POST` | `/api/orgs` | Create a new organization (sets user as owner) |
| `POST` | `/api/orgs/join` | Join an organization using a 6-character invite code |
| `GET` | `/api/orgs/members` | List members and roles for the current organization |
| `POST` | `/api/orgs/invite/regenerate` | Regenerate a new invite code (owner/admin only) |

</details>

<details>
<summary><kbd> 📋 Company Profile</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `GET` | `/api/profile/status` | Check profile existence, completeness percentage, and missing fields |
| `PUT` | `/api/profile` | Create or update organization profile (capabilities, certs, budget thresholds) |

</details>

<details>
<summary><kbd> 📄 Tender Ingestion & Analysis</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `POST` | `/api/process-tender` | Upload PDF/Scan → Docling/RapidOCR text extraction → BGE-M3 embedding → pgvector → Groq LLM extraction → 4-Phase Bid/No-Bid Evaluation |
| `POST` | `/api/upload-tender` | Alias for single tender processing |
| `POST` | `/api/tenders/process` | Standard REST route for tender processing |
| `POST` | `/api/tenders/batch` | Upload multiple tender files for asynchronous batch evaluation |
| `GET` | `/api/my-analyses` | List all historical tender analyses for the organization |
| `GET` | `/api/tenders` | REST alias for listing historical analyses |
| `GET` | `/api/analysis/{id}` | Get complete evaluation breakdown, factor scores, and extracted RFP metadata |
| `GET` | `/api/tenders/{id}` | REST alias for retrieving a specific tender analysis |
| `POST` | `/api/tenders/compare` | Compare 2+ analyzed tenders side-by-side with LLM comparative summary |

</details>

<details>
<summary><kbd> 🤖 AI Proposal Generation (LangGraph)</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `POST` | `/api/tenders/{id}/proposal` | Run LangGraph state machine (Planner → Drafter → Reviewer) to generate a tailored proposal |
| `GET` | `/api/proposals/{id}` | Retrieve generated proposal draft by ID |
| `GET` | `/api/proposals` | List all generated proposal drafts for the organization |

</details>

<details>
<summary><kbd> 🌍 Market Tender Discovery</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `GET` | `/api/market-tenders` | Query discovered public tenders with sector and deadline filters |
| `POST` | `/api/market-tenders/discover` | Trigger live scraping job across public procurement portals (CPPP, GeM) |

</details>

<details>
<summary><kbd> 📦 Batch Jobs</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `GET` | `/api/batches` | List all background batch processing jobs for the organization |
| `GET` | `/api/batches/{id}` | Get status, progress, and individual results for a batch job |

</details>

<details>
<summary><kbd> 📅 Utilities & PDF Export</kbd></summary>
<br />

| Method | Endpoint | Description |
| :----: | :------- | :---------- |
| `POST` | `/api/generate-calendar` | Generate and download an `.ics` iCalendar file for tender submission deadlines |
| `GET` | `/api/tenders/{id}/export-pdf` | Generate and download a styled evaluation report PDF via WeasyPrint & Jinja2 |

</details>

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 🗺️ Roadmap

- [X] **Phase 1: Backend API + RAG Pipeline**
  - [X] Supabase JWT authentication with dev bypass
  - [X] Company profile CRUD with completeness validation
  - [X] PDF upload → Docling extraction → Hugging Face BGE-M3 embedding → pgvector storage
  - [X] Groq GPT-OSS-120B structured data extraction
  - [X] Hybrid 4-phase Bid/No-Bid evaluation engine
  - [X] Tender analysis history (save, list, detail)
  - [X] Calendar export (.ics generation)
- [ ] **Phase 2: Next.js Frontend**
  - [ ] Landing page with animated hero + feature showcase
  - [ ] Auth flow (login, signup, session management)
  - [ ] Onboarding wizard (3-step company profile setup)
  - [ ] Dashboard with PDF upload + result display
  - [ ] Tender detail page with PWin gauge + factor breakdown
  - [ ] History page with sortable analysis table
  - [ ] Settings / profile edit page
  - [ ] Dark mode, responsive layout, micro-animations
- [ ] **Phase 3: Deployment + Polish**
  - [ ] Vercel deployment (frontend)
  - [ ] Production environment hardening
  - [ ] Performance optimization + Lighthouse audit
- [X] **Phase 4: Future Enhancements**
  - [ ] Cloud GPU migration for faster processing
  - [X] Multi-user per organization (team features)
  - [X] Tender comparison (side-by-side analysis)
  - [ ] Email notifications for upcoming deadlines
  - [X] Batch PDF upload support
  - [X] Export evaluation reports as PDF
  - [X] AI Proposal Writing Engine (LangGraph)
  - [X] Market Tender Discovery (Scraping)

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## ⚠️ Known Limitations

> Transparency is important. Here are the current constraints and how I plan to address them.

### Development Hardware

| Component     | Specification            |
| :------------ | :----------------------- |
| **GPU** | NVIDIA RTX 5050 (Laptop) |
| **RAM** | 24 GB DDR5               |
| **CPU** | Intel Core Ultra 7       |

### Current Constraints

| Limitation                 | Detail                                                                                                             | Planned Resolution                                                              |
| :------------------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| **Processing Speed** | A single ~100-page RFP takes**30–90 seconds** to fully process through the extraction + evaluation pipeline | Cloud GPU migration (RunPod / Lambda Labs) to bring this down to ~5–10 seconds |
| **Concurrency**      | Single-user experience; simultaneous uploads are processed sequentially                                            | Horizontal scaling on cloud infrastructure                                      |
| **Availability**     | Backend is only available while the dev machine is running                                                         | 24/7 cloud deployment                                                           |
| **VRAM Pressure**    | Docling's internal models require ~2–4 GB VRAM, leaving limited headroom for parallel tasks                       | Dedicated GPU server with ≥24 GB VRAM                                          |

> [!TIP]
> Despite these hardware constraints, the **core architecture is production-ready**. The RAG pipeline, Bid/No-Bid engine, auth flow, and multi-tenant data isolation are all built to scale. The only bottleneck is compute, which is a deployment concern, not an architecture concern.

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 📄 License

**© 2026 Rudra Bhavin Naik. All Rights Reserved.**

This project is shared publicly for **portfolio and educational purposes only**.

| ✅ Permitted                                                          | ❌ Not Permitted                                                           |
| :-------------------------------------------------------------------- | :------------------------------------------------------------------------- |
| View source code for personal learning                                | Copy, redistribute, or republish this codebase as your own                 |
| Reference the project in articles or academic work (with attribution) | Use this code in commercial products or services                           |
| Fork the repository for personal, non-commercial experimentation      | Create derivative works for public distribution without written permission |

See the full [`LICENSE`](LICENSE) file for details. For inquiries, [contact the author](#-author).

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 👤 Author

<div align="center">
<br />

| | |
| :-: | :--- |
| 👤 | **Rudra Bhavin Naik** |
| 🐙 | [@rudyxx007](https://github.com/rudyxx007) |
| 💼 | [LinkedIn](https://linkedin.com/in/rudy7404) |
| 📧 | [rudyop007@gmail.com](mailto:rudyop007@gmail.com) |

<br />

**Project:** [github.com/rudyxx007/TenderSync](https://github.com/rudyxx007/TenderSync)

</div>

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

## 🙏 Acknowledgments

|      | Tool                                                 | Role in TenderSync             |
| :--: | :--------------------------------------------------- | :----------------------------- |
|      | [FastAPI](https://fastapi.tiangolo.com/)              | Backend web framework          |
|      | [Supabase](https://supabase.com/)                     | Auth, database, vector storage |
|      | [Next.js](https://nextjs.org/)                        | Frontend framework             |
|      | [Tailwind CSS](https://tailwindcss.com/)              | Utility-first styling          |
|      | [shadcn/ui](https://ui.shadcn.com/)                   | Component library              |
|  📄  | [Docling](https://github.com/docling-project/docling) | PDF extraction engine          |
|  🤖  | [Groq](https://groq.com/)                             | LLM inference                  |
|  🧠  | [Hugging Face](https://huggingface.co/)               | BGE-M3 Embedding generation    |
| 🛡️ | [Shields.io](https://shields.io/)                     | README badges                  |
|  🎨  | [Skill Icons](https://skillicons.dev/)                | Tech stack visuals             |

<p align="right"><a href="#readme-top">↑ back to top</a></p>

---

<div align="center">
  <sub>Built with ❤️ and way too much ☕ for smarter RFP decisions.</sub>
  <br /><br />
  <b>If this project interests you, consider leaving a ⭐</b>
</div>

<!-- REFERENCE LINKS -->

[stars-shield]: https://img.shields.io/github/stars/rudyxx007/TenderSync?style=for-the-badge&logo=starship&logoColor=white&labelColor=0D1117&color=10B981
[stars-url]: https://github.com/rudyxx007/TenderSync/stargazers
[forks-shield]: https://img.shields.io/github/forks/rudyxx007/TenderSync?style=for-the-badge&logo=git&logoColor=white&labelColor=0D1117&color=3B82F6
[forks-url]: https://github.com/rudyxx007/TenderSync/network/members
[issues-shield]: https://img.shields.io/github/issues/rudyxx007/TenderSync?style=for-the-badge&logo=github&logoColor=white&labelColor=0D1117&color=F59E0B
[issues-url]: https://github.com/rudyxx007/TenderSync/issues
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0D1117
[linkedin-url]: https://linkedin.com/in/rudy7404
