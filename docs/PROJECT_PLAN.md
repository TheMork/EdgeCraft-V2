# Crypto-Quant Dashboard: Project Status & Scrum Plan

## 1. Executive Summary

This document outlines the current state of the Institutional Crypto Backtesting Engine project, identifies critical gaps ("Baustellen"), and proposes a structured Scrum plan to bring the project to completion.

**Current Status:** The project has a solid foundation for data ingestion (Python) and event processing (Rust). However, the user-facing components (Frontend, API) and the bridge between data and simulation are missing.

## 2. Open Construction Sites ("Baustellen")

### A. Frontend (Critical)
*   **Status:** Completely missing.
*   **Gap:** No user interface exists. Users cannot configure backtests, visualize results, or monitor real-time trading.
*   **Requirement:** Implement the architecture defined in `docs/FRONTEND_BLUEPRINT.md` (Next.js, Tailwind, Lightweight Charts).

### B. API Layer (Critical)
*   **Status:** Missing.
*   **Gap:** No mechanism to serve data to the frontend or control the simulation engine.
*   **Requirement:** Implement a FastAPI server with REST endpoints for historical data and WebSockets for real-time simulation updates.

### C. Simulation Logic (High)
*   **Status:** Basic Core (Rust EventLoop exists).
*   **Gap:**
    *   Missing `Strategy` abstract base class in Python.
    *   Missing `Runner` to orchestrate data loading -> EventLoop -> Strategy execution.
    *   Missing `Order Management System` (OMS) logic (e.g., handling fills, calculating PnL).
*   **Requirement:** Implement Python wrappers and logic to utilize the Rust core effectively.

### D. Integration (Medium)
*   **Status:** Disconnected components.
*   **Gap:** Data ingestion fills the DB, but the simulation engine doesn't read from it yet.
*   **Requirement:** Connect `QuestDBManager` to the Simulation Runner.

## 3. Scrum Project Plan

**Roles:**
*   **Product Owner:** User
*   **Scrum Master:** Lead Developer
*   **Dev Team:** AI Agent / Developer

**Sprint Duration:** 1 Week

### Sprint 1: Foundation & Core Integration
**Goal:** Establish the API and connect Data to Simulation.
*   [x] Create Project Plan & Baustellen Report.
*   [ ] **API Setup:** Initialize FastAPI project structure (`src/api`).
*   [ ] **Database Access:** Implement `get_ohlcv` in `QuestDBManager` to query historical data.
*   [ ] **Simulation Bridge:** Create `Strategy` base class and `Runner` to load data into the Rust EventLoop.
*   [ ] **Health Check:** meaningful API endpoint returning system status.

### Sprint 2: Frontend Skeleton & Basic UI
**Goal:** Initialize the User Interface.
*   [ ] **Next.js Setup:** Initialize frontend project with TypeScript and Tailwind CSS.
*   [ ] **Component Library:** Install Shadcn/UI.
*   [ ] **Layout:** Create the main dashboard layout (Sidebar, Header, Main Content).
*   [ ] **API Client:** generated or manual fetcher for the backend API.

### Sprint 3: Visualization & Streaming
**Goal:** Visualize Data and Simulation Progress.
*   [ ] **Charting:** Implement Lightweight Charts wrapper component.
*   [ ] **Data Streaming:** Implement WebSocket endpoint in FastAPI for simulation events.
*   [ ] **Frontend Integration:** Connect Chart component to WebSocket to render candles in real-time.

### Sprint 4: Advanced Logic & Reporting
**Goal:** Feature Parity with Institutional Tools.
*   [ ] **Metrics Engine:** Implement Sharpe Ratio, Drawdown, and Sortino Ratio calculations (Rust or Python).
*   [ ] **Reporting Dashboard:** Create UI for backtest results (Equity Curve, Trade List).
*   [ ] **Optimization:** implement parallel backtesting.

## 4. Definition of Done (DoD)
*   Code implemented and reviewed.
*   Unit tests passed (where applicable).
*   API endpoints documented (OpenAPI/Swagger).
*   Deployment instructions updated.
