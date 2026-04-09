import React, { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend, Area, AreaChart,
} from "recharts";
import "./App.css";

const API = "https://eustatic-crowdedly-veola.ngrok-free.dev";

/** ngrok free tier: avoid HTML interstitial blocking fetch() */
const API_HEADERS = { "ngrok-skip-browser-warning": "true" };

function apiFetch(url, options = {}) {
  return fetch(url, {
    ...options,
    headers: { ...API_HEADERS, ...(options.headers || {}) },
  });
}

/** Parse JSON and throw if status is not OK (FastAPI returns { detail: ... }). */
function readApiJson(response) {
  return response.json().then((data) => {
    if (!response.ok) {
      const d = data?.detail;
      const msg = Array.isArray(d)
        ? d.map((x) => x.msg ?? JSON.stringify(x)).join("; ")
        : (d ?? response.statusText);
      throw new Error(String(msg));
    }
    return data;
  });
}

/** Must match backend [config.TICKERS](src/config.py) for portfolio keys */
const TICKERS = ["GOOG", "AAPL", "MSFT", "AMZN", "META"];

/** Per-ticker initial cash for /api/backtest and /api/portfolio/summary ($10k each; 5 tickers => $50k total) */
const PORTFOLIO_INITIAL_CASH = 10000;
const COLORS = {
  GOOG: "#ea4335",
  GOOGL: "#ea4335",
  AAPL: "#2ecc71",
  MSFT: "#9b59b6",
  AMZN: "#f39c12",
  META: "#3498db",
};

function ActionBadge({ action }) {
  const styles = {
    BUY:  { bg: "#dcfce7", color: "#16a34a", border: "#16a34a" },
    SELL: { bg: "#fee2e2", color: "#dc2626", border: "#dc2626" },
    HOLD: { bg: "#f3f4f6", color: "#6b7280", border: "#6b7280" },
  };
  const s = styles[action] || styles.HOLD;
  return (
    <span style={{
      padding: "4px 14px", borderRadius: 20, fontSize: 13, fontWeight: 700,
      backgroundColor: s.bg, color: s.color, border: `1.5px solid ${s.border}`,
      letterSpacing: 1,
    }}>
      {action}
    </span>
  );
}

function Spinner() {
  return (
    <div className="spinner-container">
      <div className="spinner" />
    </div>
  );
}

function Card({ children, style }) {
  return <div className="card" style={style}>{children}</div>;
}

function DogSVG() {
  return (
    <svg
      viewBox="0 0 120 70"
      width="110"
      height="64"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: "block", overflow: "visible" }}
    >
      <defs>
        {/* Top-to-bottom coat gradient: highlight → mid-tone → shadow */}
        <linearGradient id="dg-body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#ecc07a" />
          <stop offset="45%"  stopColor="#d4956a" />
          <stop offset="100%" stopColor="#9a6030" />
        </linearGradient>
        <linearGradient id="dg-head" x1="0" y1="0" x2="0.4" y2="1">
          <stop offset="0%"   stopColor="#e8b87a" />
          <stop offset="100%" stopColor="#bf7d40" />
        </linearGradient>
        <linearGradient id="dg-leg-far" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#b87840" />
          <stop offset="100%" stopColor="#8a5520" />
        </linearGradient>
        <linearGradient id="dg-leg-near" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#d49050" />
          <stop offset="100%" stopColor="#a86030" />
        </linearGradient>
      </defs>

      {/* ── TAIL (behind body) ── */}
      <g className="dog-tail">
        {/* Outer tail shape — filled silhouette */}
        <path
          d="M22 38 C14 32 6 26 8 14 C9 8 14 7 17 11 C19 15 18 22 20 28 C21 31 23 35 22 38Z"
          fill="#bf7d40"
        />
        {/* Inner highlight */}
        <path
          d="M20 34 C15 28 10 21 12 13 C13 10 16 10 17 13 C17 18 18 24 20 30Z"
          fill="#d49050"
        />
      </g>

      {/* ── REAR LEGS (far side — darker, behind body) ── */}
      {/* Rear-left (far) */}
      <g className="dog-leg dog-leg-rl">
        <path
          d="M28 43 C27 50 25 55 24 61 C23 65 27 67 31 66 C34 65 33 62 32 59 C31 55 31 50 32 43Z"
          fill="url(#dg-leg-far)"
        />
        {/* knee dimple */}
        <ellipse cx="29" cy="54" rx="2" ry="1.5" fill="#9a6030" opacity="0.5" />
      </g>
      {/* Rear-right (far) */}
      <g className="dog-leg dog-leg-rr">
        <path
          d="M35 43 C34 50 32 55 31 61 C30 65 34 67 38 66 C41 65 40 62 39 59 C38 55 38 50 39 43Z"
          fill="url(#dg-leg-far)"
        />
        <ellipse cx="36" cy="54" rx="2" ry="1.5" fill="#9a6030" opacity="0.5" />
      </g>

      {/* ── BODY (organic bezier silhouette) ── */}
      {/* Underbelly fill first (shadow) */}
      <path
        className="dog-body"
        d="
          M 24 22
          C 28 14, 42 12, 60 15
          C 72 17, 80 20, 83 27
          C 86 31, 81 38, 73 41
          C 63 44, 47 44, 35 41
          C 25 40, 19 36, 18 30
          C 18 27, 21 24, 24 22 Z
        "
        fill="url(#dg-body)"
      />
      {/* Back-line highlight stroke for the arch */}
      <path
        d="M 26 20 C 38 13, 55 13, 68 17 C 74 19, 80 22, 82 27"
        fill="none"
        stroke="#ecc07a"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.55"
      />

      {/* ── FRONT LEGS (near side — lighter, in front of body) ── */}
      {/* Front-left (far) */}
      <g className="dog-leg dog-leg-fl">
        <path
          d="M58 45 C57 52 55 57 54 63 C53 67 57 68 61 67 C64 66 63 63 62 60 C61 56 61 51 62 45Z"
          fill="url(#dg-leg-far)"
        />
        <ellipse cx="59" cy="56" rx="2" ry="1.5" fill="#9a6030" opacity="0.5" />
      </g>
      {/* Front-right (near) */}
      <g className="dog-leg dog-leg-fr">
        <path
          d="M66 45 C65 52 63 57 62 63 C61 67 65 68 69 67 C72 66 71 63 70 60 C69 56 69 51 70 45Z"
          fill="url(#dg-leg-near)"
        />
        <ellipse cx="67" cy="56" rx="2" ry="1.5" fill="#9a6030" opacity="0.4" />
      </g>

      {/* ── COLLAR (app accent color) ── */}
      <rect x="72" y="34" width="16" height="5" rx="2.5" fill="#c3f73a" />
      {/* Tag */}
      <circle cx="80" cy="41" r="3" fill="#ffd700" />
      <circle cx="80" cy="41" r="1.5" fill="#e6a800" />

      {/* ── NECK ── */}
      <path
        d="M 74 22 C 78 18, 86 20, 88 26 C 90 32, 86 40, 80 42 C 76 43, 72 41, 72 38 C 72 32, 72 26, 74 22 Z"
        fill="url(#dg-body)"
      />

      {/* ── HEAD GROUP ── */}
      {/* Floppy ear (hangs behind skull, drawn first) */}
      <path
        d="M 84 12 C 78 10, 74 14, 75 22 C 76 28, 80 31, 85 30 C 88 29, 89 25, 87 19 C 86 15, 85 12, 84 12 Z"
        fill="#b06a30"
      />
      {/* Ear inner sheen */}
      <path
        d="M 84 15 C 80 14, 78 18, 79 24 C 80 28, 83 29, 85 27 C 87 25, 86 20, 85 16 Z"
        fill="#c88040"
        opacity="0.6"
      />

      {/* Skull */}
      <path
        d="M 88 12 C 94 8, 104 10, 106 18 C 108 24, 104 30, 98 32 C 93 33, 88 30, 87 24 C 86 18, 87 14, 88 12 Z"
        fill="url(#dg-head)"
      />

      {/* Muzzle / snout — wedge shape */}
      <path
        d="M 96 26 C 100 24, 108 25, 112 28 C 115 31, 113 36, 108 37 C 102 38, 96 36, 95 32 C 94 29, 95 27, 96 26 Z"
        fill="#c88040"
      />
      {/* Muzzle top highlight */}
      <path
        d="M 98 26 C 103 24, 108 25, 111 28"
        fill="none"
        stroke="#e0a860"
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.7"
      />

      {/* Nose — dark with subtle ridge */}
      <ellipse cx="112" cy="27" rx="4" ry="3" fill="#1a0802" />
      <ellipse cx="112" cy="25.5" rx="3" ry="1" fill="#2e1408" />
      {/* Nostril shine */}
      <circle cx="110.5" cy="26" r="1" fill="#3a1a08" />

      {/* Eye — dark iris with white highlight */}
      <circle cx="96" cy="18" r="3.5" fill="#1a0802" />
      <circle cx="96" cy="18" r="2"   fill="#2e1a08" />
      {/* Catch-light */}
      <circle cx="97.2" cy="16.8" r="1.1" fill="#ffffff" />

      {/* Brow ridge */}
      <path
        d="M 92 14 C 95 12, 100 13, 102 16"
        fill="none"
        stroke="#a06030"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      {/* Smile / mouth */}
      <path
        d="M 104 34 Q 108 38 112 34"
        fill="none"
        stroke="#7a4820"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

function App() {
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [tab, setTab] = useState("chart");
  const [priceData, setPriceData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [backtestData, setBacktestData] = useState(null);
  const [portfolioSummary, setPortfolioSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(null);
  const [userCash, setUserCash] = useState(10000);
  const [userShares, setUserShares] = useState(0);
  const [predLoading, setPredLoading] = useState(false);

  // Check API on mount
  useEffect(() => {
    apiFetch(`${API}/api/tickers`)
      .then(readApiJson)
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  // Fetch price data for the chart tab.
  useEffect(() => {
    if (!apiOnline || tab !== "chart") return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPriceData(null);
    apiFetch(`${API}/api/price/${selectedTicker}?period=6mo`)
      .then(readApiJson)
      .then((data) => {
        if (cancelled) return;
        setPriceData(data);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message || "Failed to load price data");
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [selectedTicker, apiOnline, tab]);

  // Fetch prediction — called manually via button or automatically on ticker/tab change.
  function fetchPrediction() {
    setPredLoading(true);
    setPrediction(null);
    apiFetch(
      `${API}/api/predict/${selectedTicker}?cash=${userCash}&shares=${userShares}`
    )
      .then(readApiJson)
      .then((d) => { setPrediction(d); setPredLoading(false); })
      .catch(() => { setPrediction(null); setPredLoading(false); });
  }

  // Auto-predict when ticker or tab changes.
  useEffect(() => {
    if (!apiOnline || tab !== "chart") return;
    fetchPrediction();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTicker, apiOnline, tab]);

  // Fetch backtest when tab = backtest
  useEffect(() => {
    if (!apiOnline || tab !== "backtest") return;
    let cancelled = false;
    setError(null);
    setLoading(true);
    setBacktestData(null);
    apiFetch(
      `${API}/api/backtest/${selectedTicker}?initial_cash=${PORTFOLIO_INITIAL_CASH}`,
    )
      .then(readApiJson)
      .then((data) => {
        if (cancelled) return;
        setBacktestData(data);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTicker, tab, apiOnline]);

  // Fetch portfolio summary when tab = portfolio
  useEffect(() => {
    if (!apiOnline || tab !== "portfolio") return;
    let cancelled = false;
    setError(null);
    setLoading(true);
    setPortfolioSummary(null);
    apiFetch(
      `${API}/api/portfolio/summary?initial_cash=${PORTFOLIO_INITIAL_CASH}`,
    )
      .then(readApiJson)
      .then((data) => {
        if (cancelled) return;
        setPortfolioSummary(data);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [tab, apiOnline]);

  // ── API Offline ──
  if (apiOnline === false) {
    return (
      <div className="offline-screen">
        <div style={{ fontSize: 48 }}>⚡</div>
        <h2>API Offline</h2>
        <p>Start the backend and ngrok tunnel, or check the API URL in App.js.</p>
        <code>cd src && python api.py</code>
        <p style={{ marginTop: 12, fontSize: 13, color: "#64748b" }}>
          Then refresh this page.
        </p>
      </div>
    );
  }

  if (apiOnline === null) return <Spinner />;

  const currentPrice = priceData?.current_price || 0;
  const changePct = priceData?.change_pct || 0;

  return (
    <div className="app">
      {/* ── HEADER ── */}
      <header className="header">
        <div className="header-left">
          <div className="logo">D³</div>
          <span className="title">
            D3QN <span className="accent">Trading</span>
          </span>
          <span className="live-badge">LIVE</span>
        </div>
        <nav className="tabs">
          {["chart", "backtest", "portfolio"].map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`tab-btn ${tab === t ? "active" : ""}`}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      <main className="main">
        {/* ── TICKER SELECTOR (portfolio covers all tickers; no per-ticker switch) ── */}
        {tab !== "portfolio" && (
          <div className="ticker-bar">
            {TICKERS.map((t) => (
              <button
                key={t}
                onClick={() => setSelectedTicker(t)}
                className="ticker-btn"
                style={{
                  background: selectedTicker === t ? (COLORS[t] ?? "#64748b") : "#1a1a2e",
                  color: selectedTicker === t ? "#fff" : "#94a3b8",
                  borderColor: selectedTicker === t ? (COLORS[t] ?? "#64748b") : "#2a2a4a",
                }}
              >
                {t}
              </button>
            ))}
          </div>
        )}

        {error && <div className="error-banner">Error: {error}</div>}

        {/* ══════════════ CHART TAB ══════════════ */}
        {tab === "chart" && (
          <div className="chart-grid">
            {/* Price chart */}
            <Card>
              {loading || !priceData ? (
                <Spinner />
              ) : (
                <>
                  <div className="price-header">
                    <span className="big-price">${currentPrice.toLocaleString()}</span>
                    <span className={`change-badge ${changePct >= 0 ? "up" : "down"}`}>
                      {changePct >= 0 ? "+" : ""}{changePct}%
                    </span>
                    <span className="ticker-label">{selectedTicker}</span>
                  </div>
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={priceData.history}>
                      <defs>
                        <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={COLORS[selectedTicker] ?? "#64748b"} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={COLORS[selectedTicker] ?? "#64748b"} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                      <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }}
                        tickFormatter={(v) => v.slice(5)} interval={20} />
                      <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={["auto", "auto"]} />
                      <Tooltip contentStyle={{
                        background: "#1a1a2e", border: "1px solid #2a2a4a", borderRadius: 10,
                        color: "#e2e8f0",
                      }} />
                      <Area type="monotone" dataKey="close" stroke={COLORS[selectedTicker] ?? "#64748b"}
                        fill="url(#cg)" strokeWidth={2} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </>
              )}
            </Card>

            {/* Signal panel */}
                <div className="signal-col">
              <Card>
                <div className="section-label">Portfolio State</div>
                <div className="pred-inputs">
                  <label>
                    Cash ($)
                    <input
                      type="number"
                      min="0"
                      value={userCash}
                      onChange={(e) => setUserCash(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Shares held
                    <input
                      type="number"
                      min="0"
                      step="1"
                      value={userShares}
                      onChange={(e) => setUserShares(Math.max(0, parseInt(e.target.value) || 0))}
                    />
                  </label>
                  <button
                    className="pred-btn"
                    onClick={fetchPrediction}
                    disabled={predLoading}
                  >
                    {predLoading ? "Loading..." : "Predict"}
                  </button>
                </div>
              </Card>
              <Card>
                <div className="section-label">D3QN LIVE SIGNAL</div>
                {predLoading ? (
                  <Spinner />
                ) : !prediction && priceData ? (
                  <p style={{ color: "#94a3b8", fontSize: 14 }}>
                    Trading signal unavailable (check API /api/predict). Price data loaded.
                  </p>
                ) : !prediction ? (
                  <Spinner />
                ) : (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
                      <ActionBadge action={prediction.action} />
                      <span style={{ fontSize: 14, color: "#94a3b8" }}>
                        {prediction.confidence}% confidence
                      </span>
                    </div>
                    {Array.isArray(prediction.q_values) && prediction.combined_index != null && (
                      <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 12 }}>
                        Predicted next trade size:{" "}
                        <strong style={{ color: "#e2e8f0" }}>
                          {prediction.q_values[prediction.combined_index]?.shares ?? 0}
                        </strong>{" "}
                        shares
                      </div>
                    )}
                    <div className="q-label">Q-VALUES (from model)</div>
                    {Array.isArray(prediction.q_values)
                      ? prediction.q_values.map((item) => {
                          const q = Number(item.q);
                          const label =
                            item.shares > 0
                              ? `${item.index} ${item.action} ${item.shares}`
                              : `${item.index} ${item.action}`;
                          return (
                            <div
                              key={`${item.index}-${item.action}-${item.shares}`}
                              className="q-row"
                              style={{ opacity: item.legal === false ? 0.45 : 1 }}
                            >
                              <span className="q-name">
                                {label}
                                {item.legal === false ? (
                                  <span style={{ color: "#64748b", fontSize: 11 }}> (illegal)</span>
                                ) : null}
                              </span>
                              <span className={`q-val ${q > 0 ? "pos" : q < 0 ? "neg" : ""}`}>
                                {`${q > 0 ? "+" : ""}${q.toFixed(4)}`}
                              </span>
                            </div>
                          );
                        })
                      : Object.entries(prediction.q_values ?? {}).map(([k, v]) => (
                          <div key={k} className="q-row">
                            <span className="q-name">{k}</span>
                            <span className={`q-val ${v > 0 ? "pos" : v < 0 ? "neg" : ""}`}>
                              {typeof v === "number"
                                ? `${v > 0 ? "+" : ""}${v.toFixed(4)}`
                                : String(v)}
                            </span>
                          </div>
                        ))}
                    <div className="model-info">
                      Price: ${prediction.price ?? "—"} | Dueling DQN + volume head
                    </div>
                  </>
                )}
              </Card>
            </div>
          </div>
        )}

        {/* ══════════════ BACKTEST TAB ══════════════ */}
        {tab === "backtest" && (
          <Card>
            {loading || !backtestData ? (
              <Spinner />
            ) : (
              <>
                <div className="backtest-header">
                  <div>
                    <div style={{ fontSize: 18, fontWeight: 700 }}>Backtest: {selectedTicker}</div>
                    <div style={{ fontSize: 13, color: "#64748b" }}>
                      D3QN vs Buy & Hold
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 32 }}>
                    <div style={{ textAlign: "right" }}>
                      <div className="stat-label">D3QN</div>
                      <div className="stat-value accent-green">
                        ${backtestData.dqn_final.toLocaleString()}
                      </div>
                      <div className={backtestData.dqn_return >= 0 ? "stat-ret pos" : "stat-ret neg"}>
                        {backtestData.dqn_return >= 0 ? "+" : ""}{backtestData.dqn_return}%
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div className="stat-label">Buy & Hold</div>
                      <div className="stat-value accent-orange">
                        ${backtestData.bh_final.toLocaleString()}
                      </div>
                      <div className={backtestData.bh_return >= 0 ? "stat-ret orange" : "stat-ret neg"}>
                        {backtestData.bh_return >= 0 ? "+" : ""}{backtestData.bh_return}%
                      </div>
                    </div>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={backtestData.portfolio}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                    <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 11 }} domain={["auto", "auto"]} />
                    <Tooltip
                      contentStyle={{
                        background: "#1a1a2e", border: "1px solid #2a2a4a",
                        borderRadius: 10, color: "#e2e8f0",
                      }}
                      formatter={(v, name) => [`$${Number(v).toLocaleString()}`, name]}
                    />
                    <Line type="monotone" dataKey="value" stroke="#c3f73a"
                      strokeWidth={2.5} dot={false} name="D3QN" />
                    <Legend />
                  </LineChart>
                </ResponsiveContainer>
              </>
            )}
          </Card>
        )}

        {/* ══════════════ PORTFOLIO TAB ══════════════ */}
        {tab === "portfolio" && (
          <>
            {loading || !portfolioSummary ? (
              <Spinner />
            ) : (
              <div className="portfolio-grid">
                {/* Summary */}
                <Card>
                  <div className="section-label">
                    TOTAL PORTFOLIO ({TICKERS.length} × ${PORTFOLIO_INITIAL_CASH.toLocaleString()} per ticker)
                  </div>
                  {portfolioSummary.period != null && (
                    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 12 }}>
                      Same window as test.py: yfinance period {portfolioSummary.period}, train/test {((portfolioSummary.train_test_split ?? 0.8) * 100).toFixed(0)}/
                      {((1 - (portfolioSummary.train_test_split ?? 0.8)) * 100).toFixed(0)} split; {portfolioSummary.simulation_rule ?? "max(240, n_test − 1 − SEQ_LEN) steps on test split"}.
                    </div>
                  )}
                  <div className="summary-row">
                    <div>
                      <div className="stat-label">D3QN</div>
                      <div className="big-value accent-green">
                        ${portfolioSummary.total_dqn.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </div>
                      <div className={portfolioSummary.total_dqn_return >= 0 ? "stat-ret pos" : "stat-ret neg"}>
                        {portfolioSummary.total_dqn_return >= 0 ? "+" : ""}{portfolioSummary.total_dqn_return}%
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div className="stat-label">Buy & Hold</div>
                      <div className="big-value accent-orange">
                        ${portfolioSummary.total_bh.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </div>
                      <div className={portfolioSummary.total_bh_return >= 0 ? "stat-ret orange" : "stat-ret neg"}>
                        {portfolioSummary.total_bh_return >= 0 ? "+" : ""}{portfolioSummary.total_bh_return}%
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Bar chart */}
                <Card>
                  <div className="section-label">RETURN COMPARISON (%)</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={TICKERS.map((t) => {
                      const row = portfolioSummary.tickers[t];
                      return {
                        ticker: t,
                        D3QN: row?.dqn_ret ?? 0,
                        "Buy&Hold": row?.bh_ret ?? 0,
                      };
                    })}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                      <XAxis dataKey="ticker" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                      <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
                      <Tooltip contentStyle={{
                        background: "#1a1a2e", border: "1px solid #2a2a4a",
                        borderRadius: 10, color: "#e2e8f0",
                      }} />
                      <Bar dataKey="D3QN" fill="#c3f73a" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Buy&Hold" fill="#f39c12" radius={[4, 4, 0, 0]} />
                      <Legend />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                {/* Table */}
                <Card style={{ gridColumn: "1 / -1" }}>
                  <div className="section-label">PER-TICKER BREAKDOWN (Live from Model)</div>
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>D3QN Final</th>
                        <th>D3QN Return</th>
                        <th>B&H Final</th>
                        <th>B&H Return</th>
                        <th>Alpha</th>
                      </tr>
                    </thead>
                    <tbody>
                      {TICKERS.map((t) => {
                        const r = portfolioSummary.tickers[t];
                        if (!r) {
                          return (
                            <tr key={t}>
                              <td style={{ fontWeight: 700, color: COLORS[t] ?? "#94a3b8" }}>{t}</td>
                              <td colSpan={5} style={{ color: "#64748b" }}>No data (ticker mismatch with API)</td>
                            </tr>
                          );
                        }
                        if (r.error) {
                          return (
                            <tr key={t}>
                              <td style={{ fontWeight: 700, color: COLORS[t] ?? "#94a3b8" }}>{t}</td>
                              <td colSpan={5} style={{ color: "#f87171", fontSize: 13 }}>{r.error}</td>
                            </tr>
                          );
                        }
                        const alpha = (r.dqn_ret - r.bh_ret).toFixed(2);
                        return (
                          <tr key={t}>
                            <td style={{ fontWeight: 700, color: COLORS[t] ?? "#94a3b8" }}>{t}</td>
                            <td className="mono">${r.dqn.toLocaleString()}</td>
                            <td className={`mono bold ${r.dqn_ret >= 0 ? "pos" : "neg"}`}>
                              {r.dqn_ret >= 0 ? "+" : ""}{r.dqn_ret}%
                            </td>
                            <td className="mono">${r.bh.toLocaleString()}</td>
                            <td className={`mono bold ${r.bh_ret >= 0 ? "orange" : "neg"}`}>
                              {r.bh_ret >= 0 ? "+" : ""}{r.bh_ret}%
                            </td>
                            <td className={`mono bold ${Number(alpha) >= 0 ? "accent" : "neg"}`}>
                              {Number(alpha) >= 0 ? "+" : ""}{alpha}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </Card>
              </div>
            )}
          </>
        )}
      </main>

      <footer className="dog-track" aria-hidden="true">
        <div className="dog-scene">
          <div className="dog-sprite">
            <div className="dog-welcome">Welcome to the CS5130 project</div>
            <DogSVG />
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;