import React, { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend, Area, AreaChart,
} from "recharts";
import "./App.css";

const API = "https://dooly-melanoid-gerald.ngrok-free.dev";

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

/** Same as /api/backtest and /api/portfolio/summary query params */
const PORTFOLIO_INITIAL_CASH = 2000;
const PORTFOLIO_TEST_DAYS = 100;
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

  // Check API on mount
  useEffect(() => {
    apiFetch(`${API}/api/tickers`)
      .then(readApiJson)
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  // Chart tab only: avoids running /predict while on other tabs (shared error + loading fights portfolio/backtest).
  // allSettled: price chart still loads if predict fails.
  useEffect(() => {
    if (!apiOnline || tab !== "chart") return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPriceData(null);
    setPrediction(null);
    Promise.allSettled([
      apiFetch(`${API}/api/price/${selectedTicker}?period=6mo`).then(readApiJson),
      apiFetch(`${API}/api/predict/${selectedTicker}`).then(readApiJson),
    ]).then(([priceResult, predResult]) => {
      if (cancelled) return;
      if (priceResult.status === "fulfilled") {
        setPriceData(priceResult.value);
      } else {
        setError(priceResult.reason?.message || "Failed to load price data");
      }
      if (predResult.status === "fulfilled") {
        setPrediction(predResult.value);
      } else {
        setPrediction(null);
      }
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedTicker, apiOnline, tab]);

  // Fetch backtest when tab = backtest
  useEffect(() => {
    if (!apiOnline || tab !== "backtest") return;
    let cancelled = false;
    setError(null);
    setLoading(true);
    setBacktestData(null);
    apiFetch(
      `${API}/api/backtest/${selectedTicker}?initial_cash=${PORTFOLIO_INITIAL_CASH}&test_days=${PORTFOLIO_TEST_DAYS}`,
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
      `${API}/api/portfolio/summary?initial_cash=${PORTFOLIO_INITIAL_CASH}&test_days=${PORTFOLIO_TEST_DAYS}`,
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
                <div className="section-label">D3QN LIVE SIGNAL</div>
                {loading ? (
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
                    {prediction.predicted_volume != null && (
                      <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 12 }}>
                        Predicted next trade size: <strong style={{ color: "#e2e8f0" }}>{prediction.predicted_volume}</strong> shares
                      </div>
                    )}
                    <div className="q-label">Q-VALUES (from model)</div>
                    {Object.entries(prediction.q_values ?? {}).map(([k, v]) => (
                      <div key={k} className="q-row">
                        <span className="q-name">{k}</span>
                        <span className={`q-val ${v > 0 ? "pos" : v < 0 ? "neg" : ""}`}>
                          {typeof v === "number" ? `${v > 0 ? "+" : ""}${v.toFixed(4)}` : String(v)}
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
                      D3QN vs Buy & Hold — ${backtestData.initial_cash.toLocaleString()} initial
                      — {backtestData.test_days} trading days
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
                    TOTAL PORTFOLIO ({TICKERS.length} × ${PORTFOLIO_INITIAL_CASH.toLocaleString()} — {PORTFOLIO_TEST_DAYS}d backtest each)
                  </div>
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
    </div>
  );
}

export default App;