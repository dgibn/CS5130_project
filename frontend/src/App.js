import React, { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Legend, Area, AreaChart,
} from "recharts";
import "./App.css";

const API = "http://localhost:8000";
const TICKERS = ["GOOGL", "AAPL", "MSFT", "AMZN", "META"];
const COLORS = {
  GOOGL: "#ea4335", AAPL: "#2ecc71", MSFT: "#9b59b6",
  AMZN: "#f39c12", META: "#3498db",
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
  const [simData, setSimData] = useState(null);
  const [simDays, setSimDays] = useState(10);
  const [simCash, setSimCash] = useState(10000);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(null);

  // Check API on mount
  useEffect(() => {
    fetch(`${API}/api/tickers`)
      .then((r) => r.json())
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  // Fetch price + prediction when ticker changes
  useEffect(() => {
    if (!apiOnline) return;
    setLoading(true);
    setError(null);
    setPriceData(null);
    setPrediction(null);
    Promise.all([
      fetch(`${API}/api/price/${selectedTicker}?period=6mo`).then((r) => r.json()),
      fetch(`${API}/api/predict/${selectedTicker}`).then((r) => r.json()),
    ])
      .then(([price, pred]) => {
        setPriceData(price);
        setPrediction(pred);
        setLoading(false);
      })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [selectedTicker, apiOnline]);

  // Fetch backtest when tab = backtest
  useEffect(() => {
    if (!apiOnline || tab !== "backtest") return;
    setLoading(true);
    setBacktestData(null);
    fetch(`${API}/api/backtest/${selectedTicker}?initial_cash=2000`)
      .then((r) => r.json())
      .then((data) => { setBacktestData(data); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [selectedTicker, tab, apiOnline]);

  // Fetch portfolio summary when tab = portfolio
  useEffect(() => {
    if (!apiOnline || tab !== "portfolio") return;
    setLoading(true);
    setPortfolioSummary(null);
    fetch(`${API}/api/portfolio/summary`)
      .then((r) => r.json())
      .then((data) => { setPortfolioSummary(data); setLoading(false); })
      .catch((e) => { setError(e.message); setLoading(false); });
  }, [tab, apiOnline]);

  // ── API Offline ──
  if (apiOnline === false) {
    return (
      <div className="offline-screen">
        <div style={{ fontSize: 48 }}>⚡</div>
        <h2>API Offline</h2>
        <p>Start the backend server first:</p>
        <code>python api.py</code>
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
          {["chart", "backtest", "simulate", "portfolio"].map((t) => (
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
        {/* ── TICKER SELECTOR ── */}
        <div className="ticker-bar">
          {TICKERS.map((t) => (
            <button
              key={t}
              onClick={() => setSelectedTicker(t)}
              className="ticker-btn"
              style={{
                background: selectedTicker === t ? COLORS[t] : "#1a1a2e",
                color: selectedTicker === t ? "#fff" : "#94a3b8",
                borderColor: selectedTicker === t ? COLORS[t] : "#2a2a4a",
              }}
            >
              {t}
            </button>
          ))}
        </div>

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
                          <stop offset="5%" stopColor={COLORS[selectedTicker]} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={COLORS[selectedTicker]} stopOpacity={0} />
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
                      <Area type="monotone" dataKey="close" stroke={COLORS[selectedTicker]}
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
                {!prediction ? (
                  <Spinner />
                ) : (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16 }}>
                      <ActionBadge action={prediction.action} />
                      <span style={{ fontSize: 14, color: "#94a3b8" }}>
                        {prediction.confidence}% confidence
                      </span>
                    </div>
                    <div className="q-label">Q-VALUES (from model)</div>
                    {Object.entries(prediction.q_values).map(([k, v]) => (
                      <div key={k} className="q-row">
                        <span className="q-name">{k}</span>
                        <span className={`q-val ${v > 0 ? "pos" : v < 0 ? "neg" : ""}`}>
                          {v > 0 ? "+" : ""}{v.toFixed(4)}
                        </span>
                      </div>
                    ))}
                    <div className="model-info">
                      Price: ${prediction.price} | Model: D3QN (Generalized)
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

        {/* ══════════════ SIMULATE TAB ══════════════ */}
        {tab === "simulate" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Input controls */}
            <Card>
              <div className="section-label">FORWARD SIMULATION SETTINGS</div>
              <div style={{ display: "flex", gap: 24, alignItems: "flex-end", flexWrap: "wrap" }}>
                <div>
                  <label style={{ fontSize: 12, color: "#64748b", display: "block", marginBottom: 6 }}>
                    Investment Amount ($)
                  </label>
                  <input
                    type="number"
                    value={simCash}
                    onChange={(e) => setSimCash(Number(e.target.value))}
                    min={100}
                    max={1000000}
                    style={{
                      background: "#0f0f23", border: "1px solid #2a2a4a", borderRadius: 10,
                      padding: "10px 16px", color: "#e2e8f0", fontSize: 16, fontWeight: 700,
                      width: 180, fontFamily: "monospace",
                    }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: "#64748b", display: "block", marginBottom: 6 }}>
                    Days Forward
                  </label>
                  <input
                    type="number"
                    value={simDays}
                    onChange={(e) => setSimDays(Number(e.target.value))}
                    min={1}
                    max={60}
                    style={{
                      background: "#0f0f23", border: "1px solid #2a2a4a", borderRadius: 10,
                      padding: "10px 16px", color: "#e2e8f0", fontSize: 16, fontWeight: 700,
                      width: 120, fontFamily: "monospace",
                    }}
                  />
                </div>
                <button
                  onClick={() => {
                    setLoading(true);
                    setSimData(null);
                    setError(null);
                    fetch(`${API}/api/simulate/${selectedTicker}?days=${simDays}&initial_cash=${simCash}&n_simulations=50`)
                      .then((r) => r.json())
                      .then((data) => { setSimData(data); setLoading(false); })
                      .catch((e) => { setError(e.message); setLoading(false); });
                  }}
                  style={{
                    padding: "10px 28px", borderRadius: 10, border: "none", cursor: "pointer",
                    background: "linear-gradient(135deg, #c3f73a, #95d600)",
                    color: "#0f0f23", fontSize: 15, fontWeight: 800,
                    letterSpacing: 0.5,
                  }}
                >
                  Run Simulation
                </button>
              </div>
              <div style={{ marginTop: 10, fontSize: 11, color: "#64748b" }}>
                Uses Monte Carlo simulation (50 paths) with D3QN agent decisions on each path.
              </div>
            </Card>

            {loading && <Spinner />}

            {simData && !loading && (
              <>
                {/* Summary cards */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                  <Card>
                    <div className="stat-label">Expected Portfolio</div>
                    <div className="big-value accent-green">
                      ${simData.final_avg_portfolio.toLocaleString()}
                    </div>
                    <div className={simData.expected_return >= 0 ? "stat-ret pos" : "stat-ret neg"}>
                      {simData.expected_return >= 0 ? "+" : ""}{simData.expected_return}%
                    </div>
                  </Card>
                  <Card>
                    <div className="stat-label">Best Case</div>
                    <div className="big-value" style={{ color: "#2ecc71" }}>
                      ${simData.final_best_portfolio.toLocaleString()}
                    </div>
                    <div className="stat-ret pos">
                      +{((simData.final_best_portfolio / simData.initial_cash - 1) * 100).toFixed(2)}%
                    </div>
                  </Card>
                  <Card>
                    <div className="stat-label">Worst Case</div>
                    <div className="big-value" style={{ color: "#e74c3c" }}>
                      ${simData.final_worst_portfolio.toLocaleString()}
                    </div>
                    <div className="stat-ret neg">
                      {((simData.final_worst_portfolio / simData.initial_cash - 1) * 100).toFixed(2)}%
                    </div>
                  </Card>
                </div>

                {/* Portfolio trajectory chart */}
                <Card>
                  <div className="section-label">
                    PROJECTED PORTFOLIO — {selectedTicker} — {simData.days} Days — ${simData.initial_cash.toLocaleString()}
                  </div>
                  <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={simData.trajectory}>
                      <defs>
                        <linearGradient id="simGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#c3f73a" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#c3f73a" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                      <XAxis dataKey="day" tick={{ fill: "#64748b", fontSize: 11 }}
                        label={{ value: "Day", position: "insideBottom", offset: -5, fill: "#64748b" }} />
                      <YAxis tick={{ fill: "#64748b", fontSize: 11 }}
                        domain={["auto", "auto"]} />
                      <Tooltip contentStyle={{
                        background: "#1a1a2e", border: "1px solid #2a2a4a",
                        borderRadius: 10, color: "#e2e8f0",
                      }}
                        formatter={(v) => [`${Number(v).toLocaleString()}`, ""]}
                      />
                      <Area type="monotone" dataKey="best_portfolio" stroke="transparent"
                        fill="#2ecc71" fillOpacity={0.08} name="Best case" />
                      <Area type="monotone" dataKey="p75_portfolio" stroke="transparent"
                        fill="#c3f73a" fillOpacity={0.1} name="75th percentile" />
                      <Line type="monotone" dataKey="avg_portfolio" stroke="#c3f73a"
                        strokeWidth={2.5} dot={false} name="Expected" />
                      <Area type="monotone" dataKey="p25_portfolio" stroke="transparent"
                        fill="#f39c12" fillOpacity={0.08} name="25th percentile" />
                      <Area type="monotone" dataKey="worst_portfolio" stroke="transparent"
                        fill="#e74c3c" fillOpacity={0.08} name="Worst case" />
                      <Legend />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>

                {/* Day-by-day actions table */}
                <Card>
                  <div className="section-label">D3QN AGENT'S PREDICTED ACTIONS</div>
                  <div style={{ maxHeight: 300, overflowY: "auto" }}>
                    <table className="results-table">
                      <thead>
                        <tr>
                          <th style={{ textAlign: "left" }}>Day</th>
                          <th>Action</th>
                          <th>Avg Price</th>
                          <th>Expected Value</th>
                          <th>Best Case</th>
                          <th>Worst Case</th>
                        </tr>
                      </thead>
                      <tbody>
                        {simData.trajectory.slice(1).map((row) => (
                          <tr key={row.day}>
                            <td style={{ textAlign: "left" }}>Day {row.day}</td>
                            <td><ActionBadge action={row.action} /></td>
                            <td className="mono">${row.avg_price.toLocaleString()}</td>
                            <td className="mono">${row.avg_portfolio.toLocaleString()}</td>
                            <td className="mono pos">${row.best_portfolio.toLocaleString()}</td>
                            <td className="mono neg">${row.worst_portfolio.toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </>
            )}
          </div>
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
                  <div className="section-label">TOTAL PORTFOLIO (5 × $2,000)</div>
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
                      <div className="stat-ret orange">
                        +{portfolioSummary.total_bh_return}%
                      </div>
                    </div>
                  </div>
                </Card>

                {/* Bar chart */}
                <Card>
                  <div className="section-label">RETURN COMPARISON (%)</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={TICKERS.map((t) => ({
                      ticker: t,
                      D3QN: portfolioSummary.tickers[t].dqn_ret,
                      "Buy&Hold": portfolioSummary.tickers[t].bh_ret,
                    }))}>
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
                        const alpha = (r.dqn_ret - r.bh_ret).toFixed(2);
                        return (
                          <tr key={t}>
                            <td style={{ fontWeight: 700, color: COLORS[t] }}>{t}</td>
                            <td className="mono">${r.dqn.toLocaleString()}</td>
                            <td className={`mono bold ${r.dqn_ret >= 0 ? "pos" : "neg"}`}>
                              {r.dqn_ret >= 0 ? "+" : ""}{r.dqn_ret}%
                            </td>
                            <td className="mono">${r.bh.toLocaleString()}</td>
                            <td className={`mono bold ${r.bh_ret >= 0 ? "orange" : "neg"}`}>
                              {r.bh_ret >= 0 ? "+" : ""}{r.bh_ret}%
                            </td>
                            <td className={`mono bold ${alpha >= 0 ? "accent" : "neg"}`}>
                              {alpha >= 0 ? "+" : ""}{alpha}%
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