import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Database,
  Download,
  ExternalLink,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  LockKeyhole,
  Menu,
  RefreshCw,
  Rocket,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Workflow,
  X,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const TIMEFRAMES = ["M5", "M15", "H1", "H4"];
const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "analytics", label: "Model analytics", icon: BarChart3 },
  { id: "training", label: "Training center", icon: Rocket },
  { id: "data", label: "Data center", icon: Database },
  { id: "predictions", label: "Predictions", icon: Activity },
  { id: "settings", label: "Settings", icon: SlidersHorizontal },
];

function readToken() {
  try {
    return window.localStorage.getItem("aurummind-api-token") || "";
  } catch {
    return "";
  }
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body) headers.set("Content-Type", "application/json");
  const token = readToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_URL}${path}`, { ...options, headers });
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text;
  }
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(
      typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `Request failed (${response.status})`,
    );
  }
  return payload;
}

function formatNumber(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : "—";
}

function formatDate(value, options = {}) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: options.dateOnly ? undefined : "short" });
}

function percent(value) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function getErrorMessage(error) {
  return error instanceof Error ? error.message : "Something went wrong.";
}

function useRemoteData(path, deps = [], enabled = true) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const reload = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    try {
      const value = await api(path);
      setData(value);
      setError("");
    } catch (requestError) {
      setError(getErrorMessage(requestError));
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, enabled, ...deps]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { data, error, loading, reload, setData };
}

function emaPoints(rows, span) {
  const alpha = 2 / (span + 1);
  let current;
  return rows.map((row) => {
    current = current === undefined ? row.close : row.close * alpha + current * (1 - alpha);
    return { ...row, [`ema${span}`]: current };
  });
}

function App() {
  const [page, setPage] = useState("overview");
  const [timeframe, setTimeframe] = useState("M15");
  const [mobileNav, setMobileNav] = useState(false);
  const [token, setToken] = useState(readToken);
  const [toast, setToast] = useState(null);
  const [busy, setBusy] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const connection = useRemoteData("/api/health", [refreshKey]);

  const notify = useCallback((message, kind = "success") => {
    setToast({ message, kind });
    window.setTimeout(() => setToast(null), 4200);
  }, []);

  const reloadAll = useCallback(() => setRefreshKey((key) => key + 1), []);

  const perform = useCallback(
    async (key, path, body, message, method = "POST") => {
      setBusy(key);
      try {
        const result = await api(path, {
          method,
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
        notify(message || `${result?.job || "Operation"} queued.`);
        reloadAll();
        return result;
      } catch (error) {
        notify(getErrorMessage(error), "error");
        return null;
      } finally {
        setBusy("");
      }
    },
    [notify, reloadAll],
  );

  const title = NAV_ITEMS.find((item) => item.id === page)?.label || "Overview";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Sparkles size={19} /></div>
          <div className="brand-copy"><strong>AURUM<span>MIND</span></strong><small>MARKET INTELLIGENCE</small></div>
          <button className="icon-button mobile-close" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={18} /></button>
        </div>
        <div className="workspace-switch"><div className="workspace-avatar">A</div><div><b>Research desk</b><small>Personal workspace</small></div><ChevronDown size={15} /></div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main navigation">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`nav-item ${page === id ? "active" : ""}`} onClick={() => { setPage(id); setMobileNav(false); }}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{id === "training" && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="market-badge"><span className="live-dot" /><span>Research mode</span><span className="badge-separator" />No execution</div>
          <button className="help-link" onClick={() => setPage("settings")}><CircleHelp size={16} />API connection help</button>
          <div className="sidebar-user"><div className="user-avatar">AM</div><div><b>AurumMind</b><small>Local instance</small></div><Settings size={16} /></div>
        </div>
      </aside>
      {mobileNav && <button className="mobile-backdrop" onClick={() => setMobileNav(false)} aria-label="Close navigation" />}

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-button menu-button" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={19} /></button><div className="breadcrumb">Workspace <span>/</span> <b>{title}</b></div></div>
          <div className="topbar-actions"><div className={`connection-pill ${connection.error ? "connection-down" : ""}`}><span className="live-dot" />API <span className="connection-label">{connection.error ? "unavailable" : "connected"}</span></div><button className="icon-button notification-button" aria-label="Notifications"><Bell size={17} /><i /></button><button className="top-avatar" title="Settings" onClick={() => setPage("settings")}>AM</button></div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div><div className="eyebrow"><span className="eyebrow-dot" />XAUUSD RESEARCH PLATFORM</div><h1>{title}</h1><p>{pageDescription(page)}</p></div>
            <div className="heading-actions"><div className="timeframe-select"><Clock3 size={15} /><select value={timeframe} onChange={(event) => setTimeframe(event.target.value)} aria-label="Select timeframe">{TIMEFRAMES.map((item) => <option key={item}>{item}</option>)}</select><ChevronDown size={14} /></div><button className="button button-outline" onClick={reloadAll}><RefreshCw size={15} /> Refresh</button></div>
          </div>

          {page === "overview" && <Overview timeframe={timeframe} refreshKey={refreshKey} onNavigate={setPage} />}
          {page === "analytics" && <Analytics refreshKey={refreshKey} perform={perform} busy={busy} />}
          {page === "training" && <TrainingCenter timeframe={timeframe} refreshKey={refreshKey} perform={perform} busy={busy} notify={notify} />}
          {page === "data" && <DataCenter timeframe={timeframe} refreshKey={refreshKey} />}
          {page === "predictions" && <PredictionHistory refreshKey={refreshKey} />}
          {page === "settings" && <SettingsPage token={token} setToken={setToken} notify={notify} refreshKey={refreshKey} />}

          <footer className="page-footer"><span>◈ AurumMind AI <span className="footer-divider">•</span> Historical research only</span><span>Not financial advice · No broker execution</span></footer>
        </div>
      </main>
      {toast && <div className={`toast toast-${toast.kind}`} role="status">{toast.kind === "error" ? <AlertCircle size={17} /> : <Check size={17} />}{toast.message}</div>}
    </div>
  );
}

function pageDescription(page) {
  return {
    overview: "A clear view of the market, model, and latest forecast.",
    analytics: "Inspect champion performance and registered model versions.",
    training: "Manage data preparation, training, and the scheduled worker.",
    data: "Review candle coverage, provider provenance, and quality checks.",
    predictions: "Track stored forecasts and their subsequently observed outcomes.",
    settings: "Connect this interface to your local API and review safe runtime settings.",
  }[page];
}

function Card({ className = "", children }) {
  return <section className={`card ${className}`}>{children}</section>;
}

function SectionTitle({ title, note, action }) {
  return <div className="section-title"><div><h2>{title}</h2>{note && <p>{note}</p>}</div>{action}</div>;
}

function MetricCard({ label, value, note, icon: Icon, tone = "" }) {
  return <Card className="metric-card"><div className="metric-top"><span>{label}</span><span className={`metric-icon ${tone}`}>{Icon && <Icon size={16} />}</span></div><strong>{value}</strong><small>{note}</small></Card>;
}

function EmptyState({ icon: Icon = Database, title, detail, action }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={21} /></div><h3>{title}</h3><p>{detail}</p>{action}</div>;
}

function ErrorNotice({ error }) {
  if (!error) return null;
  return <div className="inline-error"><AlertCircle size={16} /><span>{error}</span></div>;
}

function Loading({ label = "Loading data…" }) {
  return <div className="loading-state"><LoaderCircle size={18} className="spin" />{label}</div>;
}

function StatusBadge({ children, variant = "" }) {
  return <span className={`status-badge ${variant}`}><i />{children}</span>;
}

function Overview({ timeframe, refreshKey, onNavigate }) {
  const query = `?timeframe=${timeframe}`;
  const market = useRemoteData(`/api/market/latest${query}`, [refreshKey, timeframe]);
  const history = useRemoteData(`/api/market/history${query}&limit=260`, [refreshKey, timeframe]);
  const forecast = useRemoteData(`/api/prediction/latest${query}`, [refreshKey, timeframe]);
  const health = useRemoteData("/api/health", [refreshKey]);
  const data = useRemoteData(`/api/data/status${query}`, [refreshKey, timeframe]);

  const chartRows = useMemo(() => {
    const rows = Array.isArray(history.data) ? history.data : [];
    return emaPoints(rows, 20).map((row, index) => ({ ...row, timeLabel: new Date(row.timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric" }), idx: index }));
  }, [history.data]);

  const isStale = useMemo(() => {
    const stamp = market.data?.candle?.timestamp;
    if (!stamp) return true;
    const interval = { M5: 300, M15: 900, H1: 3600, H4: 14400 }[timeframe];
    return (Date.now() - new Date(stamp).getTime() - interval * 1000) > interval * 2000;
  }, [market.data, timeframe]);

  const prediction = forecast.data?.status === "available" ? forecast.data : null;
  const candle = market.data?.candle;
  const source = market.data?.source;

  return (
    <>
      <div className="market-strip">
        <div className="instrument"><div className="gold-icon">Au</div><div><strong>Gold / US Dollar</strong><small>XAUUSD <span>·</span> {timeframe}</small></div></div>
        <div className="strip-price"><strong>{formatNumber(candle?.close)}</strong><span className={isStale ? "text-warning" : "text-green"}>{isStale ? "STALE DATA" : "MARKET DATA"}</span></div>
        <div className="strip-meta"><span>Provider</span><strong>{source?.provider || data.data?.provider || "Not connected"}</strong></div>
        <div className="strip-meta"><span>Last completed candle</span><strong>{formatDate(candle?.timestamp)}</strong></div>
        <div className="strip-status"><StatusBadge variant={isStale ? "warning" : "success"}>{isStale ? "Needs update" : "Latest"}</StatusBadge></div>
      </div>
      <div className="metrics-grid">
        <MetricCard label="Latest close" value={formatNumber(candle?.close)} note={`Last ${timeframe} candle`} icon={Activity} tone="icon-gold" />
        <MetricCard label="Validated candles" value={data.data?.candle_count?.toLocaleString() || "—"} note={data.data?.last_update ? `Updated ${formatDate(data.data.last_update)}` : "No updates recorded"} icon={Database} tone="icon-blue" />
        <MetricCard label="Champion model" value={health.data?.champion || "Not ready"} note={health.error ? "API unavailable" : "Active registry version"} icon={Sparkles} tone="icon-purple" />
        <MetricCard label="Data quality" value={market.data?.data_quality?.quality_score != null ? `${formatNumber(market.data.data_quality.quality_score, 1)}%` : "—"} note={market.data?.data_quality?.passed_validation ? "Passed validation" : "See data center"} icon={ShieldCheck} tone="icon-green" />
      </div>
      <div className="dashboard-grid">
        <Card className="chart-card">
          <SectionTitle title="Price action" note={`${timeframe} provider candles · recent history`} action={<span className="chart-legend"><i />Close <i className="legend-ema" />EMA 20</span>} />
          {history.loading ? <Loading /> : history.error ? <ErrorNotice error={history.error} /> : chartRows.length ? (
            <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartRows} margin={{ top: 16, right: 8, left: 0, bottom: 0 }}>
              <defs><linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#d6aa59" stopOpacity={0.22} /><stop offset="100%" stopColor="#d6aa59" stopOpacity={0.01} /></linearGradient></defs>
              <CartesianGrid stroke="#202832" vertical={false} />
              <XAxis dataKey="idx" tickFormatter={(index) => chartRows[index]?.timeLabel || ""} tick={{ fill: "#737f8c", fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={45} />
              <YAxis orientation="right" domain={["auto", "auto"]} tickFormatter={(value) => Number(value).toFixed(1)} tick={{ fill: "#737f8c", fontSize: 11 }} axisLine={false} tickLine={false} width={52} />
              <Tooltip content={<PriceTooltip />} />
              <Area type="monotone" dataKey="close" stroke="#d7ad62" strokeWidth={2} fill="url(#priceFill)" dot={false} activeDot={{ r: 4, fill: "#e5bd75", stroke: "#121820" }} />
              <Line type="monotone" dataKey="ema20" stroke="#8495a4" strokeWidth={1.3} strokeDasharray="4 4" dot={false} activeDot={false} />
            </AreaChart></ResponsiveContainer></div>
          ) : <EmptyState title="No market history yet" detail="Download validated candles from the Training center to populate the chart." action={<button className="button button-gold" onClick={() => onNavigate("training")}>Open training center <ArrowUpRight size={15} /></button>} />}
          <div className="chart-footnote"><span><span className="live-dot" /> Provider candle data</span><span>Volume may represent provider tick volume</span></div>
        </Card>
        <Card className="forecast-card">
          <SectionTitle title="Latest forecast" note="Calibrated class probabilities" action={<span className="forecast-mark"><Sparkles size={15} /></span>} />
          {forecast.loading ? <Loading /> : prediction ? <ForecastPanel prediction={prediction} /> : (
            <EmptyState icon={Gauge} title="Forecast unavailable" detail={forecast.error || forecast.data?.reason || "A validated champion model and fresh candles are required."} />
          )}
          <div className="forecast-disclaimer"><ShieldCheck size={14} />Research only · no orders are placed</div>
        </Card>
      </div>
      <div className="lower-grid">
        <Card className="quality-card">
          <SectionTitle title="Data health" note="Coverage and provenance" action={<button className="text-button" onClick={() => onNavigate("data")}>View data <ArrowUpRight size={14} /></button>} />
          <div className="health-row"><div className="health-icon"><Database size={16} /></div><div className="health-copy"><b>Historical coverage</b><small>{data.data?.last_candle ? `Latest candle ${formatDate(data.data.last_candle)}` : "No processed candles available"}</small></div><StatusBadge variant={data.data?.candle_count ? "success" : "muted"}>{data.data?.candle_count ? "Available" : "Empty"}</StatusBadge></div>
          <div className="health-row"><div className="health-icon"><ShieldCheck size={16} /></div><div className="health-copy"><b>Quality gate</b><small>{market.data?.data_quality?.provider || data.data?.provider || "Awaiting provider data"}</small></div><StatusBadge variant={market.data?.data_quality?.passed_validation ? "success" : "warning"}>{market.data?.data_quality?.passed_validation ? "Passed" : "Review"}</StatusBadge></div>
          <div className="health-row"><div className="health-icon"><Workflow size={16} /></div><div className="health-copy"><b>Data pipeline</b><small>Download → validate → generate features</small></div><button className="mini-arrow" onClick={() => onNavigate("training")} aria-label="Open training center"><ArrowUpRight size={15} /></button></div>
        </Card>
        <Card className="setup-card">
          <SectionTitle title="Continue setup" note="Prepare the forecasting workflow" />
          <div className="setup-steps"><div className={`setup-step ${data.data?.candle_count ? "step-done" : ""}`}><span>{data.data?.candle_count ? <Check size={13} /> : "1"}</span><div><b>Provider history</b><small>{data.data?.candle_count ? `${data.data.candle_count.toLocaleString()} candles loaded` : "Download historical candles"}</small></div></div><div className={`setup-step ${health.data?.champion ? "step-done" : ""}`}><span>{health.data?.champion ? <Check size={13} /> : "2"}</span><div><b>Train a champion</b><small>{health.data?.champion ? "Model is active" : "Generate features, then train"}</small></div></div></div>
          <button className="button button-gold setup-button" onClick={() => onNavigate("training")}>Go to training center <ArrowUpRight size={15} /></button>
        </Card>
      </div>
      <ErrorNotice error={health.error} />
    </>
  );
}

function PriceTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  return <div className="chart-tooltip"><small>{formatDate(point?.timestamp)}</small><b>${formatNumber(point?.close)}</b><span>EMA 20 · ${formatNumber(point?.ema20)}</span></div>;
}

function ForecastPanel({ prediction }) {
  const probabilities = prediction.probabilities || {};
  const confidence = Math.max(...Object.values(probabilities).filter((value) => typeof value === "number"), 0);
  return <div className="forecast-content">
    <div className={`prediction-label prediction-${String(prediction.prediction).toLowerCase()}`}><span className="prediction-symbol">{prediction.prediction === "UP" ? <ArrowUpRight size={21} /> : prediction.prediction === "DOWN" ? <ArrowDownRight size={21} /> : <Activity size={20} />}</span><div><small>PREDICTED DIRECTION</small><strong>{prediction.prediction}</strong></div><span className="confidence-chip">{percent(confidence)} confidence</span></div>
    <div className="probability-title"><span>Class probability</span><span>0—100%</span></div>
    {["UP", "DOWN", "NEUTRAL"].map((label) => <div className="probability-row" key={label}><div><span className={`prob-dot prob-${label.toLowerCase()}`} />{label}<b>{percent(probabilities[label])}</b></div><div className="progress-track"><span className={`progress-${label.toLowerCase()}`} style={{ width: `${Math.min(100, Math.max(0, (probabilities[label] || 0) * 100))}%` }} /></div></div>)}
    <div className="forecast-details"><div><small>Model version</small><b>{prediction.model_version || "—"}</b></div><div><small>Target time</small><b>{formatDate(prediction.target_timestamp)}</b></div></div>
    <p className="forecast-meta">Input candle {formatDate(prediction.input_candle_timestamp)} · {prediction.forecast_horizon_minutes || "—"} min horizon</p>
  </div>;
}

function Analytics({ refreshKey, perform, busy }) {
  const metrics = useRemoteData("/api/models/metrics", [refreshKey]);
  const champion = useRemoteData("/api/models/champion", [refreshKey]);
  const models = useRemoteData("/api/models", [refreshKey]);
  const metric = metrics.data;
  return <>
    <div className="metrics-grid analytics-metrics">
      <MetricCard label="Champion model" value={champion.data?.version || "Not ready"} note={champion.data?.model_name || champion.error || "No promoted candidate"} icon={Sparkles} tone="icon-purple" />
      <MetricCard label="Validation macro F1" value={formatNumber(metric?.validation?.macro_f1, 3)} note="Validation window" icon={Gauge} tone="icon-gold" />
      <MetricCard label="Validation Brier score" value={formatNumber(metric?.validation?.brier_score, 3)} note="Lower is better" icon={Activity} tone="icon-blue" />
      <MetricCard label="Registered models" value={models.data?.length?.toString() || "0"} note="Candidate and champion history" icon={Workflow} tone="icon-green" />
    </div>
    {metric && <div className="analytics-grid">
      <Card><SectionTitle title="Validation metrics" note="Champion validation window" /><MetricsTable data={metric.validation} /></Card>
      <Card><SectionTitle title="Walk-forward results" note="Across chronological splits" /><MetricsTable data={metric.walk_forward} /></Card>
      <Card><SectionTitle title="Untouched holdout" note="Historical test set; not a forecast of future returns" /><MetricsTable data={metric.holdout_test} /></Card>
      <Card><SectionTitle title="Cost-aware simulation" note="Configured spread, fee, and slippage assumptions" /><MetricsTable data={metric.economic_simulation} /></Card>
    </div>}
    {metrics.error && <Card><EmptyState icon={BarChart3} title="No champion metrics yet" detail={metrics.error} /></Card>}
    <Card className="table-card">
      <SectionTitle title="Model registry" note="Locally registered candidate versions" action={<button className="button button-outline button-small" disabled={busy === "rollback" || !models.data?.some((item) => item.is_champion)} onClick={() => window.confirm("Roll back to the previous champion model?") && perform("rollback", "/api/models/rollback", undefined, "Champion rollback completed.")}><RefreshCw size={14} />Rollback champion</button>} />
      {models.loading ? <Loading /> : models.error ? <ErrorNotice error={models.error} /> : models.data?.length ? <div className="table-scroll"><table><thead><tr><th>Version</th><th>Model</th><th>Registered</th><th>Macro F1</th><th>Brier score</th><th>Status</th></tr></thead><tbody>{models.data.map((item) => <tr key={item.version}><td className="mono">{item.version}</td><td>{item.model_name}</td><td>{formatDate(item.registered_at)}</td><td>{formatNumber(item.validation_macro_f1, 3)}</td><td>{formatNumber(item.validation_brier, 3)}</td><td><StatusBadge variant={item.is_champion ? "success" : "muted"}>{item.is_champion ? "Champion" : "Candidate"}</StatusBadge></td></tr>)}</tbody></table></div> : <EmptyState icon={Workflow} title="No candidate models" detail="Train baseline challengers from the Training center after generating features." />}
    </Card>
    <div className="disclaimer-bar"><ShieldCheck size={15} />All reported metrics are historical evaluations and do not imply future performance.</div>
  </>;
}

function MetricsTable({ data }) {
  if (!data || typeof data !== "object") return <p className="muted-copy">Metrics not available.</p>;
  const rows = Object.entries(data).filter(([, value]) => typeof value === "number" || typeof value === "string");
  return <div className="metrics-list">{rows.length ? rows.slice(0, 9).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><b>{typeof value === "number" ? formatNumber(value, 3) : value}</b></div>) : <p className="muted-copy">No scalar metrics recorded.</p>}</div>;
}

function TrainingCenter({ timeframe, refreshKey, perform, busy }) {
  const status = useRemoteData(`/api/data/status?timeframe=${timeframe}`, [refreshKey, timeframe]);
  const features = useRemoteData(`/api/features/status?timeframe=${timeframe}`, [refreshKey, timeframe]);
  const jobs = useRemoteData("/api/jobs", [refreshKey]);
  const worker = useRemoteData("/api/worker", [refreshKey]);
  const settings = useRemoteData("/api/settings", [refreshKey]);
  const health = useRemoteData("/api/health", [refreshKey]);
  const [startDate, setStartDate] = useState("2020-01-01");
  const [confirmed, setConfirmed] = useState(false);
  const [optimizeConfirmed, setOptimizeConfirmed] = useState(false);
  const isReady = Number(status.data?.candle_count || 0) > 0;
  const hasFeatures = Boolean(features.data?.available);
  const hasChampion = Boolean(health.data?.champion);
  const trainingTimeframe = settings.data?.market_data?.primary_timeframe || "M15";
  const canTrain = timeframe === trainingTimeframe && hasFeatures;

  useEffect(() => {
    if (settings.data?.market_data?.historical_start) setStartDate(settings.data.market_data.historical_start);
  }, [settings.data]);

  const runDownload = () => {
    if (!confirmed) return;
    const start = `${startDate}T00:00:00Z`;
    perform("download", "/api/data/download", { start, timeframe, confirm: true }, "Historical download queued.");
    setConfirmed(false);
  };

  return <>
    <div className="setup-banner"><div className="banner-icon"><Workflow size={19} /></div><div><b>Initial setup progress</b><p>Connect verified provider candles, prepare causal features, and train a promoted model.</p></div><div className="banner-steps"><span className={isReady ? "completed" : ""}>{isReady ? <Check size={12} /> : "1"} History</span><i /><span className={hasFeatures ? "completed" : ""}>{hasFeatures ? <Check size={12} /> : "2"} Features</span><i /><span className={hasChampion ? "completed" : ""}>{hasChampion ? <Check size={12} /> : "3"} Champion</span></div></div>
    <div className="training-grid">
      <div className="training-main">
        <Card className="training-card"><SectionTitle title="Market data" note="Historical download is explicit and may take a while." action={<StatusBadge variant={isReady ? "success" : "warning"}>{isReady ? "History available" : "Setup required"}</StatusBadge>} />
          <div className="warning-note"><AlertCircle size={17} /><span>Data download requests real historical Dukascopy candles. Review provider terms and confirm before starting. Synthetic samples are not accepted for training.</span></div>
          <div className="form-grid"><label>Timeframe<select value={timeframe} onChange={() => {}} disabled><option>{timeframe}</option></select><small>Choose the timeframe from the page header.</small></label><label>History start date<input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} max={new Date().toISOString().slice(0, 10)} /><small>Configured default: {settings.data?.market_data?.historical_start || "—"}</small></label></div>
          <label className="check-row"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>I confirm starting a historical provider data download from this date.</span></label>
          <div className="button-row"><button className="button button-gold" disabled={!confirmed || Boolean(busy)} onClick={runDownload}><Download size={15} />{busy === "download" ? "Queuing…" : "Download historical data"}</button><button className="button button-outline" disabled={!isReady || Boolean(busy)} onClick={() => perform("update", "/api/data/update", { timeframe }, "Candle update queued.")}><RefreshCw size={15} />Update completed candles</button></div>
        </Card>
        <Card className="training-card"><SectionTitle title="Prepare and train" note="Run the stages in order; each operation is recorded in Jobs." />
          <div className="pipeline-steps">
            <PipelineStep number="01" title="Validate market data" detail="Run integrity and coverage checks; save a quality report." done={Boolean(status.data?.quality_report?.passed_validation)} busy={busy === "validate"} button="Queue validation" disabled={!isReady || Boolean(busy)} onClick={() => perform("validate", "/api/data/validate", { timeframe }, "Data validation queued.")} />
            <PipelineStep number="02" title="Generate features" detail="Create causal indicators from the validated candle history." done={hasFeatures} busy={busy === "features"} button={hasFeatures ? "Regenerate features" : "Generate features"} disabled={!isReady || Boolean(busy)} onClick={() => perform("features", "/api/features/generate", { timeframe }, "Feature generation queued.")} />
            <PipelineStep number="03" title="Train baseline challengers" detail={`Fit configured classifiers on the configured ${trainingTimeframe} timeframe and promote only candidates that pass checks.`} done={hasChampion} busy={busy === "train"} button="Train baseline" disabled={!canTrain || Boolean(busy)} onClick={() => perform("train", "/api/training/start", undefined, "Baseline training queued.")} />
            <PipelineStep number="04" title="Optuna optimization" detail={`Run the configured hyperparameter trial budget${settings.data?.training?.maximum_optuna_trials ? ` (${settings.data.training.maximum_optuna_trials} trials)` : ""}. This can use substantial compute.`} done={false} busy={busy === "optimize"} button="Run optimization" disabled={!canTrain || !optimizeConfirmed || Boolean(busy)} onClick={() => perform("optimize", "/api/training/optimize", undefined, "Optuna optimization queued.")} />
          </div>
          <label className="check-row optimization-check"><input type="checkbox" checked={optimizeConfirmed} onChange={(event) => setOptimizeConfirmed(event.target.checked)} /><span>I confirm the configured optimization trial budget.</span></label>
          {!canTrain && isReady && <p className="muted-copy training-prerequisite">Generate features for {timeframe} to enable this stage. Model training uses the configured {trainingTimeframe} timeframe only.</p>}
        </Card>
        <Card className="training-card"><SectionTitle title="Scheduled worker" note="Polls for completed candles and performs configured scheduled tasks." action={<StatusBadge variant={worker.data?.status === "running" ? "success" : "muted"}>{worker.data?.status || "Unknown"}</StatusBadge>} />
          <p className="worker-description">The worker does not bootstrap a large historical download. It may update data, settle past predictions, and run scheduled model jobs according to configuration.</p>
          <div className="button-row"><button className="button button-gold" disabled={worker.data?.status === "running" || Boolean(busy)} onClick={() => perform("worker-start", "/api/worker/start", undefined, "Scheduled worker started.")}><Rocket size={15} />Start worker</button><button className="button button-outline" disabled={worker.data?.status !== "running" || Boolean(busy)} onClick={() => perform("worker-stop", "/api/worker/stop", undefined, "Worker stop requested.")}><X size={15} />Stop worker</button><button className="button button-outline" disabled={Boolean(busy)} onClick={() => perform("worker-cycle", "/api/worker/cycle", undefined, "One worker cycle queued.")}><Workflow size={15} />Run one cycle</button></div>
          {worker.data?.pid && <p className="muted-copy worker-pid">Worker process ID · <span className="mono">{worker.data.pid}</span></p>}
        </Card>
      </div>
      <div className="training-side">
        <Card><SectionTitle title="Pipeline status" note={`${timeframe} data`} action={<button className="icon-button" onClick={() => { status.reload(); worker.reload(); }} aria-label="Refresh pipeline status"><RefreshCw size={15} /></button>} />
          <div className="status-list"><div><span className={`step-indicator ${isReady ? "done" : ""}`}>{isReady ? <Check size={12} /> : "1"}</span><div><b>Provider history</b><small>{status.data?.candle_count?.toLocaleString() || 0} candles · {status.data?.provider || "No provider record"}</small></div></div><div><span className={`step-indicator ${status.data?.quality_report?.passed_validation ? "done" : ""}`}>{status.data?.quality_report?.passed_validation ? <Check size={12} /> : "2"}</span><div><b>Quality validation</b><small>{status.data?.quality_report?.passed_validation ? "Latest report passed" : "Not yet validated"}</small></div></div><div><span className={`step-indicator ${hasFeatures ? "done" : ""}`}>{hasFeatures ? <Check size={12} /> : "3"}</span><div><b>Feature dataset</b><small>{features.data?.metadata?.feature_count ? `${features.data.metadata.feature_count} columns` : "Not yet generated"}</small></div></div><div><span className={`step-indicator ${hasChampion ? "done" : ""}`}>{hasChampion ? <Check size={12} /> : "4"}</span><div><b>Champion</b><small>{health.data?.champion || "Train and promote a model"}</small></div></div></div>
        </Card>
        <JobsCard data={jobs.data} loading={jobs.loading} error={jobs.error} />
      </div>
    </div>
  </>;
}

function PipelineStep({ number, title, detail, button, disabled, busy, onClick, done }) {
  return <div className="pipeline-step"><span className={`pipeline-number ${done ? "done" : ""}`}>{done ? <Check size={14} /> : number}</span><div className="pipeline-copy"><b>{title}</b><small>{detail}</small></div><button className="button button-outline button-small" disabled={disabled} onClick={onClick}>{busy ? <LoaderCircle size={14} className="spin" /> : null}{busy ? "Starting…" : button}</button></div>;
}

function JobsCard({ data, loading, error }) {
  return <Card className="jobs-card"><SectionTitle title="Background jobs" note="Recent operational activity" /><ErrorNotice error={error} />{loading ? <Loading /> : data?.length ? <div className="job-list">{data.map((job, index) => <div className="job-row" key={`${job.name}-${index}`}><div className={`job-icon job-${job.status}`}><Workflow size={14} /></div><div><b>{job.name}</b><small>{job.message || formatDate(job.started_at)}</small></div><StatusBadge variant={job.status === "running" ? "warning" : job.status === "succeeded" ? "success" : job.status === "failed" ? "danger" : "muted"}>{job.status}</StatusBadge></div>)}</div> : <p className="muted-copy">No background jobs have been recorded.</p>}</Card>;
}

function DataCenter({ timeframe, refreshKey }) {
  const status = useRemoteData(`/api/data/status?timeframe=${timeframe}`, [refreshKey, timeframe]);
  const report = status.data?.quality_report;
  const source = status.data?.latest_download;
  const metadata = source?.metadata || {};
  const safeMetadata = Object.fromEntries(Object.entries(metadata).filter(([key]) => key !== "raw_files"));
  return <>
    <div className="metrics-grid">
      <MetricCard label="Processed candles" value={status.data?.candle_count?.toLocaleString() || "0"} note={`${timeframe} validated history`} icon={Database} tone="icon-blue" />
      <MetricCard label="Latest candle" value={status.data?.last_candle ? formatDate(status.data.last_candle) : "Unavailable"} note="UTC timestamp" icon={Clock3} tone="icon-gold" />
      <MetricCard label="Data provider" value={status.data?.provider || "Not downloaded"} note={source?.price_side ? `Price side · ${source.price_side}` : "Provider provenance"} icon={ExternalLink} tone="icon-purple" />
      <MetricCard label="Quality score" value={report?.quality_score != null ? `${formatNumber(report.quality_score, 1)}%` : "Not checked"} note={report?.passed_validation ? "Passed validation" : "Review report"} icon={ShieldCheck} tone="icon-green" />
    </div>
    <div className="data-grid">
      <Card><SectionTitle title="Coverage and quality" note="Quality report from the local data pipeline" action={<StatusBadge variant={report?.passed_validation ? "success" : "warning"}>{report?.passed_validation ? "Passed" : "Not verified"}</StatusBadge>} />
        {status.loading ? <Loading /> : status.error ? <ErrorNotice error={status.error} /> : report ? <div className="metrics-list quality-metrics">{Object.entries(report).filter(([, value]) => typeof value !== "object").map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><b>{typeof value === "boolean" ? value ? "Yes" : "No" : typeof value === "number" ? formatNumber(value, 2) : String(value ?? "—")}</b></div>)}</div> : <EmptyState title="No quality report" detail="Queue data validation from the Training center." />}</Card>
      <Card><SectionTitle title="Download provenance" note="Latest recorded provider batch" />
        {source ? <><div className="provenance-overview"><div><small>Provider</small><b>{source.provider}</b></div><div><small>Instrument</small><b>{source.instrument}</b></div><div><small>Timeframe</small><b>{source.timeframe}</b></div><div><small>Downloaded</small><b>{formatDate(source.downloaded_at)}</b></div><div><small>Requested range</small><b>{formatDate(source.requested_start)} — {formatDate(source.requested_end)}</b></div><div><small>Candles in batch</small><b>{source.candle_count?.toLocaleString() || "—"}</b></div></div>{Object.keys(safeMetadata).length > 0 && <details className="metadata-details"><summary>Additional provenance details</summary><pre>{JSON.stringify(safeMetadata, null, 2)}</pre></details>}</> : <EmptyState title="No download recorded" detail="Start an explicit historical data download to see provider provenance here." />}</Card>
    </div>
    <Card className="source-note"><div className="source-note-icon"><ShieldCheck size={17} /></div><div><b>Source and limitations</b><p>The operational workflow uses Dukascopy's public historical candle feed. Provider-reported volume is not centralized exchange volume. Holiday and provider-specific closures may appear as gaps; review the quality report before relying on model output.</p></div></Card>
  </>;
}

function PredictionHistory({ refreshKey }) {
  const history = useRemoteData("/api/prediction/history?limit=5000", [refreshKey]);
  const records = Array.isArray(history.data) ? history.data : [];
  const settled = records.filter((row) => row.is_correct !== null && row.is_correct !== undefined);
  const accuracy = settled.length ? settled.filter((row) => row.is_correct).length / settled.length : null;
  return <>
    <div className="metrics-grid"><MetricCard label="Saved forecasts" value={records.length.toLocaleString()} note="Historical prediction records" icon={Activity} tone="icon-blue" /><MetricCard label="Settled forecasts" value={settled.length.toLocaleString()} note="Outcome available in stored data" icon={Check} tone="icon-green" /><MetricCard label="Paper hit rate" value={accuracy === null ? "—" : percent(accuracy)} note="Settled forecasts only" icon={Gauge} tone="icon-gold" /><MetricCard label="Execution status" value="Research only" note="No broker orders are placed" icon={ShieldCheck} tone="icon-purple" /></div>
    <Card className="table-card"><SectionTitle title="Forecast history" note="Predictions are assessed only after their target candle is stored." action={<StatusBadge variant="muted">Paper tracking only</StatusBadge>} />
      {history.loading ? <Loading /> : history.error ? <ErrorNotice error={history.error} /> : records.length ? <div className="table-scroll"><table><thead><tr><th>Generated</th><th>Input candle</th><th>Target</th><th>Prediction</th><th>Probability</th><th>Outcome</th><th>Model</th><th>Provider</th></tr></thead><tbody>{records.map((record) => <tr key={record.id}><td>{formatDate(record.prediction_timestamp)}</td><td>{formatDate(record.input_candle_timestamp)}</td><td>{formatDate(record.target_timestamp)}</td><td><StatusBadge variant={record.predicted_class === "UP" ? "success" : record.predicted_class === "DOWN" ? "danger" : "muted"}>{record.predicted_class}</StatusBadge></td><td>{Object.entries(record.probabilities || {}).map(([label, value]) => `${label} ${percent(value)}`).join(" · ")}</td><td>{record.actual_class ? <span className={record.is_correct ? "text-green" : "text-red"}>{record.actual_class} {record.is_correct ? "✓" : "×"}</span> : "Pending"}</td><td className="mono">{record.model_version}</td><td>{record.provider}</td></tr>)}</tbody></table></div> : <EmptyState icon={Activity} title="No forecasts recorded" detail="A validated champion and current data are required before the platform can save forecasts." />}
    </Card>
    <div className="disclaimer-bar"><ShieldCheck size={15} />Hit rate is a historical paper-tracking measure, not realized trading performance.</div>
  </>;
}

function SettingsPage({ token, setToken, notify, refreshKey }) {
  const settings = useRemoteData("/api/settings", [refreshKey]);
  const [draft, setDraft] = useState(token);
  const saveToken = () => {
    try {
      if (draft) window.localStorage.setItem("aurummind-api-token", draft);
      else window.localStorage.removeItem("aurummind-api-token");
      setToken(draft);
      notify(draft ? "API token saved in this browser." : "API token removed.");
    } catch {
      notify("Could not save API token in browser storage.", "error");
    }
  };
  const clearToken = () => {
    try {
      window.localStorage.removeItem("aurummind-api-token");
      setDraft("");
      setToken("");
      notify("API token removed.");
    } catch {
      notify("Could not remove API token from browser storage.", "error");
    }
  };
  return <>
    <div className="settings-grid">
      <Card><SectionTitle title="API connection" note="Connect the frontend to the local FastAPI service." action={<StatusBadge variant={settings.error ? "danger" : settings.loading ? "warning" : "success"}>{settings.error ? "Unavailable" : settings.loading ? "Checking" : "Connected"}</StatusBadge>} />
        <ErrorNotice error={settings.error} />
        <div className="connection-detail"><span>API base URL</span><b className="mono">{API_URL}</b></div>
        <p className="settings-description">Read-only data is available without a token. Actions that download data, run training, or control the worker require the server-side <code>AURUMMIND_API_TOKEN</code>.</p>
        <label className="token-label">Bearer token<div className="token-input"><LockKeyhole size={16} /><input type="password" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Enter your API token" autoComplete="off" /><button className="icon-button" onClick={saveToken} aria-label="Save API token"><Check size={16} /></button></div></label>
        <div className="settings-actions"><button className="button button-gold" onClick={saveToken}>Save token</button><button className="button button-outline" onClick={clearToken}>Clear token</button></div>
        <div className="security-note"><ShieldCheck size={16} /><span>The token is saved in this browser's local storage and sent only to the API URL shown above. Keep the API bound to localhost.</span></div>
      </Card>
      <Card><SectionTitle title="Platform configuration" note="Safe runtime settings returned by the API." />
        {settings.loading ? <Loading /> : settings.data ? <div className="settings-config">{Object.entries(settings.data).map(([section, values]) => <details key={section} open={section === "application" || section === "market_data"}><summary>{section.replaceAll("_", " ")}</summary><div>{Object.entries(values).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><b>{Array.isArray(value) ? value.join(", ") : typeof value === "object" ? JSON.stringify(value) : String(value)}</b></div>)}</div></details>)}</div> : <EmptyState icon={Settings} title="Configuration unavailable" detail="Start the API and refresh this page." />}
      </Card>
    </div>
    <Card className="security-note-card"><div className="source-note-icon"><AlertCircle size={17} /></div><div><b>Operational safety</b><p>This is a research-only forecasting workflow. It does not connect to a broker, execute trades, or provide individualized financial advice. Do not expose the API publicly without a separate security review.</p></div></Card>
  </>;
}

export default App;
