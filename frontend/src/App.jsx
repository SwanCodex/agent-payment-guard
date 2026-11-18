import React, { useState } from "react";
import {
  ShieldCheck, LayoutDashboard, ArrowLeftRight, BrainCircuit,
  ClipboardList, Bot, Settings, Search, Bell, ChevronDown,
  Activity, ShieldAlert, CircleCheck, CircleX, Clock3,
  ArrowUpRight, ArrowDownRight, Menu, X, ExternalLink,
  Zap, LockKeyhole, Fingerprint, Network, Eye, MoreHorizontal, FileUp, FileText
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  Tooltip, CartesianGrid, PieChart, Pie, Cell
} from "recharts";
import "./App.css";
import AgentPlayground from "./AgentPlayground";

const layerCatalog = [
  {id:"L1",name:"Heuristic Analysis",desc:"Rules & URL intelligence",icon:ShieldCheck,detail:"Deterministic rules and URL risk scoring"},
  {id:"L2",name:"Prompt Detection",desc:"Injection & manipulation",icon:BrainCircuit,detail:"Prompt-injection classification"},
  {id:"L3A",name:"Tabular Fraud",desc:"LightGBM ensemble",icon:Activity,detail:"Transaction fraud model"},
  {id:"L3B",name:"Graph Intelligence",desc:"Relationship analysis",icon:Network,detail:"Graph fraud detection"},
];

function verdictStatus(verdict) {
  if (verdict === "ALLOW") return "PASS";
  if (verdict === "BLOCK") return "BLOCK";
  if (verdict === "STEP_UP") return "STEP_UP";
  return "UNAVAILABLE";
}

function formatAmount(amount, currency) {
  if (amount == null || currency == null) return "N/A";
  const numeric = Number(amount);
  if (!Number.isFinite(numeric)) return "N/A";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency
  }).format(numeric);
}

function App() {
  const [page,setPage] = useState("Overview");
  const [selected,setSelected] = useState(null);
  const [query,setQuery] = useState("");
  const [mobile,setMobile] = useState(false);
  const [transactions,setTransactions] = useState([]);
  const [scans,setScans] = useState([]);
  const [summary,setSummary] = useState(null);
  const [reviews,setReviews] = useState([]);
  const [auditLogs,setAuditLogs] = useState([]);
  const [apiStatus,setApiStatus] = useState("Connecting");
  const [apiError,setApiError] = useState("");
  const [showPaymentModal,setShowPaymentModal] = useState(false);
  const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";
  const SHOWCASE_MODE = import.meta.env.VITE_SHOWCASE_MODE === "true";

  const distribution = [
    {name:"Low risk",value:summary?.risk_distribution?.low ?? 0,color:"#32d6a0"},
    {name:"Medium risk",value:summary?.risk_distribution?.medium ?? 0,color:"#f5b94c"},
    {name:"High risk",value:summary?.risk_distribution?.high ?? 0,color:"#f16d78"}
  ];
  const hasRecords = Boolean(summary?.transactions_analyzed);
  const latestScan = scans[0];
  const pipelineLayers = layerCatalog.map(layer => {
    const result = latestScan?.layers?.find(item => item.layer === layer.id);
    const unavailable = result?.reason_codes?.some(code =>
      code.includes("MODEL_UNAVAILABLE") || code.includes("_ERROR")
    );
    return {
      ...layer,
      result,
      status: result ? (unavailable ? "UNAVAILABLE" : verdictStatus(result.verdict)) : "NOT EVALUATED"
    };
  });
  const activity = transactions.reduce((points, record) => {
    const date = new Date(record.recordedAt);
    if (Number.isNaN(date.getTime())) return points;
    const time = date.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
    const point = points.find(item => item.time === time);
    if (point) point.value += 1;
    else points.push({time, value:1});
    return points;
  }, []).slice(-24);

  async function loadBackendData() {
    if (SHOWCASE_MODE) {
      const now = Date.now();
      const sampleTransactions = [
        {id:"SAMPLE-TX-001",recordType:"PAYMENT_PROPOSAL",recipient:"Northstar Office Supplies",amount:"₹4,800.00",risk:18,status:"ALLOWED",time:new Date(now-600000).toLocaleTimeString(),recordedAt:now-600000},
        {id:"SAMPLE-TX-002",recordType:"DOCUMENT_SCAN",recipient:"Invoice — Meridian Trading",amount:"N/A",risk:91,status:"BLOCKED",time:new Date(now-1800000).toLocaleTimeString(),recordedAt:now-1800000},
        {id:"SAMPLE-TX-003",recordType:"PAYMENT_PROPOSAL",recipient:"Aster Technologies",amount:"₹28,500.00",risk:57,status:"REVIEW",time:new Date(now-3600000).toLocaleTimeString(),recordedAt:now-3600000},
        {id:"SAMPLE-TX-004",recordType:"DOCUMENT_SCAN",recipient:"Purchase order — Demo",amount:"N/A",risk:12,status:"ALLOWED",time:new Date(now-5400000).toLocaleTimeString(),recordedAt:now-5400000}
      ];

      setTransactions(sampleTransactions);
      setSummary({
        transactions_analyzed:126,
        threats_detected:18,
        payments_protected:145000,
        payments_protected_currency:"INR",
        pending_reviews:4,
        risk_distribution:{low:68,medium:22,high:10}
      });
      setReviews([
        {id:"SAMPLE-REVIEW-001"},
        {id:"SAMPLE-REVIEW-002"},
        {id:"SAMPLE-REVIEW-003"},
        {id:"SAMPLE-REVIEW-004"}
      ]);
      setScans([{
        content_id:"SAMPLE-SCAN-001",
        verdict:"STEP_UP",
        risk_score:0.57,
        layers:[
          {layer:"L1",verdict:"ALLOW",reason_codes:[]},
          {layer:"L2",verdict:"ALLOW",reason_codes:[]},
          {layer:"L3A",verdict:"STEP_UP",reason_codes:[]},
          {layer:"L3B",verdict:"STEP_UP",reason_codes:[]}
        ]
      }]);
      setAuditLogs([
        {id:"SAMPLE-EVT-001",record_type:"PAYMENT_PROPOSAL",verdict:"ALLOW",status:"SIMULATED",risk_score:18,timestamp:new Date(now-600000).toISOString(),reason_codes:["SAMPLE_DATA"]},
        {id:"SAMPLE-EVT-002",record_type:"DOCUMENT_SCAN",verdict:"BLOCK",status:"SIMULATED",risk_score:91,timestamp:new Date(now-1800000).toISOString(),reason_codes:["SAMPLE_DATA","ILLUSTRATIVE_THREAT"]},
        {id:"SAMPLE-EVT-003",record_type:"PAYMENT_PROPOSAL",verdict:"STEP_UP",status:"SIMULATED",risk_score:57,timestamp:new Date(now-3600000).toISOString(),reason_codes:["SAMPLE_DATA","HUMAN_REVIEW"]},
        {id:"SAMPLE-EVT-004",record_type:"DOCUMENT_SCAN",verdict:"ALLOW",status:"SIMULATED",risk_score:12,timestamp:new Date(now-5400000).toISOString(),reason_codes:["SAMPLE_DATA"]}
      ]);
      setApiStatus("Showcase");
      setApiError("");
      return;
    }

    try {
      const responses = await Promise.all([
        fetch(`${API}/api/health`),
        fetch(`${API}/api/transactions`),
        fetch(`${API}/api/dashboard/summary`),
        fetch(`${API}/api/reviews`),
        fetch(`${API}/api/scans`),
        fetch(`${API}/api/audit-logs`)
      ]);
      if (responses.some(r => !r.ok)) throw new Error("API request failed");
      const [health, tx, stats, pending, scanRecords, logs] = await Promise.all(
        responses.map(r => r.json())
      );
      const paymentRecords = tx.map(t => ({
        ...t,
        recordType: "PAYMENT_PROPOSAL",
        risk: Number(t.risk_score ?? t.risk ?? 0),
        amount: formatAmount(t.amount, t.currency),
        status: t.status === "AWAITING_APPROVAL" ? "REVIEW" :
          ["EXECUTED", "ALLOWED"].includes(t.status) ? "ALLOWED" :
          t.status === "BLOCKED" ? "BLOCKED" : t.status,
        time: new Date(t.created_at).toLocaleTimeString(),
        recordedAt: t.created_at
      }));
      const documentRecords = scanRecords.map(scan => ({
        ...scan,
        id: scan.content_id,
        recordType: "DOCUMENT_SCAN",
        recipient: scan.metadata?.filename || scan.source_type || "PDF document",
        agent: "Document scanner",
        amount: "N/A",
        risk: Math.round(Number(scan.risk_score || 0) * 100),
        status: scan.verdict === "ALLOW" ? "ALLOWED" :
          scan.verdict === "BLOCK" ? "BLOCKED" :
          scan.verdict === "STEP_UP" ? "REVIEW" : "UNAVAILABLE",
        time: new Date(Number(scan.timestamp) * 1000).toLocaleTimeString(),
        recordedAt: Number(scan.timestamp) * 1000
      }));
      setTransactions([...paymentRecords, ...documentRecords]
        .sort((a, b) => new Date(b.recordedAt) - new Date(a.recordedAt)));
      setScans(scanRecords);
      setSummary(stats);
      setReviews(pending);
      setAuditLogs(logs);
      setApiStatus(health.status === "connected" ? "Connected" : "Disconnected");
      setApiError("");
    } catch (error) {
      setApiStatus("Disconnected");
      setApiError(error.message);
    }
  }

  React.useEffect(() => {
    loadBackendData();
    if (SHOWCASE_MODE) return undefined;
    const timer = setInterval(loadBackendData, 5000);
    return () => clearInterval(timer);
  }, []);

  const nav = [
    {section:"WORKSPACE",items:[
      {name:"Overview",icon:LayoutDashboard},
      {name:"Transactions",icon:ArrowLeftRight},
      {name:"AI Pipeline",icon:BrainCircuit},
      {name:"Agent Activity",icon:Bot},
    ]},
    {section:"SECURITY",items:[
      {name:"Payment Review",icon:ShieldAlert},
      {name:"Audit Logs",icon:ClipboardList},
    ]},
  ];

  const filtered = transactions.filter(t =>
    `${t.id} ${t.recipient} ${t.agent} ${t.recordType} ${t.status}`.toLowerCase().includes(query.toLowerCase())
  );

  const Status = ({value}) => {
    const cls = value==="ALLOWED"?"green":value==="BLOCKED"?"red":value==="REVIEW"?"amber":"blue";
    return <span className={`status ${cls}`}><span/> {value}</span>;
  };

  const Metric = ({label,value,change,icon:Icon,color,sub}) => (
    <div className="metric card">
      <div className="metric-top"><span>{label}</span><div className={`metric-icon ${color}`}><Icon size={18}/></div></div>
      <div className="metric-value">{value}</div>
      <div className="metric-bottom">{change && <span className="positive"><ArrowUpRight size={14}/>{change}</span>}<span>{sub}</span></div>
    </div>
  );

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobile?"mobile-open":""}`}>
        <div className="brand">
          <div className="brand-icon"><ShieldCheck size={25}/></div>
          <div><div className="brand-name">SENTINEL<span>.</span></div><div className="brand-sub">AI PAYMENT DEFENSE</div></div>
          <button className="mobile-close" onClick={()=>setMobile(false)}><X/></button>
        </div>
        <div className="workspace"><div className="workspace-dot"/> DEVELOPMENT ENVIRONMENT <ChevronDown size={13}/></div>
        {nav.map(group=><div className="nav-group" key={group.section}>
          <div className="nav-label">{group.section}</div>
          {group.items.map(item=><button key={item.name} className={`nav-item ${page===item.name?"active":""}`} onClick={()=>{setPage(item.name);setMobile(false);}}>
            <item.icon size={18}/>{item.name}
            {item.name==="Payment Review" && reviews.length > 0 &&
              <span className="nav-count">{reviews.length}</span>}
          </button>)}
        </div>)}
        <div className="sidebar-bottom">
          <button className="nav-item" onClick={()=>setPage("Settings")}><Settings size={18}/> Settings</button>
          <div className="user-card"><div className="avatar">SD</div><div><b>Security Admin</b><span>Administrator</span></div><MoreHorizontal size={18}/></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="top-left"><button className="mobile-menu" onClick={()=>setMobile(true)}><Menu/></button><span className="breadcrumb">Workspace</span><span className="crumb-slash">/</span><b>{page}</b></div>
          <div className="top-right">{!SHOWCASE_MODE && <button className="primary-button new-payment-button" onClick={()=>setShowPaymentModal(true)}>+ New Payment</button>}<div className="live-status"><span/> {SHOWCASE_MODE ? "SHOWCASE MODE — SAMPLE DATA" : apiStatus === "Connected" ? "BACKEND CONNECTED" : "BACKEND DISCONNECTED"}</div><button className="icon-button"><Bell size={18}/><i/></button><div className="top-avatar">SD</div></div>
        </header>

        <div className="content">
          {SHOWCASE_MODE && <div className="card info-card" style={{marginBottom:16,border:"1px solid #b98b43"}}>
            <strong>SHOWCASE MODE — ILLUSTRATIVE DATA</strong>
            <p style={{marginBottom:0}}>This public preview uses sample records and illustrative security-layer outcomes. No live backend, model inference, payment processing, or audit recording is active.</p>
          </div>}
          {apiError && <div className="card info-card">Backend: {apiError}</div>}
          {page==="Overview" ? <>
            <div className="page-heading"><div><div className="eyebrow">SECURITY COMMAND CENTER</div><h1>Overview</h1><p>Monitor autonomous payment activity and AI-driven threat detection.</p></div><div className="heading-date"><Activity size={15}/> LIVE MONITOR <span>•</span> BACKEND DATA</div></div>

            <div className="metrics-grid">
              <Metric label="Records Analyzed" value={summary?.transactions_analyzed ?? "—"} sub="payments and document scans" icon={ArrowLeftRight} color="cyan"/>
              <Metric label="Threats Detected" value={summary?.threats_detected ?? "—"} sub="BLOCK and STEP_UP results" icon={ShieldAlert} color="red-icon"/>
              <Metric label="Payments Protected" value={summary?.payments_protected && summary?.payments_protected_currency ? formatAmount(summary.payments_protected, summary.payments_protected_currency) : "N/A"} sub="executed simulated payments only" icon={LockKeyhole} color="green-icon"/>
              <Metric label="Pending Reviews" value={summary?.pending_reviews ?? "—"} sub="awaiting authorization" icon={Clock3} color="amber-icon"/>
            </div>

            <div className="charts-grid">
              <section className="card chart-card"><div className="section-head"><div><h3>Transaction Activity</h3><p>Payment volume monitored over time</p></div><button className="period-button">Today <ChevronDown size={14}/></button></div>
                <div className="chart-legend"><span><i className="legend-cyan"/> Backend records</span><span className="chart-total">{summary?.transactions_analyzed ?? "—"} <small>TOTAL</small></span></div>
                {activity.length ? <div className="area-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={activity}><defs><linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#24c9c1" stopOpacity={0.25}/><stop offset="100%" stopColor="#24c9c1" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#263445" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="time" tick={{fill:"#8192a6",fontSize:11}} axisLine={false} tickLine={false}/><YAxis allowDecimals={false} tick={{fill:"#8192a6",fontSize:11}} axisLine={false} tickLine={false} width={30}/><Tooltip contentStyle={{background:"#172436",border:"1px solid #304357",borderRadius:9,color:"#fff"}}/><Area type="monotone" dataKey="value" stroke="#27c9c1" strokeWidth={2.5} fill="url(#volumeGradient)"/></AreaChart></ResponsiveContainer></div> : <div className="empty-chart">No recorded activity yet.</div>}
              </section>

              <section className="card risk-card"><div className="section-head"><div><h3>Risk Distribution</h3><p>Classified transaction risk</p></div><button className="dots-button"><MoreHorizontal/></button></div>
                {hasRecords ? <><div className="donut-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={65} outerRadius={88} paddingAngle={4} stroke="none">{distribution.map((d,i)=><Cell key={i} fill={d.color}/>)}</Pie><Tooltip contentStyle={{background:"#172436",border:"1px solid #304357",borderRadius:9}}/></PieChart></ResponsiveContainer><div className="donut-center"><b>{summary.transactions_analyzed}</b><span>RECORDS</span></div></div><div className="risk-legend">{distribution.map(d=><div key={d.name}><span><i style={{background:d.color}}/>{d.name}</span><b>{d.value}%</b></div>)}</div></> : <div className="empty-chart">No risk classifications yet.</div>}
              </section>
            </div>

            <section className="card pipeline-card"><div className="section-head"><div><h3>AI Security Pipeline</h3><p>Multi-layer defense architecture</p></div><button className="text-link" onClick={()=>setPage("AI Pipeline")}>View pipeline <ArrowUpRight size={15}/></button></div>
              <div className="pipeline-grid">{pipelineLayers.map((layer,i)=><React.Fragment key={layer.id}><div className="pipeline-layer"><div className="layer-top"><div className="layer-icon"><layer.icon size={19}/></div><span className={`layer-state ${layer.status==="PASS"?"operational":"simulation"}`}>{layer.status}</span></div><div className="layer-id">{layer.id}</div><b>{layer.name}</b><p>{layer.result ? `Latest scan: ${layer.result.verdict}` : layer.desc}</p></div>{i!==3&&<div className="pipeline-connector"><div/></div>}</React.Fragment>)}</div></section>

            <section className="card transactions-card"><div className="section-head"><div><h3>Recent Transactions</h3><p>Latest payment security decisions</p></div><button className="text-link" onClick={()=>setPage("Transactions")}>View all transactions <ArrowUpRight size={15}/></button></div><TransactionTable rows={transactions.slice(0,4)} onSelect={setSelected} Status={Status}/></section>
          </> : page==="Transactions" ? <>
            <div className="page-heading"><div><div className="eyebrow">BACKEND RECORDS</div><h1>Transactions</h1><p>Inspect payment proposals and document scan results.</p></div></div>
            <div className="card table-card"><div className="table-toolbar"><div className="search-box"><Search size={17}/><input placeholder="Search transactions..." value={query} onChange={e=>setQuery(e.target.value)}/></div><span className="table-count">{filtered.length} transactions</span></div><TransactionTable rows={filtered} onSelect={setSelected} Status={Status}/></div>
          </> : page==="AI Pipeline" ? <>
            <div className="page-heading"><div><div className="eyebrow">DEFENSE ARCHITECTURE</div><h1>AI Security Pipeline</h1><p>Four-layer evaluation architecture powering SENTINEL.</p></div></div>
            <div className="pipeline-detail-grid">{pipelineLayers.map(layer=><div className="card detail-layer" key={layer.id}><div className="layer-icon large"><layer.icon size={24}/></div><div className="layer-id">{layer.id}</div><h3>{layer.name}</h3><p>{layer.result ? `Latest scan verdict: ${layer.result.verdict}` : layer.detail}</p><span className={`layer-state ${layer.status==="PASS"?"operational":"simulation"}`}>{layer.status}</span></div>)}</div>
            <div className="card info-card"><h3>Decision Engine</h3><p>Layer outputs are aggregated by the deterministic arbiter. Explicit BLOCK decisions take priority; unavailable models fail closed and require review. Only a final ALLOW decision can proceed to the payment execution lifecycle.</p></div>
          </> : page==="Payment Review" ? <>
            <div className="page-heading"><div><div className="eyebrow">HUMAN-IN-THE-LOOP SECURITY</div><h1>Payment Review</h1><p>Review escalated payment proposals before execution.</p></div></div>
            <div className="card empty-review"><ShieldAlert size={34}/><h3>{reviews.length} pending payment reviews</h3><p>{reviews.length ? "These payment proposals require explicit authorization." : "No payment proposals are awaiting authorization."}</p>{reviews.length > 0 && <button className="primary-button" onClick={()=>setPage("Transactions")}>View payment records</button>}</div>
          </> : page==="Audit Logs" ? <>
            <div className="page-heading"><div><div className="eyebrow">RECORDED EVENTS</div><h1>Audit Logs</h1><p>Backend-recorded document scans and payment proposal decisions.</p></div></div>
            <div className="card table-card"><div className="table-toolbar"><span className="table-count">{auditLogs.length} recorded events</span></div><AuditLogTable rows={auditLogs}/></div>
          </> : page==="Agent Activity" ? <>
            {SHOWCASE_MODE ? <section className="card info-card">
              <div className="eyebrow">READ-ONLY SHOWCASE</div>
              <h2>Agent Security Demonstrations</h2>
              <p>The local SENTINEL project includes five interactive demonstrations covering unguarded agent behaviour, guarded detection, invoice scanning, step-up verification and multi-format input.</p>
              <p>These demonstrations require the SENTINEL backend and model services, which are not connected to this public preview.</p>
              <p><strong>No live agent execution or payment action is available in showcase mode.</strong></p>
            </section> : <AgentPlayground API={API} onSubmitted={loadBackendData}/>}
          </> : <>
            <div className="page-heading"><div><div className="eyebrow">SYSTEM CONFIGURATION</div><h1>{page}</h1><p>Platform configuration and system information.</p></div></div>
            <div className="card info-card"><h3>SENTINEL development environment</h3><p>Frontend connected to the live backend records and security pipeline.</p></div>
          </>}

          <footer><span>© 2026 SENTINEL AI</span><span>SECURING THE AGENTIC ECONOMY <span className="footer-dot">●</span> DEVELOPMENT BUILD</span></footer>
        </div>
      </main>

      {!SHOWCASE_MODE && showPaymentModal && <NewPaymentModal API={API} onClose={()=>setShowPaymentModal(false)} onSubmitted={loadBackendData}/>}
      {selected&&<div className="modal-backdrop" onClick={()=>setSelected(null)}><div className="detail-modal" onClick={e=>e.stopPropagation()}><div className="modal-head"><div><div className="eyebrow">{selected.recordType}</div><h2>{selected.id}</h2></div><button className="icon-button" onClick={()=>setSelected(null)}><X/></button></div><div className="modal-amount">{selected.amount}</div><Status value={selected.status}/><div className="modal-details">{[["Record type",selected.recordType],["Source",selected.recipient],["Risk score",`${selected.risk}%`],["Timestamp",selected.time]].map(([k,v])=><div key={k}><span>{k}</span><b>{v}</b></div>)}</div><p className="modal-note">This record was returned by the SENTINEL backend. No payment action is initiated from this view.</p><button className="primary-button full" onClick={()=>setSelected(null)}>Close details</button></div></div>}
    </div>
  );
}

function TransactionTable({rows,onSelect,Status}) {
  return <div className="table-scroll"><table><thead><tr><th>RECORD</th><th>TYPE</th><th>SOURCE</th><th>AMOUNT</th><th>RISK SCORE</th><th>VERDICT</th><th>TIME</th><th></th></tr></thead><tbody>{rows.map(t=><tr key={t.id} onClick={()=>onSelect(t)}><td className="mono">{t.id}</td><td>{t.recordType}</td><td className="recipient-cell">{t.recipient}</td><td className="amount-cell">{t.amount}</td><td><div className="risk-cell"><div className="risk-track"><span style={{width:`${t.risk}%`,background:t.risk>75?"#f16d78":t.risk>40?"#f5b94c":"#32d6a0"}}/></div><b>{t.risk}%</b></div></td><td><Status value={t.status}/></td><td className="time-cell">{t.time}</td><td><button className="row-action"><Eye size={16}/></button></td></tr>)}</tbody></table>{rows.length===0&&<div className="no-results">No backend records found.</div>}</div>;
}

function AuditLogTable({rows}) {
  return <div className="table-scroll"><table><thead><tr><th>RECORD</th><th>TYPE</th><th>VERDICT</th><th>STATUS</th><th>RISK SCORE</th><th>TIME</th><th>REASONS</th></tr></thead><tbody>{rows.map(row=><tr key={`${row.record_type}-${row.id}`}><td className="mono">{row.id}</td><td>{row.record_type}</td><td>{row.verdict}</td><td>{row.status}</td><td>{row.risk_score}</td><td className="time-cell">{new Date(row.timestamp).toLocaleString()}</td><td>{(row.reason_codes || []).join(", ") || "—"}</td></tr>)}</tbody></table>{rows.length===0&&<div className="no-results">No audit records found.</div>}</div>;
}


function NewPaymentModal({API,onClose,onSubmitted}) {
  const [mode,setMode] = useState("pdf");
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState("");
  const [result,setResult] = useState(null);
  const [payment,setPayment] = useState(null);
  const [reviewed,setReviewed] = useState(false);
  const [file,setFile] = useState(null);
  const MAX_HTML_BYTES = 10 * 1024 * 1024;
  const MAX_HTML_CHARS = 100000;

  async function scanDocument() {
    if (!file) {
      setError(`Choose an ${mode.toUpperCase()} file before scanning.`);
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);
    setPayment(null);
    setReviewed(false);

    try {
      let response;
      let data;
      if (mode === "pdf") {
        const body = new FormData();
        body.append("file", file);
        response = await fetch(`${API}/v1/scan_pdf`, {
          method:"POST",
          body
        });
        data = await response.json();
      } else {
        const filename = file.name.toLowerCase();
        if (!filename.endsWith(".html") && !filename.endsWith(".htm")) {
          throw new Error("Only .html and .htm files are accepted.");
        }
        if (file.size > MAX_HTML_BYTES) {
          throw new Error("HTML file exceeds the 10 MB limit.");
        }
        const html = await file.text();
        if (!html.trim()) {
          throw new Error("The HTML file is empty.");
        }
        if (html.length > MAX_HTML_CHARS) {
          throw new Error("HTML content exceeds the 100,000 character limit.");
        }
        const contentId = `HTML-${Date.now().toString(36).toUpperCase()}`;
        response = await fetch(`${API}/v1/scan_content`, {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify({
            content_id:contentId,
            source_type:"uploaded_html",
            text:html,
            html,
            metadata:{
              filename:file.name,
              content_is_untrusted:true
            }
          })
        });
        const scan = await response.json();
        data = {
          content_id:contentId,
          filename:file.name,
          extracted_text:html,
          scan
        };
      }
      if (!response.ok) {
        throw new Error(data.detail || `${mode.toUpperCase()} scan failed`);
      }
      setResult(data);
    } catch(err) {
      setError(err.message || `Could not scan ${mode.toUpperCase()} file.`);
    } finally {
      setBusy(false);
    }
  }

  async function createProposal() {
    const fields = result?.payment_fields;
    if (!fields || result.scan?.verdict !== "ALLOW" || !reviewed ||
      !fields.amount || !fields.currency || !fields.vendor_name ||
      !fields.account_identifier) {
      return;
    }

    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/api/payments/propose`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          amount: fields.amount,
          currency: fields.currency,
          recipient_id: fields.account_identifier,
          recipient_name: fields.vendor_name,
          recipient_account_ref: fields.account_identifier,
          prompt: `Reviewed invoice ${fields.invoice_number || "without invoice number"} for ${fields.vendor_name}`,
          reference: `UPLOADED-${result.content_id}`
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Payment proposal failed");
      setPayment(data);
      onSubmitted?.();
    } catch (err) {
      setError(err.message || "Payment proposal failed.");
    } finally {
      setBusy(false);
    }
  }

  async function approveProposal() {
    if (!payment?.proposal_id || payment.status !== "AWAITING_APPROVAL") return;

    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/api/payments/${payment.proposal_id}/approve`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          reviewer_id:"payment-reviewer",
          comment:"Explicitly approved after reviewing uploaded invoice"
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Payment approval failed");
      setPayment(data);
      onSubmitted?.();
    } catch (err) {
      setError(err.message || "Payment approval failed.");
    } finally {
      setBusy(false);
    }
  }

  const transactionLayerNames = ["L1", "L2", "L3A", "L3B"];
  const paymentFields = result?.payment_fields;
  const paymentFieldsComplete = Boolean(
    paymentFields?.amount &&
    paymentFields?.currency &&
    paymentFields?.vendor_name &&
    paymentFields?.account_identifier
  );

  return (
    <div className="modal-backdrop payment-backdrop" onClick={onClose}>
      <div className="detail-modal payment-modal" onClick={e=>e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="eyebrow">SENTINEL PAYMENT WORKFLOW</div>
            <h2>Scan Payment {mode.toUpperCase()}</h2>
          </div>
          <button className="icon-button" onClick={onClose}><X/></button>
        </div>

        <div className="payment-tabs">
          {["pdf", "html"].map(type => (
            <button key={type} className={mode === type ? "active" : ""}
              onClick={() => {setMode(type);setFile(null);setResult(null);setError("");}}>
              {type === "pdf" ? <FileUp size={16}/> : <FileText size={16}/>}
              {type.toUpperCase()} Upload
            </button>
          ))}
        </div>

        <div className="pdf-upload-area">
          <FileUp size={34}/>
          <h3>Upload payment document</h3>
          <p>Select a {mode.toUpperCase()} file to extract and scan. Document scanning and payment evaluation are separate steps.</p>
          <label className="primary-button pdf-select">
            Choose {mode.toUpperCase()}
            <input key={mode} type="file"
              accept={mode === "pdf" ? ".pdf,application/pdf" : ".html,.htm,text/html"}
              onChange={e=>{setFile(e.target.files?.[0] || null);setResult(null);setError("");}}/>
          </label>
          {file && <div className="selected-file"><FileText size={17}/>{file.name}</div>}
          {error && <div className="payment-error">{error}</div>}
          <button className="primary-button full" type="button" onClick={scanDocument} disabled={busy || !file}>
            {busy ? `Scanning ${mode.toUpperCase()}...` : `Scan ${mode.toUpperCase()}`}
          </button>
        </div>

        {result && (
          <div className="payment-result">
            <div className="eyebrow">{mode.toUpperCase()} EXTRACTION AND SECURITY SCAN</div>
            <h3>Review before taking any payment action</h3>
            <div className="payment-result-grid">
              <div><span>Content ID</span><b>{result.content_id}</b></div>
              <div><span>{mode === "pdf" ? "Pages" : "Filename"}</span><b>{mode === "pdf" ? result.page_count : result.filename}</b></div>
              <div><span>Security verdict</span><b>{result.scan?.verdict || "—"}</b></div>
              <div><span>Risk score</span><b>{result.scan?.risk_score ?? "—"}</b></div>
            </div>
            {mode === "pdf" && result.payment_fields && <div className="payment-result-grid">
              {[
                ["Vendor", result.payment_fields.vendor_name],
                ["Invoice", result.payment_fields.invoice_number],
                ["Amount", result.payment_fields.amount && result.payment_fields.currency
                  ? `${result.payment_fields.currency} ${result.payment_fields.amount}`
                  : result.payment_fields.amount],
                ["Account", result.payment_fields.account_identifier],
                ["Invoice date", result.payment_fields.invoice_date]
              ].map(([label,value]) => <div key={label}><span>{label}</span><b>{value || "Not extracted"}</b></div>)}
            </div>}
            <p><b>Extracted text</b></p>
            <pre>{result.extracted_text}</pre>
            {result.scan?.layers?.length > 0 && <><p><b>Layer results</b></p>{result.scan.layers.map(layer =>
              <div className="layer-result" key={layer.layer}>
                <b>{layer.layer}</b><span>{layer.verdict}</span><small>{layer.model_name || "—"}</small>
              </div>
            )}</>}
            {result.scan?.reason_codes?.length > 0 &&
              <p>Security reasons: {result.scan.reason_codes.join(", ")}</p>}
            {result.scan?.evidence?.length > 0 &&
              <p>Evidence: {result.scan.evidence.map(item=>item.text || item.kind || JSON.stringify(item)).join(" | ")}</p>}
            {result.scan?.reason_codes?.some(code=>code.includes("MODEL_UNAVAILABLE") || code.includes("_ERROR")) &&
              <p className="payment-error">One or more security layers were unavailable. Treat this result as STEP_UP and review it manually.</p>}
            {mode === "pdf" && result.scan?.verdict === "ALLOW" && paymentFieldsComplete ? <>
              <label><input type="checkbox" checked={reviewed} onChange={event=>setReviewed(event.target.checked)}/> I reviewed the extracted payment fields.</label>
              <button className="primary-button full" onClick={createProposal} disabled={!reviewed || busy}>
                {busy ? "Evaluating payment..." : "Create reviewed payment proposal"}
              </button>
            </> : mode === "pdf" && <p className="payment-error">
              No payment proposal was created. The document must return ALLOW with complete, validated payment fields.
            </p>}
            {payment && <div className="payment-transaction-result">
              <div className="eyebrow">TRANSACTION EVALUATION</div>
              <h3>{payment.verdict} · risk {payment.risk_score}</h3>
              <p>{payment.message}</p>
              {transactionLayerNames.map(layerName => {
                const layer = (payment.layers || []).find(item=>item.layer === layerName);
                return <div className="layer-result" key={layerName}>
                  <b>{layerName}</b>
                  <span>{layer ? layer.verdict : "STEP_UP"}</span>
                  <small>{layer
                    ? `score ${Number(layer.score).toFixed(6)} · ${layer.model_name || "—"}`
                    : "MISSING — mandatory layer unavailable"}</small>
                </div>;
              })}
              {payment.status === "AWAITING_APPROVAL" && payment.verdict === "ALLOW" &&
                <button className="primary-button full" onClick={approveProposal} disabled={busy}>
                  Approve simulated payment
                </button>}
              {payment.status === "AWAITING_APPROVAL" && payment.verdict !== "ALLOW" &&
                <p className="payment-error">Execution is blocked because the aggregate decision is {payment.verdict}. Mandatory layers must be available and ALLOW.</p>}
              {payment.receipt && <p>Simulated payment confirmation: {payment.receipt.execution_id}. No real funds were transferred.</p>}
            </div>}
            <button className="primary-button full" onClick={onClose}>Close</button>
          </div>
        )}

        <p className="payment-disclaimer">Uploaded content is untrusted and is never executed. Review the extracted document and verdict before creating any payment proposal.</p>
      </div>
    </div>
  );
}

export default App;
