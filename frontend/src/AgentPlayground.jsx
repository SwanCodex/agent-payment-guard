import React, { useCallback, useEffect, useState } from "react";
import {
  Bot, Search, Globe, FileText, Shield, Zap,
  Play, XCircle, CheckCircle, Clock, Activity,
  ExternalLink, RefreshCw
} from "lucide-react";

export default function AgentPlayground({ API, onSubmitted }) {
  const [task, setTask] = useState(
    "Buy me noise-cancelling headphones under ?10,000"
  );
  const [scenario, setScenario] = useState("injection");
  const [running, setRunning] = useState(false);
  const [runId, setRunId] = useState(null);
  const [events, setEvents] = useState([]);
  const [recordedRuns, setRecordedRuns] = useState([]);
  const [runData, setRunData] = useState(null);
  const [act1, setAct1] = useState(null);
  const [act1Result, setAct1Result] = useState(null);
  const [act1SourceVisible, setAct1SourceVisible] = useState(false);
  const [act1Running, setAct1Running] = useState(false);
  const [act2Result, setAct2Result] = useState(null);
  const [act2Running, setAct2Running] = useState(false);
  const [act3File, setAct3File] = useState(null);
  const [act3Result, setAct3Result] = useState(null);
  const [act3Running, setAct3Running] = useState(false);
  const [act3Reviewed, setAct3Reviewed] = useState(false);
  const [act3Payment, setAct3Payment] = useState(null);
  const [act4Text, setAct4Text] = useState("Payee: Northstar Settlement\nIBAN: GB82WEST12345698765431\nAmount: INR 4800");
  const [act4Result, setAct4Result] = useState(null);
  const [act4Running, setAct4Running] = useState(false);
  const [format, setFormat] = useState("email");
  const [formatText, setFormatText] = useState("Sender: billing@example.test\nPayee: Northstar Office Supplies\nAmount: INR 4800");
  const [formatResult, setFormatResult] = useState(null);
  const [formatRunning, setFormatRunning] = useState(false);
  const [error, setError] = useState("");

  const loadRecordedRuns = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/agent/runs`);
      if (!response.ok) throw new Error("Unable to retrieve recorded agent runs");
      setRecordedRuns(await response.json());
    } catch (e) {
      setError(e.message);
    }
  }, [API]);

  useEffect(() => {
    loadRecordedRuns();
  }, [loadRecordedRuns]);

  useEffect(() => {
    fetch(`${API}/api/demo/act1`)
      .then(response => response.ok ? response.json() : Promise.reject(new Error("Unable to load Act 1 demo")))
      .then(setAct1)
      .catch(e => setError(e.message));
  }, [API]);

  async function runAct1() {
    setAct1Running(true);
    setAct1Result(null);
    setError("");
    try {
      const response = await fetch(`${API}/api/demo/act1/run`, {method: "POST"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Act 1 simulation failed");
      setAct1Result(data);
      onSubmitted?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setAct1Running(false);
    }
  }

  async function runAct2() {
      setAct2Running(true);
      setAct2Result(null);
      setError("");
      try {
        const response = await fetch(`${API}/api/demo/act2/run`, {method: "POST"});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Act 2 scan failed");
        setAct2Result(data);
        onSubmitted?.();
      } catch (e) {
        setError(e.message);
      } finally {
        setAct2Running(false);
      }
  }

  async function scanAct3() {
        if (!act3File) return;
        setAct3Running(true);
        setAct3Result(null);
        setError("");
        try {
          const body = new FormData();
          body.append("file", act3File);
          const response = await fetch(`${API}/v1/scan_pdf`, {method: "POST", body});
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || "PDF scan failed");
          setAct3Result(data);
          setAct3Reviewed(false);
          setAct3Payment(null);
          onSubmitted?.();
        } catch (e) {
          setError(e.message);
        } finally {
          setAct3Running(false);
        }
  }

  async function loadAct3Fixture() {
    setError("");
    try {
      const response = await fetch(`${API}/api/demo/act3/pdf`);
      if (!response.ok) throw new Error("Unable to load the clean invoice");
      const blob = await response.blob();
      setAct3File(new File([blob], "act3-clean-invoice.pdf", {type: "application/pdf"}));
      setAct3Result(null);
      setAct3Payment(null);
      setAct3Reviewed(false);
    } catch (e) {
      setError(e.message);
    }
  }

  async function proposeAct3Payment() {
          const fields = act3Result?.payment_fields;
          if (!fields || act3Result.scan?.verdict !== "ALLOW" || !act3Reviewed ||
            !fields.amount || !fields.currency || !fields.vendor_name ||
            !fields.account_identifier) return;
          setError("");
          try {
            const response = await fetch(`${API}/api/payments/propose`, {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({
                amount: fields.amount,
                currency: fields.currency,
                recipient_id: fields.account_identifier,
                recipient_name: fields.vendor_name,
                recipient_account_ref: fields.account_identifier,
                prompt: `Reviewed invoice ${fields.invoice_number || "without invoice number"} for ${fields.vendor_name}`,
                reference: "ACT3-CLEAN-INVOICE"
              })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Payment proposal failed");
            setAct3Payment(data);
            onSubmitted?.();
          } catch (e) {
            setError(e.message);
          }
        }

  async function approveAct3Payment() {
          if (!act3Payment?.proposal_id) return;
          setError("");
          try {
            const response = await fetch(`${API}/api/payments/${act3Payment.proposal_id}/approve`, {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({reviewer_id: "act3-demo-reviewer", comment: "Explicitly approved simulated disbursement"})
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Payment approval failed");
            setAct3Payment(data);
            onSubmitted?.();
          } catch (e) {
            setError(e.message);
          }
      }

  async function scanAct4() {
        setAct4Running(true);
        setAct4Result(null);
        setError("");
        try {
          const normalizedResponse = await fetch(`${API}/v1/normalize`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({source_type: "act4_text", text: act4Text})
          });
          const normalized = await normalizedResponse.json();
          const response = await fetch(`${API}/v1/scan_content`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              content_id: `ACT4-${Date.now()}`,
              source_type: "act4_text",
              text: normalized.full_text,
              metadata: {validation: normalized.validation, content_is_untrusted: true}
            })
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || "Act 4 scan failed");
          setAct4Result({...data, normalized});
          onSubmitted?.();
        } catch (e) {
          setError(e.message);
        } finally {
          setAct4Running(false);
        }
      }

  async function normalizeAndScan() {
        setFormatRunning(true);
        setFormatResult(null);
        setError("");
        try {
          const request = format === "json"
            ? {source_type: "json", json_payload: JSON.parse(formatText)}
            : {source_type: format, text: formatText, html: format === "html" ? formatText : undefined};
          const normalizedResponse = await fetch(`${API}/v1/normalize`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(request)
          });
          const normalized = await normalizedResponse.json();
          if (!normalizedResponse.ok) throw new Error(normalized.detail || "Normalization failed");
          const scanResponse = await fetch(`${API}/v1/scan_content`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              content_id: `FORMAT-${Date.now()}`,
              source_type: format,
              text: normalized.full_text,
              html: format === "html" ? formatText : undefined,
              metadata: {validation: normalized.validation, provenance: normalized.provenance, content_is_untrusted: true}
            })
          });
          const scan = await scanResponse.json();
          if (!scanResponse.ok) throw new Error(scan.detail || "Format scan failed");
          setFormatResult({normalized, scan});
          onSubmitted?.();
        } catch (e) {
          setError(e.message);
        } finally {
          setFormatRunning(false);
    }
  }

  useEffect(() => {
    if (!runId || !running) return;

    let cancelled = false;

    async function poll() {
      while (!cancelled) {
        try {
          const response = await fetch(
            `${API}/api/agent/runs/${runId}`
          );
          const data = await response.json();

          if (!response.ok) {
            throw new Error(data.detail || "Unable to retrieve agent run");
          }

          if (cancelled) return;

          setRunData(data);
          setEvents(data.events || []);

          if (data.status === "COMPLETED") {
            setRunning(false);
            loadRecordedRuns();
            onSubmitted?.();
            return;
          }

          if (data.status === "FAILED") {
            setError(data.error || "Agent execution failed");
            setRunning(false);
            return;
          }
        } catch (e) {
          if (!cancelled) {
            setError(e.message);
            setRunning(false);
          }
          return;
        }

        await new Promise(resolve => setTimeout(resolve, 700));
      }
    }

    poll();

    return () => {
      cancelled = true;
    };
  }, [runId, running, API, loadRecordedRuns]);

  async function runAgent() {
    setRunning(true);
    setEvents([]);
    setRunData(null);
    setRunId(null);
    setError("");

    try {
      const response = await fetch(`${API}/api/agent/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction: task,
          scenario
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not start browser agent");
      }

      setRunId(data.run_id);
    } catch (e) {
      setError(e.message);
      setRunning(false);
    }
  }

  const result = runData?.result;
  const verdict = result?.verdict?.toUpperCase();

  const verdictClass =
    verdict === "BLOCK" ? "blocked" :
    verdict === "ALLOW" ? "allowed" : "review";

  function eventIcon(type) {
    if (type === "search") return <Search size={16}/>;
    if (type === "browse") return <Globe size={16}/>;
    if (type === "read") return <FileText size={16}/>;
    if (type === "choose") return <Zap size={16}/>;
    if (type === "result") return <Shield size={16}/>;
    return <Bot size={16}/>;
  }

  return (
    <div className="agent-playground">
      <section className="card" style={{marginBottom: 20, padding: 20}}>
        <div className="section-head">
          <div>
            <div className="eyebrow">ACT 1 · GUARD OFF</div>
            <h3>Unguarded agent attack</h3>
            <p>The human-visible invoice hides a malicious instruction from ordinary view.</p>
          </div>
        </div>
        {act1 && <div className="agent-columns">
          <div>
            <h4>What a human sees</h4>
            <iframe
              title="Invoice preview"
              srcDoc={act1.html}
              sandbox=""
              style={{width: "100%", minHeight: 270, border: "1px solid #344054", borderRadius: 8, background: "#fff"}}
            />
            <button className="text-link" onClick={() => setAct1SourceVisible(value => !value)}>
              {act1SourceVisible ? "Hide HTML/source" : "Reveal hidden HTML/source"}
            </button>
            {act1SourceVisible && <pre style={{whiteSpace: "pre-wrap", overflowWrap: "anywhere", padding: 12, background: "#101827", color: "#d5deeb", borderRadius: 8, fontSize: 11}}>{act1.html}</pre>}
          </div>
          <div>
            <h4>What the unguarded agent extracts</h4>
            <pre style={{whiteSpace: "pre-wrap", overflowWrap: "anywhere", padding: 12, background: "#101827", color: "#d5deeb", borderRadius: 8, minHeight: 190, fontSize: 11}}>{act1.ai_extracted_text}</pre>
            <div className="result-details">
              <div><span>Beneficiary</span><b>{act1.beneficiary.name}</b></div>
              <div><span>Account</span><b>{act1.beneficiary.account_ref}</b></div>
              <div><span>Data classification</span><b>UNTRUSTED CONTENT</b></div>
            </div>
          </div>
        </div>}
        <button className="primary-button" onClick={runAct1} disabled={act1Running || !act1}>
          {act1Running ? "Running attack..." : "Run unguarded attack"}
        </button>
        {act1Result && <div style={{marginTop: 16, padding: 16, border: "1px solid #a93c4d", borderRadius: 8, background: "#351b29"}}>
          <div className="eyebrow" style={{color: "#ff8996"}}>ATTACK RESULT</div>
          <h2 style={{color: "#ff6475", margin: "8px 0"}}>PAID</h2>
          <p>{act1Result.message}</p>
          <p><b>INR {act1Result.amount}</b> → {act1Result.beneficiary.name} ({act1Result.beneficiary.account_ref})</p>
          <small>Real payment provider called: {act1Result.real_payment_provider_called ? "YES" : "NO"} · Audit event: {act1Result.audit_event_id}</small>
        </div>}
        <div style={{marginTop: 20, paddingTop: 20, borderTop: "1px solid #293c50"}}>
          <div className="eyebrow">ACT 2 · GUARD ON</div>
          <h3>Run the same attack through SENTINEL</h3>
          <p>The backend resubmits the exact HTML above to <code>/v1/scan_content</code>. No payment proposal is created.</p>
          <button className="primary-button" onClick={runAct2} disabled={act2Running || !act1}>
            {act2Running ? "Running security layers..." : "Run guarded detection"}
          </button>
          {act2Result && <div style={{marginTop: 16, padding: 16, border: "1px solid #2e9eae", borderRadius: 8, background: "#132e3a"}}>
            <div className="eyebrow">ACTUAL PIPELINE RESULT</div>
            <h2 style={{margin: "8px 0"}}>{act2Result.verdict}</h2>
            <p>Risk score: {Number(act2Result.risk_score).toFixed(4)} · Payment tool called: {act2Result.payment_tool_called ? "YES" : "NO"}</p>
            <h4>Layer evaluations</h4>
            {(act2Result.layers || []).map((layer, index) => (
              <div className="layer-result" key={`${layer.layer}-${index}`}>
                <b>{layer.layer}</b>
                <span>{layer.verdict}</span>
                <small>score {Number(layer.score).toFixed(4)} · {layer.model_name || "—"}</small>
              </div>
            ))}
            <h4>Actual reasons and evidence</h4>
            <p>{(act2Result.reason_codes || []).join(", ") || "No reason codes returned."}</p>
            {(act2Result.evidence || []).length > 0
              ? <pre style={{whiteSpace: "pre-wrap", overflowWrap: "anywhere"}}>{JSON.stringify(act2Result.evidence, null, 2)}</pre>
              : <p>No evidence span was returned by the active layers.</p>}
          </div>}
        </div>
        <div style={{marginTop: 20, paddingTop: 20, borderTop: "1px solid #293c50"}}>
          <div className="eyebrow">ACT 3 · CLEAN INVOICE</div>
          <h3>Scan a legitimate PDF invoice</h3>
          <p>Uses the existing PDF extraction endpoint. Transaction layers run when the extracted fields are submitted as a reviewed proposal.</p>
          <button className="secondary-button" onClick={loadAct3Fixture} disabled={act3Running}>Load clean invoice</button>
          <input type="file" accept=".pdf,application/pdf" onChange={event => {setAct3File(event.target.files?.[0] || null);setAct3Result(null);}}/>
          <button className="primary-button" style={{marginTop: 12}} onClick={scanAct3} disabled={act3Running || !act3File}>
            {act3Running ? "Scanning PDF..." : "Scan clean invoice"}
          </button>
          {act3Result && <div style={{marginTop: 16}}>
            <p><b>Document verdict:</b> {act3Result.scan?.verdict} · <b>Risk:</b> {act3Result.scan?.risk_score}</p>
            <p><b>Verified payment fields:</b> {JSON.stringify(act3Result.payment_fields)}</p>
            <p><b>Document security layers:</b> {JSON.stringify(act3Result.scan?.layers)}</p>
            <pre style={{whiteSpace: "pre-wrap", overflowWrap: "anywhere"}}>{act3Result.extracted_text}</pre>
            {act3Result.scan?.verdict === "ALLOW" ? <>
              {act3Result.payment_fields?.amount && act3Result.payment_fields?.currency &&
               act3Result.payment_fields?.vendor_name && act3Result.payment_fields?.account_identifier ? <>
                <label><input type="checkbox" checked={act3Reviewed} onChange={event => setAct3Reviewed(event.target.checked)}/> I reviewed this invoice and authorize the proposal.</label>
                <button className="primary-button" style={{margin: "12px 0"}} onClick={proposeAct3Payment} disabled={!act3Reviewed}>Create reviewed proposal</button>
              </> : <p>Payment execution is unavailable: this scan returned no verified payment fields. No proposal was created.</p>}
            </> : <p>Disbursement is unavailable because the actual policy did not return ALLOW.</p>}
            {act3Payment && <div style={{marginTop: 16}}>
              <p><b>Aggregate transaction verdict:</b> {act3Payment.verdict} · <b>Risk:</b> {act3Payment.risk_score}</p>
              <p><b>Transaction security layers:</b></p>
              {["L1", "L2", "L3A", "L3B"].map(layerName => {
                const layer = (act3Payment.layers || []).find(item => item.layer === layerName);
                return <p key={layerName}>
                  <b>{layerName}:</b>{" "}
                  {layer
                    ? `${layer.verdict} · score ${layer.score} · ${layer.model_name}`
                    : "MISSING — STEP_UP REQUIRED"}
                </p>;
              })}
            </div>}
            {act3Payment?.status === "AWAITING_APPROVAL" && <button className="primary-button" onClick={approveAct3Payment}>Approve simulated disbursement</button>}
            {act3Payment?.receipt && <p>Simulated payment confirmation: {act3Payment.receipt.execution_id}</p>}
          </div>}
        </div>
        <div style={{marginTop: 20, paddingTop: 20, borderTop: "1px solid #293c50"}}>
          <div className="eyebrow">ACT 4 · STEP-UP AUTHENTICATION</div>
          <h3>Validate an invoice requiring human verification</h3>
          <textarea value={act4Text} onChange={event => setAct4Text(event.target.value)} rows={4} style={{width: "100%"}}/>
          <button className="primary-button" style={{marginTop: 12}} onClick={scanAct4} disabled={act4Running}>
            {act4Running ? "Checking..." : "Run validation and security checks"}
          </button>
          {act4Result && <div style={{marginTop: 16}}>
            <h2>{act4Result.verdict === "STEP_UP" ? "STEP_UP AUTHENTICATION REQUIRED" : act4Result.verdict}</h2>
            <p>{(act4Result.reason_codes || []).join(", ") || "No reason codes returned."}</p>
            <pre style={{whiteSpace: "pre-wrap"}}>{JSON.stringify(act4Result.normalized.validation, null, 2)}</pre>
          </div>}
        </div>
        <div style={{marginTop: 20, paddingTop: 20, borderTop: "1px solid #293c50"}}>
          <div className="eyebrow">ACT 5 · MULTI-FORMAT INPUT</div>
          <h3>Normalize and scan common input formats</h3>
          <select value={format} onChange={event => setFormat(event.target.value)}>
            <option value="email">Email text</option>
            <option value="html">HTML</option>
            <option value="json">JSON transaction</option>
          </select>
          <textarea value={formatText} onChange={event => setFormatText(event.target.value)} rows={5} style={{width: "100%", marginTop: 10}}/>
          <button className="primary-button" style={{marginTop: 12}} onClick={normalizeAndScan} disabled={formatRunning}>
            {formatRunning ? "Normalizing and scanning..." : "Normalize and scan"}
          </button>
          {formatResult && <div style={{marginTop: 16}}>
            <p><b>Common pipeline verdict:</b> {formatResult.scan.verdict}</p>
            <p><b>Extracted amounts:</b> {formatResult.normalized.amounts.join(", ") || "None detected"}</p>
            <p><b>Domains:</b> {formatResult.normalized.domains.join(", ") || "None detected"}</p>
            <p><b>Embedded instructions:</b> {formatResult.normalized.embedded_instructions.join(" | ") || "None detected"}</p>
          </div>}
        </div>
      </section>

      <div className="page-heading">
        <div>
          <div className="eyebrow">AUTONOMOUS BROWSER AGENT</div>
          <h1>Agent Playground</h1>
          <p>
            A browser-operating mock shopping agent monitored by SENTINEL.
          </p>
        </div>
        <span className="mock-tag">
          <Bot size={14}/> MOCK AGENT ? NO REAL PAYMENTS
        </span>
      </div>

      <section className="card agent-task-card">
        <label>YOUR INSTRUCTION</label>
        <textarea
          value={task}
          onChange={e => setTask(e.target.value)}
          disabled={running}
        />

        <div className="agent-controls">
          <div>
            <label>TEST SCENARIO</label>
            <select
              value={scenario}
              onChange={e => setScenario(e.target.value)}
              disabled={running}
            >
              <option value="normal">Normal purchase</option>
              <option value="injection">Malicious webpage injection</option>
            </select>
          </div>

          <button
            className="primary-button"
            onClick={runAgent}
            disabled={running || !task.trim()}
          >
            {running ? <RefreshCw size={16}/> : <Play size={16}/>}
            {running ? "Agent working..." : "Run browser agent"}
          </button>
        </div>
      </section>

      <div className="agent-columns">
        <section className="card agent-activity-card">
          <div className="section-head">
            <div>
              <h3><Activity size={17}/> Browser activity</h3>
              <p>Events retrieved from the Python agent</p>
            </div>
            <span className="mock-tag">PLAYWRIGHT</span>
          </div>

          {!events.length && !recordedRuns.length ? (
            <div className="agent-empty">
              <Bot size={32}/>
              <p>No recorded agent activity</p>
              <span>Start a run to record genuine browser activity.</span>
            </div>
          ) : events.length ? (
            <div className="agent-events">
              {events.map((ev, i) => (
                <div className="agent-event" key={i}>
                  <div className="event-icon">{eventIcon(ev.type)}</div>
                  <div>
                    <b>{ev.title}</b>
                    <p>{ev.detail}</p>
                    {ev.url && (
                      <small>
                        <ExternalLink size={12}/> {ev.url}
                      </small>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="agent-events">
              {recordedRuns.map(run => (
                <div className="agent-event" key={run.run_id}>
                  <div className="event-icon"><Activity size={16}/></div>
                  <div>
                    <b>{run.run_id}</b>
                    <p>{run.instruction}</p>
                    <small>{run.status} · {run.events?.length || 0} recorded events</small>
                  </div>
                </div>
              ))}
            </div>
          )}

          {running && (
            <div className="agent-running">
              <span/> Browser agent executing
            </div>
          )}
        </section>

        <section className="card agent-result-card">
          <div className="section-head">
            <div>
              <h3><Shield size={17}/> SENTINEL decision</h3>
              <p>Actual payment workflow response</p>
            </div>
          </div>

          {!result ? (
            <div className="agent-empty">
              <Shield size={32}/>
              <p>Awaiting security evaluation</p>
              <span>The backend verdict appears here.</span>
            </div>
          ) : (
            <div className="security-result">
              <div className={`verdict-panel ${verdictClass}`}>
                <span>
                  {verdict === "BLOCK" ? <XCircle/> :
                   verdict === "ALLOW" ? <CheckCircle/> : <Clock/>}
                </span>
                <div>
                  <small>GATEWAY VERDICT</small>
                  <h2>{result.verdict}</h2>
                </div>
              </div>

              <div className="result-details">
                <div><span>Proposal ID</span><b>{result.proposal_id}</b></div>
                <div><span>Merchant</span><b>{result.recipient}</b></div>
                <div>
                  <span>Proposed amount</span>
                  <b>?{Number(result.amount).toLocaleString("en-IN")}</b>
                </div>
                <div><span>Risk score</span><b>{result.risk_score}%</b></div>
                <div><span>Payment status</span><b>{result.status}</b></div>
              </div>

              <p className="gateway-message">{result.message}</p>

              <h4>Reason codes</h4>
              {(result.reason_codes || []).length ? (
                result.reason_codes.map((r, i) => (
                  <div className="reason-code" key={i}>{r}</div>
                ))
              ) : <p>No reason codes returned.</p>}

              <h4>Layer evaluations</h4>
              {(result.layers || []).map((layer, i) => (
                <div className="layer-result" key={i}>
                  <b>{layer.layer}</b>
                  <span>{layer.verdict}</span>
                  <small>{layer.model_name || ""}</small>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {runData?.evidence && (
        <section className="card" style={{ marginTop: 20, padding: 20 }}>
          <div className="section-head">
            <div>
              <h3><Globe size={17}/> Browser evidence</h3>
              <p>Extracted from the actual ShopSphere product page</p>
            </div>
            <span className="mock-tag">CAPTURED</span>
          </div>

          <div className="result-details">
            <div>
              <span>Website</span>
              <b>{runData.evidence.page_title}</b>
            </div>
            <div>
              <span>Selected product</span>
              <b>{runData.evidence.selected_product?.name}</b>
            </div>
            <div>
              <span>Original budget</span>
              <b>?{Number(runData.evidence.budget).toLocaleString("en-IN")}</b>
            </div>
            <div>
              <span>Proposed amount</span>
              <b>?{Number(runData.evidence.proposed_amount).toLocaleString("en-IN")}</b>
            </div>
            <div>
              <span>Injection detected</span>
              <b>{runData.evidence.injection_detected ? "Yes" : "No"}</b>
            </div>
          </div>

          <h4>Captured product-page screenshot</h4>
          <img
            src={`${API}/api/agent/runs/${runId}/screenshot`}
            alt="Playwright browser screenshot"
            style={{
              width: "100%",
              borderRadius: 10,
              border: "1px solid #344054",
              marginTop: 10
            }}
          />

          <h4>Extracted product listings</h4>
          {(runData.evidence.products || []).map(product => (
            <div className="layer-result" key={product.id}>
              <b>{product.name}</b>
              <span>?{Number(product.price).toLocaleString("en-IN")}</span>
              <small>{product.id}</small>
            </div>
          ))}

          <h4>Untrusted webpage content</h4>
          <pre style={{
            whiteSpace: "pre-wrap",
            overflowWrap: "anywhere",
            padding: 14,
            borderRadius: 8,
            background: "#101827",
            color: "#d5deeb",
            fontSize: 12
          }}>
            {runData.evidence.page_content}
          </pre>
        </section>
      )}

      {error && <p className="agent-error">{error}</p>}

      <div className="agent-disclaimer">
        <Shield size={15}/>
        Mock agent and mock executor only. No real checkout, credentials or transfers.
      </div>
    </div>
  );
}
