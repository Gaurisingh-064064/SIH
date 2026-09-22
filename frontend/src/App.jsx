import React, { useEffect, useMemo, useRef, useState } from "react";
import { supabase } from "./lib/supabase";
import ForceGraph2D from "react-force-graph-2d";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = pdfWorker;

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const SOURCE_TYPES = [
  { key: "FIR", label: "FIR / Police Complaint", icon: "📄", hint: "FIR narratives, complaint text, witness statements" },
  { key: "POLICE_REPORT", label: "Police Reports", icon: "🛡️", hint: "Case notes, investigation reports, seizure or interrogation records" },
  { key: "CDR", label: "Call Detail Records", icon: "☎", hint: "Call frequency, duration, timestamps and communication patterns" },
  { key: "FINANCIAL", label: "Financial Transactions", icon: "₹", hint: "Transaction records, account activity, payment observations" },
  { key: "SURVEILLANCE", label: "Surveillance Reports", icon: "📡", hint: "Observation logs, meetings, vehicle movement, locations" },
  { key: "SOCIAL_MEDIA", label: "Social Media Intelligence", icon: "◉", hint: "Posts, handles, mentions, messages, public interactions" },
  { key: "CRIMINAL_HISTORY", label: "Criminal History Database", icon: "⚖", hint: "Prior case references, charges, convictions or aliases" },
];

const EMPTY_SOURCE = SOURCE_TYPES.reduce((acc, source) => {
  acc[source.key] = "";
  return acc;
}, {});

// Canvas can't read CSS variables, so the graph palette lives here. Keep it in
// step with the tokens at the top of styles.css (brass = --c-accent, etc.).
const GRAPH_COLORS = {
  background: "#111215",
  node: { fill: "#2b2820", stroke: "#c4a46a", hover: "#ecd9b0", halo: "rgba(196, 164, 106, 0.14)", glow: "rgba(196, 164, 106, 0.6)" },
  center: { fill: "#12302a", stroke: "#4fb886", halo: "rgba(79, 184, 134, 0.16)", glow: "rgba(79, 184, 134, 0.75)" },
  influential: { fill: "#40151a", stroke: "#e5636e", halo: "rgba(229, 99, 110, 0.20)", star: "#ef8790", glow: "rgba(229, 99, 110, 0.8)" },
  label: { text: "#ece9e2", initials: "#f4f0e6", pill: "rgba(17, 18, 21, 0.9)" },
  link: { base: "rgba(196, 164, 106, 0.45)", hover: "#e2cc9c", strongest: "#e5636e" },
  badge: {
    fill: "rgba(17, 18, 21, 0.9)", stroke: "rgba(196, 164, 106, 0.4)", text: "#eee5d0",
    hoverFill: "rgba(43, 40, 32, 0.98)", hoverStroke: "rgba(226, 204, 156, 0.85)",
    strongestFill: "rgba(64, 21, 26, 0.98)", strongestStroke: "rgba(229, 99, 110, 0.95)", strongestText: "#f5b5ba",
  },
};

// One tab per feature. Order follows the investigator's workflow.
const WORKSPACE_TABS = [
  { key: "overview", label: "Overview" },
  { key: "sources", label: "Sources" },
  { key: "network", label: "Network" },
  { key: "analytics", label: "Analytics" },
  { key: "insights", label: "Insights" },
  { key: "fir", label: "FIR Intelligence" },
  { key: "tips", label: "Tip Analysis" },
  { key: "integrity", label: "Evidence Integrity" },
  { key: "guardrails", label: "Guardrails" },
];

function initials(name = "Unknown") {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "U";
}

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function hashSourceDrafts(drafts = EMPTY_SOURCE, firLanguage = "en") {
  const payload = SOURCE_TYPES
    .map((source) => ({
      source_type: source.key,
      title: source.label,
      content: drafts[source.key] || "",
      language: source.key === "FIR" ? firLanguage : "en",
    }))
    .sort((a, b) => a.source_type.localeCompare(b.source_type));

  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  const digest = await crypto.subtle.digest("SHA-256", bytes);

  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function getUsableSession(preferredSession = null) {
  if (preferredSession?.access_token) return preferredSession;

  const { data } = await supabase.auth.getSession();
  return data?.session || null;
}

async function apiFetch(path, options = {}, session, retryOn401 = true) {
  const activeSession = await getUsableSession(session);

  const makeRequest = (requestSession) =>
    fetch(`${API}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(requestSession?.access_token
          ? { Authorization: `Bearer ${requestSession.access_token}` }
          : {}),
        ...(options.headers || {}),
      },
    });

  let response = await makeRequest(activeSession);

  // During login/token refresh, React state can briefly contain the previous
  // session (or no session) even though Supabase already has a fresh one.
  // A single retry with a freshly refreshed token prevents transient 401s
  // from breaking persisted-investigation loading.
  if (response.status === 401 && retryOn401) {
    try {
      const refreshed = await supabase.auth.refreshSession();
      const refreshedSession = refreshed?.data?.session;
      if (refreshedSession?.access_token) {
        response = await makeRequest(refreshedSession);
      }
    } catch {
      // Fall through to the normal HTTP error below.
    }
  }

  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }

  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : JSON.stringify(data?.detail || data);

    throw new Error(`HTTP ${response.status}: ${detail}`);
  }
  return data;
}


async function extractPdfText(file) {
  if (!file) return "";

  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    throw new Error("Please select a valid PDF file.");
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  const pdf = await getDocument({ data: bytes }).promise;
  const pages = [];

  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item) => ("str" in item ? item.str : ""))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();

    if (pageText) pages.push(pageText);
  }

  const extracted = pages.join("\n\n").trim();

  if (!extracted) {
    throw new Error(
      "No readable text was found in this PDF. If it is a scanned PDF, OCR support is required."
    );
  }

  return extracted;
}

function TabEmpty({ title, children, actionLabel, onAction }) {
  return (
    <section className="panel tab-empty">
      <h2>{title}</h2>
      <p>{children}</p>
      {actionLabel && (
        <button type="button" className="primary-button compact" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </section>
  );
}

export default function App() {
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [isSignup, setIsSignup] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [busyMessage, setBusyMessage] = useState("");

  const [investigations, setInvestigations] = useState([]);
  const [selected, setSelected] = useState(null);
  const [investigationFilter, setInvestigationFilter] = useState("active");
  const [activeTab, setActiveTab] = useState("overview");

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newSources, setNewSources] = useState({ ...EMPTY_SOURCE });
  const [newFirLanguage, setNewFirLanguage] = useState("en");
  const [newUploadedFiles, setNewUploadedFiles] = useState({});
  const [creating, setCreating] = useState(false);

  const [sourceDrafts, setSourceDrafts] = useState({ ...EMPTY_SOURCE });
  const [sourceLanguage, setSourceLanguage] = useState("en");
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [sourceSaveStatus, setSourceSaveStatus] = useState("Saved");
  const [uploadedFiles, setUploadedFiles] = useState({});
  const [pdfUploading, setPdfUploading] = useState({});
  const sourceHydratedRef = useRef(false);
  const hydratedCaseRef = useRef(null);

  const [firText, setFirText] = useState("");
  const [firEntities, setFirEntities] = useState([]);
  const [firAnalyzing, setFirAnalyzing] = useState(false);
  const [tipText, setTipText] = useState("");
  const [tipResult, setTipResult] = useState(null);
  const [tipAnalyzing, setTipAnalyzing] = useState(false);

  const [criminalSearch, setCriminalSearch] = useState("");
  const [criminalResults, setCriminalResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [selectedCriminal, setSelectedCriminal] = useState(null);
  const [analysisGraph, setAnalysisGraph] = useState({ nodes: [], links: [] });
  const [graph, setGraph] = useState({ nodes: [], links: [] });
  const [graphLoading, setGraphLoading] = useState(false);
  const [selectedRelationship, setSelectedRelationship] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoveredLink, setHoveredLink] = useState(null);
  const [editingSources, setEditingSources] = useState({});
  const [analysisStale, setAnalysisStale] = useState(false);
  const graphRef = useRef(null);
  const sourceSaveTimerRef = useRef(null);
  const sourceDraftsRef = useRef({ ...EMPTY_SOURCE });
  const sourceLanguageRef = useRef("en");
  const sourceSaveInFlightRef = useRef(Promise.resolve());
  const restoringCaseRef = useRef(false);
  // Tracks whose session is currently active so the auth listener below can
  // tell "same investigator, token silently refreshed" apart from "a
  // different investigator actually signed in".
  const sessionUserIdRef = useRef(null);
  const [blockchainStatus, setBlockchainStatus] = useState(null);
  const [blockchainLoading, setBlockchainLoading] = useState(false);
  const [blockchainError, setBlockchainError] = useState("");

  useEffect(() => {
    let mounted = true;

    const initialize = async () => {
      try {
        setLoading(true);
        const {
          data: { session: currentSession },
        } = await supabase.auth.getSession();
        if (!mounted) return;
        setSession(currentSession);
        sessionUserIdRef.current = currentSession?.user?.id || null;
        if (currentSession) await loadProfile(currentSession.user.id, currentSession);
      } catch (err) {
        if (mounted) setError(err.message || "Unable to initialize application.");
      } finally {
        if (mounted) setLoading(false);
      }
    };

    initialize();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, nextSession) => {
      // Supabase re-validates the session (and re-fires SIGNED_IN /
      // TOKEN_REFRESHED) whenever the browser tab regains focus. That is
      // NOT a real login or logout — treating every event as one was why
      // switching tabs reset the whole workspace and lost in-progress
      // investigation work. Only a genuine sign-out, or a different
      // investigator signing in, should clear state.
      if (event === "SIGNED_OUT" || !nextSession) {
        sessionUserIdRef.current = null;
        setSession(null);
        setProfile(null);
        setInvestigations([]);
        setSelected(null);
        resetCaseState();
        return;
      }

      const nextUserId = nextSession.user?.id || null;
      const isDifferentUser = nextUserId !== sessionUserIdRef.current;
      sessionUserIdRef.current = nextUserId;

      // Always keep the session (and its access_token) current so API
      // calls don't start using a stale token after a silent refresh.
      setSession(nextSession);

      if (isDifferentUser) {
        setProfile(null);
        setInvestigations([]);
        setSelected(null);
        resetCaseState();
        setTimeout(() => loadProfile(nextSession.user.id, nextSession), 0);
      }
      // Same investigator, just a refreshed token: nothing else resets,
      // so the active investigation, drafts, analysis and graph survive
      // a tab switch untouched.
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);


  async function verifyBlockchainIntegrity() {
  if (!selected?.id) {
    setBlockchainError("Please select an investigation first.");
    return;
  }

  setBlockchainLoading(true);
  setBlockchainError("");
  setBlockchainStatus(null);

  try {
    const activeSession = await getUsableSession(session);

    if (!activeSession?.access_token) {
      throw new Error("Access token required. Please log in again.");
    }

    const data = await apiFetch(
      `/api/investigations/${selected.id}/blockchain/verify`,
      {
        method: "GET",
      },
      activeSession
    );

    setBlockchainStatus(data);
  } catch (err) {
    console.error("Blockchain verification error:", err);

    setBlockchainError(
      err.message || "Unable to verify blockchain integrity."
    );
  } finally {
    setBlockchainLoading(false);
  }
}


  async function loadProfile(userId, activeSession = session) {
    try {
      const { data, error: profileError } = await supabase
        .from("profiles")
        .select("*")
        .eq("id", userId)
        .maybeSingle();
      if (profileError) throw profileError;
      setProfile(data);
      if (data?.is_authorized) await loadInvestigations(activeSession);
    } catch (err) {
      setError(err.message || "Unable to load profile.");
    }
  }

  async function loadInvestigations(activeSession = session) {
    // Resolve the current Supabase session at call time so a login/token
    // refresh cannot leave this request carrying an old or empty token.
    const usableSession = await getUsableSession(activeSession);
    if (!usableSession?.access_token) {
      throw new Error("Access token required");
    }

    // Investigation lists are loaded through the authenticated backend rather
    // than a broad Supabase SELECT. The backend filters by created_by so one
    // investigator can never see another investigator's case list.
    const data = await apiFetch("/api/investigations", {}, usableSession);
    setInvestigations(data || []);

    // Drop the previously selected case when switching accounts.
    const first = (data || [])[0];
    if (first) {
      setSelected((current) => {
        const stillVisible = current && (data || []).some((item) => item.id === current.id);
        return stillVisible ? current : first;
      });
    } else {
      setSelected(null);
    }
  }

  async function handleAuth(event) {
    event.preventDefault();
    setError("");
    if (!email.trim() || !password.trim()) {
      setError("Email and password are required.");
      return;
    }
    setAuthLoading(true);
    setBusyMessage(isSignup ? "Creating secure account…" : "Signing you in securely…");
    try {
      if (isSignup) {
        const { data, error: signUpError } = await supabase.auth.signUp({
          email: email.trim(),
          password,
          options: { data: { full_name: fullName.trim() } },
        });
        if (signUpError) throw signUpError;
        if (data.user && !data.session) {
          alert("Account created. Verify your email if confirmation is enabled.");
        }
      } else {
        const { data, error: signInError } = await supabase.auth.signInWithPassword({
          email: email.trim(),
          password,
        });
        if (signInError) throw signInError;
        if (data.session) {
          sessionUserIdRef.current = data.session.user.id;
          setSession(data.session);
          // Always pass the fresh sign-in session. React state updates are
          // asynchronous, so relying on `session` here can briefly send an
          // empty/stale access token during the first post-login requests.
          await loadProfile(data.session.user.id, data.session);
        }
      }
    } catch (err) {
      setError(err.message || "Authentication failed.");
    } finally {
      setAuthLoading(false);
      setBusyMessage("");
    }
  }

  useEffect(() => {
    sourceDraftsRef.current = sourceDrafts;
  }, [sourceDrafts]);

  useEffect(() => {
    sourceLanguageRef.current = sourceLanguage;
  }, [sourceLanguage]);

  function resetCaseState() {
    setAnalysis(null);
    setFirText("");
    setFirEntities([]);
    setTipText("");
    setTipResult(null);
    setCriminalSearch("");
    setCriminalResults([]);
    setShowSearchResults(false);
    setSelectedCriminal(null);
    setSelectedRelationship(null);
    setHoveredNode(null);
    setHoveredLink(null);
    setAnalysisGraph({ nodes: [], links: [] });
    setGraph({ nodes: [], links: [] });
    setSourceDrafts({ ...EMPTY_SOURCE });
    setUploadedFiles({});
  }

 async function createInvestigation(event) {
  event.preventDefault();
  setError("");
  if (!newTitle.trim()) {
    setError("Investigation title is required.");
    return;
  }
  const filledSources = SOURCE_TYPES.filter((source) => newSources[source.key]?.trim());
  if (filledSources.length === 0) {
    setError("Add at least one intelligence source before starting the investigation.");
    return;
  }

  setCreating(true);
  try {
    const created = await apiFetch(
      "/api/investigations",
      {
        method: "POST",
        body: JSON.stringify({
          title: newTitle.trim(),
          description: newDescription.trim(),
        }),
      },
      session
    );

    setInvestigations((prev) => [created, ...prev]);
    setShowCreateModal(false);

    // Save the sources BEFORE selecting the investigation. setSelected()
    // triggers the workspace GET effect, and if that fires before this
    // PUT lands, it reads back an empty case and wipes the UI state.
    await saveSourcesForInvestigation(
      created.id,
      newSources,
      newFirLanguage,
      newUploadedFiles
    );

    // Run the analysis pipeline now, as the modal promises ("Create &
    // Analyze"). Without this, sources are saved but never analyzed
    // until the user separately clicks "Run Intelligence Analysis".
    const sourcesForAnalysis = filledSources.map((source) => ({
      source_type: source.key,
      title: newUploadedFiles[source.key] || source.label,
      content: String(newSources[source.key] || ""),
      language: source.key === "FIR" ? (newFirLanguage || "en") : "en",
    }));

    let analysisResult = null;
    try {
      analysisResult = await apiFetch(
        `/api/investigations/${created.id}/analyze-sources`,
        { method: "POST", body: JSON.stringify({ sources: sourcesForAnalysis }) },
        session
      );
    } catch (err) {
      console.error("Initial analysis failed:", err);
      // Don't block investigation creation on analysis failure — the
      // investigator can retry from the workspace.
    }

    setNewTitle("");
    setNewDescription("");
    setNewSources({ ...EMPTY_SOURCE });
    setNewUploadedFiles({});
    resetCaseState();

    setSelected(created); // now safe — workspace fetch will see the saved sources
    setActiveTab("overview");

    setAnalysis(analysisResult);
    setAnalysisGraph(analysisResult?.graph || { nodes: [], links: [] });
    setAnalysisStale(false);
    setEditingSources({});
    await loadInvestigations();
  } catch (err) {
    setError(err.message || "Unable to start investigation.");
  } finally {
    setCreating(false);
  }
}


  async function handleExistingSourcePdfUpload(sourceKey, file) {
  if (!file) return;

  setError("");
  setPdfUploading((prev) => ({
    ...prev,
    [sourceKey]: true,
  }));

  try {
    // Extract text from PDF
    const extractedText = await extractPdfText(file);

    if (!extractedText || !extractedText.trim()) {
      throw new Error("No readable text was found in this PDF.");
    }

    // Create updated drafts
    const updatedDrafts = {
      ...sourceDraftsRef.current,
      [sourceKey]: extractedText,
    };

    // Update ref immediately
    sourceDraftsRef.current = updatedDrafts;

    // Update React state
    setSourceDrafts((prev) => ({
      ...prev,
      [sourceKey]: extractedText,
    }));

    // Store uploaded file name
    setUploadedFiles((prev) => ({
      ...prev,
      [sourceKey]: file.name,
    }));

    // IMPORTANT: SAVE EXTRACTED PDF TEXT TO BACKEND
    if (selected?.id) {
      await saveSourcesForInvestigation(
        selected.id,
        updatedDrafts
      );
    } else {
      throw new Error("No active investigation selected.");
    }

    setEditingSources((prev) => ({
      ...prev,
      [sourceKey]: false,
    }));

    setAnalysisStale(true);

    setSourceSaveStatus(
      "PDF uploaded and saved successfully"
    );

    console.log(
      "PDF saved successfully:",
      sourceKey,
      extractedText.length
    );

  } catch (err) {
    console.error("PDF upload error:", err);

    setError(
      err.message ||
      "Unable to extract or save the PDF."
    );
  } finally {
    setPdfUploading((prev) => ({
      ...prev,
      [sourceKey]: false,
    }));
  }
}

  async function handleNewSourcePdfUpload(sourceKey, file) {
    if (!file) return;

    setError("");
    setPdfUploading((prev) => ({ ...prev, [`new_${sourceKey}`]: true }));

    try {
      const extractedText = await extractPdfText(file);

      setNewSources((prev) => ({
        ...prev,
        [sourceKey]: extractedText,
      }));

      setNewUploadedFiles((prev) => ({
        ...prev,
        [sourceKey]: file.name,
      }));
    } catch (err) {
      setError(err.message || "Unable to extract text from the PDF.");
    } finally {
      setPdfUploading((prev) => ({ ...prev, [`new_${sourceKey}`]: false }));
    }
  }

  
     async function analyzeSourcesForExistingCase() {
  if (!selected) {
    setError("Please select an investigation first.");
    return;
  }

  setAnalysisLoading(true);
  setError("");

  try {
    // Always use the latest source data
    const latestDrafts = {
      ...sourceDraftsRef.current,
    };

    // Get all sources that actually contain data
    const filledSources = SOURCE_TYPES.filter((source) => {
      const content = latestDrafts[source.key];

      return (
        typeof content === "string" &&
        content.trim().length > 0
      );
    });

    if (filledSources.length === 0) {
      throw new Error(
        "Please upload or add at least one intelligence source before running analysis."
      );
    }

    // Prepare sources for backend
    const sources = filledSources.map((source) => ({
      source_type: source.key,

      title:
        uploadedFiles[source.key] ||
        source.label,

      content: String(
        latestDrafts[source.key] || ""
      ),

      language:
        source.key === "FIR"
          ? sourceLanguageRef.current || "en"
          : "en",
    }));

    console.log(
      "Sending sources for analysis:",
      sources
    );

    // Save sources first
    await saveSourcesForInvestigation(
      selected.id,
      latestDrafts,
      sourceLanguageRef.current
    );

    console.log(
      "Sources saved successfully"
    );

    // Run analysis
    const result = await apiFetch(
      `/api/investigations/${selected.id}/analyze-sources`,
      {
        method: "POST",
        body: JSON.stringify({
          sources,
        }),
      },
      session
    );

    console.log(
      "Analysis result:",
      result
    );

    // Store analysis
    setAnalysis(result);

    setAnalysisGraph(
      result?.graph || {
        nodes: [],
        links: [],
      }
    );

    setEditingSources({});
    setAnalysisStale(false);

    setSourceSaveStatus(
      "Analysis completed"
    );

  } catch (err) {

    console.error(
      "Analysis error:",
      err
    );

    setError(
      err.message ||
      "Source analysis failed."
    );

  } finally {

    setAnalysisLoading(false);

  }
}

      
  
  async function analyzeFIR() {
    if (!selected || !firText.trim()) {
      setError("Select an investigation and enter FIR text.");
      return;
    }
    setFirAnalyzing(true);
    setError("");
    try {
      const data = await apiFetch(
        "/api/nlp/extract",
        {
          method: "POST",
          body: JSON.stringify({
            investigation_id: selected.id,
            source_type: "FIR",
            title: "Standalone FIR Analysis",
            content: firText,
            language: sourceLanguage,
          }),
        },
        session
      );
      const entities = [];
      Object.entries(data.entities || {}).forEach(([label, values]) => {
        if (Array.isArray(values)) values.forEach((value) => entities.push({ label, text: value }));
      });
      setFirEntities(entities);
    } catch (err) {
      setError(err.message || "FIR analysis failed.");
    } finally {
      setFirAnalyzing(false);
    }
  }

  async function analyzeTip() {
    if (!selected || !tipText.trim()) {
      setError("Select an investigation and enter a tip.");
      return;
    }
    setTipAnalyzing(true);
    setError("");
    try {
      const result = await apiFetch(
        "/api/tips/analyze",
        { method: "POST", body: JSON.stringify({ investigation_id: selected.id, text: tipText }) },
        session
      );
      setTipResult(result);
    } catch (err) {
      setError(err.message || "Tip analysis failed.");
    } finally {
      setTipAnalyzing(false);
    }
  }

  // Search within the graph generated for THIS investigation.
  // No training dataset or global person database is queried here.
  useEffect(() => {
    const query = criminalSearch.trim().toLowerCase();

    if (!query) {
      setCriminalResults([]);
      setShowSearchResults(false);
      return undefined;
    }

    const matches = (analysisGraph.nodes || [])
      .filter((person) => {
        const haystack = [
          person.name,
          person.id,
          person.phone_num,
          person.vehicle_num,
          person.org,
          person.location,
        ]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();
        return haystack.includes(query);
      })
      .slice(0, 10);

    setCriminalResults(matches);
    setShowSearchResults(true);
    setSearchLoading(false);

    return undefined;
  }, [criminalSearch, analysisGraph]);

  function selectCriminal(person) {
    if (!person) return;

    setSelectedCriminal(person);
    setCriminalSearch(person.name || '');
    setCriminalResults([]);
    setShowSearchResults(false);
    setSelectedRelationship(null);

    const relatedIds = new Set([person.id]);
    const relatedLinks = (analysisGraph.links || []).filter((link) => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      const connected = sourceId === person.id || targetId === person.id;
      if (connected) {
        relatedIds.add(sourceId);
        relatedIds.add(targetId);
      }
      return connected;
    });

    setGraph({
      nodes: (analysisGraph.nodes || []).filter((node) => relatedIds.has(node.id)),
      links: relatedLinks,
    });
  }

  function resetGraphView() {
    setSelectedCriminal(null);
    setSelectedRelationship(null);
    setGraph(analysisGraph || { nodes: [], links: [] });
    setCriminalSearch('');
    setCriminalResults([]);
    setShowSearchResults(false);
  }

  useEffect(() => {
    if (!selected) return;

    let cancelled = false;

    async function loadPersistedCase() {
      restoringCaseRef.current = true;
      sourceHydratedRef.current = false;
      hydratedCaseRef.current = null;
      setSourceSaveStatus("Loading saved case…");

      // Clear only the source editor/selection for the newly opened case.
      // Do NOT clear persisted analysis here; it will be replaced after the
      // authenticated workspace response arrives.
      setSourceDrafts({ ...EMPTY_SOURCE });
      setSourceLanguage("en");
      setEditingSources({});

      try {
        // Resolve the current session at request time. This avoids the login
        // race where the selected investigation is rendered before React has
        // committed the fresh access token into component state.
        const usableSession = await getUsableSession();
        if (!usableSession?.access_token) {
          throw new Error("Access token required");
        }

        const workspace = await apiFetch(
          `/api/investigations/${selected.id}/workspace`,
          {},
          usableSession
        );

        if (cancelled) return;

        // One authenticated workspace request is the source of truth after
        // login: investigation metadata + raw source text + persisted graph +
        // last analysis. This prevents separate requests from racing while
        // Supabase is restoring the session.
        const analysisResult = workspace || {};
        let persistedSources = Array.isArray(workspace?.sources)
          ? workspace.sources
          : [];

        // Compatibility fallback for an older backend response shape.
        if (!persistedSources.length) {
          const fallback = await apiFetch(
            `/api/investigations/${selected.id}/sources`,
            {},
            usableSession
          ).catch(() => null);
          if (Array.isArray(fallback)) persistedSources = fallback;
          else if (Array.isArray(fallback?.sources)) persistedSources = fallback.sources;
        }

        console.debug(
          "Loaded persisted investigation workspace",
          selected.id,
          "sources:",
          persistedSources.length,
          "nodes:",
          workspace?.graph?.nodes?.length || 0,
          "links:",
          workspace?.graph?.links?.length || 0
        );

        const drafts = { ...EMPTY_SOURCE };

const restoredUploadedFiles = {};

let firLanguage = "en";

        persistedSources.forEach((source) => {
          const sourceType = String(
            source?.source_type ?? source?.type ?? ""
          ).trim().toUpperCase();
          const content =
            source?.content ??
            source?.raw_text ??
            source?.text ??
            source?.body ??
            "";

          if (Object.prototype.hasOwnProperty.call(drafts, sourceType)) {
            drafts[sourceType] = String(content || "");
          }
          if (
  Object.prototype.hasOwnProperty.call(
    drafts,
    sourceType
  ) &&
  String(content || "").trim()
) {

  const title =
    source?.title ||
    "";

  if (title) {

    restoredUploadedFiles[
      sourceType
    ] = title;

  }

}
          if (sourceType === "FIR" && source?.language) {
            firLanguage = source.language;
          }
        });

        sourceDraftsRef.current = drafts;
        sourceLanguageRef.current = firLanguage;
        setSourceDrafts(drafts);
        setUploadedFiles(
  restoredUploadedFiles
);
        setSourceLanguage(firLanguage);
        sourceHydratedRef.current = true;
        hydratedCaseRef.current = selected.id;

        const persistedGraph = analysisResult?.graph || { nodes: [], links: [] };
        setAnalysisGraph(persistedGraph);
        setGraph(persistedGraph);
        setSelectedCriminal(null);
        setSelectedRelationship(null);

        setAnalysis(
          analysisResult?.sources?.length ||
          analysisResult?.candidate_relationships?.length ||
          persistedGraph.nodes.length ||
          persistedGraph.links.length
            ? analysisResult
            : null
        );

        const savedHash = analysisResult?.source_snapshot_hash;
        const currentHash = await hashSourceDrafts(drafts, firLanguage);
        setAnalysisStale(Boolean(savedHash && currentHash && savedHash !== currentHash));

        setEditingSources({});
        setSourceSaveStatus(
          persistedSources.some((source) => String(source?.content || "").trim())
            ? "Saved"
            : "No saved source data"
        );
      } catch (err) {
        if (!cancelled) {
          console.warn("Saved case load failed:", err.message);
          setSourceSaveStatus("Unable to load saved case");
        }
      } finally {
        restoringCaseRef.current = false;
      }
    }

    loadPersistedCase();

    return () => { cancelled = true; };
  }, [selected?.id, session?.access_token]);

  async function saveSourcesForInvestigation(
  investigationId,
  drafts = sourceDraftsRef.current,
  firLanguage = sourceLanguageRef.current,
  fileNames = uploadedFiles
) {
  if (!investigationId) return;

  const snapshot = {
    ...EMPTY_SOURCE,
    ...(drafts || {}),
  };

  const sources = SOURCE_TYPES
    .filter((source) =>
      String(snapshot[source.key] || "").trim()
    )
    .map((source) => ({
      source_type: source.key,

      title:
        fileNames?.[source.key] ||
        source.label,

      content:
        String(
          snapshot[source.key] || ""
        ),

      language:
        source.key === "FIR"
          ? firLanguage || "en"
          : "en",
    }));

  sourceSaveInFlightRef.current =
    sourceSaveInFlightRef.current
      .catch(() => {})
      .then(async () => {

        setSourceSaveStatus("Saving…");

        const activeSession =
          await getUsableSession(session);

        if (!activeSession?.access_token) {
          throw new Error(
            "Access token required"
          );
        }

        const result =
          await apiFetch(
            `/api/investigations/${investigationId}/sources`,
            {
              method: "PUT",
              body: JSON.stringify({
                sources,
              }),
            },
            activeSession
          );

        const restoredFiles = {};

        (
          result?.sources || []
        ).forEach((item) => {

          const type =
            String(
              item?.source_type || ""
            )
              .trim()
              .toUpperCase();

          if (
            type &&
            item?.title &&
            item?.content
          ) {
            restoredFiles[type] =
              item.title;
          }

        });

        if (
          Object.keys(restoredFiles).length
        ) {

          setUploadedFiles((prev) => ({
            ...prev,
            ...restoredFiles,
          }));

        }

        setSourceSaveStatus("Saved");

      });

  return sourceSaveInFlightRef.current;
}

  useEffect(() => {
    if (
      !selected?.id ||
      !session?.access_token ||
      restoringCaseRef.current ||
      !sourceHydratedRef.current ||
      hydratedCaseRef.current !== selected.id
    ) return;

    if (sourceSaveTimerRef.current) clearTimeout(sourceSaveTimerRef.current);

    sourceSaveTimerRef.current = setTimeout(() => {
      saveSourcesForInvestigation(selected.id, sourceDraftsRef.current, sourceLanguageRef.current).catch((err) => {
        console.warn("Source autosave failed:", err.message);
        setSourceSaveStatus("Save failed");
      });
    }, 700);

    return () => {
      if (sourceSaveTimerRef.current) clearTimeout(sourceSaveTimerRef.current);
    };
  }, [selected?.id, session?.access_token, sourceDrafts, sourceLanguage]);

  useEffect(() => {
    const graphApi = graphRef.current;
    if (!graphApi) return;

    const charge = graphApi.d3Force("charge");
    const link = graphApi.d3Force("link");
    const center = graphApi.d3Force("center");

    // Large separation between people + gentle centering.
    // This is deliberately much looser than the default force layout.
    charge?.strength?.(-2600);
    charge?.distanceMax?.(1400);
    link?.distance?.(430);
    link?.strength?.(0.65);
    center?.strength?.(0.08);

    graphApi.d3ReheatSimulation?.();
  }, [graph.nodes.length, graph.links.length, activeTab]);

  function closeInvestigation() {
    if (!selected) return;
    // This UI intentionally keeps case closure as an explicit state operation.
    apiFetch(`/api/investigations/${selected.id}/close`, { method: "POST" }, session)
      .then(() => loadInvestigations())
      .catch((err) => setError(err.message || "Unable to close investigation."));
  }

  async function signOut() {
    setAuthLoading(true);
    setBusyMessage("Signing you out securely…");
    sourceHydratedRef.current = false;
    hydratedCaseRef.current = null;
    setError("");
    try {
      // Flush the current source drafts before invalidating the session.
      if (selected?.id) {
        const usableSession = await getUsableSession();
        if (usableSession?.access_token) {
          await saveSourcesForInvestigation(
            selected.id,
            sourceDraftsRef.current,
            sourceLanguageRef.current
          );
        }
      }
      await supabase.auth.signOut();
      resetCaseState();
      setSession(null);
      setProfile(null);
      setInvestigations([]);
      setSelected(null);
    } catch (err) {
      setError(err.message || "Unable to sign out.");
    } finally {
      setAuthLoading(false);
      setBusyMessage("");
    }
  }

  const visibleInvestigations = useMemo(() => {
    return investigations.filter((item) => {
      if (investigationFilter === "all") return true;
      return item.status === investigationFilter;
    });
  }, [investigations, investigationFilter]);

  const entityTotal = analysis
    ? Object.values(analysis.entity_counts || {}).reduce((sum, value) => sum + Number(value || 0), 0)
    : 0;

  const filledSourceCount = SOURCE_TYPES.filter((source) => sourceDrafts[source.key]?.trim()).length;
  const suspiciousCount = analysis?.suspicious_patterns?.length || 0;

  // Small status chips on the tabs that have something worth flagging.
  const tabBadges = {
    sources: analysisStale
      ? { text: "!", tone: "warn", label: "Evidence changed since the last analysis" }
      : filledSourceCount
        ? { text: `${filledSourceCount}/${SOURCE_TYPES.length}`, tone: "neutral", label: `${filledSourceCount} of ${SOURCE_TYPES.length} sources added` }
        : null,
    analytics: suspiciousCount
      ? { text: String(suspiciousCount), tone: "warn", label: `${suspiciousCount} suspicious signal(s)` }
      : null,
    integrity: blockchainStatus
      ? blockchainStatus.valid
        ? { text: "✓", tone: "ok", label: "Blockchain verified" }
        : { text: "!", tone: "danger", label: "Integrity violation detected" }
      : null,
  };

  function handleTabKeyDown(event) {
    const keys = WORKSPACE_TABS.map((tab) => tab.key);
    const index = keys.indexOf(activeTab);
    let next = null;
    if (event.key === "ArrowRight") next = keys[(index + 1) % keys.length];
    else if (event.key === "ArrowLeft") next = keys[(index - 1 + keys.length) % keys.length];
    else if (event.key === "Home") next = keys[0];
    else if (event.key === "End") next = keys[keys.length - 1];
    if (!next) return;
    event.preventDefault();
    setActiveTab(next);
    document.getElementById(`tab-${next}`)?.focus();
  }


  // Investigator Assistance: derive visual and analytical insights from the
  // same evidence-backed analysis already returned by the backend.
  const investigatorInsights = useMemo(() => {
    const relationships = Array.isArray(analysis?.candidate_relationships)
      ? analysis.candidate_relationships
      : [];
    const influential = Array.isArray(analysis?.influential_persons)
      ? analysis.influential_persons
      : [];
    const suspicious = Array.isArray(analysis?.suspicious_patterns)
      ? analysis.suspicious_patterns
      : [];

    const personName = (value) => String(value || "Unknown").trim();
    const getA = (item) => personName(item.person_a_name || item.person_a_id || item.person_a_key);
    const getB = (item) => personName(item.person_b_name || item.person_b_id || item.person_b_key);
    const confidence = (item) => Number(item.model_confidence ?? item.confidence ?? 0);

    // Highlight ALL relationships tied at the highest DISPLAYED percentage.
    // The UI shows rounded whole percentages (for example 76.6% and 76.9%
    // can both display as 77%), so comparison must use the same rounded value
    // instead of the hidden raw floating-point confidence.
    const confidencePercent = (item) => Math.round(confidence(item) * 100);

    const strongestScore = relationships.length
      ? Math.max(...relationships.map((item) => confidencePercent(item)))
      : null;

    const strongestRelationships = strongestScore == null
      ? []
      : relationships.filter(
          (item) => confidencePercent(item) === strongestScore
        );

    // Kept for existing summary/recommendation UI compatibility.
    const strongestRelationship = strongestRelationships[0] || null;

    const activity = {};
    relationships.forEach((item) => {
      [getA(item), getB(item)].forEach((name) => {
        if (!activity[name]) activity[name] = { name, calls: 0, transactions: 0, amount: 0, meetings: 0, links: 0, confidence: 0 };
        activity[name].calls += Number(item.calls ?? item.phone_call_count ?? 0);
        activity[name].transactions += Number(item.transactions ?? item.transaction_count ?? 0);
        activity[name].amount += Number(item.amount ?? item.total_transaction_amount ?? 0);
        activity[name].meetings += Number(item.meetings ?? item.meeting_count ?? 0);
        activity[name].links += 1;
        activity[name].confidence += confidence(item);
      });
    });

    const activityPeople = Object.values(activity);
    const mostFinancial = [...activityPeople].sort((a, b) => b.amount - a.amount)[0] || null;
    const mostCommunication = [...activityPeople].sort((a, b) => b.calls - a.calls)[0] || null;
    const highestRisk = [...relationships]
      .sort((a, b) => {
        const ar = String(a.risk_level || "").toLowerCase() === "high" ? 3 : String(a.risk_level || "").toLowerCase() === "medium" ? 2 : 1;
        const br = String(b.risk_level || "").toLowerCase() === "high" ? 3 : String(b.risk_level || "").toLowerCase() === "medium" ? 2 : 1;
        return br - ar || confidence(b) - confidence(a);
      })[0] || null;

    const recommendations = [];
    if (suspicious.length) recommendations.push(`Review ${suspicious.length} suspicious relationship signal(s) and validate them against the underlying evidence.`);
    if (strongestRelationship) recommendations.push(`Prioritize the link between ${getA(strongestRelationship)} and ${getB(strongestRelationship)} because it has the strongest current evidence score.`);
    if (mostFinancial && mostFinancial.amount > 0) recommendations.push(`Trace financial activity connected to ${mostFinancial.name}, including linked transactions and counterparties.`);
    if (mostCommunication && mostCommunication.calls > 0) recommendations.push(`Review repeated communication involving ${mostCommunication.name} and compare call activity with other source timelines.`);
    if (!recommendations.length) recommendations.push("Add more structured communication, financial, or surveillance evidence to generate stronger analytical leads.");

    return {
      influential: influential[0] || null,
      strongestRelationship,
      strongestRelationships,
      strongestScore,

      // Used by the graph to highlight the #1 network-central person.
      influentialPersonId:
        influential[0]?.person_id ||
        influential[0]?.id ||
        influential[0]?.person_key ||
        null,
      influentialPersonName: influential[0]?.name || null,

      highestRisk,
      mostFinancial,
      mostCommunication,
      suspiciousCount: suspicious.length,
      relationshipCount: relationships.length,
      recommendations: recommendations.slice(0, 4),
      relationshipBars: [...relationships].sort((a, b) => confidence(b) - confidence(a)).slice(0, 5),
      getA,
      getB,
      confidence,
      confidencePercent,
    };
  }, [analysis]);

  // Visual prioritization helpers. IDs are preferred because graph labels can
  // change; names are kept as a fallback for older persisted analyses.
  const normalizeGraphValue = (value) => String(value || "").trim().toLowerCase();

  const isInfluentialGraphNode = (node) => {
    const influentialId = normalizeGraphValue(investigatorInsights.influentialPersonId);
    const influentialName = normalizeGraphValue(investigatorInsights.influentialPersonName);
    return (
      (influentialId && normalizeGraphValue(node?.id) === influentialId) ||
      (influentialName && normalizeGraphValue(node?.name) === influentialName)
    );
  };

  const isStrongestGraphLink = (link) => {
    const strongestRelationships = investigatorInsights.strongestRelationships || [];
    if (!strongestRelationships.length || !link) return false;

    const source = typeof link.source === "object" ? link.source : { id: link.source };
    const target = typeof link.target === "object" ? link.target : { id: link.target };

    const linkPair = [
      normalizeGraphValue(source?.id || source?.name),
      normalizeGraphValue(target?.id || target?.name),
    ].sort().join("|");

    const graphNames = [
      normalizeGraphValue(source?.name || source?.id),
      normalizeGraphValue(target?.name || target?.id),
    ].sort().join("|");

    return strongestRelationships.some((strongest) => {
      const strongestPair = [
        normalizeGraphValue(
          strongest.person_a_id || strongest.person_a_key || strongest.person_a_name
        ),
        normalizeGraphValue(
          strongest.person_b_id || strongest.person_b_key || strongest.person_b_name
        ),
      ].sort().join("|");

      if (linkPair && strongestPair && linkPair === strongestPair) return true;

      // Fallback for graph links that preserve names rather than person IDs.
      const strongestNames = [
        normalizeGraphValue(investigatorInsights.getA(strongest)),
        normalizeGraphValue(investigatorInsights.getB(strongest)),
      ].sort().join("|");

      return Boolean(graphNames && strongestNames && graphNames === strongestNames);
    });
  
  };
  if (loading) {
    return (
      <div className="app-shell center-screen">
        <div className="loading-card">
          <div className="brand-mark large">N</div>
          <div className="eyebrow">SECURE WORKSPACE</div>
          <h1>NyayaNet</h1>
          <p>Initializing investigative intelligence environment…</p>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="app-shell center-screen">
        <div className="auth-card">
          <div className="brand-row">
            <div className="brand-mark">N</div>
            <div>
              <div className="brand-title">NyayaNet</div>
              <div className="brand-subtitle">AI-POWERED CRIMINAL NETWORK ANALYSIS</div>
            </div>
          </div>
          <div className="auth-header">
            <span className="eyebrow">AUTHORIZED ACCESS</span>
            <h1>{isSignup ? "Create Investigator Account" : "Secure Login"}</h1>
            <p>Investigative intelligence workspace for authorized personnel.
            Use these credentials for testing:
            Email:- nyayanet@gmail.com
            Password:- Nyayanet </p>
          </div>
          {error && <div className="error-box">{error}</div>}
          <form onSubmit={handleAuth} className="stack-form">
            {isSignup && (
              <label>Full Name<input value={fullName} onChange={(e) => setFullName(e.target.value)} /></label>
            )}
            <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="investigator@example.gov" /></label>
            <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
            <button className="primary-button" disabled={authLoading}>{authLoading ? "Authenticating…" : isSignup ? "Create Account" : "Login Securely"}</button>
          </form>
          <button className="link-button" onClick={() => { setIsSignup((value) => !value); setError(""); }}>
            {isSignup ? "Already have an account? Login" : "Need an account? Create one"}
          </button>
        </div>
      </div>
    );
  }

  if (!profile?.is_authorized) {
    return (
      <div className="app-shell center-screen">
        <div className="auth-card restricted-card">
          <div className="restricted-icon">🔒</div>
          <span className="eyebrow">ACCESS CONTROL</span>
          <h1>Access Restricted</h1>
          <p>Your account is authenticated but is not currently authorized for the investigative workspace.</p>
          <button className="ghost-button" onClick={signOut}>Sign Out</button>
        </div>
      </div>
    );
  }

  return (
    <>
      {(authLoading || analysisLoading || creating) && (
        <div className="global-busy-overlay" role="status" aria-live="polite">
          <div className="global-busy-card">
            <div className="busy-spinner" aria-hidden="true" />
            <strong>
              {busyMessage ||
                (analysisLoading
                  ? "Running intelligence analysis…"
                  : creating
                    ? "Creating and analyzing investigation…"
                    : "Working securely…")}
            </strong>
            <span>Please wait…</span>
          </div>
        </div>
      )}
      <div className="app-shell dashboard-shell">
      <aside className="sidebar">
        <div>
          <div className="brand-row sidebar-brand">
            <div className="brand-mark">N</div>
            <div>
              <div className="brand-title">NyayaNet</div>
              <div className="brand-subtitle">INVESTIGATIVE INTELLIGENCE</div>
            </div>
          </div>

          <div className="sidebar-heading-row">
            <span>INVESTIGATIONS</span>
            <button className="small-primary" onClick={() => { setError(""); setShowCreateModal(true); }}>+ New</button>
          </div>

          <div className="filter-pills">
            {[
              ["active", "Ongoing"],
              ["closed", "Closed"],
              ["all", "All"],
            ].map(([value, label]) => (
              <button key={value} className={investigationFilter === value ? "active" : ""} onClick={() => setInvestigationFilter(value)}>
                {label}
              </button>
            ))}
          </div>

          <div className="investigation-list">
            {visibleInvestigations.length === 0 ? (
              <div className="sidebar-empty">No {investigationFilter === "all" ? "" : investigationFilter} investigations.</div>
            ) : visibleInvestigations.map((investigation) => (
              <button
                key={investigation.id}
                className={`investigation-item ${selected?.id === investigation.id ? "active" : ""}`}
                onClick={() => {
                  setSelected(investigation);
                  resetCaseState();
                }}
              >
                <span className="investigation-code">{investigation.investigation_code}</span>
                <strong>{investigation.title}</strong>
                <span className={`status-badge ${investigation.status}`}>{investigation.status}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="user-card">
            <div className="user-avatar">{initials(profile.full_name || session.user.email)}</div>
            <div>
              <strong>{profile.full_name || session.user.email}</strong>
              <span>{profile.role || "investigator"}</span>
            </div>
          </div>
          <button className="logout-button" onClick={signOut}>Sign Out</button>
          <div className="security-note">🔐 Authorized access • investigative actions are audited</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">SECURE INVESTIGATIVE WORKSPACE</div>
            <h1>Criminal Network Analysis</h1>
            <p>Collect intelligence, extract entities, discover candidate relationships, and surface suspicious patterns.</p>
          </div>
          <div className="authorized-pill"><span /> AUTHORIZED</div>
        </header>

        {error && <div className="error-box main-error"><span>{error}</span><button onClick={() => setError("")}>×</button></div>}

        {!selected ? (
          <section className="empty-dashboard panel">
            <div className="empty-icon">+</div>
            <div className="eyebrow">INVESTIGATION WORKSPACE</div>
            <h2>Start a New Investigation</h2>
            <p>Create a case, provide the available intelligence sources, and let NyayaNet build the analytical view.</p>
            <button className="primary-button compact" onClick={() => setShowCreateModal(true)}>Start New Investigation</button>
          </section>
        ) : (
          <>
            <section className="case-banner panel">
              <div>
                <div className="eyebrow">ACTIVE CASE</div>
                <h2>{selected.title}</h2>
                <p>{selected.description || "No case description provided."}</p>
              </div>
              <div className="case-meta">
                <span>INVESTIGATION ID</span>
                <strong>{selected.investigation_code}</strong>
                <button className="ghost-button small" onClick={closeInvestigation} disabled={selected.status === "closed"}>
                  {selected.status === "closed" ? "Investigation Closed" : "Close Investigation"}
                </button>
              </div>
            </section>

            {analysis?.warnings?.length > 0 && (
              <div className="error-box main-error analysis-warning">
                <span>{analysis.warnings.join(" ")}</span>
                <button onClick={() => setAnalysis((prev) => (prev ? { ...prev, warnings: [] } : prev))}>×</button>
              </div>
            )}

            <div className="tabs" role="tablist" aria-label="Investigation workspace" onKeyDown={handleTabKeyDown}>
              {WORKSPACE_TABS.map((tab) => {
                const badge = tabBadges[tab.key];
                const isActive = activeTab === tab.key;
                return (
                  <button
                    key={tab.key}
                    type="button"
                    role="tab"
                    id={`tab-${tab.key}`}
                    aria-selected={isActive}
                    aria-controls="workspace-panel"
                    tabIndex={isActive ? 0 : -1}
                    className="tab"
                    onClick={() => setActiveTab(tab.key)}
                  >
                    {tab.label}
                    {badge && <span className={`tab-badge ${badge.tone}`} title={badge.label}>{badge.text}</span>}
                  </button>
                );
              })}
            </div>

            <div className="tab-panel" role="tabpanel" id="workspace-panel" aria-labelledby={`tab-${activeTab}`} key={activeTab}>

              {/* ===== OVERVIEW ===== */}
              {activeTab === "overview" && (
                <>
                  <section className="stats-grid">
                    <div className="stat-card"><span>SOURCES PROCESSED</span><strong>{analysis?.sources?.length || 0}</strong><small>documents in this analysis</small></div>
                    <div className="stat-card"><span>ENTITIES EXTRACTED</span><strong>{entityTotal}</strong><small>people, places, vehicles & more</small></div>
                    <div className="stat-card"><span>CANDIDATE LINKS</span><strong>{analysis?.candidate_relationships?.length || 0}</strong><small>model-scored analytical leads</small></div>
                    <div className="stat-card alert-stat"><span>SUSPICIOUS PATTERNS</span><strong>{analysis?.suspicious_patterns?.length || 0}</strong><small>requires investigator review</small></div>
                  </section>

                  {analysis ? (
                    <>
                      <section className="panel investigation-summary-panel">
                        <div className="section-header">
                          <div>
                            <div className="eyebrow">CASE SUMMARY</div>
                            <h2>Investigation Overview</h2>
                            <p>{analysis.summary_text || "Persistent analytical snapshot for this investigation."}</p>
                          </div>
                          <div className="summary-run-badge">
                            {analysis.analysis_run?.created_at
                              ? `Last analyzed ${new Date(analysis.analysis_run.created_at).toLocaleString()}`
                              : "Saved case data"}
                          </div>
                        </div>
                        <div className="summary-grid">
                          <div className="summary-stat"><span>PEOPLE</span><strong>{analysis.graph?.nodes?.length || 0}</strong></div>
                          <div className="summary-stat"><span>RELATIONSHIPS</span><strong>{analysis.graph?.links?.length || 0}</strong></div>
                          <div className="summary-stat"><span>COMMUNITIES</span><strong>{analysis.community_count || 0}</strong></div>
                          <div className="summary-stat summary-alert"><span>SUSPICIOUS SIGNALS</span><strong>{analysis.suspicious_patterns?.length || 0}</strong></div>
                        </div>
                      </section>
                    </>
                  ) : (
                    <TabEmpty title="No analysis yet" actionLabel="Add sources" onAction={() => setActiveTab("sources")}>
                      Add at least one intelligence source and run the analysis to see the case overview here.
                    </TabEmpty>
                  )}
                </>
              )}

              {/* ===== SOURCES ===== */}
              {activeTab === "sources" && (
                <>
                  <section className="panel source-panel">
                    <div className="section-header">
                      <div>
                        <div className="eyebrow">DATA INGESTION</div>
                        <h2>Add Intelligence Sources</h2>
                        <p>Provide available intelligence. The system keeps the original text, extracts entities, and builds candidate evidence across sources.</p>
                      </div>
                      <div className="pipeline-badge">INGEST → NLP → GRAPH → ANALYTICS</div>
                    </div>

                    <div className="source-grid">
                      {SOURCE_TYPES.map((source) => (
                        <div className="source-card" key={source.key}>
                          <div className="source-card-head">
                            <span className="source-icon">{source.icon}</span>
                            <div><strong>{source.label}</strong><small>{source.hint}</small></div>
                          </div>
                          <div className="source-edit-row">
                            <span className={`source-status ${sourceDrafts[source.key]?.trim() ? "has-content" : ""}`}>
                              {sourceDrafts[source.key]?.trim()
                                ? "Saved to case"
                                : "No record added"}
                            </span>

                            <button
                              type="button"
                              className="ghost-button small source-edit-button"
                              onClick={() =>
                                setEditingSources((prev) => ({
                                  ...prev,
                                  [source.key]: !prev[source.key],
                                }))
                              }
                            >
                              {editingSources[source.key] ? "Done" : "Edit"}
                            </button>
                          </div>

                          <div className="source-upload-row">
                            <label className="ghost-button small source-upload-button">
                              {pdfUploading[source.key] ? "Reading PDF…" : "Upload PDF"}
                              <input
                                type="file"
                                accept=".pdf,application/pdf"
                                hidden
                                onChange={(e) => {
                                  const file = e.target.files?.[0];
                                  handleExistingSourcePdfUpload(source.key, file);
                                  e.target.value = "";
                                }}
                              />
                            </label>
                            {uploadedFiles[source.key] && (

        <div className="uploaded-file-info">

          <span className="pdf-icon">
            📄
          </span>

          <div>

            <small>
              UPLOADED SOURCE
            </small>

            <strong>
              {uploadedFiles[source.key]}
            </strong>

          </div>

        </div>

      )}
                          </div>

                          <textarea
                            rows={5}
                            placeholder={`Paste ${source.label.toLowerCase()} here…`}
                            value={sourceDrafts[source.key]}
                            readOnly={!editingSources[source.key]}
                            onChange={(e) => {
                              const value = e.target.value;
                              sourceDraftsRef.current = {
                                ...sourceDraftsRef.current,
                                [source.key]: value,
                              };
                              setAnalysisStale(true);
                              setSourceDrafts((prev) => ({
                                ...prev,
                                [source.key]: value,
                              }));
                            }}
                          />

                          {source.key === "FIR" && (
                            <select
                              value={sourceLanguage}
                              disabled={!editingSources.FIR}
                              onChange={(e) => {
                                setAnalysisStale(true);
                                setSourceLanguage(e.target.value);
                              }}
                            >
                              <option value="en">FIR language: English</option>
                              <option value="hi">FIR language: Hindi</option>
                              <option value="pa">FIR language: Punjabi</option>
                            </select>
                          )}
                        </div>
                      ))}
                    </div>

                    <div className="source-actions">
                      <button className="ghost-button" onClick={() => selected && saveSourcesForInvestigation(selected.id, sourceDraftsRef.current, sourceLanguageRef.current)} disabled={analysisLoading}>Save All Sources</button>
                      <button className="primary-button" onClick={analyzeSourcesForExistingCase} disabled={analysisLoading}>
                        {analysisLoading ? "Running Intelligence Analysis…" : "Run Intelligence Analysis"}
                      </button>
                      <span>
                        {sourceSaveStatus} · At least one source is required.
                        Add only the sources available for the case.
                        {analysisStale && (
                          <strong className="analysis-stale">
                            {" "}Evidence changed — run analysis again to refresh
                            the graph and analytics.
                          </strong>
                        )}
                      </span>
                    </div>
                  </section>
                </>
              )}

              {/* ===== NETWORK EXPLORER ===== */}
              {activeTab === "network" && (
                <>
                  <section className="panel network-workspace">
                    <div className="section-header">
                      <div>
                        <div className="eyebrow">NETWORK INTELLIGENCE</div>
                        <h2>Criminal Network Explorer</h2>
                        <p>NyayaNet builds this network automatically from the intelligence submitted to this investigation. Search is used to focus the generated network on a subject.</p>
                      </div>
                      <div className="legend"><span><i className="legend-dot selected" /> Selected Subject</span><span><i className="legend-dot connected" /> Connected Person</span><span className="legend-influential">● Most Influential Person</span><span className="legend-strongest">━ Strongest Relationship</span><span>Hover a node for profile details</span></div>
                    </div>

                    <div className="network-topbar">
                      <div className="search-panel">
                        <label>FOCUS WITHIN GENERATED NETWORK</label>
                        <div className="search-input-wrap">
                          <span>⌕</span>
                          <input
                            value={criminalSearch}
                            placeholder="Search a person from this investigation…"
                            onChange={(e) => setCriminalSearch(e.target.value)}
                            onFocus={() => criminalResults.length && setShowSearchResults(true)}
                          />
                          {criminalSearch && <button onClick={resetGraphView}>×</button>}
                          {searchLoading && <em>Searching…</em>}
                        </div>
                        {showSearchResults && (
                          <div className="search-results">
                            <div className="search-results-title">MATCHING RECORDS <span>{criminalResults.length}</span></div>
                            {criminalResults.length ? criminalResults.map((person) => (
                              <button key={person.id} className="search-result-row" onClick={() => selectCriminal(person)}>
                                <div className="result-avatar">{initials(person.name)}</div>
                                <div className="result-main"><strong>{person.name}</strong><small>{person.person_id} · {person.location || "Location unavailable"}</small><span>☎ {person.phone_num || "No phone"}</span></div>
                                <span className="result-arrow">›</span>
                              </button>
                            )) : <div className="empty-inline">No matching person found.</div>}
                          </div>
                        )}
                      </div>

                      {selectedCriminal && (
                        <div className="subject-summary">
                          <div className="subject-avatar">{initials(selectedCriminal.name)}</div>
                          <div className="subject-main"><span>SELECTED SUBJECT</span><strong>{selectedCriminal.name}</strong><small>{selectedCriminal.person_id} · {selectedCriminal.location || "Location unavailable"}</small></div>
                          <div className="subject-metric"><span>AGE</span><strong>{selectedCriminal.age || "—"}</strong></div>
                          <div className="subject-metric"><span>CONNECTIONS</span><strong>{selectedCriminal ? Math.max(0, graph.nodes.length - 1) : analysisGraph.links.length}</strong></div>
                          <div className="subject-metric"><span>PHONE</span><strong>{selectedCriminal.phone_num || "—"}</strong></div>
                          <div className="subject-metric"><span>VEHICLE</span><strong>{selectedCriminal.vehicle_num || "—"}</strong></div>
                        </div>
                      )}
                    </div>

                    <div className={`network-body ${selectedRelationship ? "with-details" : ""}`}>
                      <div className="graph-panel">
                        <div className="graph-titlebar"><div><span>RELATIONSHIP MAP</span><strong>{selectedCriminal ? `${Math.max(0, graph.nodes.length - 1)} connections in focus` : `${analysisGraph.links.length} candidate connections generated from submitted evidence`}</strong></div><button className="ghost-button small" onClick={() => { resetGraphView(); setTimeout(() => graphRef.current?.zoomToFit?.(500, 80), 0); }}>Reset View</button></div>
                        <div className="graph-canvas">
                          {graphLoading ? (
                            <div className="graph-placeholder"><div className="spinner" /><h3>Building subject network…</h3><p>Combining relationship records and model signals.</p></div>
                          ) : graph.nodes.length === 0 ? (
                            <div className="graph-placeholder"><div className="placeholder-icon">◌</div><h3>No network generated yet</h3><p>Submit at least one intelligence source and run analysis. The graph is generated only from this investigation's submitted evidence.</p></div>
                          ) : (
                            <ForceGraph2D
                              ref={graphRef}
                              graphData={graph}
                              backgroundColor={GRAPH_COLORS.background}
                              enableNodeDrag
                              cooldownTicks={360}
                              warmupTicks={100}
                              d3AlphaDecay={0.012}
                              d3VelocityDecay={0.2}
                              nodeRelSize={7}

                              // ONLY PEOPLE are rendered as graph nodes.
                              // The full name is the node label; all other person
                              // attributes stay inside the hover tooltip.
                              nodeLabel={(node) => `
                                <div class="node-tooltip">
                                  <div class="node-tooltip-head">
                                    <div class="node-tooltip-avatar">${escapeHtml(initials(node.name))}</div>
                                    <div>
                                      <strong>${escapeHtml(node.name || "Unknown")}</strong>
                                      <span>PERSON</span>
                                    </div>
                                  </div>
                                  <div class="node-tooltip-grid">
                                    <div><span>AGE</span><b>${escapeHtml(node.age ?? "—")}</b></div>
                                    <div><span>LOCATION</span><b>${escapeHtml(node.location || "—")}</b></div>
                                    <div><span>PHONE</span><b>${escapeHtml(node.phone_num || "—")}</b></div>
                                    <div><span>VEHICLE</span><b>${escapeHtml(node.vehicle_num || "—")}</b></div>
                                    <div><span>ORGANIZATION</span><b>${escapeHtml(node.org || "—")}</b></div>
                                    <div><span>CRIME RECORDED</span><b>${escapeHtml(node.crime_recorded || "—")}</b></div>
                                    <div><span>SOURCES</span><b>${escapeHtml((node.source_types || []).join(" • ") || "—")}</b></div>
                                  </div>
                                </div>
                              `}

                              // Relationship hover shows the connection type, why the
                              // people are connected, evidence counts and confidence.
                              linkLabel={(link) => `
                                <div class="edge-tooltip">
                                  <strong>${escapeHtml(link.relationship_type || "Evidence-linked Association")}</strong>
                                  <span class="edge-confidence">
                                    Potential relationship confidence:
                                    ${link.confidence != null
                                      ? `${Math.round(Number(link.confidence) * 100)}%`
                                      : "N/A"}
                                  </span>
                                  <p>${escapeHtml(link.reason || link.relationship_description || "Evidence-backed relationship.")}</p>
                                  <div class="edge-evidence">
                                    ${Number(link.calls || 0) > 0 ? `<span>☎ ${Number(link.calls)} call(s)</span>` : ""}
                                    ${Number(link.transactions || 0) > 0 ? `<span>₹ ${Number(link.transactions)} transaction(s)</span>` : ""}
                                    ${Number(link.meetings || 0) > 0 ? `<span>● ${Number(link.meetings)} meeting(s)</span>` : ""}
                                  </div>
                                </div>
                              `}

                              nodeCanvasObject={(node, ctx, globalScale) => {
                                const isHovered = hoveredNode === node;
                                const isCenter = Boolean(node.is_center);
                                const isInfluential = isInfluentialGraphNode(node);
                                const radius = isHovered || isCenter || isInfluential ? 16 : 12;

                                ctx.save();

                                if (isHovered || isCenter || isInfluential) {
                                  ctx.beginPath();
                                  ctx.arc(node.x, node.y, radius + 8, 0, Math.PI * 2);
                                  ctx.fillStyle = isInfluential
                                    ? GRAPH_COLORS.influential.halo
                                    : isCenter
                                      ? GRAPH_COLORS.center.halo
                                      : GRAPH_COLORS.node.halo;
                                  ctx.fill();
                                }

                                // Soft glow: strong for the states that matter, a faint bloom otherwise.
                                ctx.shadowColor = isInfluential
                                  ? GRAPH_COLORS.influential.glow
                                  : isCenter ? GRAPH_COLORS.center.glow : GRAPH_COLORS.node.glow;
                                ctx.shadowBlur = isHovered || isCenter || isInfluential ? 24 : 9;

                                ctx.beginPath();
                                ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                                ctx.fillStyle = isInfluential
                                  ? GRAPH_COLORS.influential.fill
                                  : isCenter ? GRAPH_COLORS.center.fill : GRAPH_COLORS.node.fill;
                                ctx.fill();
                                ctx.strokeStyle = isInfluential
                                  ? GRAPH_COLORS.influential.stroke
                                  : isCenter
                                    ? GRAPH_COLORS.center.stroke
                                    : isHovered
                                      ? GRAPH_COLORS.node.hover
                                      : GRAPH_COLORS.node.stroke;
                                ctx.lineWidth = isHovered ? 3 : 2;
                                ctx.stroke();
                                ctx.shadowBlur = 0;

                                ctx.font = "700 11px Inter, system-ui, sans-serif";
                                ctx.fillStyle = GRAPH_COLORS.label.initials;
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText(initials(node.name), node.x, node.y);

                                if (isInfluential) {
                                  ctx.font = "700 10px Inter, system-ui, sans-serif";
                                  ctx.fillStyle = GRAPH_COLORS.influential.star;
                                  ctx.fillText("★", node.x, node.y - radius - 7);
                                }

                                // Keep names readable but compact. No IDs or entity
                                // metadata are painted onto the graph.
                                const name = node.name || "Unknown";
                                const nameSize = Math.max(
                                  10,
                                  Math.min(13, 12 / Math.max(globalScale, 0.8))
                                );
                                ctx.font = `600 ${nameSize}px Inter, system-ui, sans-serif`;

                                const textWidth = ctx.measureText(name).width;
                                const pillWidth = textWidth + 14;
                                const pillHeight = nameSize + 10;
                                const pillY = node.y + radius + 7;

                                ctx.fillStyle = GRAPH_COLORS.label.pill;
                                ctx.beginPath();
                                ctx.roundRect(
                                  node.x - pillWidth / 2,
                                  pillY,
                                  pillWidth,
                                  pillHeight,
                                  6
                                );
                                ctx.fill();

                                ctx.fillStyle = GRAPH_COLORS.label.text;
                                ctx.textBaseline = "middle";
                                ctx.fillText(name, node.x, pillY + pillHeight / 2);

                                ctx.restore();
                              }}

                              // Keep the graph clean: no permanent relationship prose.
                              // Confidence is always visible as a compact badge on the
                              // relationship itself, while full evidence appears on hover.
                              linkCanvasObjectMode={() => "after"}
                              linkCanvasObject={(link, ctx, globalScale) => {
                                const source = link.source;
                                const target = link.target;
                                if (!source || !target) return;
                                if (typeof source.x !== "number" || typeof target.x !== "number") return;

                                const isStrongest = isStrongestGraphLink(link);
                                const confidence = link.confidence != null
                                  ? `${Math.round(Number(link.confidence) * 100)}%`
                                  : "—";

                                const x = (source.x + target.x) / 2;
                                const y = (source.y + target.y) / 2;
                                const fontSize = Math.max(9, Math.min(12, 10 / Math.max(globalScale, 0.8)));

                                ctx.save();
                                ctx.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
                                const label = confidence;
                                const width = ctx.measureText(label).width + 14;
                                const height = fontSize + 9;

                                ctx.fillStyle = isStrongest
                                  ? GRAPH_COLORS.badge.strongestFill
                                  : hoveredLink === link
                                    ? GRAPH_COLORS.badge.hoverFill
                                    : GRAPH_COLORS.badge.fill;
                                ctx.strokeStyle = isStrongest
                                  ? GRAPH_COLORS.badge.strongestStroke
                                  : hoveredLink === link
                                    ? GRAPH_COLORS.badge.hoverStroke
                                    : GRAPH_COLORS.badge.stroke;
                                ctx.lineWidth = hoveredLink === link ? 1.5 : 1;

                                if (isStrongest || hoveredLink === link) {
                                  ctx.shadowColor = isStrongest ? GRAPH_COLORS.influential.glow : GRAPH_COLORS.node.glow;
                                  ctx.shadowBlur = 14;
                                }

                                ctx.beginPath();
                                ctx.roundRect(
                                  x - width / 2,
                                  y - height / 2,
                                  width,
                                  height,
                                  5
                                );
                                ctx.fill();
                                ctx.stroke();
                                ctx.shadowBlur = 0;

                                ctx.fillStyle = isStrongest ? GRAPH_COLORS.badge.strongestText : GRAPH_COLORS.badge.text;
                                ctx.textAlign = "center";
                                ctx.textBaseline = "middle";
                                ctx.fillText(label, x, y);
                                ctx.restore();
                              }}

                              linkWidth={(link) =>
                                isStrongestGraphLink(link)
                                  ? 4.2
                                  : hoveredLink === link ? 3.5 : 1.6
                              }
                              linkColor={(link) =>
                                isStrongestGraphLink(link)
                                  ? GRAPH_COLORS.link.strongest
                                  : hoveredLink === link ? GRAPH_COLORS.link.hover : GRAPH_COLORS.link.base
                              }
                              linkDirectionalArrowLength={7}
                              linkDirectionalArrowRelPos={1}
                              linkCurvature={0.08}

                              onNodeHover={(node) => setHoveredNode(node || null)}
                              onLinkHover={(link) => setHoveredLink(link || null)}
                              onNodeClick={(node) => setSelectedCriminal(node)}
                              onLinkClick={(link) => setSelectedRelationship(link)}

                              onNodeDragEnd={(node) => {
                                node.fx = null;
                                node.fy = null;
                                graphRef.current?.d3ReheatSimulation?.();
                              }}

                              onEngineStop={() => {
                                graphRef.current?.zoomToFit?.(700, 110);
                              }}
                            />
                          )}
                          <div className="graph-help">Generated from current case evidence • Drag nodes • Scroll to zoom • Click a relationship for evidence details • Hover a node for profile information</div>
                        </div>
                      </div>

                      {selectedRelationship && (
                        <aside className="relationship-panel">
                          <div className="relationship-panel-head"><div><span className="eyebrow">RELATIONSHIP INTELLIGENCE</span><h3>Connection Details</h3></div><button onClick={() => setSelectedRelationship(null)}>×</button></div>
                          <div className="relationship-subjects">
                            <div>
                              <span>PERSON A</span>
                              <strong>
                                {graph.nodes.find((n) => n.id === (
                                  typeof selectedRelationship.source === "object"
                                    ? selectedRelationship.source.id
                                    : selectedRelationship.source
                                ))?.name || selectedRelationship.source}
                              </strong>
                            </div>
                            <div className="relationship-arrow">↔</div>
                            <div>
                              <span>PERSON B</span>
                              <strong>
                                {graph.nodes.find((n) => n.id === (
                                  typeof selectedRelationship.target === "object"
                                    ? selectedRelationship.target.id
                                    : selectedRelationship.target
                                ))?.name || selectedRelationship.target}
                              </strong>
                            </div>
                          </div>
                          <div className="relationship-type-block"><span>RELATIONSHIP TYPE</span><strong>{selectedRelationship.relationship_type || "Potential Relationship"}</strong><em>{selectedRelationship.confidence != null ? `${Math.round(selectedRelationship.confidence * 100)}% analytical confidence` : "Confidence unavailable"}</em></div>
                          <div className="relationship-metrics"><div><span>PHONE CALLS</span><strong>{selectedRelationship.calls || 0}</strong></div><div><span>TRANSACTIONS</span><strong>{selectedRelationship.transactions || 0}</strong></div><div><span>MEETINGS</span><strong>{selectedRelationship.meetings || 0}</strong></div><div><span>TRANSACTION VALUE</span><strong>₹{Number(selectedRelationship.total_transaction_amount || 0).toLocaleString("en-IN")}</strong></div></div>
                          <div className="relationship-evidence"><span>EVIDENCE EXPLANATION</span><p>{selectedRelationship.relationship_description || selectedRelationship.reason || "No explanation is available for this candidate link."}</p></div>
                          <div className="lead-warning">Analytical lead only. This score does not establish criminal guilt or prove the stated relationship.</div>
                        </aside>
                      )}
                    </div>
                  </section>
                </>
              )}

              {/* ===== ANALYTICS ===== */}
              {activeTab === "analytics" && (
                <>
                  <section className="analytics-grid">
                    <div className="panel analysis-card">
                      <div className="section-header compact-header"><div><div className="eyebrow">NETWORK INTELLIGENCE</div><h2>Influential Individuals</h2></div></div>
                      <div className="rank-list">
                        {(analysis?.influential_persons || []).slice(0, 6).map((person, index) => (
                          <div className="rank-row" key={person.person_id}>
                            <span className="rank-number">{index + 1}</span>
                            <div><strong>{person.name}</strong><small>{person.person_id}</small></div>
                            <div className="rank-score">{Math.round((person.influence_score || 0) * 100)}<small>influence</small></div>
                          </div>
                        ))}
                        {!analysis?.influential_persons?.length && <div className="empty-inline">Run intelligence analysis to identify network-central individuals.</div>}
                      </div>
                    </div>

                    <div className="panel analysis-card">
                      <div className="section-header compact-header"><div><div className="eyebrow">PATTERN DETECTION</div><h2>Suspicious Activity Signals</h2></div></div>
                      <div className="pattern-list">
                        {(analysis?.suspicious_patterns || []).slice(0, 6).map((pattern, index) => (
                          <div className="pattern-row" key={`${pattern.person_a_id}-${pattern.person_b_id}-${index}`}>
                            <div className="pattern-icon">!</div>
                            <div><strong>{pattern.person_a_id} ↔ {pattern.person_b_id}</strong><small>{(pattern.reasons || []).join(" • ") || "Unusual activity combination"}</small></div>
                            <span>{Math.round((pattern.confidence || 0) * 100)}%</span>
                          </div>
                        ))}
                        {!analysis?.suspicious_patterns?.length && <div className="empty-inline">No suspicious combinations surfaced yet.</div>}
                      </div>
                    </div>
                  </section>
                </>
              )}

              {/* ===== INSIGHTS (Investigator Assistance) ===== */}
              {activeTab === "insights" && (
                analysis ? (
                  <>
                  <section className="panel investigator-assistance-panel">
                    <div className="section-header assistance-title">
                      <div>
                        <div className="eyebrow">INVESTIGATOR ASSISTANCE</div>
                        <h2>Visual & Analytical Insights</h2>
                        <p>Evidence-backed insights to help investigators prioritize relationships, risks and next actions.</p>
                      </div>
                    </div>

                    <div className="assistance-insight-grid">
                      <div className="assist-card"><span>MOST INFLUENTIAL PERSON</span><strong>{investigatorInsights.influential?.name || "Not identified"}</strong><small>{investigatorInsights.influential ? `${Math.round(Number(investigatorInsights.influential.influence_score || 0) * 100)} influence score` : "Run analysis with connected entities"}</small></div>
                      <div className="assist-card"><span>STRONGEST RELATIONSHIP</span><strong>{investigatorInsights.strongestRelationship ? `${investigatorInsights.getA(investigatorInsights.strongestRelationship)} ↔ ${investigatorInsights.getB(investigatorInsights.strongestRelationship)}` : "Not identified"}</strong><small>{investigatorInsights.strongestRelationship ? `${Math.round(investigatorInsights.confidence(investigatorInsights.strongestRelationship) * 100)}% evidence confidence` : "No relationship evidence yet"}</small></div>
                      <div className="assist-card"><span>HIGHEST RISK LEAD</span><strong>{investigatorInsights.highestRisk ? `${investigatorInsights.getA(investigatorInsights.highestRisk)} ↔ ${investigatorInsights.getB(investigatorInsights.highestRisk)}` : "Not identified"}</strong><small>{investigatorInsights.highestRisk?.risk_level ? `${investigatorInsights.highestRisk.risk_level} risk classification` : "No high-risk relationship surfaced"}</small></div>
                      <div className="assist-card"><span>SUSPICIOUS SIGNALS</span><strong>{investigatorInsights.suspiciousCount}</strong><small>{investigatorInsights.relationshipCount} evidence-backed relationship(s) analyzed</small></div>
                    </div>

                    <div className="assistance-lower">
                      <div className="assist-subpanel">
                        <div className="eyebrow">VISUAL PRIORITIZATION</div><h3>Relationship Strength</h3>
                        {investigatorInsights.relationshipBars.length ? investigatorInsights.relationshipBars.map((item, index) => {
                          const score = Math.round(investigatorInsights.confidence(item) * 100);
                          const isStrongest = investigatorInsights.strongestRelationships?.some(
                            (strongest) => strongest === item
                          );
                          return <div className={`relationship-bar-row ${isStrongest ? "strongest-relationship-row" : ""}`} key={`${investigatorInsights.getA(item)}-${investigatorInsights.getB(item)}-${index}`}><div className="relationship-bar-label"><span>{investigatorInsights.getA(item)} ↔ {investigatorInsights.getB(item)} {isStrongest && <b className="strongest-label">STRONGEST</b>}</span><strong>{score}%</strong></div><div className="relationship-bar-track"><div className="relationship-bar-fill" style={{ width: `${Math.max(2, Math.min(score, 100))}%` }} /></div></div>;
                        }) : <div className="assist-empty">Relationship strength will appear after candidate relationships are generated.</div>}
                      </div>
                      <div className="assist-subpanel">
                        <div className="eyebrow">NEXT ACTIONS</div><h3>Investigator Recommendations</h3>
                        {investigatorInsights.recommendations.map((recommendation, index) => <div className="recommendation-row" key={index}><span className="recommendation-icon">✓</span><span>{recommendation}</span></div>)}
                      </div>
                    </div>
                  </section>
                  </>
                ) : (
                  <TabEmpty title="No insights yet" actionLabel="Go to Sources" onAction={() => setActiveTab("sources")}>
                    Run the intelligence analysis to rank relationships, surface the highest-risk lead and get recommended next steps.
                  </TabEmpty>
                )
              )}

              {/* ===== FIR INTELLIGENCE ===== */}
              {activeTab === "fir" && (
                <section className="panel utility-panel tab-narrow">
                    <div className="section-header compact-header"><div><div className="eyebrow">NLP ENGINE</div><h2>Standalone FIR Intelligence</h2></div></div>
                    <textarea className="utility-textarea" rows={7} value={firText} onChange={(e) => setFirText(e.target.value)} placeholder="Paste an additional FIR / report for focused entity extraction…" />
                    <div className="utility-actions"><select value={sourceLanguage} onChange={(e) => setSourceLanguage(e.target.value)}><option value="en">English</option><option value="hi">Hindi</option><option value="pa">Punjabi</option></select><button className="primary-button" onClick={analyzeFIR} disabled={firAnalyzing}>{firAnalyzing ? "Analyzing…" : "Extract Entities"}</button></div>
                    {firEntities.length > 0 && <div className="entity-list">{firEntities.map((entity, index) => <div className="entity-chip" key={`${entity.label}-${entity.text}-${index}`}><span>{entity.label}</span><strong>{entity.text}</strong></div>)}</div>}
                </section>
              )}

              {/* ===== TIP ANALYSIS ===== */}
              {activeTab === "tips" && (
                <section className="panel utility-panel tab-narrow">
                    <div className="section-header compact-header"><div><div className="eyebrow">INTELLIGENCE SEED</div><h2>Tip → Network</h2></div></div>
                    <textarea className="utility-textarea" rows={7} value={tipText} onChange={(e) => setTipText(e.target.value)} placeholder="Enter a small tip or lead…" />
                    <button className="primary-button" onClick={analyzeTip} disabled={tipAnalyzing}>{tipAnalyzing ? "Analyzing…" : "Analyze Tip"}</button>
                    {tipResult && <pre className="tip-result">{JSON.stringify(tipResult, null, 2)}</pre>}
                </section>
              )}

              {/* ===== EVIDENCE INTEGRITY (BLOCKCHAIN) ===== */}
              {activeTab === "integrity" && (
                <>
      <section className="panel blockchain-panel">

        <div className="section-header">
          <div>
            <div className="eyebrow">
              EVIDENCE SECURITY
            </div>

            <h2>
              🔐 Evidence Integrity & Blockchain
            </h2>

            <p>
              Verify investigation evidence and blockchain
              records for unauthorized modifications.
            </p>
          </div>

          <button
            type="button"
            className="primary-button"
            onClick={verifyBlockchainIntegrity}
            disabled={
              blockchainLoading || !selected?.id
            }
          >
            {blockchainLoading
              ? "Verifying..."
              : "🔄 Verify Integrity"}
          </button>
        </div>


        {/* ERROR */}

        {blockchainError && (

          <div className="error-box main-error">

            <span>
              ❌ {blockchainError}
            </span>

          </div>

        )}


        {/* DEFAULT STATE */}

        {!blockchainStatus &&
          !blockchainError &&
          !blockchainLoading && (

            <div className="empty-state">

              <div className="empty-icon">
                🔐
              </div>

              <strong>
                Blockchain verification not run
              </strong>

              <p>
                Verify the evidence chain to ensure
                investigation data has not been modified.
              </p>

            </div>

          )}


        {/* LOADING */}

        {blockchainLoading && (

          <div className="empty-state">

            <div className="empty-icon">
              ⏳
            </div>

            <strong>
              Verifying Blockchain...
            </strong>

            <p>
              Checking evidence hashes and
              blockchain integrity.
            </p>

          </div>

        )}


        {/* RESULT */}

        {blockchainStatus && (

          <>

            {/* STATUS */}

            <div
              className={
                blockchainStatus.valid
                  ? "blockchain-status verified"
                  : "blockchain-status invalid"
              }
            >

              <div className="blockchain-status-icon">

                {blockchainStatus.valid
                  ? "🟢"
                  : "🔴"}

              </div>

              <div>

                <h3>

                  {blockchainStatus.valid
                    ? "Blockchain Verified"
                    : "Integrity Violation Detected"}

                </h3>

                <p>
                  {blockchainStatus.message}
                </p>

              </div>

            </div>


            {/* STATISTICS */}

            <div className="stats-grid blockchain-stats">

              <div className="stat-card">

                <span>
                  TOTAL BLOCKS
                </span>

                <strong>
                  {blockchainStatus.total_blocks || 0}
                </strong>

                <small>
                  evidence records secured
                </small>

              </div>


              <div className="stat-card">

                <span>
                  VERIFIED BLOCKS
                </span>

                <strong>
                  {blockchainStatus.verified_blocks || 0}
                </strong>

                <small>
                  successfully validated
                </small>

              </div>


              <div
                className={
                  blockchainStatus.tampering_detected
                    ? "stat-card alert-stat"
                    : "stat-card"
                }
              >

                <span>
                  EVIDENCE TAMPERING
                </span>

                <strong>

                  {blockchainStatus.tampering_detected
                    ? "YES"
                    : "NO"}

                </strong>

                <small>

                  {blockchainStatus.tampering_detected
                    ? "unauthorized changes detected"
                    : "no modification detected"}

                </small>

              </div>

            </div>


            {/* TAMPERED BLOCKS */}

            {blockchainStatus.tampering_detected &&
              blockchainStatus.tampered_blocks?.length > 0 && (

                <div className="tampered-evidence">

                  <div className="section-header">

                    <div>

                      <div className="eyebrow">
                        SECURITY ALERT
                      </div>

                      <h3>
                        ⚠️ Tampered Evidence Detected
                      </h3>

                    </div>

                  </div>


                  {blockchainStatus.tampered_blocks.map(
                    (block, index) => (

                      <div
                        className="tampered-item"
                        key={
                          `${block.evidence_id}-${index}`
                        }
                      >

                        <div>

                          <strong>

                            {block.title ||
                              block.source_type ||
                              "Unknown Evidence"}

                          </strong>


                          <small>

                            Evidence ID:
                            {" "}
                            {block.evidence_id}

                          </small>


                          <small>

                            Source:
                            {" "}
                            {block.source_type}

                          </small>

                        </div>


                        <div className="tamper-errors">

                          {block.errors?.map(
                            (error, errorIndex) => (

                              <div
                                key={errorIndex}
                              >

                                🔴 {error}

                              </div>

                            )
                          )}

                        </div>

                      </div>

                    )
                  )}

                </div>

              )}

          </>

        )}

      </section>
                </>
              )}

              {/* ===== ANALYTICAL GUARDRAILS ===== */}
              {activeTab === "guardrails" && (
                <>
                  <section className="panel methodology-panel">
                    <div><div className="eyebrow">ANALYTICAL GUARDRAILS</div><h2>How NyayaNet interprets evidence</h2></div>
                    <div className="guardrail-grid"><div><strong>Candidate relationship score</strong><p>Ranks evidence-backed links using observable communication, transaction, meeting and shared-entity signals.</p></div><div><strong>Suspicious pattern detection</strong><p>Flags unusual combinations of activity for investigator review; it does not declare guilt.</p></div><div><strong>Network influence</strong><p>Uses graph-centrality measures to identify structurally influential nodes, not “most criminal” people.</p></div></div>
                  </section>
                </>
              )}
            </div>
          </>
        )}
      </main>

      {showCreateModal && (
        <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && !creating && setShowCreateModal(false)}>
          <form className="create-modal" onSubmit={createInvestigation}>
            <div className="modal-head"><div><div className="eyebrow">NEW INVESTIGATION</div><h2>Start Investigation Workspace</h2><p>Enter the case details and all available intelligence sources. NyayaNet will process them immediately after the case is created.</p></div><button type="button" onClick={() => !creating && setShowCreateModal(false)}>×</button></div>
            <div className="modal-grid-top"><label>Investigation Title<input value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="e.g. Network analysis — Sector 18" /></label><label>Case Purpose / Description<textarea rows={3} value={newDescription} onChange={(e) => setNewDescription(e.target.value)} placeholder="Scope, objective, known lead, or case summary…" /></label></div>
            <div className="modal-source-header"><div><span>INTELLIGENCE SOURCES</span><small>Provide the sources available for this case. FIR is not the only accepted input.</small></div><select value={newFirLanguage} onChange={(e) => setNewFirLanguage(e.target.value)}><option value="en">FIR: English</option><option value="hi">FIR: Hindi</option><option value="pa">FIR: Punjabi</option></select></div>
            <div className="modal-source-grid">
              {SOURCE_TYPES.map((source) => (
                <div className="modal-source-card" key={source.key}>
                  <div>
                    <span>{source.icon}</span>
                    <strong>{source.label}</strong>
                  </div>

                  <div className="source-upload-row">
                    <label className="ghost-button small source-upload-button">
                      {pdfUploading[`new_${source.key}`] ? "Reading PDF…" : "Upload PDF"}
                      <input
                        type="file"
                        accept=".pdf,application/pdf"
                        hidden
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          handleNewSourcePdfUpload(source.key, file);
                          e.target.value = "";
                        }}
                      />
                    </label>

                    {newUploadedFiles[source.key] && (
                      <small className="uploaded-file-name">
                        PDF loaded: {newUploadedFiles[source.key]}
                      </small>
                    )}
                  </div>

                  <textarea
                    rows={4}
                    value={newSources[source.key]}
                    onChange={(e) =>
                      setNewSources((prev) => ({
                        ...prev,
                        [source.key]: e.target.value,
                      }))
                    }
                    placeholder={`Enter or paste ${source.label.toLowerCase()}…`}
                  />
                </div>
              ))}
            </div>
            <div className="modal-foot"><span>At least one source must be supplied. Original evidence is stored with an integrity hash.</span><div><button type="button" className="ghost-button" onClick={() => setShowCreateModal(false)} disabled={creating}>Cancel</button><button type="submit" className="primary-button" disabled={creating}>{creating ? "Creating & Analyzing…" : "Create & Analyze Investigation"}</button></div></div>
          </form>
        </div>
      )}
    </div>
    </>
  );
}