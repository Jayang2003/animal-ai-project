import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import logo from "../assets/logo.jpeg";

const API = "http://127.0.0.1:8000";

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" width="15" height="15">
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#1565a8" strokeWidth="2" strokeLinecap="round">
      <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
      <polyline points="13 2 13 9 20 9" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#a0b8cc" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#378ADD" strokeWidth="1.8" strokeLinecap="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

function E5LogoLarge() {
  return (
    <svg viewBox="0 0 80 100" width="36" height="45">
      <image href={logo} width="80" height="100" />
    </svg>
  );
}

function E5LogoMini() {
  return (
    <svg viewBox="0 0 80 100" width="40" height="45">
      <image href={logo} width="80" height="100" />
    </svg>
  );
}

function ThinkingDots() {
  const [frame, setFrame] = useState(0);
  const frames = ["...", ".. ", "..."];
  useEffect(() => {
    const id = setInterval(() => setFrame((f) => (f + 1) % 3), 400);
    return () => clearInterval(id);
  }, []);
  return <span style={{ color: "#6a8aaa", fontSize: 13 }}>Thinking{frames[frame]}</span>;
}

function formatTime(timestamp) {
  if (timestamp) {
    const d = new Date(timestamp);
    if (!isNaN(d)) return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function makeSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function getStorageKey(username) {
  return `chat_sessions_${username}`;
}

function getImportedFlagKey(username) {
  return `chat_history_imported_${username}`;
}

function buildHistory(messages) {
  return messages
    .filter((m) => m.role === "user" || m.role === "ai")
    .map((m) => ({
      role: m.role === "ai" ? "assistant" : "user",
      content: m.role === "ai" ? m.content.replace(/<[^>]+>/g, "") : m.content || "",
    }))
    .filter((m) => m.content.trim() !== "");
}

function buildMessagesFromPairs(pairs) {
  const rebuilt = [];
  pairs.forEach((p) => {
    rebuilt.push({ role: "user", content: p.q, files: [] });
    rebuilt.push({ role: "ai", content: p.a });
  });
  return rebuilt;
}

function getSessionPreview(session) {
  if (!session?.pairs?.length) return "Empty chat";
  return (session.pairs[0]?.q || "").slice(0, 60);
}

export default function Dashboard() {
  const username = localStorage.getItem("username") || "admin";

  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputVal, setInputVal] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [showWelcome, setShowWelcome] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editingLabel, setEditingLabel] = useState("");
  const [hoveredSessionId, setHoveredSessionId] = useState(null);

  const contentRef  = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const canSend = inputVal.trim() !== "" || uploadedFiles.length > 0;

  useEffect(() => {
    async function loadData() {
      try {
        const storageKey     = getStorageKey(username);
        const importedFlagKey = getImportedFlagKey(username);
        const saved = localStorage.getItem(storageKey);

        if (saved) {
          const parsedSessions = JSON.parse(saved);
          if (Array.isArray(parsedSessions) && parsedSessions.length > 0) {
            setSessions(parsedSessions);
            setCurrentSession(parsedSessions[0]);
            setMessages(buildMessagesFromPairs(parsedSessions[0].pairs || []));
            setShowWelcome(false);
            setHistoryLoading(false);
            return;
          }
        }

        const alreadyImported = localStorage.getItem(importedFlagKey);
        if (!alreadyImported) {
          const res = await axios.get(`${API}/sessions`, { params: { username } });
          const rows = res.data;

          if (rows && rows.length > 0) {
            const sessionsWithMessages = await Promise.all(
              rows.map(async (s) => {
                const msgRes = await axios.get(`${API}/messages`, {
                  params: { session_id: s.id },
                });
                return {
                  id:    s.id,
                  label: s.label,
                  time:  formatTime(s.updated_at),
                  pairs: msgRes.data.map((m) => ({ q: m.question, a: m.answer })),
                };
              })
            );

            setSessions(sessionsWithMessages);
            setCurrentSession(sessionsWithMessages[0]);
            setMessages(buildMessagesFromPairs(sessionsWithMessages[0].pairs || []));
            setShowWelcome(false);
            localStorage.setItem(storageKey, JSON.stringify(sessionsWithMessages));
          }
          localStorage.setItem(importedFlagKey, "true");
        }
      } catch (err) {
        console.error("Failed to load history:", err);
      } finally {
        setHistoryLoading(false);
      }
    }
    loadData();
  }, [username]);

  useEffect(() => {
    if (!historyLoading) {
      localStorage.setItem(getStorageKey(username), JSON.stringify(sessions));
    }
  }, [sessions, username, historyLoading]);

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [messages]);

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 100) + "px";
  }, []);

  function handleFiles(e) {
    setUploadedFiles(Array.from(e.target.files));
    e.target.value = "";
  }

  function removeFile(i) {
    setUploadedFiles((prev) => prev.filter((_, idx) => idx !== i));
  }

  function loadSession(session) {
    setCurrentSession(session);
    setMessages(buildMessagesFromPairs(session.pairs || []));
    setShowWelcome(false);
  }

  function newSession() {
    const newChat = {
      id:    makeSessionId(),
      label: `New Chat ${sessions.length + 1}`,
      time:  formatTime(),
      pairs: [],
    };
    setSessions((prev) => [newChat, ...prev]);
    setCurrentSession(newChat);
    setMessages([]);
    setShowWelcome(true);
    setInputVal("");
    setUploadedFiles([]);
    setTimeout(autoResize, 0);
  }

  async function sendMessage() {
    if (!canSend) return;

    const filesSnapshot = [...uploadedFiles];
    const text = inputVal.trim();
    let activeSession = currentSession;

    if (!activeSession) {
      activeSession = {
        id:    makeSessionId(),
        label: `New Chat ${sessions.length + 1}`,
        time:  formatTime(),
        pairs: [],
      };
      setSessions((prev) => [activeSession, ...prev]);
      setCurrentSession(activeSession);
    }

    setShowWelcome(false);

    // ✅ Set messages ONCE here
    const newMessages = [
      ...messages,
      { role: "user", content: text, files: filesSnapshot },
      { role: "typing" },
    ];
    setMessages(newMessages);
    setInputVal("");
    setUploadedFiles([]);
    setTimeout(autoResize, 0);

    const historySnapshot = buildHistory(
      newMessages.filter((m) => m.role !== "typing")
    );

    try {
      let answer = "";

      if (filesSnapshot.length > 0) {
        const formData = new FormData();
        formData.append("file", filesSnapshot[0]);
        formData.append("username", username);
        if (text) formData.append("message", text);
        formData.append("history", JSON.stringify(historySnapshot));

        const res = await axios.post(`${API}/predict`, formData, {
  headers: { "Content-Type": "multipart/form-data" },
});

const data = res.data;

// ✅ NEW: handle invalid response
if (!data || typeof data !== "object") {
  answer = "Error: Invalid server response";
}

// ✅ NEW: handle unknown case (IMPORTANT FIX)
else if (data.animal === "unknown") {
  answer = `
**Animal:** Unknown

${data.message || "Unsupported or unclear animal image"}
`;
}

// ✅ NORMAL CASE
else {
  const animal     = data.animal || "Unknown";
  const breed      = data.breed  || "Unknown";
  const confidence = data.breed_result?.confidence != null
    ? (data.breed_result.confidence * 100).toFixed(2)
    : "N/A";

  const info        = data.breed_info || {};
  const origin      = info.origin;
  const lifeSpan    = info.life_span;
  const temperament = info.temperament;
  const food        = info.food;
  const care        = info.care;
  const description = info.description;

  answer = [
    `**Animal:** ${animal}`,
    `**Breed:** ${breed}`,
    `**Confidence:** ${confidence}%`,
    "",
    description ? description : "",
    origin      ? `**Origin:** ${origin}` : "",
    lifeSpan    ? `**Life Span:** ${lifeSpan}` : "",
    temperament ? `**Temperament:** ${temperament}` : "",
    food        ? `**Food:** ${food}` : "",
    care        ? `**Care:** ${care}` : "",
  ]
    .filter(Boolean)
    .join("\n\n");
}

        if (res.data.error) {
          answer = `Error: ${res.data.error}`;
        } else {
          const animal     = res.data.animal || "Unknown";
          const breed      = res.data.breed  || "Unknown";
          const confidence = res.data.breed_result?.confidence != null
            ? (res.data.breed_result.confidence * 100).toFixed(2)
            : "N/A";

          const info        = res.data.breed_info || {};
          const origin      = info.origin      || null;
          const lifeSpan    = info.life_span   || null;
          const temperament = info.temperament || null;
          const food        = info.food        || null;
          const care        = info.care        || null;
          const description = info.description || null;

          answer = [
            `**Animal:** ${animal}`,
            `**Breed:** ${breed}`,
            `**Confidence:** ${confidence}%`,
            "",
            description ? `${description}`                  : "",
            origin      ? `**Origin:** ${origin}`           : "",
            lifeSpan    ? `**Life Span:** ${lifeSpan}`      : "",
            temperament ? `**Temperament:** ${temperament}` : "",
            food        ? `**Food:** ${food}`               : "",
            care        ? `**Care:** ${care}`               : "",
          ]
            .filter(Boolean)
            .join("\n\n");
        }
      } else {
        // ✅ Real chatbot call
        const res = await axios.post(`${API}/chat`, {
          message:    text,
          username,
          session_id: activeSession.id,
          history:    historySnapshot,
          context:    {},
        });
        answer = res.data.response;
      }

      const displayQuestion =
        text || `📎 ${filesSnapshot.map((f) => f.name).join(", ")}`;

      const newPair = { q: displayQuestion, a: answer };

      const updatedSession = {
        ...activeSession,
        time:  formatTime(),
        label: activeSession.pairs.length === 0
          ? displayQuestion.slice(0, 30) || activeSession.label
          : activeSession.label,
        pairs: [...activeSession.pairs, newPair],
      };

      setCurrentSession(updatedSession);

      setSessions((prev) => {
        const exists = prev.some((s) => s.id === updatedSession.id);
        if (!exists) return [updatedSession, ...prev];
        return [updatedSession, ...prev.filter((s) => s.id !== updatedSession.id)];
      });

      // ✅ Set messages ONCE here — replaces typing with answer
      setMessages([
        ...newMessages.filter((m) => m.role !== "typing"),
        { role: "ai", content: answer },
      ]);

      // ✅ Save to DB
      try {
        await axios.post(`${API}/save_session`, {
          session_id: updatedSession.id,
          username,
          label:      updatedSession.label,
        });
        await axios.post(`${API}/save_message`, {
          session_id: updatedSession.id,
          username,
          question:   displayQuestion,
          answer,
        });
      } catch (dbErr) {
        console.error("DB sync failed:", dbErr);
      }

    } catch (err) {
      console.error(err);
      const errMsg = err?.response?.data?.detail || "Error occurred ❌";
      setMessages([
        ...newMessages.filter((m) => m.role !== "typing"),
        { role: "ai", content: errMsg },
      ]);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function handleLogout() {
    localStorage.removeItem("username");
    window.location.href = "/";
  }

  function startRename(e, session) {
    e.stopPropagation();
    setEditingSessionId(session.id);
    setEditingLabel(session.label || "");
  }

  function confirmRename(session) {
    const trimmed = editingLabel.trim();
    if (!trimmed) { setEditingSessionId(null); return; }
    const updated = { ...session, label: trimmed };
    setSessions((prev) => prev.map((s) => (s.id === session.id ? updated : s)));
    if (currentSession?.id === session.id) setCurrentSession(updated);
    setEditingSessionId(null);
    axios.post(`${API}/rename`, { session_id: session.id, new_name: trimmed }).catch(() => {});
  }

  function handleRenameKey(e, session) {
    if (e.key === "Enter") confirmRename(session);
    if (e.key === "Escape") setEditingSessionId(null);
  }

  function deleteSession(e, session) {
    e.stopPropagation();
    if (!window.confirm(`Delete "${session.label || "this chat"}"?`)) return;
    setSessions((prev) => {
      const remaining = prev.filter((s) => s.id !== session.id);
      if (currentSession?.id === session.id) {
        if (remaining.length > 0) {
          setCurrentSession(remaining[0]);
          setMessages(buildMessagesFromPairs(remaining[0].pairs || []));
          setShowWelcome(false);
        } else {
          setCurrentSession(null);
          setMessages([]);
          setShowWelcome(true);
        }
      }
      return remaining;
    });
    axios.post(`${API}/delete`, { session_id: session.id }).catch(() => {});
  }

  return (
    <div style={styles.root}>
      <aside style={styles.sidebar}>
        <div style={styles.sbLogo}>
          <E5LogoLarge />
          <div style={styles.logoWrap}>
            <span style={styles.logoBrand}>emerging5</span>
            <span style={styles.logoTag}>AI Assistant</span>
          </div>
        </div>

        <div style={styles.sbLabel}>Chat History</div>

        <div style={styles.sbHistory}>
          {historyLoading ? (
            <div style={styles.sbEmpty}>Loading history…</div>
          ) : sessions.length === 0 ? (
            <div style={styles.sbEmpty}>
              No conversations yet.<br />Start by asking a question below.
            </div>
          ) : (
            sessions.map((s) => {
              const last      = s.pairs[s.pairs.length - 1];
              const isActive  = currentSession?.id === s.id;
              const isEditing = editingSessionId === s.id;
              const isHovered = hoveredSessionId === s.id;

              return (
                <div
                  key={s.id}
                  style={{ ...styles.hi, ...(isActive ? styles.hiActive : {}), position: "relative" }}
                  onClick={() => !isEditing && loadSession(s)}
                  onMouseEnter={() => setHoveredSessionId(s.id)}
                  onMouseLeave={() => setHoveredSessionId(null)}
                >
                  {isEditing ? (
                    <input
                      autoFocus
                      value={editingLabel}
                      onChange={(e) => setEditingLabel(e.target.value)}
                      onBlur={() => confirmRename(s)}
                      onKeyDown={(e) => handleRenameKey(e, s)}
                      onClick={(e) => e.stopPropagation()}
                      style={styles.renameInput}
                    />
                  ) : (
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
                      <div style={{ ...styles.hiQ, flex: 1, minWidth: 0 }}>{s.label || "New Chat"}</div>
                      {(isHovered || isActive) && (
                        <div style={styles.sessionActions} onClick={(e) => e.stopPropagation()}>
                          <button title="Rename" style={styles.iconBtn} onClick={(e) => startRename(e, s)}>
                            <PencilIcon />
                          </button>
                          <button title="Delete" style={{ ...styles.iconBtn, ...styles.iconBtnDelete }} onClick={(e) => deleteSession(e, s)}>
                            <TrashIcon />
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                  {!isEditing && (
                    <>
                      <div style={styles.hiA}>
                        {last ? last.a.replace(/<[^>]+>/g, "").slice(0, 60) + "…" : getSessionPreview(s)}
                      </div>
                      <div style={styles.hiMeta}>
                        <div style={styles.hiDot} />
                        {s.time}
                      </div>
                    </>
                  )}
                </div>
              );
            })
          )}
        </div>

        <div style={styles.sbBottom}>
          <button style={styles.newBtn} onClick={newSession}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" width="14" height="14">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New Session
          </button>
        </div>
      </aside>

      <div style={styles.main}>
        <header style={styles.topbar}>
          <div style={styles.topbarLeft}>
            <div style={styles.topbarDot} />
            <div>
              <div style={styles.topbarTitle}>AI Assistant</div>
              <div style={styles.topbarSub}>emerging5 Platform</div>
            </div>
          </div>
          <div style={styles.topbarActions}>
            <button style={{ ...styles.tbBtn, ...styles.tbSolid }} onClick={handleLogout}>
              Logout
            </button>
          </div>
        </header>

        <div style={styles.content} ref={contentRef}>
          <div style={styles.sectionLabel}>Upload &amp; Analysis</div>
          <h1 style={styles.pageTitle}>AI Response</h1>
          <p style={styles.pageSub}>Upload an animal image or ask a question to get started.</p>

          <div style={styles.convoWrap}>
            {showWelcome && messages.length === 0 && (
              <div style={styles.welcome}>
                <div style={styles.welcomeIcon}><ChatIcon /></div>
                <p style={{ fontSize: 14, color: "#6a8aaa" }}>
                  Upload an animal image or type a question below to begin.
                </p>
              </div>
            )}

            {messages.map((msg, idx) => {
              if (msg.role === "user") {
                return (
                  <div key={idx} style={styles.msgUser}>
                    <div style={styles.bubbleUser}>
                      {msg.files && msg.files.length > 0 && (
                        <div style={{ fontSize: 11, opacity: 0.75, marginBottom: 4 }}>
                          {msg.files.map((f) => `📎 ${f.name}`).join(" · ")}
                        </div>
                      )}
                      {msg.content}
                    </div>
                  </div>
                );
              }

              if (msg.role === "typing") {
                return (
                  <div key={idx} style={styles.msgAi}>
                    <div style={styles.aiAvatar}><E5LogoMini /></div>
                    <div style={styles.bubbleAi}>
                      <div style={styles.bubbleLabel}>emerging5 AI</div>
                      <ThinkingDots />
                    </div>
                  </div>
                );
              }

              return (
                <div key={idx} style={styles.msgAi}>
                  <div style={styles.aiAvatar}><E5LogoMini /></div>
                  <div style={styles.bubbleAi}>
                    <div style={styles.bubbleLabel}>emerging5 AI</div>
                    <ReactMarkdown
                      components={{
                        p:     ({ children }) => <p style={{ margin: "4px 0", lineHeight: 1.6 }}>{children}</p>,
                        strong:({ children }) => <strong style={{ color: "#0C447C" }}>{children}</strong>,
                        ul:    ({ children }) => <ul style={{ paddingLeft: 18, margin: "6px 0" }}>{children}</ul>,
                        ol:    ({ children }) => <ol style={{ paddingLeft: 18, margin: "6px 0" }}>{children}</ol>,
                        li:    ({ children }) => <li style={{ marginBottom: 4, lineHeight: 1.6 }}>{children}</li>,
                        table: ({ children }) => <table style={{ borderCollapse: "collapse", width: "100%", margin: "8px 0", fontSize: 13 }}>{children}</table>,
                        th:    ({ children }) => <th style={{ background: "#E6F1FB", color: "#0C447C", padding: "6px 10px", border: "1px solid #B5D4F4", textAlign: "left" }}>{children}</th>,
                        td:    ({ children }) => <td style={{ padding: "5px 10px", border: "1px solid #B5D4F4" }}>{children}</td>,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={styles.inputZone}>
          {uploadedFiles.length > 0 && (
            <div style={styles.fileChipsRow}>
              {uploadedFiles.map((f, i) => (
                <div key={i} style={styles.fileChip}>
                  <FileIcon />
                  {f.name}
                  <span style={styles.fileChipX} onClick={() => removeFile(i)}>✕</span>
                </div>
              ))}
            </div>
          )}

          <div style={styles.inputRow}>
            <label style={styles.uploadBtn}>
              <input
                ref={fileInputRef}
                type="file"
                style={{ display: "none" }}
                accept="image/*,.pdf,.doc,.docx,.txt,.csv,.xlsx"
                multiple
                onChange={handleFiles}
              />
              <UploadIcon />
            </label>

            <textarea
              ref={textareaRef}
              style={styles.textarea}
              rows={1}
              placeholder="Ask a question or upload an animal image…"
              value={inputVal}
              onChange={(e) => { setInputVal(e.target.value); autoResize(); }}
              onKeyDown={handleKey}
            />

            <button
              style={{ ...styles.sendBtn, ...(canSend ? {} : styles.sendBtnDisabled) }}
              onClick={sendMessage}
              disabled={!canSend}
            >
              <SendIcon />
            </button>
          </div>

          <div style={styles.inputHint}>
            <InfoIcon />
            Press Enter to send &nbsp;·&nbsp; Shift+Enter for new line &nbsp;·&nbsp; Upload files with the button
          </div>
        </div>
      </div>
    </div>
  );
}

// ── STYLES ─────────────────────────────────────────────────────────────────
const C = {
  blue900: "#042C53", blue800: "#0C447C", blue600: "#185FA5",
  blue400: "#378ADD", blue200: "#85B7EB", blue100: "#B5D4F4",
  blue50:  "#E6F1FB", sidebar: "#071828", sidebar2: "#0c2035",
  text: "#0d1e35", muted: "#6a8aaa", border: "rgba(30,127,212,0.15)",
};

const styles = {
  root:          { fontFamily: "'DM Sans', sans-serif", background: "#f0f6fd", color: C.text, height: "100vh", overflow: "hidden", display: "flex" },
  sidebar:       { width: 260, minWidth: 260, background: C.sidebar, display: "flex", flexDirection: "column", height: "100vh", borderRight: "1px solid rgba(55,138,221,0.12)" },
  sbLogo:        { padding: "20px 18px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", gap: 11 },
  logoWrap:      { display: "flex", flexDirection: "column", gap: 2 },
  logoBrand:     { fontFamily: "'Rajdhani', sans-serif", fontSize: 18, fontWeight: 700, color: "#fff", letterSpacing: ".04em" },
  logoTag:       { fontSize: 9, fontWeight: 600, letterSpacing: ".16em", textTransform: "uppercase", color: C.blue400 },
  sbLabel:       { padding: "14px 16px 5px", fontSize: 9, fontWeight: 600, letterSpacing: ".15em", textTransform: "uppercase", color: "#4a6a8a", opacity: 0.8 },
  sbHistory:     { flex: 1, overflowY: "auto", padding: "4px 10px 8px" },
  sbEmpty:       { padding: 16, textAlign: "center", color: "#2a4a68", fontSize: 12, opacity: 0.6 },
  hi:            { padding: "10px 11px", borderRadius: 8, cursor: "pointer", marginBottom: 2, borderLeft: "2px solid transparent", transition: "background .18s" },
  hiActive:      { background: "rgba(55,138,221,0.14)", borderLeftColor: C.blue400 },
  hiQ:           { fontSize: 12, fontWeight: 500, color: "#b8d4ef", lineHeight: 1.45, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" },
  hiA:           { fontSize: 11, color: "#4a7aa0", marginTop: 4, lineHeight: 1.4, display: "-webkit-box", WebkitLineClamp: 1, WebkitBoxOrient: "vertical", overflow: "hidden", fontStyle: "italic" },
  hiMeta:        { fontSize: 10, color: "#3a5a78", marginTop: 5, display: "flex", alignItems: "center", gap: 5 },
  hiDot:         { width: 5, height: 5, borderRadius: "50%", background: C.blue400, flexShrink: 0 },
  sbBottom:      { padding: 12, borderTop: "1px solid rgba(255,255,255,0.05)" },
  newBtn:        { width: "100%", padding: "10px 14px", background: `linear-gradient(135deg,${C.blue800},${C.blue600})`, color: "#fff", border: "none", borderRadius: 8, fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 600, cursor: "pointer", letterSpacing: ".02em", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 },
  main:          { flex: 1, display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" },
  topbar:        { background: "#fff", borderBottom: `1px solid ${C.border}`, padding: "0 28px", height: 58, display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 },
  topbarLeft:    { display: "flex", alignItems: "center", gap: 14 },
  topbarDot:     { width: 8, height: 8, borderRadius: "50%", background: "#22c55e", flexShrink: 0 },
  topbarTitle:   { fontSize: 14, fontWeight: 600, color: C.text },
  topbarSub:     { fontSize: 12, color: C.muted },
  topbarActions: { display: "flex", gap: 8, alignItems: "center" },
  tbBtn:         { padding: "7px 16px", borderRadius: 8, fontSize: 12.5, fontWeight: 600, fontFamily: "'DM Sans', sans-serif", cursor: "pointer", transition: "all .18s", letterSpacing: ".02em" },
  tbSolid:       { background: C.blue600, border: "none", color: "#fff" },
  content:       { flex: 1, overflowY: "auto", padding: "32px 36px 0" },
  sectionLabel:  { fontSize: 11, fontWeight: 600, letterSpacing: ".14em", textTransform: "uppercase", color: C.blue400, marginBottom: 14 },
  pageTitle:     { fontFamily: "'Rajdhani', sans-serif", fontSize: 34, fontWeight: 700, color: C.blue900, letterSpacing: ".02em", marginBottom: 6 },
  pageSub:       { fontSize: 14, color: C.muted, marginBottom: 28, fontWeight: 400 },
  convoWrap:     { marginBottom: 20 },
  welcome:       { textAlign: "center", padding: "40px 20px", opacity: 0.55 },
  welcomeIcon:   { margin: "0 auto 16px", width: 56, height: 56, borderRadius: "50%", background: C.blue50, border: `2px solid ${C.blue100}`, display: "flex", alignItems: "center", justifyContent: "center" },
  msgUser:       { display: "flex", justifyContent: "flex-end", marginBottom: 16 },
  bubbleUser:    { background: C.blue600, color: "#fff", padding: "12px 16px", borderRadius: "14px 14px 4px 14px", maxWidth: "72%", fontSize: 14, lineHeight: 1.55 },
  msgAi:         { display: "flex", gap: 10, alignItems: "flex-start", marginBottom: 16 },
  aiAvatar:      { width: 32, height: 32, borderRadius: "4px", background: C.blue50, border: `2px solid ${C.blue200}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 },
  bubbleAi:      { background: "#fff", border: `1px solid ${C.border}`, color: C.text, padding: "12px 16px", borderRadius: "4px 14px 14px 14px", maxWidth: "78%", fontSize: 14, lineHeight: 1.6 },
  bubbleLabel:   { fontSize: 10.5, fontWeight: 600, letterSpacing: ".08em", textTransform: "uppercase", color: C.blue400, marginBottom: 6 },
  inputZone:     { background: "#fff", borderTop: `1px solid ${C.border}`, padding: "14px 24px 16px", flexShrink: 0 },
  fileChipsRow:  { display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 },
  fileChip:      { padding: "4px 10px 4px 8px", background: C.blue50, border: `1px solid ${C.blue100}`, borderRadius: 6, fontSize: 11.5, color: C.blue800, display: "flex", alignItems: "center", gap: 5 },
  fileChipX:     { cursor: "pointer", color: C.muted, fontSize: 13, lineHeight: 1, marginLeft: 2 },
  inputRow:      { display: "flex", alignItems: "flex-end", gap: 10, background: "#f8fbff", border: `1.5px solid ${C.blue100}`, borderRadius: 12, padding: "10px 12px" },
  uploadBtn:     { width: 32, height: 32, borderRadius: 8, border: `1px solid ${C.blue200}`, background: "#fff", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: C.blue600 },
  textarea:      { flex: 1, border: "none", background: "transparent", fontFamily: "'DM Sans', sans-serif", fontSize: 14, color: C.text, resize: "none", outline: "none", minHeight: 24, maxHeight: 100, lineHeight: 1.5, padding: "4px 0" },
  sendBtn:       { width: 36, height: 36, borderRadius: 9, border: "none", background: C.blue600, color: "#fff", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "all .18s" },
  sendBtnDisabled: { opacity: 0.45, cursor: "not-allowed" },
  renameInput:   { width: "100%", background: "rgba(255,255,255,0.1)", border: "1px solid rgba(55,138,221,0.5)", borderRadius: 5, color: "#e0eaf5", fontSize: 12, fontFamily: "'DM Sans', sans-serif", padding: "3px 7px", outline: "none", boxSizing: "border-box", marginBottom: 2 },
  sessionActions:{ display: "flex", gap: 3, flexShrink: 0 },
  iconBtn:       { width: 22, height: 22, borderRadius: 5, border: "none", background: "rgba(55,138,221,0.18)", color: "#7ab8e8", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", padding: 0, transition: "background .15s, color .15s" },
  iconBtnDelete: { background: "rgba(220,60,60,0.18)", color: "#e87a7a" },
  inputHint:     { fontSize: 11, color: "#a0b8cc", padding: "6px 4px 0", display: "flex", alignItems: "center", gap: 5 },
};