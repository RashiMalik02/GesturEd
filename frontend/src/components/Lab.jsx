// frontend/src/components/Lab.jsx

import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';

if (!document.getElementById('lab-styles')) {
  const tag = document.createElement('style');
  tag.id = 'lab-styles';
  tag.textContent = `
    @keyframes pulse        { 0%,100%{opacity:1}  50%{opacity:.3} }
    @keyframes fadeSlideUp  { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
    @keyframes revealGlow   { 0%{box-shadow:0 0 0 rgba(74,222,128,0)} 60%{box-shadow:0 0 32px rgba(74,222,128,.35)} 100%{box-shadow:0 0 12px rgba(74,222,128,.15)} }
  `;
  document.head.appendChild(tag);
}

function buildRevealMessage(chemical, reactionType) {
  if (!chemical) return null;
  const { label, formula, type } = chemical;
  const paperColor = reactionType === 'red_litmus' ? 'Red' : 'Blue';
  if (type === 'neutral') {
    return { headline: 'No Reaction Observed', body: `The ${paperColor} Litmus paper did not change colour because ${label} (${formula}) is a neutral substance — it has no acidic or basic properties to trigger a reaction.`, verdict: 'NEUTRAL', color: '#a3a3a3' };
  }
  if (type === 'acid' && reactionType === 'blue_litmus') {
    return { headline: 'Acid Detected!', body: `The Blue Litmus paper turned Red because ${label} (${formula}) is an acid. Acids donate protons (H⁺), which causes blue litmus to change colour.`, verdict: 'ACID CONFIRMED', color: '#f87171' };
  }
  if (type === 'base' && reactionType === 'red_litmus') {
    return { headline: 'Base Detected!', body: `The Red Litmus paper turned Blue because ${label} (${formula}) is a base. Bases accept protons (OH⁻), which causes red litmus to change colour.`, verdict: 'BASE CONFIRMED', color: '#38bdf8' };
  }
  const typeLabel = type === 'acid' ? 'an acid' : 'a base';
  return { headline: 'No Change — Wrong Litmus', body: `${label} (${formula}) is ${typeLabel}, but this test used ${paperColor} Litmus. To observe a colour change, use ${type === 'acid' ? 'Blue' : 'Red'} Litmus with this substance.`, verdict: 'NO CHANGE', color: '#fbbf24' };
}

function getWsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const host  = import.meta.env.VITE_WS_HOST || window.location.host;
  return `${proto}://${host}/ws/lab/`;
}

const FRAME_W = 640;
const FRAME_H = 480;

const s = {
  page: { minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', padding: '1.5rem', gap: '1rem', paddingTop: '2rem' },
  topBar: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', maxWidth: '820px' },
  backBtn: { background: 'none', border: '1px solid var(--border)', color: 'var(--text-muted)', padding: '0.45rem 1.1rem', borderRadius: '3px', fontSize: '0.72rem', fontFamily: 'var(--mono)', letterSpacing: '0.1em', textTransform: 'uppercase', cursor: 'pointer' },
  statusRow: { display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.7rem', fontFamily: 'var(--mono)', letterSpacing: '0.1em', textTransform: 'uppercase' },
  liveDot: { width: '7px', height: '7px', borderRadius: '50%', animation: 'pulse 1.5s ease-in-out infinite' },
  bubbleSection: { width: '100%', maxWidth: '820px' },
  bubbleLabel: { fontSize: '0.62rem', fontFamily: 'var(--mono)', color: 'var(--text-muted)', letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: '0.6rem', display: 'block' },
  bubbleRow: { display: 'flex', flexWrap: 'wrap', gap: '0.5rem' },
  bubble: (active, loading) => ({ padding: '0.38rem 0.9rem', borderRadius: '999px', border: `1px solid ${active ? 'var(--accent-blue)' : 'rgba(255,255,255,0.12)'}`, background: active ? 'rgba(56,189,248,0.12)' : 'transparent', color: active ? 'var(--accent-blue)' : 'var(--text-muted)', fontSize: '0.78rem', fontFamily: 'var(--mono)', cursor: loading ? 'wait' : 'pointer', transition: 'all 0.15s ease', letterSpacing: '0.04em', opacity: loading ? 0.5 : 1, whiteSpace: 'nowrap' }),
  streamCard: { width: '100%', maxWidth: '820px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', boxShadow: 'var(--glow-blue)' },
  windowBar: { display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.55rem 1rem', borderBottom: '1px solid var(--border)' },
  wDot: (c) => ({ width: '10px', height: '10px', borderRadius: '50%', background: c, opacity: 0.7 }),
  wTitle: { marginLeft: 'auto', fontSize: '0.62rem', fontFamily: 'var(--mono)', color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase' },
  canvas: { display: 'block', width: '100%', minHeight: '280px', background: '#000' },
  revealBanner: (c) => ({ width: '100%', maxWidth: '820px', background: 'var(--surface)', border: `1px solid ${c}55`, borderRadius: '6px', overflow: 'hidden', animation: 'fadeSlideUp 0.5s ease, revealGlow 1.2s ease' }),
  revealHeader: (c) => ({ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.7rem 1.2rem', background: `${c}12`, borderBottom: `1px solid ${c}33` }),
  revealHeadline: (c) => ({ fontSize: '0.85rem', fontWeight: 700, color: c, fontFamily: 'var(--sans)', letterSpacing: '0.01em' }),
  revealVerdict: (c) => ({ fontSize: '0.62rem', fontFamily: 'var(--mono)', color: c, letterSpacing: '0.18em', textTransform: 'uppercase', padding: '0.2rem 0.6rem', border: `1px solid ${c}55`, borderRadius: '999px' }),
  revealBody: { padding: '0.9rem 1.2rem', fontSize: '0.85rem', color: 'var(--text-primary)', lineHeight: 1.65, fontFamily: 'var(--sans)' },
  hint: { fontSize: '0.68rem', fontFamily: 'var(--mono)', color: 'var(--text-muted)', letterSpacing: '0.06em' },
};

export default function Lab() {
  const navigate    = useNavigate();
  const stopCalled  = useRef(false);
  const pollRef     = useRef(null);

  const wsRef         = useRef(null);
  const streamRef     = useRef(null);
  const videoRef      = useRef(null);
  const canvasRef     = useRef(null);
  const offCanvasRef  = useRef(null);
  const wsReady       = useRef(false);
  const frameTimerRef = useRef(null);
  const sendingRef    = useRef(false);

  const [chemicals, setChemicals]       = useState([]);
  const [activeId, setActiveId]         = useState(null);
  const [loadingChem, setLoadingChem]   = useState(null);
  const [revealData, setRevealData]     = useState(null);
  const [reactionType, setReactionType] = useState(null);
  const [wsStatus, setWsStatus]         = useState('connecting');

  const startPipeline = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: FRAME_W, height: FRAME_H },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const off = document.createElement('canvas');
      off.width  = FRAME_W;
      off.height = FRAME_H;
      offCanvasRef.current = off;

      const ws = new WebSocket(getWsUrl());
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      function sendFrame() {
        if (!wsReady.current || ws.readyState !== WebSocket.OPEN || sendingRef.current) return;
        sendingRef.current = true;
        const ctx = off.getContext('2d');
        ctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);
        off.toBlob((blob) => {
          if (!blob) { sendingRef.current = false; return; }
          blob.arrayBuffer().then((buf) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(buf);
            sendingRef.current = false;
          });
        }, 'image/jpeg', 0.5);
      }

      ws.onopen = () => {
        wsReady.current = true;
        setWsStatus('live');
        frameTimerRef.current = setInterval(sendFrame, 66);
      };

      ws.onmessage = (evt) => {
        createImageBitmap(new Blob([evt.data], { type: 'image/jpeg' })).then((bitmap) => {
          const canvas = canvasRef.current;
          if (canvas) {
            canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            bitmap.close();
          }
        });
      };

      ws.onerror = () => setWsStatus('error');
      ws.onclose = () => { wsReady.current = false; setWsStatus('error'); };

    } catch (err) {
      console.error('Camera / WebSocket error:', err);
      setWsStatus('error');
    }
  }, []);

  const stopPipeline = useCallback(() => {
    wsReady.current = false;
    if (frameTimerRef.current) { clearInterval(frameTimerRef.current); frameTimerRef.current = null; }
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
  }, []);

  useEffect(() => {
    api.get('/reactions/chemicals/').then((r) => setChemicals(r.data.chemicals)).catch(() => {});
    api.get('/reactions/current/').then((r) => setReactionType(r.data.active_reaction)).catch(() => {});
    startPipeline();
    return () => stopPipeline();
  }, [startPipeline, stopPipeline]);

  useEffect(() => {
    pollRef.current = setInterval(async () => {
      try {
        const { data } = await api.get('/reactions/status/');
        if (data.complete) {
          clearInterval(pollRef.current);
          setRevealData(buildRevealMessage(data.chemical, data.reaction_type || reactionType));
        }
      } catch {}
    }, 1000);
    return () => clearInterval(pollRef.current);
  }, [reactionType]);

  const handleSelectChemical = async (chem) => {
    if (loadingChem || revealData) return;
    setLoadingChem(chem.id);
    setRevealData(null);
    try {
      await api.post('/reactions/set-chemical/', { chemical_id: chem.id });
      setActiveId(chem.id);
    } catch {}
    finally { setLoadingChem(null); }
  };

  const handleBack = async () => {
    if (stopCalled.current) return;
    stopCalled.current = true;
    clearInterval(pollRef.current);
    stopPipeline();
    try { await api.post('/reactions/stop/'); } finally { navigate('/dashboard'); }
  };

  useEffect(() => {
    const onUnload = () => navigator.sendBeacon('/api/reactions/stop/', new Blob([JSON.stringify({})], { type: 'application/json' }));
    window.addEventListener('beforeunload', onUnload);
    return () => {
      window.removeEventListener('beforeunload', onUnload);
      if (!stopCalled.current) {
        stopCalled.current = true;
        clearInterval(pollRef.current);
        stopPipeline();
        api.post('/reactions/stop/').catch(() => {});
      }
    };
  }, [stopPipeline]);

  const statusColor = wsStatus === 'live' ? 'var(--accent-green)' : wsStatus === 'error' ? '#f87171' : '#fbbf24';
  const statusLabel = wsStatus === 'live' ? 'Live Stream' : wsStatus === 'error' ? 'Connection Error' : 'Connecting…';

  return (
    <div style={s.page}>
      <video ref={videoRef} style={{ display: 'none' }} playsInline muted />

      <div style={s.topBar}>
        <button style={s.backBtn} onClick={handleBack}>← Back</button>
        <div style={{ ...s.statusRow, color: statusColor }}>
          <span style={{ ...s.liveDot, background: statusColor }} />
          {statusLabel}
        </div>
      </div>

      <div style={s.bubbleSection}>
        <span style={s.bubbleLabel}>// Select substance for the test tube</span>
        <div style={s.bubbleRow}>
          {chemicals.length === 0 && <span style={{ ...s.bubble(false, false), cursor: 'default' }}>Loading…</span>}
          {chemicals.map((c) => (
            <button key={c.id} style={s.bubble(activeId === c.id, loadingChem === c.id)} onClick={() => handleSelectChemical(c)} disabled={!!loadingChem || !!revealData} title={c.label}>
              {c.id}
            </button>
          ))}
        </div>
      </div>

      <div style={s.streamCard}>
        <div style={s.windowBar}>
          <span style={s.wDot('#f87171')} />
          <span style={s.wDot('#fbbf24')} />
          <span style={s.wDot('#4ade80')} />
          <span style={s.wTitle}>{activeId ? `// loaded: ${activeId}` : '// webcam feed — select a substance'}</span>
        </div>
        <canvas ref={canvasRef} width={FRAME_W} height={FRAME_H} style={s.canvas} />
      </div>

      {revealData && (
        <div style={s.revealBanner(revealData.color)}>
          <div style={s.revealHeader(revealData.color)}>
            <span style={s.revealHeadline(revealData.color)}>{revealData.headline}</span>
            <span style={s.revealVerdict(revealData.color)}>{revealData.verdict}</span>
          </div>
          <p style={s.revealBody}>{revealData.body}</p>
        </div>
      )}

      <p style={s.hint}>{revealData ? 'Click ← Back to run another experiment.' : 'Select a substance · tilt hand to pour · watch the litmus paper.'}</p>
    </div>
  );
}