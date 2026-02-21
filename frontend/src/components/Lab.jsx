// src/components/Lab.jsx
import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api';

const styles = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem',
    gap: '1.5rem',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '100%',
    maxWidth: '820px',
  },
  backBtn: {
    background: 'none',
    border: '1px solid var(--border)',
    color: 'var(--text-muted)',
    padding: '0.5rem 1.2rem',
    borderRadius: '3px',
    fontSize: '0.75rem',
    fontFamily: 'var(--mono)',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    transition: 'color 0.2s, border-color 0.2s',
    cursor: 'pointer',
  },
  statusDot: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.72rem',
    fontFamily: 'var(--mono)',
    color: 'var(--accent-green)',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
  },
  dot: {
    width: '7px',
    height: '7px',
    borderRadius: '50%',
    background: 'var(--accent-green)',
    animation: 'pulse 1.5s ease-in-out infinite',
  },
  streamWrapper: {
    width: '100%',
    maxWidth: '820px',
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    overflow: 'hidden',
    boxShadow: 'var(--glow-blue)',
    position: 'relative',
  },
  streamBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    padding: '0.6rem 1rem',
    borderBottom: '1px solid var(--border)',
  },
  barDot: (color) => ({
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    background: color,
    opacity: 0.7,
  }),
  barTitle: {
    marginLeft: 'auto',
    fontSize: '0.65rem',
    fontFamily: 'var(--mono)',
    color: 'var(--text-muted)',
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
  },
  stream: {
    display: 'block',
    width: '100%',
    maxWidth: '100%',
    minHeight: '300px',
    background: '#000',
  },
  hint: {
    fontSize: '0.72rem',
    fontFamily: 'var(--mono)',
    color: 'var(--text-muted)',
    letterSpacing: '0.08em',
  },
};

// Inject keyframe animation once
const styleTag = document.createElement('style');
styleTag.textContent = `@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.3 } }`;
document.head.appendChild(styleTag);

export default function Lab() {
  const navigate = useNavigate();
  const stopCalled = useRef(false); // prevent double-call on strict mode remount
  const imgRef = useRef(null); // Reference for the image element

  const handleBack = () => {
    // Force the browser to instantly abort the stream connection
    if (imgRef.current) {
      imgRef.current.src = ''; 
    }
    // Navigate immediately. The unmount will trigger the useEffect cleanup.
    navigate('/dashboard');
  };

  // Stop the reaction if the user closes/refreshes the tab
  useEffect(() => {
    const handleUnload = () => {
      // navigator.sendBeacon does not easily send cross-origin credentials.
      // fetch with keepalive: true ensures the Django session cookie is sent on tab close.
      fetch('http://localhost:8000/api/reactions/stop/', {
        method: 'POST',
        keepalive: true, 
        credentials: 'include', 
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      }).catch(err => console.error("Unload error:", err));
    };
    
    window.addEventListener('beforeunload', handleUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleUnload);
      
      // Force stream abort if unmounted via the browser's native back button
      if (imgRef.current) {
        imgRef.current.src = '';
      }
      
      // Also stop on React unmount (e.g. browser back button or handleBack)
      if (!stopCalled.current) {
        stopCalled.current = true;
        api.post('/reactions/stop/').catch(() => {});
      }
    };
  }, []);

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <button style={styles.backBtn} onClick={handleBack}>
          ← Back
        </button>
        <div style={styles.statusDot}>
          <span style={styles.dot} />
          Live Stream
        </div>
      </div>

      <div style={styles.streamWrapper}>
        {/* Fake window chrome bar */}
        <div style={styles.streamBar}>
          <span style={styles.barDot('#f87171')} />
          <span style={styles.barDot('#fbbf24')} />
          <span style={styles.barDot('#4ade80')} />
          <span style={styles.barTitle}>// webcam feed</span>
        </div>

        {/*
          crossOrigin="use-credentials" is REQUIRED.
          Without it, the browser strips the session cookie on cross-origin
          image requests (port 5173 → 8000), and Django returns 401.
        */}
        <img
          ref={imgRef} // Attach the ref to the image tag
          src="/api/reactions/video-feed/"
          alt="Virtual Lab Stream"
          style={styles.stream}
        />
      </div>

      <p style={styles.hint}>
        Stream stops automatically when you leave this page.
      </p>
    </div>
  );
}