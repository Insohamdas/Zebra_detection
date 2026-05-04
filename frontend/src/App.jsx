import { useEffect, useState, useRef } from 'react';
import { 
  UploadCloud, 
  AlertCircle, 
  Fingerprint, 
  RefreshCw, 
  Video, 
  Copy
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [videoJob, setVideoJob] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!videoJob?.job_id || videoJob.status === 'completed' || videoJob.status === 'failed') {
      return undefined;
    }

    let cancelled = false;
    const pollVideoStatus = async () => {
      try {
        const response = await fetch(`${API_URL}/video-status/${videoJob.job_id}`);
        if (!response.ok) {
          const errData = await response.json();
          throw new Error(errData.detail || 'Video status unavailable');
        }

        const data = await response.json();
        if (cancelled) return;

        setVideoJob(data);
        if (data.status === 'completed') {
          setResult(data);
          setLoading(false);
        } else if (data.status === 'failed') {
          setError(data.error || 'Video processing failed');
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      }
    };

    pollVideoStatus();
    const intervalId = window.setInterval(pollVideoStatus, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [API_URL, videoJob?.job_id, videoJob?.status]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && (droppedFile.type.startsWith('image/') || droppedFile.type.startsWith('video/'))) {
      handleFileSelection(droppedFile);
    }
  };

  const handleFileInput = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      handleFileSelection(selectedFile);
    }
  };

  const handleFileSelection = (selectedFile) => {
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setError(null);
    setResult(null);
    setVideoJob(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setVideoJob(null);

    const isVideo = file.type.startsWith('video/');
    const endpoint = isVideo ? '/process-video' : '/identify';
    const fieldName = isVideo ? 'video' : 'image';
    let shouldContinueVideoPolling = false;

    const formData = new FormData();
    formData.append(fieldName, file);

    try {
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Processing failed');
      }

      const data = await response.json();
      if (isVideo) {
        shouldContinueVideoPolling = true;
        setVideoJob({
          ...data,
          sampled_frames: 0,
          estimated_total_samples: 0,
          progress: 0,
        });
        return;
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      if (!shouldContinueVideoPolling) {
        setLoading(false);
      }
    }
  };

  const resetState = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setVideoJob(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isVideoFile = file?.type.startsWith('video/');
  const videoProgressPercent = Math.round((videoJob?.progress || 0) * 100);

  return (
    <div className="app-wrapper">
      <div className="bg-gradient"></div>
      
      <header className="header">
        <div className="container nav-container">
          <a href="/" className="logo-area">
            <div className="logo-icon">
              <Fingerprint size={24} />
            </div>
            <span>ZEBRAID</span>
          </a>
          <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.5, fontFamily: 'monospace' }}>V1.3.5.B</div>
        </div>
      </header>

      <main className="main-content container">
        {!result && !error && (
          <section className="hero" style={{ position: 'relative' }}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <span className="project-tagline" style={{ color: 'var(--text-muted)', fontSize: '0.75rem', opacity: 0.8 }}>Software version V1.3.5.B</span>
              <h1 className="heading-xl text-gradient">
                Individual Recognition. <br />
                Continental Scale.
              </h1>
              <p className="hero-subtitle">
                A Stripe-Based Biometric Identification System for Individual Zebra Recognition at Continental Scale.
              </p>
            </motion.div>
          </section>
        )}

        <AnimatePresence mode="wait">
          {!result && !error ? (
            <motion.div 
              key="uploader"
              className="upload-section"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95, filter: "blur(10px)" }}
              transition={{ duration: 0.4 }}
            >
              <div 
                className={`dropzone ${isDragging ? 'active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !file && fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  accept="image/jpeg, image/png, video/mp4, video/x-m4v, video/*" 
                  style={{ display: 'none' }} 
                  ref={fileInputRef}
                  onChange={handleFileInput}
                />
                
                {loading && (
                  <div className="loading-overlay">
                    <span className="loader"></span>
                    <p className="dropzone-title">{isVideoFile ? 'Analyzing Video Streams...' : 'Extracting Stripe Embeddings...'}</p>
                    {isVideoFile && videoJob && (
                      <div className="video-progress">
                        <div className="video-progress-track" aria-hidden="true">
                          <div
                            className="video-progress-bar"
                            style={{ width: `${videoProgressPercent}%` }}
                          ></div>
                        </div>
                        <div className="video-progress-meta">
                          <span>{videoProgressPercent}%</span>
                          <span>
                            {videoJob.sampled_frames || 0}
                            {videoJob.estimated_total_samples ? ` / ${videoJob.estimated_total_samples}` : ''}
                            {' '}samples
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {!loading && !file && (
                  <div className="dropzone-content">
                    <div className="upload-icon-wrapper">
                      <UploadCloud size={32} />
                    </div>
                    <div>
                      <h3 className="dropzone-title">Identification Portal</h3>
                      <p className="dropzone-subtitle">Drop your zebra images or video files here</p>
                    </div>
                    <button className="btn-primary" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                      Choose Source File
                    </button>
                  </div>
                )}

                {!loading && file && (
                  <div className="dropzone-content">
                    <div className={isVideoFile ? 'selected-video-preview' : 'selected-image-preview'}>
                      {isVideoFile ? (
                        <div className="selected-video-icon">
                          <Video size={70} strokeWidth={1.7} />
                        </div>
                      ) : (
                        <img src={preview} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                      <button className="btn-reset" style={{ marginTop: 0 }} onClick={(e) => { e.stopPropagation(); resetState(); }}>
                        Reset
                      </button>
                      <button className="btn-primary" onClick={(e) => { e.stopPropagation(); handleUpload(); }}>
                        {isVideoFile ? 'Run Video Analysis' : 'Verify Identity'}
                      </button>
                    </div>
                  </div>
                )}
              </div>

            </motion.div>
          ) : result ? (
            <motion.div 
              key="result"
              className={result.unique_zebras ? 'sequence-results-shell' : 'single-result-shell'}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              {result.unique_zebras ? (
                <div className="sequence-results-panel">
                  <div className="sequence-results-header">
                    <div className="sequence-results-title-group">
                      <span className="sequence-results-kicker">VIDEO ANALYSIS COMPLETE</span>
                      <h2>Sequence Results</h2>
                      <p>Verified identities from processed video samples.</p>
                    </div>
                    <button className="btn-primary sequence-results-action" onClick={resetState}>
                      <RefreshCw size={18} /> New Analysis
                    </button>
                  </div>

                  <div className="sequence-summary-grid">
                    <div className="sequence-summary-item">
                      <span>INDIVIDUALS</span>
                      <strong>{result.unique_zebras.length}</strong>
                    </div>
                    <div className="sequence-summary-item">
                      <span>SAMPLES</span>
                      <strong>{result.total_frames_processed}</strong>
                    </div>
                    <div className="sequence-summary-item">
                      <span>AVG CONFIDENCE</span>
                      <strong>
                        {(result.unique_zebras.reduce((sum, zebra) => sum + zebra.confidence, 0) / Math.max(result.unique_zebras.length, 1) * 100).toFixed(1)}%
                      </strong>
                    </div>
                  </div>

                  <div className="sequence-card-grid">
                    {result.unique_zebras.map((zebra, idx) => (
                      <motion.div 
                        key={idx} 
                        className="sequence-zebra-card"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.05 }}
                      >
                        <div className="sequence-card-topline">
                          <div className={`sequence-status-pill ${zebra.is_new ? 'is-new' : 'is-match'}`}>
                            {zebra.is_new ? 'New Profile' : 'Registry Match'}
                          </div>
                          <button 
                            className="copy-id-button sequence-copy-button"
                            onClick={() => navigator.clipboard.writeText(zebra.zebra_id)}
                            title="Copy ID"
                            aria-label="Copy biometric ID"
                          >
                            <Copy size={16} strokeWidth={2.2} />
                          </button>
                        </div>
                        <div className="sequence-zebra-id">{zebra.zebra_id}</div>
                        <div className="sequence-card-metrics">
                          <div className="sequence-card-metric">
                            <span>CONFIDENCE</span>
                            <strong>{(zebra.confidence * 100).toFixed(1)}%</strong>
                          </div>
                          <div className="sequence-card-metric">
                            <span>STATUS</span>
                            <strong>{zebra.is_new ? 'ENROLLED' : 'SYNCED'}</strong>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="single-result-card">
                  <div className="single-result-preview">
                    <img src={preview} alt="Input" />
                  </div>
                  
                  <div className="single-result-details">
                    <div className="single-result-status">
                      <span>
                        {result.is_new ? 'New Registration' : 'Registry Match Confirmed'}
                      </span>
                    </div>

                    <h2 className="single-result-label">BIOMETRIC ID</h2>
                    <div className="single-result-id-row">
                      <div className="single-result-id">{result.zebra_id}</div>
                      <button 
                        className="copy-id-button"
                        onClick={() => navigator.clipboard.writeText(result.zebra_id)}
                        title="Copy ID"
                        aria-label="Copy biometric ID"
                      >
                        <Copy size={17} strokeWidth={2.2} />
                      </button>
                    </div>
                    
                    <div className="single-result-metrics">
                      <div className="single-result-metric is-primary">
                        <div className="single-result-metric-label">MATCH SCORE</div>
                        <div className="single-result-metric-value">{(result.confidence * 100).toFixed(1)}%</div>
                      </div>
                      <div className="single-result-metric">
                        <div className="single-result-metric-label">DATABASE</div>
                        <div className="single-result-metric-value">{result.is_new ? 'ENROLLED' : 'VERIFIED'}</div>
                      </div>
                    </div>

                    <button className="btn-primary single-result-action" onClick={resetState}>
                      <RefreshCw size={20} /> Perform New Search
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div 
              key="error"
              className="glass-card result-card"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ textAlign: 'center', maxWidth: '600px', margin: '0 auto' }}
            >
              <AlertCircle size={64} color="var(--danger)" style={{ margin: '0 auto 1.5rem', opacity: 0.5 }} />
              <h2 className="heading-xl" style={{ fontSize: '2.5rem' }}>Processing Aborted</h2>
              <p style={{ color: 'var(--text-muted)', marginBottom: '2.5rem', fontSize: '1.1rem' }}>
                {error === 'low_quality' 
                  ? 'Input quality insufficient for biometric extraction. Please ensure the target is well-lit and in focus.' 
                  : error === 'no_zebra'
                  ? 'No target species detected in this capture. The system only processes individual zebra flanks.'
                  : error}
              </p>
              <button className="btn-primary" onClick={resetState}>
                Re-submit Data
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="new-age-footer">
        <div className="container">
          <div className="footer-inner">
            <div className="footer-left">
              <div className="footer-logo-minimal">
                <Fingerprint size={20} />
                <span>ZEBRAID</span>
              </div>
              <span className="footer-divider"></span>
              <span className="footer-version-tag">SYSTEM V1.3.5.B</span>
            </div>
            
            <div className="footer-right">
              <p style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>© 2026 ZEBRAID. All Rights Reserved.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
