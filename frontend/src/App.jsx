import { useState, useRef } from 'react';
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
  const fileInputRef = useRef(null);

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
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const isVideo = file.type.startsWith('video/');
    const endpoint = isVideo ? '/process-video' : '/identify';
    const fieldName = isVideo ? 'video' : 'image';

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
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isVideoFile = file?.type.startsWith('video/');

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
                    <div style={{ width: '240px', height: '240px', borderRadius: 'var(--radius-xl)', overflow: 'hidden', boxShadow: 'var(--shadow-xl)', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {isVideoFile ? (
                        <div style={{ color: '#fff', textAlign: 'center' }}>
                          <Video size={64} strokeWidth={1.5} />
                          <p style={{ fontSize: '0.9rem', marginTop: '1rem', opacity: 0.7 }}>{file.name}</p>
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
              className={result.unique_zebras ? 'results-container sequence-results-shell' : 'single-result-shell'}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              {result.unique_zebras ? (
                <div style={{ width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '2.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '1.5rem' }}>
                    <div>
                      <h2 style={{ fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.02em' }}>Sequence Results</h2>
                      <p style={{ color: 'var(--text-muted)' }}>Verified {result.unique_zebras.length} individuals from {result.total_frames_processed} samples.</p>
                    </div>
                    <button className="btn-primary" onClick={resetState}>
                      <RefreshCw size={18} /> New Analysis
                    </button>
                  </div>

                  <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
                    {result.unique_zebras.map((zebra, idx) => (
                      <motion.div 
                        key={idx} 
                        style={{ background: 'white', borderRadius: '1.25rem', padding: '1.5rem', border: '1px solid var(--border)' }}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.05 }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
                          <div style={{ background: zebra.is_new ? 'rgba(16, 185, 129, 0.1)' : 'rgba(37, 99, 235, 0.1)', color: zebra.is_new ? 'var(--success)' : 'var(--primary)', padding: '0.4rem 0.8rem', borderRadius: '8px', fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase' }}>
                            {zebra.is_new ? 'New Profile' : 'Registry Match'}
                          </div>
                          <button 
                            onClick={() => navigator.clipboard.writeText(zebra.zebra_id)}
                            style={{ color: 'var(--primary)', background: 'rgba(37, 99, 235, 0.1)', padding: '0.5rem', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s ease', border: '1px solid transparent' }}
                            onMouseOver={(e) => e.currentTarget.style.borderColor = 'var(--primary)'}
                            onMouseOut={(e) => e.currentTarget.style.borderColor = 'transparent'}
                            title="Copy ID"
                          >
                            <Copy size={16} />
                          </button>
                        </div>
                        <div style={{ fontSize: '1.1rem', fontWeight: 800, fontFamily: 'monospace', marginBottom: '1.25rem' }}>{zebra.zebra_id}</div>
                        <div style={{ display: 'flex', gap: '1.5rem' }}>
                          <div>
                            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 600 }}>CONFIDENCE</div>
                            <div style={{ fontSize: '1rem', fontWeight: 700 }}>{(zebra.confidence * 100).toFixed(1)}%</div>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 600 }}>STATUS</div>
                            <div style={{ fontSize: '1rem', fontWeight: 700 }}>{zebra.is_new ? 'ENROLLED' : 'SYNCED'}</div>
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
