import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, Fingerprint, RefreshCw, Video } from 'lucide-react';
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
        </div>
      </header>

      <main className="main-content container">
        <motion.div 
          className="hero"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="heading-xl text-gradient">Identify individuals with precision.</h1>
          <p className="hero-subtitle">
            Upload an image or video of a zebra to instantly cross-reference our global FAISS registry and determine unique identities.
          </p>
        </motion.div>

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
                    <p className="dropzone-title">{isVideoFile ? 'Scanning video frames...' : 'Analyzing biometric markers...'}</p>
                  </div>
                )}

                {!loading && !file && (
                  <div className="dropzone-content">
                    <div className="upload-icon-wrapper">
                      <UploadCloud size={32} />
                    </div>
                    <div>
                      <h3 className="dropzone-title">Drag & drop media here</h3>
                      <p className="dropzone-subtitle">supports images and videos</p>
                    </div>
                    <button className="btn-primary" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                      Select File
                    </button>
                  </div>
                )}

                {!loading && file && (
                  <div className="dropzone-content">
                    <div style={{ width: '200px', height: '200px', borderRadius: '1rem', overflow: 'hidden', boxShadow: 'var(--shadow-md)', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      {isVideoFile ? (
                        <div style={{ color: '#fff', textAlign: 'center' }}>
                          <Video size={48} />
                          <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>Video Clip</p>
                        </div>
                      ) : (
                        <img src={preview} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      )}
                    </div>
                    <h3 className="dropzone-title">{file.name}</h3>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                      <button className="btn-reset" onClick={(e) => { e.stopPropagation(); resetState(); }}>
                        Cancel
                      </button>
                      <button className="btn-primary" onClick={(e) => { e.stopPropagation(); handleUpload(); }}>
                        {isVideoFile ? 'Process Video' : 'Identify Zebra'}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          ) : result ? (
            <motion.div 
              key="result"
              className="results-container"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              {result.unique_zebras ? (
                // Video Result View
                <div style={{ width: '100%' }}>
                  <div className="glass-card" style={{ padding: '2rem', marginBottom: '2rem' }}>
                    <div className="status-badge status-match" style={{ width: 'fit-content', marginBottom: '1rem' }}>
                      <CheckCircle2 size={18} />
                      Video Scan Complete
                    </div>
                    <h2 className="heading-xl" style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>
                      {result.unique_zebras.length} Unique Zebras Found
                    </h2>
                    <p style={{ color: 'var(--text-muted)' }}>
                      Processed {result.total_frames_processed} frames from the video.
                    </p>
                  </div>

                  <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))' }}>
                    {result.unique_zebras.map((zebra, idx) => (
                      <div key={idx} className="glass-card result-card" style={{ marginTop: 0 }}>
                        <div className={`status-badge ${zebra.is_new ? 'status-new' : 'status-match'}`}>
                          {zebra.is_new ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}
                          {zebra.is_new ? 'New Individual' : 'Existing Match'}
                        </div>
                        <h2 className="metric-label" style={{ marginTop: '1rem' }}>Identity</h2>
                        <div className="id-display" style={{ fontSize: '1.2rem' }}>{zebra.zebra_id}</div>
                        <div className="metric-item" style={{ marginTop: '1rem' }}>
                          <span className="metric-label">Match Confidence</span>
                          <span className="metric-value">{(zebra.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div style={{ textAlign: 'center', marginTop: '3rem' }}>
                    <button className="btn-primary" onClick={resetState}>
                      <RefreshCw size={18} /> Process Another Video
                    </button>
                  </div>
                </div>
              ) : (
                // Single Image Result View
                <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', justifyContent: 'center' }}>
                  <div className="image-preview">
                    <img src={preview} alt="Analyzed Zebra" />
                  </div>
                  <div className="glass-card result-card">
                    <div className={`status-badge ${result.is_new ? 'status-new' : 'status-match'}`}>
                      {result.is_new ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}
                      {result.is_new ? 'New Individual Discovered' : 'Match Found'}
                    </div>
                    
                    <h2 className="metric-label">Assigned Identity</h2>
                    <div className="id-display">{result.zebra_id}</div>
                    
                    <div className="metrics-grid">
                      <div className="metric-item">
                        <span className="metric-label">Confidence Score</span>
                        <span className="metric-value">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">Status</span>
                        <span className="metric-value">{result.is_new ? 'Enrolled' : 'Verified'}</span>
                      </div>
                    </div>

                    <button className="btn-reset" onClick={resetState}>
                      <RefreshCw size={18} /> Process Another
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
              <AlertCircle size={48} color="var(--danger)" style={{ margin: '0 auto 1.5rem' }} />
              <h2 className="heading-xl" style={{ fontSize: '2rem' }}>Analysis Failed</h2>
              <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
                {error === 'low_quality' 
                  ? 'The image quality is too low for accurate identification. Please ensure the image is sharp and the subject is clearly visible.' 
                  : error === 'no_zebra'
                  ? 'No zebra was detected in this image. Please upload a clear picture of a zebra.'
                  : error}
              </p>
              <button className="btn-primary" onClick={resetState}>
                Try Again
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default App;
