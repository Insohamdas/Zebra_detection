import { useState, useRef, useEffect } from 'react';
import { Upload, X, CheckCircle, AlertCircle, ArrowRight, Camera, Video, VideoOff } from 'lucide-react';
import { Link } from 'react-router-dom';
import { api } from '../api/mock';
import type { Zebra, Sighting } from '../types';

export default function Identify() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ zebra: Zebra; sighting: Sighting; isNew: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      stopCamera();
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      stopCamera();
    }
  };

  const startCamera = async () => {
    try {
      setCameraError(null);
      setCameraLoading(true);
      setResult(null);

      console.log('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      console.log('Camera access granted, stream:', stream);
      console.log('Stream active:', stream.active);
      console.log('Video tracks:', stream.getVideoTracks());

      if (videoRef.current && stream) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setFile(null);
        setPreview(null);

        // Set a timeout to show camera active after 2 seconds if video doesn't load
        const timeoutId = setTimeout(() => {
          console.log('Timeout: Forcing camera active state');
          setCameraActive(true);
          setCameraLoading(false);
        }, 2000);

        // Try to play immediately
        try {
          console.log('Attempting to play video...');
          await videoRef.current.play();
          clearTimeout(timeoutId);
          console.log('Video playing successfully');
          setCameraActive(true);
          setCameraLoading(false);
        } catch (playError) {
          console.error('Play error:', playError);
          // Wait for metadata
          videoRef.current.onloadedmetadata = async () => {
            clearTimeout(timeoutId);
            console.log('Video metadata loaded');
            try {
              if (videoRef.current) {
                await videoRef.current.play();
                console.log('Video playing after metadata');
                setCameraActive(true);
                setCameraLoading(false);
              }
            } catch (err) {
              console.error('Error playing after metadata:', err);
              // Force show anyway
              setCameraActive(true);
              setCameraLoading(false);
            }
          };
        }
      } else {
        throw new Error('Video element not available');
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraError(`Unable to access camera: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setCameraActive(false);
      setCameraLoading(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
          if (blob) {
            const capturedFile = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            setFile(capturedFile);
            setPreview(URL.createObjectURL(capturedFile));
            stopCamera();
          }
        }, 'image/jpeg', 0.95);
      }
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const handleIdentify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.identifyZebra(file);
      setResult(data);
    } catch (error) {
      console.error(error);
      if (error instanceof Error && error.message === 'NO_ZEBRA_DETECTED') {
        setError('No zebra detected in this image. Please upload an image containing a zebra.');
      } else {
        setError('Failed to analyze image. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Identify Zebra</h1>
        <p className="text-gray-500">Upload an image or use your camera to detect and identify individual zebras.</p>
      </div>

      {/* Camera/Upload Toggle */}
      <div className="flex gap-3">
        <button
          onClick={() => !cameraActive && fileInputRef.current?.click()}
          disabled={cameraActive}
          className={`btn ${!cameraActive && !preview ? 'btn-primary' : 'btn-secondary'} flex-1`}
        >
          <Upload size={18} className="mr-2" />
          Upload Image
        </button>
        <button
          onClick={cameraActive ? stopCamera : startCamera}
          className={`btn ${cameraActive ? 'btn-danger' : 'btn-secondary'} flex-1`}
        >
          {cameraActive ? (
            <>
              <VideoOff size={18} className="mr-2" />
              Stop Camera
            </>
          ) : (
            <>
              <Video size={18} className="mr-2" />
              Use Camera
            </>
          )}
        </button>
      </div>

      {cameraError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle size={20} className="text-red-600" />
          <p className="text-sm text-red-800">{cameraError}</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Upload/Camera Section */}
        <div className="space-y-4">
          <canvas ref={canvasRef} className="hidden" />

          {cameraActive ? (
            <div className="relative border-2 border-primary-500 rounded-xl overflow-hidden bg-black shadow-xl">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full rounded-lg"
                style={{
                  minHeight: '400px',
                  maxHeight: '600px',
                  width: '100%',
                  objectFit: 'cover',
                  display: 'block',
                  backgroundColor: '#000'
                }}
              />
              <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-3 px-4">
                <button
                  onClick={capturePhoto}
                  className="btn btn-primary shadow-2xl hover:scale-105 transition-transform"
                >
                  <Camera size={20} className="mr-2" />
                  Capture Photo
                </button>
              </div>
              <div className="absolute top-3 left-3 bg-red-600 text-white px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-2 shadow-lg">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
                LIVE
              </div>
              <div className="absolute top-3 right-3 bg-blue-600 text-white px-2 py-1 rounded text-xs">
                {videoRef.current?.videoWidth}x{videoRef.current?.videoHeight}
              </div>
            </div>
          ) : cameraLoading ? (
            <div className="border-2 border-dashed border-primary-500 rounded-xl p-8 bg-primary-50">
              <div className="flex flex-col items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mb-4"></div>
                <p className="text-primary-900 font-medium">Starting camera...</p>
                <p className="text-sm text-primary-700 mt-2">Please allow camera access if prompted</p>
              </div>
            </div>
          ) : (
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                preview ? 'border-gray-300 bg-gray-50' : 'border-gray-300 hover:border-primary-500 hover:bg-primary-50 cursor-pointer'
              }`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => !preview && fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleFileSelect}
              />

              {preview ? (
                <div className="relative">
                  <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-sm" />
                  <button
                    onClick={(e) => { e.stopPropagation(); reset(); }}
                    className="absolute -top-2 -right-2 p-1 bg-white rounded-full shadow-md border border-gray-200 hover:bg-gray-100"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <div className="py-8">
                  <div className="w-16 h-16 bg-primary-50 text-primary-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Upload size={32} />
                  </div>
                  <p className="text-lg font-medium text-gray-900">Click to upload or drag and drop</p>
                  <p className="text-sm text-gray-500 mt-1">SVG, PNG, JPG or GIF (max. 800x400px)</p>
                </div>
              )}
            </div>
          )}

          {file && !result && (
            <button
              onClick={handleIdentify}
              disabled={loading}
              className="w-full btn btn-primary py-3 text-base"
            >
              {loading ? 'Processing...' : 'Identify Zebra'}
            </button>
          )}
        </div>

        {/* Results Section */}
        <div>
          {loading && (
            <div className="h-full flex flex-col items-center justify-center p-8 border border-gray-200 rounded-xl bg-white">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mb-4"></div>
              <p className="text-gray-500 font-medium">Analyzing patterns...</p>
              <p className="text-xs text-gray-400 mt-2">Comparing with 1,245 known individuals</p>
            </div>
          )}

          {!loading && error && (
            <div className="h-full flex flex-col items-center justify-center p-8 border-2 border-red-200 rounded-xl bg-red-50 animate-fade-in">
              <div className="w-16 h-16 bg-red-100 text-red-600 rounded-full flex items-center justify-center mb-4">
                <AlertCircle size={32} />
              </div>
              <p className="text-red-900 font-bold text-lg mb-2">No Zebra Detected</p>
              <p className="text-red-700 text-center text-sm max-w-sm">{error}</p>
              <button
                onClick={reset}
                className="btn btn-secondary mt-4"
              >
                Try Another Image
              </button>
            </div>
          )}

          {!loading && !result && !error && !file && (
            <div className="h-full flex flex-col items-center justify-center p-8 border border-dashed border-gray-200 rounded-xl bg-gray-50 text-gray-400">
              <p>Results will appear here</p>
            </div>
          )}

          {!loading && result && (
            <div className="card overflow-hidden animate-fade-in">
              <div className={`p-4 ${result.isNew ? 'bg-blue-50 border-b border-blue-100' : 'bg-green-50 border-b border-green-100'}`}>
                <div className="flex items-center gap-3">
                  {result.isNew ? (
                    <AlertCircle className="text-blue-600" />
                  ) : (
                    <CheckCircle className="text-green-600" />
                  )}
                  <div>
                    <h3 className={`font-bold ${result.isNew ? 'text-blue-900' : 'text-green-900'}`}>
                      {result.isNew ? 'New Zebra Detected' : 'Match Found'}
                    </h3>
                    <p className={`text-sm ${result.isNew ? 'text-blue-700' : 'text-green-700'}`}>
                      Confidence: {(result.sighting.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-6 space-y-6">
                {/* Show the uploaded/captured image */}
                <div>
                  <p className="text-sm text-gray-500 mb-2">Your Uploaded Image:</p>
                  <img
                    src={preview || ''}
                    alt="Uploaded zebra"
                    className="w-full rounded-lg object-cover border-2 border-gray-300"
                    style={{ maxHeight: '300px' }}
                  />
                </div>

                <div className="flex items-center gap-4 pt-4 border-t border-gray-100">
                  <div className="w-20 h-20 bg-primary-100 rounded-lg flex items-center justify-center">
                    <CheckCircle size={40} className="text-primary-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 uppercase tracking-wider font-medium">Zebra ID</p>
                    <p className="text-3xl font-bold text-gray-900">{result.zebra.id}</p>
                    <p className="text-sm text-gray-500 mt-1">
                      {result.zebra.sightingsCount} previous sightings
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
                  <div>
                    <p className="text-xs text-gray-500">Last Seen</p>
                    <p className="font-medium">{new Date(result.zebra.lastSeen).toLocaleDateString()}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500">Location</p>
                    <p className="font-medium">{result.sighting.location}</p>
                  </div>
                </div>

                <div className="pt-4">
                  <Link
                    to={`/zebras/${result.zebra.id}`}
                    className="btn btn-secondary w-full justify-between group"
                  >
                    View Full Profile
                    <ArrowRight size={16} className="text-gray-400 group-hover:text-gray-900 transition-colors" />
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
