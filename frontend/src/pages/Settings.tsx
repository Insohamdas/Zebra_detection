import { useState } from 'react';
import { Save, Bell, Globe, Database, Shield, Camera, Download, CheckCircle, Key, RefreshCw } from 'lucide-react';

type SettingsSection = 'detection' | 'camera' | 'notifications' | 'storage' | 'regional' | 'api';

export default function Settings() {
  const [activeSection, setActiveSection] = useState<SettingsSection>('detection');
  const [threshold, setThreshold] = useState(0.85);
  const [requireConfirmation, setRequireConfirmation] = useState(true);
  const [autoBackup, setAutoBackup] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(false);
  const [emailAddress, setEmailAddress] = useState('');
  const [cameraAutoStart, setCameraAutoStart] = useState(true);
  const [retentionDays, setRetentionDays] = useState(90);
  const [maxDetectionsPerImage, setMaxDetectionsPerImage] = useState(10);
  const [timezone, setTimezone] = useState('UTC');
  const [language, setLanguage] = useState('en');
  const [apiKey, setApiKey] = useState('sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz');
  const [showApiKey, setShowApiKey] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const handleSave = () => {
    // Simulate API call
    setSaveSuccess(true);
    console.log('Settings saved:', {
      threshold,
      requireConfirmation,
      autoBackup,
      emailNotifications,
      emailAddress,
      cameraAutoStart,
      retentionDays,
      maxDetectionsPerImage,
      timezone,
      language,
    });

    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const handleReset = () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      setThreshold(0.85);
      setRequireConfirmation(true);
      setAutoBackup(true);
      setEmailNotifications(false);
      setEmailAddress('');
      setCameraAutoStart(true);
      setRetentionDays(90);
      setMaxDetectionsPerImage(10);
      setTimezone('UTC');
      setLanguage('en');
    }
  };

  const handleExportData = async () => {
    setIsExporting(true);

    try {
      // Fetch CSV from backend
      const response = await fetch('http://localhost:5001/export/csv');

      if (!response.ok) {
        throw new Error('Failed to export data');
      }

      // Get the blob from response
      const blob = await response.blob();

      // Download file
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `zebra-database-${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export data. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleRegenerateApiKey = () => {
    if (confirm('Are you sure you want to regenerate your API key? The old key will stop working immediately.')) {
      const newKey = 'sk-proj-' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
      setApiKey(newKey);
      setShowApiKey(true);
      setTimeout(() => setShowApiKey(false), 5000);
    }
  };

  const sections = [
    { id: 'detection' as const, label: 'Detection Settings', icon: Shield },
    { id: 'camera' as const, label: 'Camera Settings', icon: Camera },
    { id: 'notifications' as const, label: 'Notifications', icon: Bell },
    { id: 'storage' as const, label: 'Data & Storage', icon: Database },
    { id: 'regional' as const, label: 'Regional Settings', icon: Globe },
    { id: 'api' as const, label: 'API Access', icon: Key },
  ];

  return (
    <div className="max-w-4xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-500">Configure system parameters and preferences</p>
      </div>

      {saveSuccess && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3 animate-fade-in">
          <CheckCircle size={20} className="text-green-600" />
          <p className="text-sm font-medium text-green-800">Settings saved successfully!</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <div className="card p-4 space-y-1">
            {sections.map(section => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                  activeSection === section.id
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-gray-700 hover:bg-gray-50'
                }`}
              >
                <section.icon size={16} />
                {section.label}
              </button>
            ))}
          </div>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Detection Parameters */}
          {activeSection === 'detection' && (
            <div className="card">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Shield size={20} className="text-primary-600" />
                  Detection Parameters
                </h2>
              </div>
              <div className="p-6 space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Match Confidence Threshold
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min="0.5"
                      max="0.99"
                      step="0.01"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                    />
                    <span className="text-sm font-semibold text-gray-900 min-w-[3rem] text-right">
                      {(threshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-gray-500">
                    Higher values reduce false positives but might miss some matches. Current: {threshold >= 0.9 ? 'Very High' : threshold >= 0.8 ? 'High' : threshold >= 0.7 ? 'Medium' : 'Low'}
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Maximum Detections Per Image
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={maxDetectionsPerImage}
                    onChange={(e) => setMaxDetectionsPerImage(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)))}
                    className="input w-32"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Limit the number of zebras detected in a single image (1-50)
                  </p>
                </div>

                <div className="flex items-center justify-between py-3 border-t border-gray-100">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Require Multiple Sightings</label>
                    <p className="text-xs text-gray-500">Need 2+ sightings to confirm new zebra ID</p>
                  </div>
                  <button
                    onClick={() => setRequireConfirmation(!requireConfirmation)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      requireConfirmation ? 'bg-primary-600' : 'bg-gray-300'
                    }`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      requireConfirmation ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Camera Settings */}
          {activeSection === 'camera' && (
            <div className="card">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Camera size={20} className="text-primary-600" />
                  Camera Settings
                </h2>
              </div>
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between py-3">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Auto-start Cameras</label>
                    <p className="text-xs text-gray-500">Automatically start all cameras on system boot</p>
                  </div>
                  <button
                    onClick={() => setCameraAutoStart(!cameraAutoStart)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      cameraAutoStart ? 'bg-primary-600' : 'bg-gray-300'
                    }`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      cameraAutoStart ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                <p className="text-sm text-gray-600 bg-blue-50 border border-blue-100 rounded-lg p-3">
                  {cameraAutoStart ? '✓ Cameras will automatically start when the system boots' : '✗ Cameras will need to be manually started after system boot'}
                </p>
              </div>
            </div>
          )}

          {/* Notifications */}
          {activeSection === 'notifications' && (
            <div className="card">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Bell size={20} className="text-primary-600" />
                  Notifications
                </h2>
              </div>
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between py-3">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Email Notifications</label>
                    <p className="text-xs text-gray-500">Receive alerts for new zebra detections</p>
                  </div>
                  <button
                    onClick={() => setEmailNotifications(!emailNotifications)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      emailNotifications ? 'bg-primary-600' : 'bg-gray-300'
                    }`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      emailNotifications ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>

                {emailNotifications && (
                  <div className="animate-fade-in">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address
                    </label>
                    <input
                      type="email"
                      placeholder="ranger@park.org"
                      value={emailAddress}
                      onChange={(e) => setEmailAddress(e.target.value)}
                      className="input w-full"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Notifications will be sent to this email address
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Data & Storage */}
          {activeSection === 'storage' && (
            <div className="card">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Database size={20} className="text-primary-600" />
                  Data & Storage
                </h2>
              </div>
              <div className="p-6 space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Data Retention Period
                  </label>
                  <select
                    value={retentionDays}
                    onChange={(e) => setRetentionDays(parseInt(e.target.value))}
                    className="input w-full"
                  >
                    <option value={30}>30 days</option>
                    <option value={90}>90 days (Recommended)</option>
                    <option value={180}>180 days</option>
                    <option value={365}>1 year</option>
                    <option value={-1}>Forever</option>
                  </select>
                  <p className="mt-1 text-xs text-gray-500">
                    Automatically delete sighting records older than this period
                  </p>
                </div>

                <div className="flex items-center justify-between py-3 border-t border-gray-100">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Automatic Backups</label>
                    <p className="text-xs text-gray-500">Daily backup of database and images</p>
                  </div>
                  <button
                    onClick={() => setAutoBackup(!autoBackup)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      autoBackup ? 'bg-primary-600' : 'bg-gray-300'
                    }`}
                  >
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      autoBackup ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>

                <div className="border-t border-gray-100 pt-4">
                  <button
                    onClick={handleExportData}
                    disabled={isExporting}
                    className="btn btn-secondary w-full"
                  >
                    {isExporting ? (
                      <>
                        <RefreshCw size={16} className="mr-2 animate-spin" />
                        Exporting...
                      </>
                    ) : (
                      <>
                        <Download size={16} className="mr-2" />
                        Export All Data (CSV)
                      </>
                    )}
                  </button>
                  <p className="mt-2 text-xs text-gray-500 text-center">
                    Export all zebra records and sightings
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Regional Settings */}
          {activeSection === 'regional' && (
            <div className="card">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Globe size={20} className="text-primary-600" />
                  Regional Settings
                </h2>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Timezone
                  </label>
                  <select
                    value={timezone}
                    onChange={(e) => setTimezone(e.target.value)}
                    className="input w-full"
                  >
                    <option value="UTC">UTC (Coordinated Universal Time)</option>
                    <option value="America/New_York">Eastern Time (US & Canada)</option>
                    <option value="America/Los_Angeles">Pacific Time (US & Canada)</option>
                    <option value="Europe/London">London (GMT/BST)</option>
                    <option value="Africa/Nairobi">Nairobi (EAT)</option>
                    <option value="Africa/Johannesburg">Johannesburg (SAST)</option>
                  </select>
                  <p className="mt-1 text-xs text-gray-500">
                    All timestamps will be displayed in this timezone
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Language
                  </label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="input w-full"
                  >
                    <option value="en">English</option>
                    <option value="sw">Kiswahili (Swahili)</option>
                    <option value="fr">Français (French)</option>
                    <option value="es">Español (Spanish)</option>
                  </select>
                  <p className="mt-1 text-xs text-gray-500">
                    Interface language (requires page reload)
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* API Access */}
          {activeSection === 'api' && (
            <div className="card">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Key size={20} className="text-primary-600" />
                  API Access
                </h2>
              </div>
              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    API Endpoint
                  </label>
                  <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                    <code className="text-sm text-green-400 font-mono whitespace-pre">
                      {`curl -X POST https://api.zebraid.system/v1/identify \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "image=@zebra.jpg"`}
                    </code>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    API Key
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={showApiKey ? apiKey : apiKey.substring(0, 10) + '***********************************'}
                      readOnly
                      className="input flex-1 font-mono text-sm"
                    />
                    <button
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="btn btn-secondary px-4"
                    >
                      {showApiKey ? 'Hide' : 'Show'}
                    </button>
                    <button
                      onClick={handleRegenerateApiKey}
                      className="btn btn-secondary"
                    >
                      <RefreshCw size={16} />
                    </button>
                  </div>
                  <p className="mt-2 text-xs text-gray-500">
                    Keep your API key secure. Never share it publicly.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Save Button */}
          <div className="flex justify-end gap-3 pt-4 sticky bottom-0 bg-gray-50 p-4 rounded-lg border border-gray-200">
            <button onClick={handleReset} className="btn btn-secondary">
              Reset to Defaults
            </button>
            <button onClick={handleSave} className="btn btn-primary">
              <Save size={16} className="mr-2" />
              Save All Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
