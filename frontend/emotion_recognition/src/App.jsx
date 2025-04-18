import React, { useState, useRef } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null); // clear previous result
  };

  const handlePredict = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('audio', file);

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setResult(null);
    setLoading(false);
    fileInputRef.current.value = null; // Reset file input manually
  };

  return (
    <div style={{       
      backgroundColor: '#f9f9fb',
      minHeight: '100vh',
      width: '100vw',
      boxSizing: 'border-box'
      }}>
      {/* Header Bar */}
      <div style={{
        backgroundColor: '#de0000',
        color: '#fff',
        padding: '1rem 2rem',
        fontSize: '1.25rem',
        fontWeight: 'bold'
      }}>
        🎤 Bimodal Emotion Recognition Demo
      </div>

      {/* Main Content */}
      <div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif', color: "#000"}}>
        <h2>Upload Audio to Predict Emotion</h2>

        {/* <input type="file" accept="audio/*" onChange={handleFileChange} />
        <br /><br /> */}
        {/* Hidden File Input */}
        <input
          type="file"
          accept="audio/*"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        {/* Custom Button */}
        <button
          onClick={() => fileInputRef.current.click()}
          style={{
            padding: '0.5rem 1rem',
            fontSize: '1rem',
            marginBottom: '1rem',
            backgroundColor: '#333',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          {file ? `Selected: ${file.name}` : 'Choose Audio File'}
        </button>

        <br />

        <button
          onClick={handlePredict}
          disabled={!file || loading}
          style={{
            padding: '0.5rem 1rem',
            fontSize: '1rem',
            backgroundColor: '#333',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>

        <button
          onClick={handleClear}
          style={{
            padding: '0.5rem 1rem',
            fontSize: '1rem',
            backgroundColor: '#333',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            marginLeft: '0.5rem'
          }}
        >
          Clear
        </button>

        {result && (
          <div style={{ marginTop: '2rem' }}>
            <h3>Prediction Result</h3>
            <p><strong>Speech Model Prediction:</strong> {result.emotion_label}</p>
            <p><strong>Text Model Prediction:</strong> {result.emotion_label}</p>
            <p><strong>Emotion:</strong> {result.emotion_label}</p>

            <h4>Confidence Scores</h4>
            <ul>
              {result.confidence_scores.map((score, index) => (
                <li key={index}>{['Negative', 'Neutral', 'Positive'][index]}: {score.toFixed(4)}</li>
              ))}
            </ul>

            <h4>Time (s)</h4>
            <p>Speech Model: {result.cnn_time.toFixed(4)}</p>
            <p>Text Model: {result.nb_time.toFixed(4)}</p>

            <h4>Memory Usage (bytes)</h4>
            <p>Speech Model: {result.cnn_mem} | Peak: {result.cnn_peak}</p>
            <p>Text Model: {result.nb_mem} | Peak: {result.nb_peak}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;