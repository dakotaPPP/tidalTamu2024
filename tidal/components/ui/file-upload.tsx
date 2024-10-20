import React, { useState } from 'react';
import axios from 'axios';

const FileUploadComponent: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState<string>('');
  const [uploadStatus, setUploadStatus] = useState<string>('');

  // Handle file selection
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!selectedFile || !patientId) {
      setUploadStatus('Please select a file and enter a patient ID');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('patientId', patientId);

    try {
      const response = await axios.post('http://localhost:3001/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.status === 200) {
        setUploadStatus('File uploaded successfully');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus('File upload failed');
    }
  };

  return (
    <div>
      <h2>Upload File</h2>
      <input type="file" onChange={handleFileChange} />
      <input
        type="text"
        placeholder="Enter Patient ID"
        value={patientId}
        onChange={(e) => setPatientId(e.target.value)}
      />
      <button onClick={handleUpload}>Upload File</button>

      {uploadStatus && <p>{uploadStatus}</p>}
    </div>
  );
};

export default FileUploadComponent;