import React, { useState } from 'react';
import axios from 'axios';

const FileRemoveComponent: React.FC = () => {
  const [fileKey, setFileKey] = useState<string>('');  // Store the S3 key of the file to be removed
  const [removeStatus, setRemoveStatus] = useState<string>('');

  // Handle file removal
  const handleRemove = async () => {
    if (!fileKey) {
      setRemoveStatus('Please enter the file key (S3 key) of the file to remove');
      return;
    }

    try {
      const response = await axios.delete(`/api/delete/${fileKey}`);
      if (response.status === 204) {
        setRemoveStatus('File removed successfully');
      } else {
        setRemoveStatus('Failed to remove file');
      }
    } catch (error) {
      console.error('Error removing file:', error);
      setRemoveStatus('File removal failed');
    }
  };

  return (
    <div>
      <h2>Remove File</h2>
      <input
        type="text"
        placeholder="Enter S3 File Key"
        value={fileKey}
        onChange={(e) => setFileKey(e.target.value)}
      />
      <button onClick={handleRemove}>Remove File</button>

      {removeStatus && <p>{removeStatus}</p>}
    </div>
  );
};

export default FileRemoveComponent;