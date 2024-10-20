

    const handleProcessFile = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!s3Key) {
      setStatus('Please enter an S3 key');
      return;
    }

    try {
      const response = await fetch(`/api/process-file?s3Key=${s3Key}`, {
        method: 'GET',
      });

      if (response.ok) {
        const result = await response.json();
        setData(result); // Store the processed data in state
        setStatus('File processed successfully');
      } else {
        setStatus('Failed to process the file');
      }
    } catch (error) {
      console.error('Error processing file:', error);
      setStatus('Error processing file');
    }
  };

  return (
    <div>
      <h2>Process File from S3</h2>
      
      <form onSubmit={handleProcessFile}>
        <input
          type="text"
          placeholder="Enter S3 Key"
          value={s3Key}
          onChange={(e) => setS3Key(e.target.value)}
        />
        <button type="submit">Process File</button>
      </form>

      {status && <p>{status}</p>}

      <h3>Processed Results:</h3>
      <ul>
        {data.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  );
};

export default ResultsPage