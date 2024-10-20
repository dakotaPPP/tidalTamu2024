import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import { S3Client, PutObjectCommand, DeleteObjectCommand } from "@aws-sdk/client-s3";
import patientRoutes from './routes/patients';
import pool from './config/database';

dotenv.config();

// Initialize Express, S3 Client, and Multer
const upload = multer();  // Multer to handle file uploads in memory

const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});


const app = express();

// Function to upload file to S3
// Function to upload file to S3
async function uploadFileToS3(fileBuffer: Buffer, fileName: string, patientId: string): Promise<string | null> {
  try {
    console.log('Uploading file to S3...');
    const s3Key = `${Date.now()}_${fileName}_${patientId}`;  // Generate unique S3 key

    const uploadParams = {
      Bucket: process.env.S3_BUCKET_NAME!,
      Region: process.env.AWS_REGION,
      Key: s3Key,
      Body: fileBuffer,  // Upload file as a buffer
    };

    const command = new PutObjectCommand(uploadParams);
    const response = await s3Client.send(command);

    console.log(`File uploaded with status: ${response.$metadata.httpStatusCode}`);
    console.log(`S3 Object Key: ${s3Key}`);
    
    return s3Key;
  } catch (err: any) {
    console.error(`Error uploading file: ${err.message}`);
    return null;
  }
}

// Upload Endpoint
app.post('/api/upload', upload.single('file'), async (req, res) => {
  console.log("File uploaded successfully");
  const file = req.file;  // Multer stores the uploaded file in req.file
  const { patientId } = req.body;

  // Check if file and patient ID are present
  if (!file || !patientId) {
    return res.status(400).json({ message: 'File and patient ID are required' });
  }

  try {
    const s3Key = await uploadFileToS3(file.buffer, file.originalname, patientId);  // Use file.buffer and file.originalname from multer
    if (s3Key) {
      res.status(200).json({ message: 'File uploaded successfully', s3Key });
    } else {
      res.status(500).json({ message: 'File upload failed' });
    }
  } catch (err) {
    res.status(500).json({ message: 'Error uploading file' });
  }
});

async function removeFileFromS3(s3Key: string): Promise<string | null> {
  try {
    const deleteParams = {
      Bucket: process.env.S3_BUCKET_NAME!,
      Region: process.env.AWS_REGION,
      Key: s3Key,
    };

    const command = new DeleteObjectCommand(deleteParams);
    const response = await s3Client.send(command);

    console.log(`${s3Key} deleted successfully from S3`);
    console.log(`Status Code: ${response.$metadata.httpStatusCode}`);
    
    return s3Key;
  } catch (err: any) {
    console.error(`Error Status: ${err?.$metadata?.httpStatusCode}`);
    return null;
  }
}

// Remove file endpoint (delete from S3 and SQL)
app.delete('/api/delete/:fileId', async (req, res) => {
  const { fileId } = req.params;

  try {
    // Step 1: Fetch the S3 key from SQL database using the fileId
    const queryResult = await pool.query('SELECT s3_key FROM files WHERE id = $1', [fileId]);
    const s3Key = queryResult.rows[0]?.s3_key;

    if (!s3Key) {
      return res.status(404).json({ message: 'File not found' });
    }

    // Step 2: Remove the file from S3
    const deletionResult = await removeFileFromS3(s3Key);
    if (!deletionResult) {
      return res.status(500).json({ message: 'Failed to delete file from S3' });
    }

    // Step 3: Remove the entry from the SQL database
    await pool.query('DELETE FROM files WHERE id = $1', [fileId]);
    console.log(`File ${s3Key} removed from SQL database`);

    res.status(204).json({ message: 'File removed successfully' });
  } catch (err) {
    console.error('Error removing file:', err);
    res.status(500).json({ message: 'Error removing file' });
  }
});

const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.use('/api/patients', patientRoutes);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
