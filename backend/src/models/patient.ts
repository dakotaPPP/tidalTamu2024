import pool from '../config/database';
import { spawn } from 'child_process';
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { Readable } from 'stream';
import { run } from 'node:test';

const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

export interface Patient {
  pid: string;
  name: string;
  education_year: number;
  sex: string;
  iq: number;
}

export interface Eeg {
  id: string;
  name: string;
  date: Date;
  csv_file: String;
}



export async function getAllPatients(): Promise<Patient[]> {
  const result = await pool.query('SELECT * FROM patient');
  return result.rows;
}

export async function addPatient(patient: Omit<Patient, 'pid'>): Promise<Patient> {
  const { name, education_year, sex, iq } = patient;
  const result = await pool.query(
    'INSERT INTO patient (name, education_year, sex, iq) VALUES ($1, $2, $3, $4) RETURNING *',
    [name, education_year, sex, iq]
  );
  return result.rows[0];
}

export async function updatePatient(pid: string, patient: Partial<Patient>): Promise<Patient> {
  const { name, education_year, sex, iq } = patient;
  const result = await pool.query(
    'UPDATE patient SET name = COALESCE($1, name), education_year = COALESCE($2, education_year), sex = COALESCE($3, sex), iq = COALESCE($4, iq) WHERE pid = $5 RETURNING *',
    [name, education_year, sex, iq, pid]
  );
  return result.rows[0];
}

export async function deletePatient(pid: string): Promise<void> {
  await pool.query('DELETE FROM patient WHERE pid = $1', [pid]);
}

export async function getPatientEEGCount(pid: string): Promise<number> {
  const result = await pool.query('SELECT COUNT(*) FROM eeg WHERE pid = $1', [pid]);
  return parseInt(result.rows[0].count);
}


export async function fetchEEGData(pid: string): Promise<Eeg[]> {
  const result = await pool.query('SELECT id, name, date, csv_data FROM eeg WHERE pid = $1', [pid]);
  return result.rows.map(row => ({
    id: row.id,
    name: row.name,
    date: new Date(row.date),
    csv_file: row.csv_data,
  }));
}

export async function addEEGData(pid: string, eegData: Eeg): Promise<Eeg> {
  const { name, date, csv_file } = eegData;
  const result = await pool.query(
    'INSERT INTO eeg (pid, name, date, csv_data) VALUES ($1, $2, $3, $4) RETURNING *',
    [pid, name, date, csv_file]
  );
  return result.rows[0];
}

export async function deleteEEGData(pid: string, eegId: string): Promise<void> {
  await pool.query('DELETE FROM eeg WHERE pid = $1 AND id = $2', [pid, eegId]);
}

export async function runPythonScript(s3Key: string): Promise<string> {
  try {
    // Step 1: Fetch the CSV file from S3
    const command = new GetObjectCommand({
      Bucket: process.env.S3_BUCKET_NAME!,
      Key: s3Key as string,
    });

    const { Body } = await s3Client.send(command);

    const fileStream = ensureStream(Body); // Ensure the body is treated as a stream

    // Step 2: Send the CSV data to the Python script
    const pythonProcess = spawn('python3', ['src/python/script.py'], {
      stdio: ['pipe', 'pipe', 'inherit'],
    })
    
    fileStream.pipe(pythonProcess.stdin);

    // Capture the output from the Python script
    let result = '';
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });
    return result;

  } catch (error) {
    console.error('Error processing file:', error);
    throw error;
  }
}

function ensureStream(body: any): Readable {
  if (body instanceof Readable) {
    return body;
  } else if (body instanceof Uint8Array) {
    return Readable.from(body); // Convert buffer/Uint8Array to stream
  } else {
    throw new Error('Unsupported Body type for streaming');
  }
}

runPythonScript('1729438108219_eeg1.csv_2')