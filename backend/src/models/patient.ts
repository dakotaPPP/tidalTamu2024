import pool from '../config/database';

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
