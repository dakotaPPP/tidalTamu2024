import express from 'express';
import { getAllPatients, addPatient, updatePatient, deletePatient, getPatientEEGCount, fetchEEGData, addEEGData, deleteEEGData } from '../models/patient';

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    const patients = await getAllPatients();
    const patientsWithEEGCount = await Promise.all(
      patients.map(async (patient) => ({
        ...patient,
        eegFilesCount: await getPatientEEGCount(patient.pid),
      }))
    );
    res.json(patientsWithEEGCount);
  } catch (error) {
    res.status(500).json({ error: 'Error fetching patients' });
  }
});

router.post('/', async (req, res) => {
  try {
    const newPatient = await addPatient(req.body);
    res.status(201).json(newPatient);
  } catch (error) {
    res.status(500).json({ error: 'Error adding patient' });
  }
});

router.put('/:pid', async (req, res) => {
  try {
    const updatedPatient = await updatePatient(req.params.pid, req.body);
    res.json(updatedPatient);
  } catch (error) {
    res.status(500).json({ error: 'Error updating patient' });
  }
});

router.delete('/:pid', async (req, res) => {
  try {
    await deletePatient(req.params.pid);
    res.status(204).send();
  } catch (error) {
    res.status(500).json({ error: 'Error deleting patient' });
  }
});

router.get('/:pid/eeg-data', async (req, res) => {
  try {
    const eegData = await fetchEEGData(req.params.pid);
    res.json(eegData);
  } catch (error) {
    res.status(500).json({ error: 'Error fetching EEG data' });
  }
});

router.post('/:pid/eeg-data', async (req, res) => {
  try {
    const newEEG = await addEEGData(req.params.pid, req.body);
    res.status(201).json(newEEG);
  } catch (error) {
    res.status(500).json({ error: 'Error adding EEG data' });
  }
})

router.delete('/:pid/eeg-data/:eegId', async (req, res) => {
  try {
    await deleteEEGData(req.params.pid, req.params.eegId);
    res.status(204).send();
  } catch (error) {
    res.status(500).json({ error: 'Error deleting EEG data' });
  }
});

export default router;
