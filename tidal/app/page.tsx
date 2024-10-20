'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Label } from "@/components/ui/label"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { format } from "date-fns"
import { Calendar as CalendarIcon, MoreHorizontal, UserCircle, Upload, Download, RefreshCw, Trash2, Plus } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { cn } from "@/lib/utils"
import FileUploadComponent from '@/components/ui/file-upload'
import FileRemoveComponent from '@/components/ui/file-remove'

// Define the Patient interface
interface Patient {
  pid: string;
  name: string;
  education_year: number;
  sex: string;
  iq: number;
  eegFilesCount: number;
}

export interface Eeg {
  id: string;
  name: string;
  date: Date;
  csv_file: String;
}
  
// API functions
const API_URL = 'http://localhost:3001/api';

async function fetchPatients(): Promise<Patient[]> {
  const response = await fetch(`${API_URL}/patients`);
  if (!response.ok) {
    throw new Error('Failed to fetch patients');
  }
  return response.json();
}

async function addPatient(patientData: Omit<Patient, 'pid' | 'eegFilesCount'>): Promise<Patient> {
  const response = await fetch(`${API_URL}/patients`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(patientData),
  });
  if (!response.ok) {
    throw new Error('Failed to add patient');
  }
  return response.json();
}

async function updatePatient(pid: string, patientData: Partial<Patient>): Promise<Patient> {
  const response = await fetch(`${API_URL}/patients/${pid}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(patientData),
  });
  if (!response.ok) {
    throw new Error('Failed to update patient');
  }
  return response.json();
}

async function deletePatient(pid: string): Promise<void> {
  const response = await fetch(`${API_URL}/patients/${pid}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to delete patient');
  }
}

async function fetchEEGData(pid: string): Promise<any[]> {
  const response = await fetch(`${API_URL}/patients/${pid}/eeg-data`);
  if (!response.ok) {
    throw new Error('Failed to fetch EEG data');
  }
  return response.json();
}

//todo
async function addEEGData(pid: string, eegData: any): Promise<any> {
  const response = await fetch(`${API_URL}/patients/${pid}/eeg-data`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(eegData),
  });
  if (!response.ok) {
    throw new Error('Failed to add EEG data');
  }
  return response.json();
}

//todo
async function updateEEGData(pid: string, eegData: any): Promise<any> {
  const response = await fetch(`${API_URL}/patients/${pid}/eeg-data`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(eegData),
  });
  if (!response.ok) {
    throw new Error('Failed to update EEG data');
  }
  return response.json();
}

async function deleteEEGData(pid: string, eegId: string): Promise<void> {
  const response = await fetch(`${API_URL}/patients/${pid}/eeg-data/${eegId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to delete EEG data');
  }
}

export default function Page() {
  const [patients, setPatients] = useState<Patient[]>([])
  const [isAddingPatient, setIsAddingPatient] = useState(false)
  const [newPatient, setNewPatient] = useState({ name: '', id: '', education_year: '', sex: '', iq: '' })
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isManageEEGOpen, setIsManageEEGOpen] = useState(false)
  const [currentPatientEEG, setCurrentPatientEEG] = useState<any[]>([])
  const [currentPatientId, setCurrentPatientId] = useState('')
  const [isPatientDetailsOpen, setIsPatientDetailsOpen] = useState(false)
  const [currentPatient, setCurrentPatient] = useState<Patient | null>(null)
  const [isRemovePatientOpen, setIsRemovePatientOpen] = useState(false)
  const [patientToRemove, setPatientToRemove] = useState<string | null>(null)
  const [isUploadingEEG, setIsUploadingEEG] = useState(false)
  const [isReplacingEEG, setIsReplacingEEG] = useState(false)
  const [eegToReplace, setEegToReplace] = useState<Eeg | null>(null)
  const [newEEG, setNewEEG] = useState({ name: '', date: new Date() })
  const [isUploadEEGDialogOpen, setIsUploadEEGDialogOpen] = useState(false)
  useEffect(() => {
    fetchPatients()
      .then(setPatients)
      .catch(error => console.error('Error fetching patients:', error));
  }, [])

  const handleAddPatient = async () => {
    setIsAddingPatient(true)
    try {
      const patientData = {
        name: newPatient.name,
        education_year: parseInt(newPatient.education_year),
        sex: newPatient.sex,
        iq: parseInt(newPatient.iq),
      }
      const addedPatient = await addPatient(patientData)
      setPatients([...patients, { ...addedPatient, eegFilesCount: 0 }])
      setNewPatient({ name: '', id: '', education_year: '', sex: '', iq: '' })
      setSelectedFiles([])
    } catch (error) {
      console.error('Error adding patient:', error)
    }
    setIsAddingPatient(false)
  }

  const handleManageEEG = async (patientId: string) => {
    setCurrentPatientId(patientId)
    // Fetch EEG data for the patient (you'll need to implement this API endpoint)
    const eegData = await fetchEEGData(patientId)
    setCurrentPatientEEG(eegData)
    // For now, we'll use an empty array
    setIsManageEEGOpen(true)
  }

  const handlePatientDetails = (patient: Patient) => {
    setCurrentPatient(patient)
    setIsPatientDetailsOpen(true)
  }

  const handleUpdatePatient = async () => {
    if (currentPatient) {
      try {
        const updatedPatient = await updatePatient(currentPatient.pid, currentPatient)
        setPatients(patients.map(p => p.pid === updatedPatient.pid ? { ...updatedPatient, eegFilesCount: p.eegFilesCount } : p))
        setIsPatientDetailsOpen(false)
      } catch (error) {
        console.error('Error updating patient:', error)
      }
    }
  }

  const handleRemovePatient = (patientId: string) => {
    setPatientToRemove(patientId)
    setIsRemovePatientOpen(true)
  }

  const confirmRemovePatient = async () => {
    if (patientToRemove) {
      try {
        await deletePatient(patientToRemove)
        setPatients(patients.filter(p => p.pid !== patientToRemove))
        setIsRemovePatientOpen(false)
        setPatientToRemove(null)
      } catch (error) {
        console.error('Error removing patient:', error)
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files))
    }
  }

  const handleDownloadEEG = (eegId: string) => {
    // Implement download logic here
    console.log('Downloading EEG:', eegId)
  }

  // todo
  const handleReplaceEEG = (eegId: string) => {
    //setEegToReplace(eegId)
    setIsReplacingEEG(true)
  }

  const handleRemoveEEG = async (eegId: string) => {
    // Implement remove EEG logic here
    await deleteEEGData(currentPatientId, eegId)
    const updatedEEGData = await fetchEEGData(currentPatientId)
    setCurrentPatientEEG(updatedEEGData)
  }

  const handleUploadEEG = async () => {
    setIsUploadingEEG(true)
    try {
      // Implement upload EEG logic here
      console.log('Uploading EEG for patient:', currentPatientId, 'Name:', newEEG.name, 'Date:', newEEG.date)
      // After successful upload, you might want to refresh the EEG list
      const updatedEEGData = await fetchEEGData(currentPatientId);
      setCurrentPatientEEG(updatedEEGData);
    } catch (error) {
      console.error('Error uploading EEG:', error)
    }
    setIsUploadingEEG(false)
    setIsUploadEEGDialogOpen(false)
    setNewEEG({ name: '', date: new Date() })
  }

  const handleReplaceEEGConfirm = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0 && eegToReplace) {
      setIsReplacingEEG(true)
      // Implement replace EEG logic here
      console.log('Replacing EEG:', eegToReplace)
      setIsReplacingEEG(false)
      setEegToReplace(null)
    }
  }

  const handleViewEEG = (pid : string, eegId: string) => {
    // now open eeg-report.tsx
    console.log('Viewing EEG:', eegId)
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-primary text-primary-foreground p-4 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-primary-foreground rounded-full"></div>
          <span className="text-xl font-bold">Your Logo</span>
        </div>
        <div className="flex items-center space-x-2">
          <UserCircle className="w-6 h-6" />
          <span>Account</span>
        </div>
      </header>

      <main className="flex-grow p-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold mb-4">Patient List</h1>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Patient ID</TableHead>
                <TableHead>Patient Name</TableHead>
                <TableHead>Education Year</TableHead>
                <TableHead>Sex</TableHead>
                <TableHead>IQ</TableHead>
                <TableHead>EEG Files</TableHead>
                <TableHead className="w-[100px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {patients.map((patient) => (
                <TableRow key={patient.pid}>
                  <TableCell>{patient.pid}</TableCell>
                  <TableCell>{patient.name}</TableCell>
                  <TableCell>{patient.education_year}</TableCell>
                  <TableCell>{patient.sex}</TableCell>
                  <TableCell>{patient.iq}</TableCell>
                  <TableCell>{patient.eegFilesCount}</TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" className="h-8 w-8 p-0">
                          <span className="sr-only">Open menu</span>
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleManageEEG(patient.pid)}>
                          Manage EEG
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handlePatientDetails(patient)}>
                          Patient Details
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleRemovePatient(patient.pid)}>
                          Remove Patient
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <Dialog>
          <DialogTrigger asChild>
            <Button>Add Patient</Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Add New Patient</DialogTitle>
              <DialogDescription>
                Enter the patient's details here. Click save when you're done.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">
                  Name
                </Label>
                <Input
                  id="name"
                  value={newPatient.name}
                  onChange={(e) => setNewPatient({ ...newPatient, name: e.target.value })}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="education_year" className="text-right">
                  Education Year
                </Label>
                <Input
                  id="education_year"
                  type="number"
                  value={newPatient.education_year}
                  onChange={(e) => setNewPatient({ ...newPatient, education_year: e.target.value })}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="sex" className="text-right">
                  Sex
                </Label>
                <Select onValueChange={(value) => setNewPatient({ ...newPatient, sex: value })}>
                  <SelectTrigger className="col-span-3">
                    <SelectValue placeholder="Select sex" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Male">Male</SelectItem>
                    <SelectItem value="Female">Female</SelectItem>
                    <SelectItem value="Other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="iq" className="text-right">
                  IQ
                </Label>
                <Input
                  id="iq"
                  type="number"
                  value={newPatient.iq}
                  onChange={(e) => setNewPatient({ ...newPatient, iq: e.target.value })}
                  className="col-span-3"
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={handleAddPatient} disabled={isAddingPatient}>
                {isAddingPatient ? 'Adding...' : 'Add Patient'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={isManageEEGOpen} onOpenChange={setIsManageEEGOpen}>
          <DialogContent className="sm:max-w-[725px]">
            <DialogHeader>
              <DialogTitle>Manage EEG Data</DialogTitle>
              <DialogDescription>
                View and manage EEG data for the selected patient.
              </DialogDescription>
            </DialogHeader>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>ID</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {currentPatientEEG.map((eeg) => (
                  <TableRow key={eeg.id} onClick={() => handleViewEEG(currentPatientId, eeg.id)}>
                    <TableCell>{eeg.name}</TableCell>
                    <TableCell>{eeg.id}</TableCell>
                    <TableCell>{eeg.date}</TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" className="h-8 w-8 p-0">
                            <span className="sr-only">Open menu</span>
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handleDownloadEEG(eeg.id)}>
                            <Download className="mr-2 h-4 w-4" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleReplaceEEG(eeg.id)}>
                            <RefreshCw className="mr-2 h-4 w-4" />
                            Replace
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => handleRemoveEEG(eeg.id)}>
                            <Trash2 className="mr-2 h-4 w-4" />
                            Remove
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <div className="mt-4">
              <FileUploadComponent></FileUploadComponent>
              <FileRemoveComponent></FileRemoveComponent>
            </div>
          </DialogContent>
        </Dialog>

        <Dialog open={isUploadEEGDialogOpen} onOpenChange={setIsUploadEEGDialogOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Upload New EEG</DialogTitle>
              <DialogDescription>
                Enter the details for the new EEG file.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="eegName" className="text-right">
                  Name
                </Label>
                <Input
                  id="eegName"
                  value={newEEG.name}
                  onChange={(e) => setNewEEG({ ...newEEG, name: e.target.value })}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="eegDate" className="text-right">
                  Date
                </Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant={"outline"}
                      className={cn(
                        "w-[280px] justify-start text-left font-normal",
                        !newEEG.date && "text-muted-foreground"
                      )}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {newEEG.date ? format(newEEG.date, "PPP") : <span>Pick a date</span>}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0">
                    <Calendar
                      mode="single"
                      selected={newEEG.date}
                      onSelect={(date) => date && setNewEEG({ ...newEEG, date })}
                      initialFocus
                    />
                  </PopoverContent>
                </Popover>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="eegFile" className="text-right">
                  File
                </Label>
                <Input
                  id="eegFile"
                  type="file"
                  className="col-span-3"
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={handleUploadEEG} disabled={isUploadingEEG}>
                {isUploadingEEG ? 'Uploading...' : 'Upload'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={isPatientDetailsOpen} onOpenChange={setIsPatientDetailsOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Patient Details</DialogTitle>
              <DialogDescription>
                Update patient information.
              </DialogDescription>
            </DialogHeader>
            {currentPatient && (
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="patientName" className="text-right">
                    Name
                  </Label>
                  <Input
                    id="patientName"
                    value={currentPatient.name}
                    onChange={(e) => setCurrentPatient({ ...currentPatient, name: e.target.value })}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="patientEducationYear" className="text-right">
                    Education Year
                  </Label>
                  <Input
                    id="patientEducationYear"
                    type="number"
                    value={currentPatient.education_year}
                    onChange={(e) => setCurrentPatient({ ...currentPatient, education_year: parseInt(e.target.value) })}
                    className="col-span-3"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="patientSex" className="text-right">
                    Sex
                  </Label>
                  <Select
                    value={currentPatient.sex}
                    onValueChange={(value) => setCurrentPatient({ ...currentPatient, sex: value })}
                  >
                    <SelectTrigger className="col-span-3">
                      <SelectValue placeholder="Select sex" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Male">Male</SelectItem>
                      <SelectItem value="Female">Female</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="patientIQ" className="text-right">
                    IQ
                  </Label>
                  <Input
                    id="patientIQ"
                    type="number"
                    value={currentPatient.iq}
                    onChange={(e) => setCurrentPatient({ ...currentPatient, iq: parseInt(e.target.value) })}
                    className="col-span-3"
                  />
                </div>
              </div>
            )}
            <DialogFooter>
              <Button onClick={handleUpdatePatient}>Update Patient</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <AlertDialog open={isRemovePatientOpen} onOpenChange={setIsRemovePatientOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Are you sure?</AlertDialogTitle>
              <AlertDialogDescription>
                This action cannot be undone. This will permanently delete the patient's data and all associated EEG files.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={confirmRemovePatient}>Continue</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <Dialog open={isReplacingEEG} onOpenChange={(open) => {
          setIsReplacingEEG(open)
          if (!open) setEegToReplace(null)
        }}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Replace EEG File</DialogTitle>
              <DialogDescription>
                Select a new EEG file to replace the existing one.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <Input
                type="file"
                id="replaceEEG"
                onChange={handleReplaceEEGConfirm}
              />
            </div>
            <DialogFooter>
              <Button onClick={() => setIsReplacingEEG(false)}>Cancel</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </main>
    </div>
  )
}
/*
"use client"

import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LabelList, Cell, PolarAngleAxis, PolarGrid, PolarRadiusAxis, RadarChart, Radar } from "recharts"
import { Button } from "@/components/ui/button"
import { useParams } from "react-router-dom"

// Function to allocate data from array to variables
const allocateData = (dataArray: number[]) => {
  if (dataArray.length !== 12) {
    console.error("Input array must have exactly 12 elements");
    return null;
  }

  return {
    healthy: dataArray[3],
    addictive: Math.max(dataArray[0], dataArray[4]), // Max of Alcohol Abuse and Behavioral addiction
    traumaStress: Math.max(dataArray[1], dataArray[9], dataArray[10]), // Max of Acute stress, PTSD, and Adjustment disorder
    mood: Math.max(dataArray[2], dataArray[11]), // Max of Depression and Bipolar disorder
    obsessiveCompulsive: dataArray[5],
    schizophrenia: dataArray[6],
    anxiety: Math.max(dataArray[7], dataArray[8]), // Max of Panic disorder and Social anxiety
  };
};

// Sample data array (replace this with your actual data input)
const sampleDataArray = [30, 40, 35, 80, 25, 20, 15, 35, 40, 45, 30, 25];

// Updated sample data with new categories and subcategories
const createDisorderData = (allocatedData: ReturnType<typeof allocateData>) => {
  if (!allocatedData) return [];

  return [
    { name: "Healthy", value: allocatedData.healthy, subcategories: {} },
    { name: "Addictive", value: allocatedData.addictive, subcategories: { "Alcohol Abuse": sampleDataArray[0], "Behavioral Addiction": sampleDataArray[4] } },
    { name: "Trauma/Stress", value: allocatedData.traumaStress, subcategories: { "Acute Stress": sampleDataArray[1], "PTSD": sampleDataArray[9], "Adjustment Disorder": sampleDataArray[10] } },
    { name: "Mood", value: allocatedData.mood, subcategories: { "Depression": sampleDataArray[2], "Bipolar Disorder": sampleDataArray[11] } },
    { name: "Obsessive Compulsive", value: allocatedData.obsessiveCompulsive, subcategories: {} },
    { name: "Schizophrenia", value: allocatedData.schizophrenia, subcategories: {} },
    { name: "Anxiety", value: allocatedData.anxiety, subcategories: { "Panic Disorder": sampleDataArray[7], "Social Anxiety": sampleDataArray[8] } },
  ];
};

const COLORS = ['#8884d8', '#83a6ed', '#8dd1e1', '#82ca9d', '#a4de6c', '#ffc658', '#ff8042']


// Updated CustomTooltip component for BarChart
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-white p-4 rounded shadow">
        <p className="font-bold">{data.name}</p>
        <p className="text-black">Overall: {data.value}%</p>
        {Object.entries(data.subcategories).map(([key, value]) => (
          <p key={key} className="text-black">{key}: {value as number}%</p>
        ))}
      </div>
    )
  }
  return null
}

// Updated RadarTooltip component
const RadarTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-white p-2 rounded shadow">
        <p className="text-black">{data.name}</p>
        <p className="text-black">Likelihood: {data.value}%</p>
      </div>
    )
  }
  return null
}

export default function Component() {
  const params = useParams();
  const { patientId, eegId } = params;
  console.log(patientId, eegId)
  const [disorderData, setDisorderData] = useState<any[]>([]);

  useEffect(() => {
    const allocatedData = allocateData(sampleDataArray);
    if (allocatedData) {
      setDisorderData(createDisorderData(allocatedData));
    }
  }, []);

  return (
    <div className="container mx-auto p-4 min-h-screen">
      <h1 className="text-xl font-bold mb-10 mt-6 text-center">EEG Data Analysis Results</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mt-28">
          <div>
            <h2 className="text-lg font-semibold mb-4 text-center">Category Likelihood</h2>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={disorderData}
                  margin={{ top: 10, right: 30, left: 20, bottom: 5 }}
                >
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis dataKey="name" type="category" width={140} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value">
                    {disorderData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                    <LabelList dataKey="value" position="right" formatter={(value: any) => `${value}%`} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <h2 className="text-lg font-semibold mb-4 text-center">Category Distribution</h2>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart outerRadius="80%" data={disorderData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="name" />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar name="Category Likelihood" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                  <Tooltip content={<RadarTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
      </div>

      <div className="mt-12 flex justify-between items-center">
        <Button>Previous</Button>
        <p className="text-base font-semibold text-right">EEG ID: EEG12345</p>
      </div>
    </div>
  )
}*/