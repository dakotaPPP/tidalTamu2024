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
import { MoreHorizontal, UserCircle, Upload, Download, RefreshCw, Trash2, Plus } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select" 

// Mock server actions
async function addPatient(patientData: FormData) {
  await new Promise(resolve => setTimeout(resolve, 1000))
  console.log('Patient added:', Object.fromEntries(patientData))
  return { success: true }
}

async function fetchPatients() {
  await new Promise(resolve => setTimeout(resolve, 500))
  return [
    { pid: '001', name: 'John Doe', educationYear: 12, sex: 'Male', iq: 105, eegFilesCount: 2 },
    { pid: '002', name: 'Jane Smith', educationYear: 16, sex: 'Female', iq: 120, eegFilesCount: 3 },
  ]
}

async function fetchEEGData(patientId: string) {
  await new Promise(resolve => setTimeout(resolve, 500))
  return [
    { id: '1', name: 'EEG_001', date: '2023-05-01' },
    { id: '2', name: 'EEG_002', date: '2023-05-15' },
  ]
}

async function updatePatient(patientId: string, patientData: any) {
  await new Promise(resolve => setTimeout(resolve, 500))
  console.log('Patient updated:', patientId, patientData)
  return { success: true }
}

async function removePatient(patientId: string) {
  await new Promise(resolve => setTimeout(resolve, 500))
  console.log('Patient removed:', patientId)
  return { success: true }
}

async function removeEEG(eegId: string) {
  await new Promise(resolve => setTimeout(resolve, 500))
  console.log('EEG removed:', eegId)
  return { success: true }
}

async function uploadEEG(patientId: string, file: File) {
  await new Promise(resolve => setTimeout(resolve, 1000))
  console.log('EEG uploaded for patient:', patientId, file.name)
  return { success: true, id: Date.now().toString(), name: file.name, date: new Date().toISOString().split('T')[0] }
}

async function replaceEEG(eegId: string, file: File) {
  await new Promise(resolve => setTimeout(resolve, 1000))
  console.log('EEG replaced:', eegId, file.name)
  return { success: true }
}

async function openEEG(eegId: string) {
  await new Promise(resolve => setTimeout(resolve, 1000))
  console.log('EEG opened:', eegId)
}

export default function Dashboard() {
  const [patients, setPatients] = useState<any[]>([])
  const [isAddingPatient, setIsAddingPatient] = useState(false)
  const [newPatient, setNewPatient] = useState({ name: '', id: '', educationYear: '', sex: '', iq: '' })
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isManageEEGOpen, setIsManageEEGOpen] = useState(false)
  const [currentPatientEEG, setCurrentPatientEEG] = useState<any[]>([])
  const [currentPatientId, setCurrentPatientId] = useState('')
  const [isPatientDetailsOpen, setIsPatientDetailsOpen] = useState(false)
  const [currentPatient, setCurrentPatient] = useState<any>(null)
  const [isRemovePatientOpen, setIsRemovePatientOpen] = useState(false)
  const [patientToRemove, setPatientToRemove] = useState<string | null>(null)
  const [isUploadingEEG, setIsUploadingEEG] = useState(false)
  const [isReplacingEEG, setIsReplacingEEG] = useState(false)
  const [eegToReplace, setEegToReplace] = useState<string | null>(null)

  useEffect(() => {
    fetchPatients().then(setPatients)
  }, [])

  const handleAddPatient = async () => {
    setIsAddingPatient(true)
    const formData = new FormData()
    Object.entries(newPatient).forEach(([key, value]) => {
      formData.append(key, value)
    })
    selectedFiles.forEach((file, index) => {
      formData.append(`eegFile${index}`, file)
    })

    const result = await addPatient(formData)
    if (result.success) {
      fetchPatients().then(setPatients)
      setNewPatient({ name: '', id: '', educationYear: '', sex: '', iq: '' })
      setSelectedFiles([])
    }
    setIsAddingPatient(false)
  }

  const handleManageEEG = async (patientId: string) => {
    setCurrentPatientId(patientId)
    const eegData = await fetchEEGData(patientId)
    setCurrentPatientEEG(eegData)
    setIsManageEEGOpen(true)
  }

  const handlePatientDetails = (patient: any) => {
    setCurrentPatient(patient)
    setIsPatientDetailsOpen(true)
  }

  const handleUpdatePatient = async () => {
    if (currentPatient) {
      await updatePatient(currentPatient.pid, currentPatient)
      fetchPatients().then(setPatients)
      setIsPatientDetailsOpen(false)
    }
  }

  const handleRemovePatient = (patientId: string) => {
    setPatientToRemove(patientId)
    setIsRemovePatientOpen(true)
  }

  const confirmRemovePatient = async () => {
    if (patientToRemove) {
      await removePatient(patientToRemove)
      fetchPatients().then(setPatients)
      setIsRemovePatientOpen(false)
      setPatientToRemove(null)
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

  const handleReplaceEEG = (eegId: string) => {
    setEegToReplace(eegId)
    setIsReplacingEEG(true)
  }

  const handleRemoveEEG = async (eegId: string) => {
    await removeEEG(eegId)
    const updatedEEGData = await fetchEEGData(currentPatientId)
    setCurrentPatientEEG(updatedEEGData)
  }

  const handleUploadEEG = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setIsUploadingEEG(true)
      const file = e.target.files[0]
      const result = await uploadEEG(currentPatientId, file)
      if (result.success) {
        setCurrentPatientEEG([...currentPatientEEG, result])
      }
      setIsUploadingEEG(false)
    }
  }

  const handleReplaceEEGConfirm = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0 && eegToReplace) {
      setIsReplacingEEG(true)
      const file = e.target.files[0]
      const result = await replaceEEG(eegToReplace, file)
      if (result.success) {
        const updatedEEGData = await fetchEEGData(currentPatientId)
        setCurrentPatientEEG(updatedEEGData)
      }
      setIsReplacingEEG(false)
      setEegToReplace(null)
    }
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
                  <TableCell>{patient.educationYear}</TableCell>
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
                <Label htmlFor="id" className="text-right">
                  ID
                </Label>
                <Input
                  id="id"
                  value={newPatient.id}
                  onChange={(e) => setNewPatient({ ...newPatient, id: e.target.value })}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="educationYear" className="text-right">
                  Education Year
                </Label>
                <Input
                  id="educationYear"
                  type="number"
                  value={newPatient.educationYear}
                  onChange={(e) => setNewPatient({ ...newPatient, educationYear: e.target.value })}
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
              <div className="grid  grid-cols-4 items-center gap-4">
                <Label htmlFor="eegData" className="text-right">
                  EEG Data
                </Label>
                <div className="col-span-3">
                  <Input
                    id="eegData"
                    type="file"
                    multiple
                    className="hidden"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="mr-2 h-4 w-4" />
                    {selectedFiles.length > 0 ? `${selectedFiles.length} file(s) selected` : 'Upload EEG Files'}
                  </Button>
                </div>
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
                  <TableRow key={eeg.id} onClick={() => openEEG(eeg.id)}>
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
              <Input
                type="file"
                id="uploadEEG"
                className="hidden"
                onChange={handleUploadEEG}
              />
              <Button
                onClick={() => document.getElementById('uploadEEG')?.click()}
                disabled={isUploadingEEG}
              >
                <Plus className="mr-2 h-4 w-4" />
                {isUploadingEEG ? 'Uploading...' : 'Upload New EEG'}
              </Button>
            </div>
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
                    value={currentPatient.educationYear}
                    onChange={(e) => setCurrentPatient({ ...currentPatient, educationYear: e.target.value })}
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
                    onChange={(e) => setCurrentPatient({ ...currentPatient, iq: e.target.value })}
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