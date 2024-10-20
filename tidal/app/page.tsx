'use client'

import { useState, useRef } from 'react'
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
import { Label } from "@/components/ui/label"
import { MoreHorizontal, UserCircle, Upload } from "lucide-react"

// Mock server action
async function addPatient(patientData: FormData) {
  // Simulate a database operation
  await new Promise(resolve => setTimeout(resolve, 1000))
  console.log('Patient added:', Object.fromEntries(patientData))
  return { success: true }
}

export default function Dashboard() {
  const [patients, setPatients] = useState([
    { id: '001', name: 'John Doe', eegFiles: [] },
    { id: '002', name: 'Jane Smith', eegFiles: [] },
  ])
  const [isAddingPatient, setIsAddingPatient] = useState(false)
  const [newPatient, setNewPatient] = useState({ name: '', id: '' })
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleAddPatient = async () => {
    setIsAddingPatient(true)
    const formData = new FormData()
    formData.append('name', newPatient.name)
    formData.append('id', newPatient.id)
    selectedFiles.forEach((file, index) => {
      formData.append(`eegFile${index}`, file)
    })

    const result = await addPatient(formData)
    if (result.success) {
      setPatients([...patients, { ...newPatient, eegFiles: selectedFiles }])
      setNewPatient({ name: '', id: '' })
      setSelectedFiles([])
    }
    setIsAddingPatient(false)
  }

  const handleRemovePatient = (id: string) => {
    setPatients(patients.filter(patient => patient.id !== id))
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files))
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
                <TableHead>EEG Files</TableHead>
                <TableHead className="w-[100px]">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {patients.map((patient) => (
                <TableRow key={patient.id}>
                  <TableCell>{patient.id}</TableCell>
                  <TableCell>{patient.name}</TableCell>
                  <TableCell>{patient.eegFiles.length} file(s)</TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" className="h-8 w-8 p-0">
                          <span className="sr-only">Open menu</span>
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => alert('Opening EEG Reports')}>
                          Open EEG Reports
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleRemovePatient(patient.id)}>
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
      </main>
    </div>
  )
}
