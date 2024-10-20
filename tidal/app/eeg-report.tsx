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
}