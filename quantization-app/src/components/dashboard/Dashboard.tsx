"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WeightDistribution } from "../visualizers/WeightDistribution";
import { QuantizationComparison } from "../visualizers/QuantizationComparison";
import { InteractivePlayground } from "../visualizers/InteractivePlayground";
import { ValueRepresentation } from "../visualizers/ValueRepresentation";
import { StepByStepQuantization } from "../visualizers/StepByStepQuantization";
import { ErrorAnalysis } from "../visualizers/ErrorAnalysis";
import { MemoryBenefits } from "../visualizers/MemoryBenefits";
import { AdvancedMethods } from "../visualizers/AdvancedMethods";
import { DecisionGuide } from "../visualizers/DecisionGuide";
import { Benchmarks } from "../visualizers/Benchmarks";
import { HardwareGuide } from "../visualizers/HardwareGuide";
import { RealModelQuantization } from "../visualizers/RealModelQuantization";

const TABS = [
  { value: "formats", label: "Formats", icon: "🔢", desc: "Number representations" },
  { value: "distribution", label: "Weights", icon: "📊", desc: "Weight distribution" },
  { value: "steps", label: "Steps", icon: "🔬", desc: "Step-by-step walkthrough" },
  { value: "comparison", label: "Compare", icon: "⚖️", desc: "Method comparison" },
  { value: "errors", label: "Errors", icon: "📉", desc: "Error analysis" },
  { value: "memory", label: "Memory", icon: "💾", desc: "Memory savings" },
  { value: "playground", label: "Playground", icon: "🎮", desc: "Interactive demo" },
  { value: "advanced", label: "Advanced", icon: "🧬", desc: "FP8, GPTQ, SmoothQuant" },
  { value: "guide", label: "Guide", icon: "🧭", desc: "Decision wizard" },
  { value: "benchmarks", label: "Benchmarks", icon: "⚡", desc: "Performance data" },
  { value: "hardware", label: "Hardware", icon: "🖥️", desc: "GPU comparison" },
  { value: "real-model", label: "Real LLM", icon: "🤖", desc: "Quantize TinyLlama-1.1B" },
];

export function Dashboard() {
  const [gpuInfo, setGpuInfo] = useState<{ gpu_name?: string; cuda_available?: boolean } | null>(null);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${apiUrl}/device-info`)
      .then((r) => r.json())
      .then((data) => setGpuInfo(data))
      .catch(() => setGpuInfo(null));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/30">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-background/80 border-b shadow-sm">
        <div className="max-w-[1400px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                Quantization Visualizer
              </h1>
              <p className="text-sm text-muted-foreground mt-0.5">
                Interactive exploration of AI model quantization &mdash; from number formats to memory savings
              </p>
            </div>
            <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground">
              {gpuInfo?.cuda_available && (
                <span className="px-2 py-1 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 font-semibold animate-pulse">
                  GPU: {gpuInfo.gpu_name}
                </span>
              )}
              <span className="px-2 py-1 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
                ResNet50 FC Layer
              </span>
              <span className="px-2 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                2,048,000 params
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        <Tabs defaultValue="formats" className="w-full">
          {/* Navigation Tabs */}
          <TabsList className="w-full h-auto flex flex-wrap gap-1 p-1.5 bg-muted/60 backdrop-blur-sm rounded-xl shadow-inner mb-6">
            {TABS.map((tab) => (
              <TabsTrigger
                key={tab.value}
                value={tab.value}
                className="flex-1 min-w-[120px] py-2.5 px-3 rounded-lg data-[state=active]:bg-background data-[state=active]:shadow-md data-[state=active]:border data-[state=active]:border-border/50 transition-all duration-200 hover:bg-background/50 group"
              >
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-base group-data-[state=active]:scale-110 transition-transform">
                    {tab.icon}
                  </span>
                  <span className="text-xs font-medium">{tab.label}</span>
                </div>
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="formats" className="mt-0 animate-in fade-in-50 duration-300">
            <ValueRepresentation />
          </TabsContent>

          <TabsContent value="distribution" className="mt-0 animate-in fade-in-50 duration-300">
            <WeightDistribution />
          </TabsContent>

          <TabsContent value="steps" className="mt-0 animate-in fade-in-50 duration-300">
            <StepByStepQuantization />
          </TabsContent>

          <TabsContent value="comparison" className="mt-0 animate-in fade-in-50 duration-300">
            <QuantizationComparison />
          </TabsContent>

          <TabsContent value="errors" className="mt-0 animate-in fade-in-50 duration-300">
            <ErrorAnalysis />
          </TabsContent>

          <TabsContent value="memory" className="mt-0 animate-in fade-in-50 duration-300">
            <MemoryBenefits />
          </TabsContent>

          <TabsContent value="playground" className="mt-0 animate-in fade-in-50 duration-300">
            <InteractivePlayground />
          </TabsContent>

          <TabsContent value="advanced" className="mt-0 animate-in fade-in-50 duration-300">
            <AdvancedMethods />
          </TabsContent>

          <TabsContent value="guide" className="mt-0 animate-in fade-in-50 duration-300">
            <DecisionGuide />
          </TabsContent>

          <TabsContent value="benchmarks" className="mt-0 animate-in fade-in-50 duration-300">
            <Benchmarks />
          </TabsContent>

          <TabsContent value="hardware" className="mt-0 animate-in fade-in-50 duration-300">
            <HardwareGuide />
          </TabsContent>

          <TabsContent value="real-model" className="mt-0 animate-in fade-in-50 duration-300">
            <RealModelQuantization />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
