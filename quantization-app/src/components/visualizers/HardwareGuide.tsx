"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getHardwareComparison } from "@/lib/api";
import type { HardwareResponse, GPUInfo } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  Cell,
} from "recharts";
import { Cpu, Loader2, AlertTriangle } from "lucide-react";

// --- Static fallback data ---
const FALLBACK_GPUS: GPUInfo[] = [
  { name: "A100 80GB", fp32_tflops: 19.5, fp16_tflops: 312, int8_tops: 624, fp8_tops: 624, memory_gb: 80, year: 2020 },
  { name: "H100 SXM", fp32_tflops: 67, fp16_tflops: 989, int8_tops: 1979, fp8_tops: 3958, memory_gb: 80, year: 2022 },
  { name: "RTX 4090", fp32_tflops: 82.6, fp16_tflops: 165, int8_tops: 661, fp8_tops: 661, memory_gb: 24, year: 2022 },
  { name: "RTX 3090", fp32_tflops: 35.6, fp16_tflops: 71, int8_tops: 284, fp8_tops: null, memory_gb: 24, year: 2020 },
  { name: "L40S", fp32_tflops: 91.6, fp16_tflops: 183, int8_tops: 733, fp8_tops: 733, memory_gb: 48, year: 2023 },
  { name: "A6000", fp32_tflops: 38.7, fp16_tflops: 77.4, int8_tops: 310, fp8_tops: null, memory_gb: 48, year: 2020 },
  { name: "MI300X", fp32_tflops: 81.7, fp16_tflops: 1307, int8_tops: 2614, fp8_tops: 5222, memory_gb: 192, year: 2023 },
];

const PRECISION_COLORS: Record<string, string> = {
  fp32: "#ef4444",
  fp16: "#f59e0b",
  int8: "#10b981",
  fp8: "#8b5cf6",
};

// --- Custom Tooltip ---
function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name: string; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-900 text-white px-3 py-2 rounded-lg shadow-xl text-xs border border-gray-700">
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
        </p>
      ))}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function HardwareGuide() {
  const { data, isLoading, refetch, isError } = useQuery({
    queryKey: ["hardware-comparison"],
    queryFn: getHardwareComparison,
    enabled: false,
  });

  const gpus = data?.gpus || FALLBACK_GPUS;

  const throughputData = gpus.map((gpu) => ({
    name: gpu.name,
    FP32: gpu.fp32_tflops,
    FP16: gpu.fp16_tflops,
    INT8: gpu.int8_tops,
    FP8: gpu.fp8_tops || 0,
  }));

  const memoryData = gpus.map((gpu) => ({
    name: gpu.name,
    memory_gb: gpu.memory_gb,
  }));

  const speedupData = gpus.map((gpu) => ({
    name: gpu.name,
    int8_vs_fp32: (gpu.int8_tops / gpu.fp32_tflops).toFixed(1) + "x",
    int8_vs_fp32_val: gpu.int8_tops / gpu.fp32_tflops,
    fp8_vs_fp32: gpu.fp8_tops ? (gpu.fp8_tops / gpu.fp32_tflops).toFixed(1) + "x" : "N/A",
    fp8_vs_fp32_val: gpu.fp8_tops ? gpu.fp8_tops / gpu.fp32_tflops : 0,
  }));

  return (
    <div className="space-y-8">
      {/* Throughput Comparison */}
      <Card className="border-0 shadow-lg overflow-hidden rounded-xl">
        <CardHeader className="bg-gradient-to-r from-rose-50 to-pink-50 dark:from-rose-950/30 dark:to-pink-950/30">
          <div className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-rose-600" />
            <CardTitle className="text-xl">GPU Throughput Comparison</CardTitle>
          </div>
          <CardDescription>
            Compare compute throughput across precisions for popular GPUs (TFLOPS / TOPS)
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          <div className="flex items-center gap-3">
            <Button
              onClick={() => refetch()}
              disabled={isLoading}
              variant="outline"
              className="hover:shadow-md transition-shadow"
            >
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Refresh from API
            </Button>
            {!data && (
              <span className="text-xs text-muted-foreground">(Showing static reference data)</span>
            )}
            {isError && (
              <span className="text-xs text-amber-600 flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" /> API unavailable, using fallback
              </span>
            )}
          </div>

          {/* Grouped bar chart */}
          <div>
            <h4 className="text-sm font-medium mb-2">Compute Throughput by Precision</h4>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={throughputData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-15} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: "TFLOPS / TOPS", angle: -90, position: "insideLeft", fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar dataKey="FP32" name="FP32 (TFLOPS)" fill={PRECISION_COLORS.fp32} radius={[2, 2, 0, 0]} />
                <Bar dataKey="FP16" name="FP16 (TFLOPS)" fill={PRECISION_COLORS.fp16} radius={[2, 2, 0, 0]} />
                <Bar dataKey="INT8" name="INT8 (TOPS)" fill={PRECISION_COLORS.int8} radius={[2, 2, 0, 0]} />
                <Bar dataKey="FP8" name="FP8 (TOPS)" fill={PRECISION_COLORS.fp8} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Speedup callouts */}
          <div>
            <h4 className="text-sm font-medium mb-3">Quantization Speedup vs FP32</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {speedupData.map((gpu) => (
                <div
                  key={gpu.name}
                  className="p-3 rounded-lg bg-gradient-to-br from-muted/50 to-muted/20 border hover:shadow-md transition-shadow"
                >
                  <p className="text-xs font-medium text-muted-foreground">{gpu.name}</p>
                  <div className="mt-1 space-y-0.5">
                    <p className="text-sm">
                      INT8: <span className="font-bold text-green-600 dark:text-green-400">{gpu.int8_vs_fp32}</span> faster
                    </p>
                    <p className="text-sm">
                      FP8:{" "}
                      <span className={`font-bold ${gpu.fp8_vs_fp32_val > 0 ? "text-purple-600 dark:text-purple-400" : "text-muted-foreground"}`}>
                        {gpu.fp8_vs_fp32}
                      </span>
                      {gpu.fp8_vs_fp32_val > 0 ? " faster" : ""}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Memory Comparison */}
      <Card className="border-0 shadow-lg overflow-hidden rounded-xl">
        <CardHeader className="bg-gradient-to-r from-indigo-50 to-violet-50 dark:from-indigo-950/30 dark:to-violet-950/30">
          <div className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-indigo-600" />
            <CardTitle className="text-xl">GPU Memory Comparison</CardTitle>
          </div>
          <CardDescription>
            Available VRAM determines what model sizes can fit at different precisions
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={memoryData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis type="number" tick={{ fontSize: 11 }} unit=" GB" />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={90} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="memory_gb" name="VRAM (GB)" fill="#6366f1" radius={[0, 4, 4, 0]}>
                {memoryData.map((_, i) => (
                  <Cell key={i} fill={memoryData[i].memory_gb >= 80 ? "#8b5cf6" : memoryData[i].memory_gb >= 48 ? "#6366f1" : "#a5b4fc"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* What fits where */}
          <div className="p-4 bg-muted/30 rounded-lg">
            <h4 className="text-sm font-medium mb-2">What Model Sizes Fit?</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 text-xs">
              {gpus.map((gpu) => {
                const fp16Fit = Math.floor((gpu.memory_gb * 0.85) / 2); // ~85% usable, 2 bytes/param
                const int4Fit = Math.floor((gpu.memory_gb * 0.85) / 0.5); // 0.5 bytes/param
                return (
                  <div key={gpu.name} className="flex items-center gap-2 p-2 bg-background rounded border">
                    <span className="font-medium w-20">{gpu.name}</span>
                    <span className="text-muted-foreground">
                      FP16: ~{fp16Fit}B | INT4: ~{int4Fit}B params
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Full specs table */}
      <Card className="border-0 shadow-lg overflow-hidden rounded-xl">
        <CardHeader>
          <CardTitle className="text-base">Full Specifications</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto rounded-lg border">
            <table className="w-full text-xs">
              <thead className="bg-muted">
                <tr>
                  <th className="px-3 py-2 text-left">GPU</th>
                  <th className="px-3 py-2 text-right">Year</th>
                  <th className="px-3 py-2 text-right">FP32 (TFLOPS)</th>
                  <th className="px-3 py-2 text-right">FP16 (TFLOPS)</th>
                  <th className="px-3 py-2 text-right">INT8 (TOPS)</th>
                  <th className="px-3 py-2 text-right">FP8 (TOPS)</th>
                  <th className="px-3 py-2 text-right">VRAM (GB)</th>
                </tr>
              </thead>
              <tbody>
                {gpus.map((gpu, i) => (
                  <tr key={i} className="border-t hover:bg-muted/30 transition-colors">
                    <td className="px-3 py-1.5 font-medium">{gpu.name}</td>
                    <td className="px-3 py-1.5 text-right">{gpu.year}</td>
                    <td className="px-3 py-1.5 text-right font-mono">{gpu.fp32_tflops}</td>
                    <td className="px-3 py-1.5 text-right font-mono">{gpu.fp16_tflops}</td>
                    <td className="px-3 py-1.5 text-right font-mono">{gpu.int8_tops}</td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {gpu.fp8_tops !== null ? gpu.fp8_tops : <span className="text-muted-foreground">--</span>}
                    </td>
                    <td className="px-3 py-1.5 text-right">{gpu.memory_gb}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
