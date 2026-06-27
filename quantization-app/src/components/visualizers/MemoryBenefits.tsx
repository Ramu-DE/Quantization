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
import { getMemoryComparison } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";
import { Loader2, HardDrive, Zap, TrendingDown } from "lucide-react";

const RESNET_FC_PARAMS = 2048000;

const FORMAT_COLORS: Record<string, string> = {
  float32: "#ef4444",
  float16: "#f59e0b",
  bfloat16: "#8b5cf6",
  int8: "#10b981",
  int4: "#06b6d4",
};

export function MemoryBenefits() {
  const [numElements] = useState(RESNET_FC_PARAMS);

  const {
    data: memoryData,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["memory-comparison", numElements],
    queryFn: () => getMemoryComparison(numElements),
    enabled: false,
  });

  const staticFormats = [
    { name: "float32", bits: 32, memory_mb: (numElements * 4) / (1024 * 1024), compression_ratio: 1.0 },
    { name: "float16", bits: 16, memory_mb: (numElements * 2) / (1024 * 1024), compression_ratio: 2.0 },
    { name: "bfloat16", bits: 16, memory_mb: (numElements * 2) / (1024 * 1024), compression_ratio: 2.0 },
    { name: "int8", bits: 8, memory_mb: (numElements * 1) / (1024 * 1024), compression_ratio: 4.0 },
    { name: "int4", bits: 4, memory_mb: (numElements * 0.5) / (1024 * 1024), compression_ratio: 8.0 },
  ];

  const formats = memoryData
    ? memoryData.formats.map((fmt) => ({
        name: fmt.format_name,
        bits: fmt.bits_per_element,
        memory_mb: fmt.megabytes,
        compression_ratio: fmt.compression_ratio,
      }))
    : staticFormats;
  const fp32Memory = formats[0]?.memory_mb || 0;

  const chartData = formats.map((fmt) => ({
    name: fmt.name,
    memory_mb: Number(fmt.memory_mb.toFixed(2)),
    compression: fmt.compression_ratio,
    bits: fmt.bits,
  }));

  return (
    <div className="space-y-6">
      {/* Savings highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { ratio: "4x", format: "INT8", color: "green", icon: Zap, memory: fp32Memory / 4 },
          { ratio: "8x", format: "INT4", color: "cyan", icon: TrendingDown, memory: fp32Memory / 8 },
          { ratio: "2x", format: "BFloat16", color: "purple", icon: HardDrive, memory: fp32Memory / 2 },
        ].map((item) => (
          <Card
            key={item.format}
            className="border-0 shadow-md hover:shadow-xl transition-all duration-300 group cursor-default overflow-hidden"
          >
            <div className={`h-1 bg-${item.color}-500`} style={{ backgroundColor: item.color === "green" ? "#10b981" : item.color === "cyan" ? "#06b6d4" : "#8b5cf6" }} />
            <CardContent className="p-5 text-center">
              <item.icon className={`h-8 w-8 mx-auto mb-3 group-hover:scale-110 transition-transform`} style={{ color: item.color === "green" ? "#10b981" : item.color === "cyan" ? "#06b6d4" : "#8b5cf6" }} />
              <p className="text-3xl font-black" style={{ color: item.color === "green" ? "#10b981" : item.color === "cyan" ? "#06b6d4" : "#8b5cf6" }}>
                {item.ratio}
              </p>
              <p className="text-sm font-semibold mt-1">{item.format} Compression</p>
              <p className="text-xs text-muted-foreground mt-2">
                {fp32Memory.toFixed(2)} MB → {item.memory.toFixed(2)} MB
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Chart */}
      <Card className="border-0 shadow-lg overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-950/30 dark:to-gray-950/30 pb-4">
          <CardTitle className="text-2xl flex items-center gap-2">
            <HardDrive className="h-5 w-5 text-slate-600" />
            Memory Usage Comparison
          </CardTitle>
          <CardDescription className="text-base">
            Memory required to store {numElements.toLocaleString()} parameters
            (ResNet50 FC layer) across formats
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          <Button
            onClick={() => refetch()}
            disabled={isLoading}
            className="shadow-md hover:shadow-lg transition-all"
          >
            {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Calculate from API
          </Button>

          <div className="bg-muted/20 rounded-xl p-4 border">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                <XAxis
                  type="number"
                  tick={{ fontSize: 11 }}
                  label={{ value: "Memory (MB)", position: "insideBottom", offset: -5, fontSize: 11 }}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  tick={{ fontSize: 12, fontWeight: "bold" }}
                  width={80}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "10px",
                    boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
                    padding: "10px 14px",
                  }}
                  formatter={(value: number) => [`${value.toFixed(2)} MB`, "Memory"]}
                />
                <Bar dataKey="memory_mb" radius={[0, 6, 6, 0]}>
                  {chartData.map((entry, idx) => (
                    <Cell
                      key={idx}
                      fill={FORMAT_COLORS[entry.name] || "#6b7280"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Detailed table */}
      <Card className="border-0 shadow-lg">
        <CardHeader>
          <CardTitle>Detailed Breakdown</CardTitle>
          <CardDescription>
            Memory savings for {numElements.toLocaleString()} parameters
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-xl border">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-muted/50">
                  <th className="text-left py-3 px-4 font-semibold">Format</th>
                  <th className="text-center py-3 px-4 font-semibold">Bits</th>
                  <th className="text-right py-3 px-4 font-semibold">Memory (MB)</th>
                  <th className="text-right py-3 px-4 font-semibold">Savings</th>
                  <th className="text-center py-3 px-4 font-semibold">Compression</th>
                </tr>
              </thead>
              <tbody>
                {formats.map((fmt) => (
                  <tr
                    key={fmt.name}
                    className="border-t hover:bg-primary/5 transition-colors cursor-default"
                  >
                    <td className="py-3 px-4 font-medium flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: FORMAT_COLORS[fmt.name] || "#6b7280" }}
                      />
                      {fmt.name}
                    </td>
                    <td className="text-center py-3 px-4 font-mono">{fmt.bits}</td>
                    <td className="text-right py-3 px-4 font-mono font-bold">
                      {fmt.memory_mb.toFixed(2)}
                    </td>
                    <td className="text-right py-3 px-4">
                      {fmt.compression_ratio > 1 ? (
                        <span className="text-green-600 font-bold">
                          -{((1 - 1 / fmt.compression_ratio) * 100).toFixed(0)}%
                        </span>
                      ) : (
                        <span className="text-muted-foreground">baseline</span>
                      )}
                    </td>
                    <td className="text-center py-3 px-4">
                      <span
                        className="inline-flex items-center rounded-full px-3 py-1 text-xs font-bold text-white"
                        style={{ backgroundColor: FORMAT_COLORS[fmt.name] || "#6b7280" }}
                      >
                        {fmt.compression_ratio}x
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Impact callout */}
          <div className="mt-6 p-5 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/20 dark:to-emerald-950/20 rounded-xl border border-green-200 dark:border-green-800">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-2">
              Real-World Impact
            </h4>
            <p className="text-sm text-green-700 dark:text-green-300 leading-relaxed">
              ResNet50 FC layer: <strong>2,048,000 parameters</strong>.
              FP32 → INT8 reduces from <strong>{fp32Memory.toFixed(2)} MB</strong> to{" "}
              <strong>{(fp32Memory / 4).toFixed(2)} MB</strong> (4x savings).
              For models with billions of parameters (e.g., LLMs), this means fitting
              on a single GPU instead of requiring multiple GPUs or expensive cloud infrastructure.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
