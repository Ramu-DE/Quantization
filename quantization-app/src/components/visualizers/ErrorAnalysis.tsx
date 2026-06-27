"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getErrorDistribution } from "@/lib/api";
import type { ErrorDistributionResponse } from "@/lib/api";
import { useQuantizationStore } from "@/store/quantizationStore";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Loader2, AlertTriangle } from "lucide-react";
import { formatScientific } from "@/lib/utils";

const METHODS = [
  { key: "symmetric", label: "Symmetric INT8", color: "#3b82f6", desc: "Maps range symmetrically around zero" },
  { key: "asymmetric", label: "Asymmetric INT8", color: "#10b981", desc: "Uses full range with zero-point offset" },
  { key: "clipped", label: "Clipped INT8", color: "#f59e0b", desc: "Clips outliers before quantizing" },
  { key: "bfloat16", label: "BFloat16", color: "#8b5cf6", desc: "Truncates mantissa to 7 bits" },
];

export function ErrorAnalysis() {
  const { weights, clipMin, clipMax } = useQuantizationStore();
  const [results, setResults] = useState<Record<string, ErrorDistributionResponse>>({});
  const [isRunning, setIsRunning] = useState(false);

  const handleAnalyze = async () => {
    if (weights.length === 0) return;
    setIsRunning(true);
    try {
      const promises = METHODS.map(async (m) => {
        const resp = m.key === "clipped"
          ? await getErrorDistribution(weights, m.key, clipMin, clipMax)
          : await getErrorDistribution(weights, m.key);
        return { key: m.key, data: resp };
      });
      const all = await Promise.all(promises);
      const newResults: Record<string, ErrorDistributionResponse> = {};
      all.forEach(({ key, data }) => {
        newResults[key] = data;
      });
      setResults(newResults);
    } catch (err) {
      console.error("Error analysis failed:", err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-lg">
        <CardHeader className="pb-4">
          <CardTitle className="text-2xl">Quantization Error Distribution</CardTitle>
          <CardDescription className="text-base">
            Compare how quantization errors are distributed across different
            methods. INT8 methods produce uniform error distributions, while
            BFloat16 concentrates errors near zero.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Button
              onClick={handleAnalyze}
              disabled={weights.length === 0 || isRunning}
              size="lg"
              className="shadow-md hover:shadow-lg transition-shadow"
            >
              {isRunning && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Analyze Error Distributions
            </Button>
            {weights.length === 0 && (
              <div className="flex items-center gap-2 text-sm text-amber-600 dark:text-amber-400">
                <AlertTriangle className="h-4 w-4" />
                Load weights from the Weight Distribution tab first
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Error Distribution Charts - 2x2 grid */}
      {Object.keys(results).length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {METHODS.map((method) => {
            const data = results[method.key];
            if (!data) return null;

            const chartData = data.bin_centers.map((center, idx) => ({
              error: center.toFixed(6),
              count: data.counts[idx],
            }));

            return (
              <Card
                key={method.key}
                className="border-0 shadow-md hover:shadow-xl transition-all duration-300 group"
              >
                <CardHeader className="pb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: method.color }}
                    />
                    <CardTitle className="text-base">{method.label}</CardTitle>
                  </div>
                  <CardDescription className="text-xs">
                    {method.desc}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis
                        dataKey="error"
                        tick={{ fontSize: 9 }}
                        interval="preserveStartEnd"
                      />
                      <YAxis tick={{ fontSize: 9 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                          fontSize: "12px",
                        }}
                        formatter={(value: number) => [value.toLocaleString(), "Frequency"]}
                        labelFormatter={(label) => `Error: ${label}`}
                      />
                      <Bar
                        dataKey="count"
                        fill={method.color}
                        opacity={0.8}
                        radius={[2, 2, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>

                  {/* Stats grid with hover effect */}
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    {[
                      { label: "Min Error", value: data.min_error },
                      { label: "Max Error", value: data.max_error },
                      { label: "Mean", value: data.mean_error },
                      { label: "Std Dev", value: data.std_error },
                    ].map((stat) => (
                      <div
                        key={stat.label}
                        className="p-2 bg-muted/50 rounded-md hover:bg-muted transition-colors cursor-default"
                      >
                        <span className="text-[10px] text-muted-foreground block">
                          {stat.label}
                        </span>
                        <span className="font-mono text-xs">
                          {stat.value.toFixed(6)}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Pattern indicator */}
                  <div className="mt-3 text-xs px-2 py-1.5 rounded-full inline-flex items-center gap-1.5"
                    style={{
                      backgroundColor: `${method.color}15`,
                      color: method.color,
                    }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: method.color }} />
                    {method.key === "bfloat16"
                      ? "Concentrated at zero (truncation)"
                      : "Uniform distribution (rounding)"}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Comparison Stats Table */}
      {Object.keys(results).length === METHODS.length && (
        <Card className="border-0 shadow-lg">
          <CardHeader>
            <CardTitle>Error Statistics Comparison</CardTitle>
            <CardDescription>
              Side-by-side comparison matching the AMD notebook output format
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-muted/50">
                    <th className="text-left py-3 px-4 font-semibold">Method</th>
                    <th className="text-right py-3 px-4 font-semibold">Min Error</th>
                    <th className="text-right py-3 px-4 font-semibold">Max Error</th>
                    <th className="text-right py-3 px-4 font-semibold">Mean Error</th>
                    <th className="text-right py-3 px-4 font-semibold">Std Error</th>
                    <th className="text-center py-3 px-4 font-semibold">Pattern</th>
                  </tr>
                </thead>
                <tbody>
                  {METHODS.map((method) => {
                    const data = results[method.key];
                    if (!data) return null;
                    return (
                      <tr
                        key={method.key}
                        className="border-t hover:bg-muted/30 transition-colors"
                      >
                        <td className="py-3 px-4 font-medium">
                          <span
                            className="inline-block w-3 h-3 rounded-full mr-2 align-middle"
                            style={{ backgroundColor: method.color }}
                          />
                          {method.label}
                        </td>
                        <td className="text-right py-3 px-4 font-mono text-xs">
                          {formatScientific(data.min_error)}
                        </td>
                        <td className="text-right py-3 px-4 font-mono text-xs">
                          {formatScientific(data.max_error)}
                        </td>
                        <td className="text-right py-3 px-4 font-mono text-xs">
                          {formatScientific(data.mean_error)}
                        </td>
                        <td className="text-right py-3 px-4 font-mono text-xs">
                          {formatScientific(data.std_error)}
                        </td>
                        <td className="text-center py-3 px-4">
                          <span
                            className="text-xs px-2 py-0.5 rounded-full"
                            style={{
                              backgroundColor: `${method.color}15`,
                              color: method.color,
                            }}
                          >
                            {method.key === "bfloat16" ? "Concentrated" : "Uniform"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
