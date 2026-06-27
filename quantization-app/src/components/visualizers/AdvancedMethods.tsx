"use client";

import { useState, useCallback } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { simulateFP8, simulateGPTQ, simulateSmoothQuant } from "@/lib/api";
import type { FP8Response, GPTQResponse, SmoothQuantResponse } from "@/lib/api";
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
import { Cpu, Zap, Brain, Loader2, AlertTriangle } from "lucide-react";

// --- Helpers ---
function generateRandomWeights(n: number, min = -1, max = 1): number[] {
  return Array.from({ length: n }, () => Math.random() * (max - min) + min);
}

function generateRandomMatrix(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() - 0.5) * 2)
  );
}

function generateOutlierMatrix(rows: number, cols: number, outlierCol: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, (_, c) =>
      c === outlierCol ? (Math.random() - 0.5) * 20 : (Math.random() - 0.5) * 2
    )
  );
}

// --- FP8 local fallback simulation ---
function localFP8Simulate(weights: number[], format: "e4m3" | "e5m2"): FP8Response {
  const maxVal = format === "e4m3" ? 448 : 57344;
  const minVal = format === "e4m3" ? -448 : -57344;
  const mantissaBits = format === "e4m3" ? 3 : 2;
  const exponentBits = format === "e4m3" ? 4 : 5;

  const quantized = weights.map((w) => {
    const clamped = Math.max(minVal, Math.min(maxVal, w));
    const precision = Math.pow(2, -mantissaBits);
    return Math.round(clamped / precision) * precision;
  });
  const errors = weights.map((w, i) => Math.abs(w - quantized[i]));
  const mse = errors.reduce((s, e) => s + e * e, 0) / errors.length;
  const rss = errors.reduce((s, e) => s + e * e, 0);

  return {
    original: weights,
    quantized,
    errors,
    mse,
    rss,
    format_info: {
      name: format === "e4m3" ? "E4M3" : "E5M2",
      exponent_bits: exponentBits,
      mantissa_bits: mantissaBits,
      max_value: maxVal,
      min_value: format === "e4m3" ? 0.001953125 : 0.0000610352,
    },
  };
}

// --- GPTQ local fallback simulation ---
function localGPTQSimulate(weights: number[][], bits: number): GPTQResponse {
  const rows = weights.length;
  const cols = weights[0].length;
  const levels = Math.pow(2, bits);
  const quantized = weights.map((row) => [...row]);
  const steps: GPTQResponse["steps"] = [];

  for (let c = 0; c < cols; c++) {
    const origCol = weights.map((r) => r[c]);
    const colMin = Math.min(...quantized.map((r) => r[c]));
    const colMax = Math.max(...quantized.map((r) => r[c]));
    const scale = (colMax - colMin) / (levels - 1) || 1;

    const quantCol = quantized.map((r) => {
      const q = Math.round((r[c] - colMin) / scale) * scale + colMin;
      return q;
    });

    const error = quantCol.reduce((s, q, i) => s + Math.pow(q - quantized[i][c], 2), 0) / rows;

    // Apply compensation to remaining columns
    for (let r = 0; r < rows; r++) {
      const residual = quantized[r][c] - quantCol[r];
      for (let j = c + 1; j < cols; j++) {
        quantized[r][j] += residual * 0.1;
      }
      quantized[r][c] = quantCol[r];
    }

    steps.push({
      column: c,
      original_values: origCol,
      quantized_values: quantCol,
      error,
      compensation_applied: c < cols - 1,
    });
  }

  const totalMse =
    weights.flat().reduce((s, v, i) => s + Math.pow(v - quantized.flat()[i], 2), 0) /
    (rows * cols);

  return {
    steps,
    original_matrix: weights,
    quantized_matrix: quantized,
    total_mse: totalMse,
    compression_ratio: 32 / bits,
  };
}

// --- SmoothQuant local fallback ---
function localSmoothQuant(weights: number[][], activations: number[][], alpha: number): SmoothQuantResponse {
  const cols = weights[0].length;
  const smoothFactors: number[] = [];

  for (let c = 0; c < cols; c++) {
    const wMax = Math.max(...weights.map((r) => Math.abs(r[c])));
    const aMax = Math.max(...activations.map((r) => Math.abs(r[c])));
    const s = Math.pow(aMax, alpha) / Math.pow(wMax, 1 - alpha);
    smoothFactors.push(s || 1);
  }

  const smoothedWeights = weights.map((row) =>
    row.map((v, c) => v * smoothFactors[c])
  );
  const smoothedActivations = activations.map((row) =>
    row.map((v, c) => v / smoothFactors[c])
  );

  const flatW = weights.flat();
  const flatSW = smoothedWeights.flat();
  const flatA = activations.flat();
  const flatSA = smoothedActivations.flat();

  return {
    original_weights: weights,
    smoothed_weights: smoothedWeights,
    original_activations: activations,
    smoothed_activations: smoothedActivations,
    smooth_factors: smoothFactors,
    alpha,
    weight_range_before: { min: Math.min(...flatW), max: Math.max(...flatW) },
    weight_range_after: { min: Math.min(...flatSW), max: Math.max(...flatSW) },
    activation_range_before: { min: Math.min(...flatA), max: Math.max(...flatA) },
    activation_range_after: { min: Math.min(...flatSA), max: Math.max(...flatSA) },
  };
}

// --- Custom Tooltip ---
function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name: string; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-900 text-white px-3 py-2 rounded-lg shadow-xl text-xs border border-gray-700">
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(6) : p.value}
        </p>
      ))}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function AdvancedMethods() {
  return (
    <div className="space-y-8">
      <FP8Section />
      <GPTQSection />
      <SmoothQuantSection />
    </div>
  );
}

// =============================================================================
// FP8 SECTION
// =============================================================================
function FP8Section() {
  const [format, setFormat] = useState<"e4m3" | "e5m2">("e4m3");
  const [fp8Data, setFp8Data] = useState<FP8Response | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSimulate = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    const weights = generateRandomWeights(20);
    try {
      const result = await simulateFP8(weights, format);
      setFp8Data(result);
    } catch {
      // Fallback to local simulation
      setFp8Data(localFP8Simulate(weights, format));
    } finally {
      setIsLoading(false);
    }
  }, [format]);

  const errorChartData = fp8Data
    ? fp8Data.errors.map((e, i) => ({ index: i, error: e }))
    : [];

  return (
    <Card className="border-0 shadow-lg overflow-hidden rounded-xl">
      <CardHeader className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-950/30 dark:to-purple-950/30">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-violet-600" />
          <CardTitle className="text-xl">FP8 Quantization</CardTitle>
        </div>
        <CardDescription>
          Simulate 8-bit floating point formats (E4M3 for inference, E5M2 for training)
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        {/* Controls */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex gap-2">
            <Button
              variant={format === "e4m3" ? "default" : "outline"}
              onClick={() => setFormat("e4m3")}
              className="hover:shadow-md transition-shadow"
            >
              E4M3
            </Button>
            <Button
              variant={format === "e5m2" ? "default" : "outline"}
              onClick={() => setFormat("e5m2")}
              className="hover:shadow-md transition-shadow"
            >
              E5M2
            </Button>
          </div>
          <Button
            onClick={handleSimulate}
            disabled={isLoading}
            size="lg"
            className="shadow-md hover:shadow-lg transition-shadow"
          >
            {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Simulate FP8 (20 random values)
          </Button>
        </div>

        {error && (
          <div className="flex items-center gap-2 p-3 bg-amber-50 dark:bg-amber-950/30 rounded-lg border border-amber-200 dark:border-amber-800">
            <AlertTriangle className="h-4 w-4 text-amber-600" />
            <span className="text-sm text-amber-700 dark:text-amber-300">{error}</span>
          </div>
        )}

        {fp8Data && (
          <div className="space-y-6">
            {/* Format Info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { label: "Format", value: fp8Data.format_info.name },
                { label: "Exponent Bits", value: fp8Data.format_info.exponent_bits },
                { label: "Mantissa Bits", value: fp8Data.format_info.mantissa_bits },
                { label: "Max Value", value: fp8Data.format_info.max_value.toFixed(2) },
              ].map((item) => (
                <div key={item.label} className="p-3 bg-muted/50 rounded-lg text-center">
                  <p className="text-xs text-muted-foreground">{item.label}</p>
                  <p className="text-lg font-semibold">{item.value}</p>
                </div>
              ))}
            </div>

            {/* Table */}
            <div className="max-h-64 overflow-auto rounded-lg border">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-muted">
                  <tr>
                    <th className="px-3 py-2 text-left">#</th>
                    <th className="px-3 py-2 text-right">Original</th>
                    <th className="px-3 py-2 text-right">Quantized</th>
                    <th className="px-3 py-2 text-right">Error</th>
                  </tr>
                </thead>
                <tbody>
                  {fp8Data.original.map((v, i) => (
                    <tr key={i} className="border-t hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-1.5">{i}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{v.toFixed(6)}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{fp8Data.quantized[i].toFixed(6)}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-red-600">
                        {fp8Data.errors[i].toFixed(8)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Error bar chart */}
            <div>
              <h4 className="text-sm font-medium mb-2">Error per Value</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={errorChartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="index" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="error" name="Abs Error" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* MSE */}
            <div className="flex gap-4 text-sm">
              <span className="px-3 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 rounded-full">
                MSE: {fp8Data.mse.toExponential(4)}
              </span>
              <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full">
                RSS: {fp8Data.rss.toExponential(4)}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// GPTQ SECTION
// =============================================================================
function GPTQSection() {
  const [gptqData, setGptqData] = useState<GPTQResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  const handleRun = useCallback(async () => {
    setIsLoading(true);
    setActiveStep(0);
    const matrix = generateRandomMatrix(6, 6);
    try {
      const result = await simulateGPTQ(matrix, 4, 6);
      setGptqData(result);
    } catch {
      setGptqData(localGPTQSimulate(matrix, 4));
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getHeatColor = (value: number, maxAbs: number) => {
    const normalized = value / (maxAbs || 1);
    if (normalized > 0) return `rgba(59, 130, 246, ${Math.abs(normalized)})`;
    return `rgba(239, 68, 68, ${Math.abs(normalized)})`;
  };

  const maxAbs = gptqData
    ? Math.max(...gptqData.original_matrix.flat().map(Math.abs))
    : 1;

  return (
    <Card className="border-0 shadow-lg overflow-hidden rounded-xl">
      <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/30 dark:to-teal-950/30">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-emerald-600" />
          <CardTitle className="text-xl">GPTQ Column-wise Quantization</CardTitle>
        </div>
        <CardDescription>
          Watch GPTQ quantize a matrix column by column, compensating errors in remaining columns
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        <Button
          onClick={handleRun}
          disabled={isLoading}
          size="lg"
          className="shadow-md hover:shadow-lg transition-shadow"
        >
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Generate 6x6 Random Matrix & Run GPTQ
        </Button>

        {gptqData && (
          <div className="space-y-6">
            {/* Matrices side by side */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Original */}
              <div>
                <h4 className="text-sm font-medium mb-2">Original Matrix</h4>
                <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(6, 1fr)` }}>
                  {gptqData.original_matrix.flat().map((v, i) => (
                    <div
                      key={i}
                      className="aspect-square flex items-center justify-center text-[10px] font-mono rounded"
                      style={{ backgroundColor: getHeatColor(v, maxAbs) }}
                      title={v.toFixed(4)}
                    >
                      {v.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
              {/* Quantized */}
              <div>
                <h4 className="text-sm font-medium mb-2">Quantized Matrix (4-bit)</h4>
                <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(6, 1fr)` }}>
                  {gptqData.quantized_matrix.flat().map((v, i) => (
                    <div
                      key={i}
                      className="aspect-square flex items-center justify-center text-[10px] font-mono rounded"
                      style={{ backgroundColor: getHeatColor(v, maxAbs) }}
                      title={v.toFixed(4)}
                    >
                      {v.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Step slider */}
            <div>
              <h4 className="text-sm font-medium mb-2">
                Step {activeStep + 1} / {gptqData.steps.length} - Column {gptqData.steps[activeStep].column}
              </h4>
              <input
                type="range"
                min={0}
                max={gptqData.steps.length - 1}
                value={activeStep}
                onChange={(e) => setActiveStep(Number(e.target.value))}
                className="w-full"
              />
              <div className="mt-2 p-3 bg-muted/50 rounded-lg text-sm space-y-1">
                <p>
                  <strong>Error:</strong> {gptqData.steps[activeStep].error.toExponential(4)}
                </p>
                <p>
                  <strong>Compensation applied:</strong>{" "}
                  {gptqData.steps[activeStep].compensation_applied ? "Yes (to remaining columns)" : "No (last column)"}
                </p>
              </div>
            </div>

            {/* Summary */}
            <div className="flex gap-4 text-sm flex-wrap">
              <span className="px-3 py-1 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-full">
                Total MSE: {gptqData.total_mse.toExponential(4)}
              </span>
              <span className="px-3 py-1 bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300 rounded-full">
                Compression: {gptqData.compression_ratio.toFixed(1)}x
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// =============================================================================
// SMOOTHQUANT SECTION
// =============================================================================
function SmoothQuantSection() {
  const [alpha, setAlpha] = useState(0.5);
  const [sqData, setSqData] = useState<SmoothQuantResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleRun = useCallback(async () => {
    setIsLoading(true);
    const weights = generateRandomMatrix(4, 4);
    const activations = generateOutlierMatrix(4, 4, 2); // outlier in column 2
    try {
      const result = await simulateSmoothQuant(weights, activations, alpha);
      setSqData(result);
    } catch {
      setSqData(localSmoothQuant(weights, activations, alpha));
    } finally {
      setIsLoading(false);
    }
  }, [alpha]);

  const channelData = sqData
    ? sqData.smooth_factors.map((_, c) => ({
        channel: `Ch ${c}`,
        weight_before: Math.max(...sqData.original_weights.map((r) => Math.abs(r[c]))),
        weight_after: Math.max(...sqData.smoothed_weights.map((r) => Math.abs(r[c]))),
        activation_before: Math.max(...sqData.original_activations.map((r) => Math.abs(r[c]))),
        activation_after: Math.max(...sqData.smoothed_activations.map((r) => Math.abs(r[c]))),
      }))
    : [];

  return (
    <Card className="border-0 shadow-lg overflow-hidden rounded-xl">
      <CardHeader className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30">
        <div className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-amber-600" />
          <CardTitle className="text-xl">SmoothQuant</CardTitle>
        </div>
        <CardDescription>
          Migrate quantization difficulty from activations to weights using per-channel scaling
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        {/* Controls */}
        <div className="flex items-center gap-6 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <label className="text-sm font-medium block mb-1">
              Alpha: {alpha.toFixed(2)}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>0.0 (all on weights)</span>
              <span>1.0 (all on activations)</span>
            </div>
          </div>
          <Button
            onClick={handleRun}
            disabled={isLoading}
            size="lg"
            className="shadow-md hover:shadow-lg transition-shadow"
          >
            {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Run SmoothQuant
          </Button>
        </div>

        {sqData && (
          <div className="space-y-6">
            {/* Range comparison */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 bg-muted/50 rounded-lg text-center">
                <p className="text-xs text-muted-foreground">Weight Range (Before)</p>
                <p className="text-sm font-mono">
                  [{sqData.weight_range_before.min.toFixed(3)}, {sqData.weight_range_before.max.toFixed(3)}]
                </p>
              </div>
              <div className="p-3 bg-muted/50 rounded-lg text-center">
                <p className="text-xs text-muted-foreground">Weight Range (After)</p>
                <p className="text-sm font-mono">
                  [{sqData.weight_range_after.min.toFixed(3)}, {sqData.weight_range_after.max.toFixed(3)}]
                </p>
              </div>
              <div className="p-3 bg-muted/50 rounded-lg text-center">
                <p className="text-xs text-muted-foreground">Activation Range (Before)</p>
                <p className="text-sm font-mono">
                  [{sqData.activation_range_before.min.toFixed(3)}, {sqData.activation_range_before.max.toFixed(3)}]
                </p>
              </div>
              <div className="p-3 bg-muted/50 rounded-lg text-center">
                <p className="text-xs text-muted-foreground">Activation Range (After)</p>
                <p className="text-sm font-mono">
                  [{sqData.activation_range_after.min.toFixed(3)}, {sqData.activation_range_after.max.toFixed(3)}]
                </p>
              </div>
            </div>

            {/* Channel bar chart */}
            <div>
              <h4 className="text-sm font-medium mb-2">Max Absolute Value per Channel (Before vs After)</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={channelData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="activation_before" name="Activation (Before)" fill="#f59e0b" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="activation_after" name="Activation (After)" fill="#d97706" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="weight_before" name="Weight (Before)" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="weight_after" name="Weight (After)" fill="#1d4ed8" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Smooth factors */}
            <div className="p-3 bg-muted/50 rounded-lg">
              <p className="text-xs text-muted-foreground mb-1">Smooth Factors (per channel)</p>
              <div className="flex gap-2 flex-wrap">
                {sqData.smooth_factors.map((f, i) => (
                  <span key={i} className="px-2 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded text-xs font-mono">
                    Ch{i}: {f.toFixed(4)}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
