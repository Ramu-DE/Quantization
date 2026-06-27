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
import { Label } from "@/components/ui/label";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  ZAxis,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ReferenceLine,
  Area,
  AreaChart,
  ComposedChart,
} from "recharts";
import { Loader2, Brain, Sparkles, AlertTriangle, ArrowRight, Cpu, Zap, HardDrive } from "lucide-react";
import api from "@/lib/api";

interface ModelInfo {
  model_name: string;
  architecture: string;
  num_parameters: number;
  num_layers: number;
  hidden_size: number;
  vocab_size: number;
  fp32_size_mb: number;
  layers?: { name: string; shape: number[]; numel: number; mean: number; std: number; min: number; max: number }[];
}

interface QuantizeResult {
  model_name: string;
  bits: number;
  scheme: string;
  group_size: number;
  num_parameters: number;
  num_layers_quantized: number;
  fp32_size_mb: number;
  quantized_size_mb: number;
  compression_ratio: number;
  avg_mse: number;
  baseline_perplexity: number;
  quantized_perplexity: number;
  perplexity_increase: number;
  perplexity_increase_pct: number;
  original_generation: string;
  quantized_generation: string;
  test_prompt: string;
  time_seconds: number;
  top_degraded_layers: { name: string; shape: number[]; mse: number; max_error: number; numel: number }[];
}

interface CompareResult {
  model_name: string;
  num_parameters: number;
  fp32_size_mb: number;
  baseline_perplexity: number;
  test_prompt: string;
  methods: {
    label: string;
    bits: number;
    scheme: string;
    group_size: number;
    mse: number;
    perplexity: number;
    perplexity_increase: number;
    size_mb: number;
    compression_ratio: number;
    generation: string;
    time_seconds: number;
  }[];
}

interface VisualizeResult {
  layer_name: string;
  shape: number[];
  bits: number;
  num_elements: number;
  num_unique_original: number;
  num_unique_quantized: number;
  mse: number;
  max_error: number;
  mean_abs_error: number;
  original_histogram: { counts: number[]; centers: number[] };
  quantized_histogram: { counts: number[]; centers: number[] };
  error_histogram: { counts: number[]; centers: number[] };
  sample_original: number[];
  sample_quantized: number[];
  sample_error: number[];
  grid_lines: number[];
  statistics: {
    original_min: number; original_max: number; original_std: number;
    quantized_min: number; quantized_max: number; error_std: number;
  };
}

export function RealModelQuantization() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loadingInfo, setLoadingInfo] = useState(false);
  const [quantizeResult, setQuantizeResult] = useState<QuantizeResult | null>(null);
  const [loadingQuantize, setLoadingQuantize] = useState(false);
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [loadingCompare, setLoadingCompare] = useState(false);
  const [visualizeResult, setVisualizeResult] = useState<VisualizeResult | null>(null);
  const [loadingVisualize, setLoadingVisualize] = useState(false);
  const [deepVizResult, setDeepVizResult] = useState<any | null>(null);
  const [loadingDeepViz, setLoadingDeepViz] = useState(false);

  const [bits, setBits] = useState(8);
  const [groupSize, setGroupSize] = useState(128);
  const [scheme, setScheme] = useState("symmetric");
  const [prompt, setPrompt] = useState("The future of AI is");

  const handleLoadModel = async () => {
    setLoadingInfo(true);
    try {
      const resp = await api.get("/api/real-model/info");
      setModelInfo(resp.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingInfo(false);
    }
  };

  const handleQuantize = async () => {
    setLoadingQuantize(true);
    try {
      const resp = await api.post("/api/real-model/quantize", {
        bits,
        scheme,
        group_size: groupSize,
        test_prompt: prompt,
      });
      setQuantizeResult(resp.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingQuantize(false);
    }
  };

  const handleCompare = async () => {
    setLoadingCompare(true);
    try {
      const resp = await api.post("/api/real-model/compare", {
        test_prompt: prompt,
      });
      setCompareResult(resp.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingCompare(false);
    }
  };

  const handleVisualize = async () => {
    setLoadingVisualize(true);
    try {
      const resp = await api.post("/api/real-model/visualize-quantization", {
        layer_name: "model.layers.0.self_attn.q_proj.weight",
        bits,
        scheme,
        group_size: groupSize,
        num_samples: 800,
      });
      setVisualizeResult(resp.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingVisualize(false);
    }
  };

  const handleDeepVisualize = async () => {
    setLoadingDeepViz(true);
    try {
      const resp = await api.post("/api/real-model/deep-visualize", {
        bits,
        scheme,
        group_size: groupSize,
        prompt: prompt || "Once upon a time",
      });
      setDeepVizResult(resp.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingDeepViz(false);
    }
  };

  const COLORS = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#06b6d4"];

  return (
    <div className="space-y-6">
      {/* Model Loading */}
      <Card className="border-0 shadow-lg overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-violet-50 to-fuchsia-50 dark:from-violet-950/30 dark:to-fuchsia-950/30 pb-4">
          <CardTitle className="text-2xl flex items-center gap-2">
            <Brain className="h-6 w-6 text-violet-600" />
            Real Model Quantization — TinyLlama 1.1B
          </CardTitle>
          <CardDescription className="text-base">
            Download and quantize a real 1.1 billion parameter LLM on NVIDIA L4 GPU.
            See exactly what happens to the weights during quantization.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          <div className="flex items-center gap-4">
            <Button
              onClick={handleLoadModel}
              disabled={loadingInfo}
              size="lg"
              className="shadow-md hover:shadow-lg transition-all"
            >
              {loadingInfo && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {modelInfo ? "Reload Model Info" : "Load TinyLlama-1.1B"}
            </Button>
            {loadingInfo && (
              <span className="text-sm text-muted-foreground">
                First load downloads ~2GB model onto GPU...
              </span>
            )}
          </div>

          {modelInfo && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[
                  { label: "Parameters", value: `${(modelInfo.num_parameters / 1e9).toFixed(2)}B`, icon: "🧠" },
                  { label: "FP32 Size", value: `${(modelInfo.fp32_size_mb / 1024).toFixed(1)} GB`, icon: "💾" },
                  { label: "Layers", value: modelInfo.num_layers.toString(), icon: "📚" },
                  { label: "Architecture", value: modelInfo.architecture.toUpperCase(), icon: "🏗️" },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="p-4 bg-muted/50 rounded-xl border hover:border-primary/20 transition-all"
                  >
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      {stat.icon} {stat.label}
                    </p>
                    <p className="text-lg font-bold mt-1">{stat.value}</p>
                  </div>
                ))}
              </div>

              {/* Weight Statistics Visualization */}
              {modelInfo.layers && modelInfo.layers.length > 0 && (
                <div className="bg-muted/20 rounded-xl p-4 border">
                  <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <Zap className="h-4 w-4 text-amber-500" />
                    Weight Distribution Across Layers (what gets quantized)
                  </h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <ComposedChart data={modelInfo.layers.slice(0, 15).map((l, i) => ({
                      name: l.name.split(".").slice(-2).join("."),
                      std: l.std,
                      range: l.max - l.min,
                      params: l.numel / 1e6,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                      <XAxis dataKey="name" tick={{ fontSize: 8 }} interval={0} angle={-30} height={60} />
                      <YAxis yAxisId="left" tick={{ fontSize: 9 }} label={{ value: "Std Dev", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
                      <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9 }} label={{ value: "Range", angle: 90, position: "insideRight", style: { fontSize: 9 } }} />
                      <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid hsl(var(--border))" }} />
                      <Bar yAxisId="left" dataKey="std" fill="#8b5cf6" opacity={0.7} radius={[3, 3, 0, 0]} name="Std Dev" />
                      <Line yAxisId="right" type="monotone" dataKey="range" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Value Range" />
                    </ComposedChart>
                  </ResponsiveContainer>
                  <p className="text-xs text-muted-foreground mt-2 text-center">
                    Layers with higher std/range are harder to quantize — they need more bits to represent accurately
                  </p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quantization Controls */}
      {modelInfo && (
        <Card className="border-0 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-amber-500" />
              Quantize Model
            </CardTitle>
            <CardDescription>
              Apply real quantization to all 1.1B parameters and measure the impact
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-3 p-4 bg-muted/30 rounded-xl border">
                <Label className="font-semibold">
                  Bit Width: <span className="text-primary font-mono">{bits}-bit</span>
                </Label>
                {/* Clickable bit buttons */}
                <div className="flex gap-2">
                  {[2, 3, 4, 8].map((b) => (
                    <button
                      key={b}
                      onClick={() => setBits(b)}
                      className={`flex-1 py-2 rounded-lg text-sm font-bold transition-all ${
                        bits === b
                          ? "bg-primary text-primary-foreground shadow-md scale-105"
                          : "bg-muted hover:bg-muted/80 border"
                      }`}
                    >
                      {b}-bit
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-muted-foreground">
                  {bits === 8 ? "Safe — minimal quality loss" :
                   bits >= 4 ? "Good — noticeable compression" :
                   "Aggressive — may degrade quality"}
                </p>
                {/* Visual: what bit-width means */}
                <div className="mt-2 p-2 bg-background rounded border text-xs">
                  <div className="flex justify-between text-muted-foreground">
                    <span>Representable values:</span>
                    <span className="font-mono font-bold">{Math.pow(2, bits)}</span>
                  </div>
                  <div className="mt-1 h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-500 via-amber-500 to-red-500 transition-all"
                      style={{ width: `${(bits / 8) * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[9px] mt-0.5 text-muted-foreground">
                    <span>2-bit (4 vals)</span>
                    <span>8-bit (256 vals)</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3 p-4 bg-muted/30 rounded-xl border">
                <Label className="font-semibold">
                  Group Size: <span className="text-primary font-mono">{groupSize}</span>
                </Label>
                {/* Clickable group size buttons */}
                <div className="flex gap-2">
                  {[32, 64, 128, 256].map((g) => (
                    <button
                      key={g}
                      onClick={() => setGroupSize(g)}
                      className={`flex-1 py-2 rounded-lg text-sm font-bold transition-all ${
                        groupSize === g
                          ? "bg-primary text-primary-foreground shadow-md scale-105"
                          : "bg-muted hover:bg-muted/80 border"
                      }`}
                    >
                      {g}
                    </button>
                  ))}
                </div>
                <p className="text-[10px] text-muted-foreground">
                  Smaller = better accuracy, more overhead
                </p>
                {/* Visual: group size illustration */}
                <div className="mt-2 p-2 bg-background rounded border">
                  <p className="text-[9px] text-muted-foreground mb-1">Weights grouped for shared scale:</p>
                  <div className="flex gap-0.5">
                    {Array.from({ length: Math.min(8, Math.ceil(256 / groupSize)) }).map((_, i) => (
                      <div key={i} className="flex-1 h-4 rounded-sm" style={{ backgroundColor: COLORS[i % COLORS.length], opacity: 0.6 }} />
                    ))}
                  </div>
                  <p className="text-[9px] text-muted-foreground mt-1">
                    {Math.ceil(2048 / groupSize)} groups per row of 2048 weights
                  </p>
                </div>
              </div>

              <div className="space-y-3 p-4 bg-muted/30 rounded-xl border">
                <Label className="font-semibold">Scheme</Label>
                <div className="flex gap-2 mt-2">
                  {["symmetric", "asymmetric"].map((s) => (
                    <button
                      key={s}
                      onClick={() => setScheme(s)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                        scheme === s
                          ? "bg-primary text-primary-foreground shadow"
                          : "bg-muted hover:bg-muted/80"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                {/* Visual: symmetric vs asymmetric */}
                <div className="mt-2 p-2 bg-background rounded border text-[9px]">
                  {scheme === "symmetric" ? (
                    <div>
                      <p className="font-semibold">Symmetric: zero_point = 0</p>
                      <div className="flex items-center gap-1 mt-1">
                        <span className="text-red-500">-max</span>
                        <div className="flex-1 h-3 bg-gradient-to-r from-red-200 via-gray-100 to-blue-200 rounded-full relative">
                          <div className="absolute inset-y-0 left-1/2 w-0.5 bg-black" />
                        </div>
                        <span className="text-blue-500">+max</span>
                      </div>
                      <p className="text-muted-foreground mt-1">Best for weights (centered at 0)</p>
                    </div>
                  ) : (
                    <div>
                      <p className="font-semibold">Asymmetric: zero_point ≠ 0</p>
                      <div className="flex items-center gap-1 mt-1">
                        <span className="text-red-500">min</span>
                        <div className="flex-1 h-3 bg-gradient-to-r from-amber-200 via-green-200 to-blue-200 rounded-full relative">
                          <div className="absolute inset-y-0 left-1/3 w-0.5 bg-black" />
                        </div>
                        <span className="text-blue-500">max</span>
                      </div>
                      <p className="text-muted-foreground mt-1">Best for activations (shifted range)</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Prompt */}
            <div className="space-y-2">
              <Label className="font-semibold">Test Prompt</Label>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full px-4 py-2 rounded-lg border bg-background focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                placeholder="Enter a prompt to compare generations..."
              />
            </div>

            {/* Action buttons */}
            <div className="flex gap-4">
              <Button
                onClick={handleQuantize}
                disabled={loadingQuantize}
                size="lg"
                className="shadow-md hover:shadow-lg transition-all"
              >
                {loadingQuantize && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {loadingQuantize ? "Quantizing on GPU..." : `Quantize to ${bits}-bit`}
              </Button>

              <Button
                onClick={handleCompare}
                disabled={loadingCompare}
                variant="outline"
                size="lg"
                className="shadow-md hover:shadow-lg transition-all"
              >
                {loadingCompare && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {loadingCompare ? "Comparing (~30s on GPU)..." : "Compare All Methods"}
              </Button>

              <Button
                onClick={handleVisualize}
                disabled={loadingVisualize}
                variant="secondary"
                size="lg"
                className="shadow-md hover:shadow-lg transition-all bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30 border border-purple-200 dark:border-purple-700"
              >
                {loadingVisualize && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {loadingVisualize ? "Reading GPU weights..." : `Inspect Weights (${bits}-bit)`}
              </Button>

              <Button
                onClick={handleDeepVisualize}
                disabled={loadingDeepViz}
                variant="secondary"
                size="lg"
                className="shadow-md hover:shadow-lg transition-all bg-gradient-to-r from-emerald-100 to-cyan-100 dark:from-emerald-900/30 dark:to-cyan-900/30 border border-emerald-200 dark:border-emerald-700"
              >
                {loadingDeepViz && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {loadingDeepViz ? "Computing on GPU..." : `Deep Visualize (${bits}-bit)`}
              </Button>
            </div>

            {loadingQuantize && (
              <div className="p-4 bg-amber-50 dark:bg-amber-950/20 rounded-xl border border-amber-200 dark:border-amber-800">
                <p className="text-sm text-amber-800 dark:text-amber-200 font-semibold mb-2">What&apos;s happening on the GPU right now:</p>
                <ol className="text-xs text-amber-700 dark:text-amber-300 space-y-1 list-decimal list-inside">
                  <li>For each of 156 weight matrices: compute max(|w|) per group of {groupSize}</li>
                  <li>Calculate scale = max / (2^{bits - 1} - 1) = max / {Math.pow(2, bits - 1) - 1}</li>
                  <li>Quantize: q = clamp(round(w / scale), -{Math.pow(2, bits - 1)}, {Math.pow(2, bits - 1) - 1})</li>
                  <li>Dequantize: w&apos; = q * scale (introduces rounding error)</li>
                  <li>Measure perplexity degradation on evaluation text</li>
                </ol>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Single Quantization Results with Visualizations */}
      {quantizeResult && (
        <Card className="border-0 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30">
            <CardTitle>Quantization Results — {quantizeResult.bits}-bit {quantizeResult.scheme}</CardTitle>
          </CardHeader>
          <CardContent className="pt-6 space-y-6">
            {/* Key metrics */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {[
                { label: "Compression", value: `${quantizeResult.compression_ratio}x`, color: "text-green-600" },
                { label: "FP32 Size", value: `${(quantizeResult.fp32_size_mb / 1024).toFixed(1)}GB`, color: "" },
                { label: "Quantized", value: `${(quantizeResult.quantized_size_mb / 1024).toFixed(2)}GB`, color: "text-blue-600" },
                { label: "Perplexity +", value: `${quantizeResult.perplexity_increase_pct}%`, color: quantizeResult.perplexity_increase_pct < 5 ? "text-green-600" : "text-red-600" },
                { label: "GPU Time", value: `${quantizeResult.time_seconds}s`, color: "text-purple-600" },
              ].map((stat) => (
                <div key={stat.label} className="p-3 bg-muted/50 rounded-xl border">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{stat.label}</p>
                  <p className={`text-lg font-bold mt-0.5 ${stat.color}`}>{stat.value}</p>
                </div>
              ))}
            </div>

            {/* VISUALIZATION 1: Memory Size Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-muted/20 rounded-xl p-4 border">
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <HardDrive className="h-4 w-4" />
                  Memory Footprint
                </h4>
                <ResponsiveContainer width="100%" height={150}>
                  <BarChart layout="vertical" data={[
                    { name: "FP32", size: quantizeResult.fp32_size_mb / 1024, fill: "#ef4444" },
                    { name: `INT${quantizeResult.bits}`, size: quantizeResult.quantized_size_mb / 1024, fill: "#10b981" },
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis type="number" tick={{ fontSize: 10 }} unit=" GB" />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fontWeight: "bold" }} width={50} />
                    <Tooltip formatter={(v: number) => [`${v.toFixed(2)} GB`, "Size"]} />
                    <Bar dataKey="size" radius={[0, 6, 6, 0]}>
                      <Cell fill="#ef4444" />
                      <Cell fill="#10b981" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-center text-muted-foreground">
                  Saved: <span className="font-bold text-green-600">{((quantizeResult.fp32_size_mb - quantizeResult.quantized_size_mb) / 1024).toFixed(2)} GB</span> ({((1 - quantizeResult.quantized_size_mb / quantizeResult.fp32_size_mb) * 100).toFixed(0)}% reduction)
                </p>
              </div>

              {/* VISUALIZATION 2: Perplexity Impact */}
              <div className="bg-muted/20 rounded-xl p-4 border">
                <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  Quality Impact (Perplexity)
                </h4>
                <div className="flex items-center justify-center gap-6 py-4">
                  <div className="text-center">
                    <div className="w-20 h-20 rounded-full border-4 border-blue-500 flex items-center justify-center bg-blue-50 dark:bg-blue-950/30">
                      <span className="text-lg font-bold">{quantizeResult.baseline_perplexity.toFixed(1)}</span>
                    </div>
                    <p className="text-xs mt-2 text-muted-foreground">FP32</p>
                  </div>
                  <div className="flex flex-col items-center">
                    <ArrowRight className="h-6 w-6 text-muted-foreground" />
                    <span className={`text-xs font-bold mt-1 px-2 py-0.5 rounded-full ${
                      quantizeResult.perplexity_increase_pct < 3 ? "bg-green-100 text-green-700" :
                      quantizeResult.perplexity_increase_pct < 10 ? "bg-amber-100 text-amber-700" :
                      "bg-red-100 text-red-700"
                    }`}>
                      +{quantizeResult.perplexity_increase_pct}%
                    </span>
                  </div>
                  <div className="text-center">
                    <div className={`w-20 h-20 rounded-full border-4 flex items-center justify-center ${
                      quantizeResult.perplexity_increase_pct < 3 ? "border-green-500 bg-green-50 dark:bg-green-950/30" :
                      quantizeResult.perplexity_increase_pct < 10 ? "border-amber-500 bg-amber-50 dark:bg-amber-950/30" :
                      "border-red-500 bg-red-50 dark:bg-red-950/30"
                    }`}>
                      <span className="text-lg font-bold">{quantizeResult.quantized_perplexity.toFixed(1)}</span>
                    </div>
                    <p className="text-xs mt-2 text-muted-foreground">INT{quantizeResult.bits}</p>
                  </div>
                </div>
                <p className="text-xs text-center text-muted-foreground">
                  Lower perplexity = better. &lt;5% increase is considered negligible.
                </p>
              </div>
            </div>

            {/* VISUALIZATION 3: Most Affected Layers */}
            {quantizeResult.top_degraded_layers && quantizeResult.top_degraded_layers.length > 0 && (
              <div className="bg-muted/20 rounded-xl p-4 border">
                <h4 className="text-sm font-semibold mb-3">
                  Most Affected Layers (highest quantization error)
                </h4>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={quantizeResult.top_degraded_layers.slice(0, 8).map((l) => ({
                    name: l.name.split(".").slice(-3).join("."),
                    mse: l.mse,
                    max_error: l.max_error,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="name" tick={{ fontSize: 8 }} interval={0} angle={-20} height={50} />
                    <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => v.toExponential(1)} />
                    <Tooltip
                      contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid hsl(var(--border))" }}
                      formatter={(v: number) => [v.toExponential(3), ""]}
                    />
                    <Bar dataKey="mse" fill="#f59e0b" radius={[3, 3, 0, 0]} name="MSE" />
                    <Bar dataKey="max_error" fill="#ef4444" radius={[3, 3, 0, 0]} name="Max Error" />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-muted-foreground mt-2 text-center">
                  K_proj (key projection) layers typically suffer most — they have wider weight distributions
                </p>
              </div>
            )}

            {/* VISUALIZATION 4: How Quantization Works diagram */}
            <div className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-900/50 dark:to-slate-800/50 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-3">How It Works: The Quantization Pipeline</h4>
              <div className="flex items-center justify-between gap-2 overflow-x-auto py-2">
                {[
                  { step: "1", title: "Original Weight", desc: `FP32: -0.0156`, color: "bg-blue-500" },
                  { step: "→", title: "Divide by Scale", desc: `w/s = ${(-0.0156 / (0.15 / (Math.pow(2, bits-1)-1))).toFixed(1)}`, color: "bg-purple-500" },
                  { step: "→", title: "Round", desc: `round(${(-0.0156 / (0.15 / (Math.pow(2, bits-1)-1))).toFixed(1)}) = ${Math.round(-0.0156 / (0.15 / (Math.pow(2, bits-1)-1)))}`, color: "bg-amber-500" },
                  { step: "→", title: "Clamp", desc: `[${-Math.pow(2, bits-1)}, ${Math.pow(2, bits-1)-1}]`, color: "bg-red-500" },
                  { step: "→", title: "Stored as INT", desc: `${bits}-bit integer`, color: "bg-green-500" },
                  { step: "→", title: "Dequantize", desc: `q × scale ≈ original`, color: "bg-teal-500" },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-1">
                    {item.step === "→" ? (
                      <ArrowRight className="h-4 w-4 text-muted-foreground shrink-0" />
                    ) : null}
                    {item.step !== "→" && (
                      <div className="text-center min-w-[100px]">
                        <div className={`${item.color} text-white text-[9px] font-bold px-2 py-0.5 rounded-full mb-1`}>
                          Step {item.step}
                        </div>
                        <p className="text-[10px] font-semibold">{item.title}</p>
                        <p className="text-[9px] text-muted-foreground font-mono">{item.desc}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-2 text-center">
                The rounding step introduces irreversible error. With {bits} bits, each weight maps to one of {Math.pow(2, bits)} discrete levels.
              </p>
            </div>

            {/* Generation Comparison */}
            <div className="space-y-3">
              <h4 className="font-semibold text-sm flex items-center gap-2">
                Text Generation Comparison
                <span className="text-xs text-muted-foreground font-normal">
                  (prompt: &ldquo;{quantizeResult.test_prompt}&rdquo;)
                </span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-xl border border-blue-200 dark:border-blue-800">
                  <p className="text-[10px] uppercase tracking-wider text-blue-600 mb-2 font-semibold">Original (FP32) — 4.2 GB</p>
                  <p className="text-sm leading-relaxed">{quantizeResult.original_generation}</p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-xl border border-green-200 dark:border-green-800">
                  <p className="text-[10px] uppercase tracking-wider text-green-600 mb-2 font-semibold">Quantized ({quantizeResult.bits}-bit) — {(quantizeResult.quantized_size_mb / 1024).toFixed(1)} GB</p>
                  <p className="text-sm leading-relaxed">{quantizeResult.quantized_generation}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Multi-method Comparison Results */}
      {compareResult && (
        <Card className="border-0 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/30 dark:to-purple-950/30">
            <CardTitle>Method Comparison — What&apos;s Happening Underneath</CardTitle>
            <CardDescription>
              {compareResult.methods.length} quantization configurations on {(compareResult.num_parameters / 1e9).toFixed(2)}B parameters — same weights, different precision
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-6">

            {/* VISUALIZATION: Tradeoff Chart */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-muted/20 rounded-xl p-4 border">
                <h4 className="text-sm font-semibold mb-3">Size vs Quality Tradeoff</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis type="number" dataKey="size" name="Size" unit=" MB" tick={{ fontSize: 9 }} label={{ value: "Model Size (MB)", position: "insideBottom", offset: -5, style: { fontSize: 10 } }} />
                    <YAxis type="number" dataKey="perplexity" name="Perplexity" tick={{ fontSize: 9 }} label={{ value: "Perplexity", angle: -90, position: "insideLeft", style: { fontSize: 10 } }} domain={["dataMin - 1", "dataMax + 1"]} />
                    <ZAxis type="number" dataKey="bits" range={[100, 400]} />
                    <Tooltip
                      contentStyle={{ fontSize: 11, borderRadius: 8 }}
                      formatter={(v: number, name: string) => [name === "size" ? `${v} MB` : v.toFixed(2), name === "size" ? "Size" : "Perplexity"]}
                    />
                    <Scatter data={[
                      { size: compareResult.fp32_size_mb, perplexity: compareResult.baseline_perplexity, bits: 32, name: "FP32" },
                      ...compareResult.methods.map((m) => ({ size: m.size_mb, perplexity: m.perplexity, bits: m.bits, name: m.label })),
                    ]} fill="#8b5cf6">
                      {[{ bits: 32 }, ...compareResult.methods].map((m, i) => (
                        <Cell key={i} fill={i === 0 ? "#3b82f6" : COLORS[(i - 1) % COLORS.length]} />
                      ))}
                    </Scatter>
                    <ReferenceLine y={compareResult.baseline_perplexity} stroke="#3b82f6" strokeDasharray="5 5" label={{ value: "FP32 baseline", position: "right", style: { fontSize: 9 } }} />
                  </ScatterChart>
                </ResponsiveContainer>
                <p className="text-xs text-center text-muted-foreground">
                  Ideal = bottom-left (small size, low perplexity). Points near the baseline = lossless compression.
                </p>
              </div>

              <div className="bg-muted/20 rounded-xl p-4 border">
                <h4 className="text-sm font-semibold mb-3">Compression vs Error (Radar)</h4>
                <ResponsiveContainer width="100%" height={250}>
                  <RadarChart data={compareResult.methods.map((m) => ({
                    method: m.label.replace("Symmetric ", "S").replace("Asymmetric", "Asym").replace("(group=", "g").replace(")", ""),
                    compression: m.compression_ratio,
                    quality: Math.max(0, 10 - m.perplexity_increase * 2),
                    speed: 10 - m.time_seconds,
                  }))}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="method" tick={{ fontSize: 8 }} />
                    <PolarRadiusAxis tick={{ fontSize: 8 }} />
                    <Radar name="Compression" dataKey="compression" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                    <Radar name="Quality (10 - ppl_increase)" dataKey="quality" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Perplexity Bar Chart */}
            <div className="bg-muted/20 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-3">Perplexity by Method (lower = better)</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={[
                  { name: "FP32\n(baseline)", perplexity: compareResult.baseline_perplexity, fill: "#3b82f6" },
                  ...compareResult.methods.map((m, i) => ({
                    name: m.label.replace(" Symmetric", "").replace(" (", "\n("),
                    perplexity: m.perplexity,
                    fill: COLORS[i % COLORS.length],
                  })),
                ]}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-15} />
                  <YAxis domain={[0, "dataMax + 5"]} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v: number) => [v.toFixed(4), "Perplexity"]} contentStyle={{ borderRadius: 8 }} />
                  <ReferenceLine y={compareResult.baseline_perplexity} stroke="#3b82f6" strokeDasharray="5 5" />
                  <Bar dataKey="perplexity" radius={[4, 4, 0, 0]}>
                    <Cell fill="#3b82f6" />
                    {compareResult.methods.map((_, idx) => (
                      <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-center text-muted-foreground">
                INT3 explodes because 8 discrete levels cannot represent nuanced weight differences. INT4/INT8 are nearly lossless.
              </p>
            </div>

            {/* VISUALIZATION: What Each Bit-Width Means */}
            <div className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-900/50 dark:to-slate-800/50 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-3">What&apos;s Happening: Quantization Grid Resolution</h4>
              <div className="space-y-3">
                {[
                  { bits: 8, levels: 256, desc: "Fine grid — barely any rounding error" },
                  { bits: 4, levels: 16, desc: "Coarse grid — some values merge together" },
                  { bits: 3, levels: 8, desc: "Very coarse — many distinct weights collapse to same integer" },
                ].map((item) => (
                  <div key={item.bits} className="flex items-center gap-3">
                    <span className="text-xs font-mono font-bold w-12">{item.bits}-bit</span>
                    <div className="flex-1 h-6 bg-muted rounded relative overflow-hidden">
                      {Array.from({ length: Math.min(item.levels, 32) }).map((_, i) => (
                        <div
                          key={i}
                          className="absolute top-0 bottom-0 w-px bg-primary/60"
                          style={{ left: `${(i / Math.min(item.levels, 32)) * 100}%` }}
                        />
                      ))}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-[9px] font-bold bg-background/80 px-1 rounded">{item.levels} levels</span>
                      </div>
                    </div>
                    <span className="text-[10px] text-muted-foreground w-48">{item.desc}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                Each vertical line = one representable value. A float32 weight gets snapped to the nearest line.
                The distance between the original value and the line is the <strong>quantization error</strong>.
              </p>
            </div>

            {/* Results table */}
            <div className="overflow-x-auto rounded-xl border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-muted/50">
                    <th className="text-left py-3 px-4 font-semibold">Method</th>
                    <th className="text-center py-3 px-4 font-semibold">Bits</th>
                    <th className="text-right py-3 px-4 font-semibold">Perplexity</th>
                    <th className="text-right py-3 px-4 font-semibold">Increase</th>
                    <th className="text-right py-3 px-4 font-semibold">Size (MB)</th>
                    <th className="text-center py-3 px-4 font-semibold">Compression</th>
                    <th className="text-right py-3 px-4 font-semibold">MSE</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t bg-blue-50/30 dark:bg-blue-950/10">
                    <td className="py-2 px-4 font-medium">FP32 Baseline</td>
                    <td className="text-center py-2 px-4">32</td>
                    <td className="text-right py-2 px-4 font-mono font-bold">{compareResult.baseline_perplexity}</td>
                    <td className="text-right py-2 px-4">—</td>
                    <td className="text-right py-2 px-4 font-mono">{compareResult.fp32_size_mb}</td>
                    <td className="text-center py-2 px-4">1.0x</td>
                    <td className="text-right py-2 px-4">0</td>
                  </tr>
                  {compareResult.methods.map((m, idx) => (
                    <tr key={idx} className="border-t hover:bg-primary/5 transition-colors">
                      <td className="py-2 px-4 font-medium flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                        {m.label}
                      </td>
                      <td className="text-center py-2 px-4 font-mono">{m.bits}</td>
                      <td className="text-right py-2 px-4 font-mono font-bold">{m.perplexity}</td>
                      <td className="text-right py-2 px-4">
                        <span className={m.perplexity_increase < 0.5 ? "text-green-600" : m.perplexity_increase < 2 ? "text-amber-600" : "text-red-600"}>
                          {m.perplexity_increase >= 0 ? "+" : ""}{m.perplexity_increase.toFixed(4)}
                        </span>
                      </td>
                      <td className="text-right py-2 px-4 font-mono">{m.size_mb}</td>
                      <td className="text-center py-2 px-4">
                        <span className="inline-flex items-center rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-0.5 text-xs font-bold text-green-800 dark:text-green-200">
                          {m.compression_ratio}x
                        </span>
                      </td>
                      <td className="text-right py-2 px-4 font-mono text-xs">{m.mse.toExponential(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Generation samples */}
            <div className="space-y-3">
              <h4 className="font-semibold text-sm">Generated Text Samples (prompt: &ldquo;{compareResult.test_prompt}&rdquo;)</h4>
              <div className="grid grid-cols-1 gap-2">
                {compareResult.methods.map((m, idx) => (
                  <div key={idx} className="p-3 rounded-lg border hover:border-primary/20 transition-colors">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                      <span className="text-xs font-bold">{m.label}</span>
                      <span className="text-[10px] text-muted-foreground">({m.bits}-bit, {m.compression_ratio}x compression)</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{m.generation}</p>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Weight Visualization - What NVIDIA GPUs actually do */}
      {visualizeResult && (
        <Card className="border-0 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/30 dark:to-pink-950/30">
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-purple-600" />
              Inside the GPU: Real Weight Values Before &amp; After Quantization
            </CardTitle>
            <CardDescription>
              Layer: <code className="bg-muted px-1 rounded">{visualizeResult.layer_name}</code> — {visualizeResult.num_elements.toLocaleString()} weights, {visualizeResult.bits}-bit
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-6">

            {/* Key stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 bg-blue-50 dark:bg-blue-950/20 rounded-xl border border-blue-200 dark:border-blue-800">
                <p className="text-[10px] uppercase tracking-wider text-blue-600">Unique values BEFORE</p>
                <p className="text-lg font-bold text-blue-700">{visualizeResult.num_unique_original.toLocaleString()}</p>
                <p className="text-[9px] text-muted-foreground">FP32 has billions of possible values</p>
              </div>
              <div className="p-3 bg-green-50 dark:bg-green-950/20 rounded-xl border border-green-200 dark:border-green-800">
                <p className="text-[10px] uppercase tracking-wider text-green-600">Unique values AFTER</p>
                <p className="text-lg font-bold text-green-700">{visualizeResult.num_unique_quantized.toLocaleString()}</p>
                <p className="text-[9px] text-muted-foreground">Only {Math.pow(2, visualizeResult.bits)} levels per group</p>
              </div>
              <div className="p-3 bg-amber-50 dark:bg-amber-950/20 rounded-xl border border-amber-200 dark:border-amber-800">
                <p className="text-[10px] uppercase tracking-wider text-amber-600">Avg Error per Weight</p>
                <p className="text-lg font-bold text-amber-700">{visualizeResult.mean_abs_error.toExponential(2)}</p>
                <p className="text-[9px] text-muted-foreground">How much each value shifted</p>
              </div>
              <div className="p-3 bg-red-50 dark:bg-red-950/20 rounded-xl border border-red-200 dark:border-red-800">
                <p className="text-[10px] uppercase tracking-wider text-red-600">Worst-case Error</p>
                <p className="text-lg font-bold text-red-700">{visualizeResult.max_error.toFixed(4)}</p>
                <p className="text-[9px] text-muted-foreground">Max rounding damage</p>
              </div>
            </div>

            {/* Before vs After Histogram */}
            <div className="bg-muted/20 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-1">Weight Distribution: Before (blue) vs After (green) Quantization</h4>
              <p className="text-xs text-muted-foreground mb-3">
                The smooth blue curve shows original FP32 weights. The spiky green shows quantized values — they cluster at discrete grid points.
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={visualizeResult.original_histogram.centers.map((c, i) => ({
                  x: c,
                  original: visualizeResult.original_histogram.counts[i],
                  quantized: visualizeResult.quantized_histogram.counts[i] || 0,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis
                    dataKey="x"
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    tick={{ fontSize: 9 }}
                    tickFormatter={(v) => v.toFixed(3)}
                    label={{ value: "Weight Value", position: "insideBottom", offset: -5, style: { fontSize: 10 } }}
                  />
                  <YAxis tick={{ fontSize: 9 }} label={{ value: "Count", angle: -90, position: "insideLeft", style: { fontSize: 9 } }} />
                  <Tooltip contentStyle={{ fontSize: 10, borderRadius: 8 }} />
                  <Area
                    dataKey="original"
                    type="monotone"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.3}
                    name="Original (FP32)"
                  />
                  <Area
                    dataKey="quantized"
                    type="monotone"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.3}
                    name={`Quantized (${visualizeResult.bits}-bit)`}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Scatter: Original vs Quantized values */}
            <div className="bg-muted/20 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-1">Each Dot = One Weight: Original (x) vs Quantized (y)</h4>
              <p className="text-xs text-muted-foreground mb-3">
                Perfect quantization = all dots on the diagonal line. Dots off the line = rounding error. Notice the &quot;staircase&quot; — quantized values snap to discrete levels.
              </p>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis type="number" dataKey="x" name="Original" tick={{ fontSize: 9 }} domain={["dataMin", "dataMax"]}
                    label={{ value: "Original FP32 Value", position: "insideBottom", offset: -5, style: { fontSize: 10 } }} />
                  <YAxis type="number" dataKey="y" name="Quantized" tick={{ fontSize: 9 }} domain={["dataMin", "dataMax"]}
                    label={{ value: "After Quantization", angle: -90, position: "insideLeft", style: { fontSize: 10 } }} />
                  <Tooltip
                    contentStyle={{ fontSize: 10, borderRadius: 8 }}
                    formatter={(v: number, name: string) => [v.toFixed(6), name === "x" ? "Original" : "Quantized"]}
                  />
                  <ReferenceLine segment={[
                    { x: visualizeResult.statistics.original_min, y: visualizeResult.statistics.original_min },
                    { x: visualizeResult.statistics.original_max, y: visualizeResult.statistics.original_max },
                  ]} stroke="#ef4444" strokeDasharray="5 5" />
                  <Scatter
                    data={visualizeResult.sample_original.map((o, i) => ({ x: o, y: visualizeResult.sample_quantized[i] }))}
                    fill="#8b5cf6"
                    fillOpacity={0.5}
                    r={2}
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-xs text-center text-muted-foreground">
                Red dashed line = perfect (no error). Purple dots = actual quantized values. The horizontal &quot;bands&quot; are the {Math.pow(2, visualizeResult.bits)} quantization levels.
              </p>
            </div>

            {/* Error Distribution */}
            <div className="bg-muted/20 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-1">Quantization Error Distribution (the &quot;noise&quot; added to every weight)</h4>
              <p className="text-xs text-muted-foreground mb-3">
                This is the damage. Each weight shifts by this much. Centered at 0 = no systematic bias, just random rounding noise.
              </p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={visualizeResult.error_histogram.centers.map((c, i) => ({
                  x: c,
                  count: visualizeResult.error_histogram.counts[i],
                }))}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis dataKey="x" tick={{ fontSize: 9 }} tickFormatter={(v) => v.toFixed(4)}
                    label={{ value: "Error (original - quantized)", position: "insideBottom", offset: -5, style: { fontSize: 10 } }} />
                  <YAxis tick={{ fontSize: 9 }} />
                  <Tooltip contentStyle={{ fontSize: 10, borderRadius: 8 }} formatter={(v: number) => [v, "Count"]} />
                  <ReferenceLine x={0} stroke="#ef4444" strokeWidth={2} />
                  <Bar dataKey="count" fill="#f59e0b" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-xs text-center text-muted-foreground">
                Error std: <span className="font-mono font-bold">{visualizeResult.statistics.error_std.toExponential(2)}</span> — this is the &quot;noise floor&quot; that the model must tolerate.
              </p>
            </div>

            {/* Quantization Grid */}
            <div className="bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-900/50 dark:to-slate-800/50 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-2">The Quantization Grid (actual values from the GPU)</h4>
              <p className="text-xs text-muted-foreground mb-3">
                These are the ONLY values that can exist after quantization. Every original weight gets snapped to the nearest one:
              </p>
              <div className="relative h-12 bg-muted rounded-lg overflow-hidden border">
                {visualizeResult.grid_lines.map((v, i) => {
                  const min = visualizeResult.statistics.original_min;
                  const max = visualizeResult.statistics.original_max;
                  const pct = ((v - min) / (max - min)) * 100;
                  return (
                    <div
                      key={i}
                      className="absolute top-0 bottom-0 w-px bg-purple-500/70"
                      style={{ left: `${Math.max(0, Math.min(100, pct))}%` }}
                    />
                  );
                })}
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-xs font-bold bg-background/90 px-2 py-0.5 rounded shadow">
                    {visualizeResult.grid_lines.length} visible grid lines ({Math.pow(2, visualizeResult.bits)} per group × {Math.ceil(2048 / (visualizeResult.shape[1] > 0 ? groupSize : 128))} groups)
                  </span>
                </div>
              </div>
              <div className="flex justify-between text-[9px] text-muted-foreground mt-1">
                <span>{visualizeResult.statistics.original_min.toFixed(4)}</span>
                <span>0</span>
                <span>{visualizeResult.statistics.original_max.toFixed(4)}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                With {visualizeResult.bits}-bit: max gap between grid lines ≈ {((visualizeResult.statistics.original_max - visualizeResult.statistics.original_min) / Math.pow(2, visualizeResult.bits)).toFixed(5)} — any original value within this gap maps to the same integer.
              </p>
            </div>

            {/* Sample values table */}
            <div className="bg-muted/20 rounded-xl p-4 border">
              <h4 className="text-sm font-semibold mb-2">Sample Weight Values (first 10 from layer)</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="bg-muted/50">
                      <th className="py-2 px-3 text-left">#</th>
                      <th className="py-2 px-3 text-right">Original (FP32)</th>
                      <th className="py-2 px-3 text-right">Quantized ({visualizeResult.bits}-bit)</th>
                      <th className="py-2 px-3 text-right">Error</th>
                      <th className="py-2 px-3 text-left">Visual</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visualizeResult.sample_original.slice(0, 10).map((orig, i) => {
                      const quant = visualizeResult.sample_quantized[i];
                      const err = Math.abs(orig - quant);
                      const errPct = visualizeResult.statistics.original_std > 0 ? (err / visualizeResult.statistics.original_std) * 100 : 0;
                      return (
                        <tr key={i} className="border-t">
                          <td className="py-1.5 px-3 text-muted-foreground">{i + 1}</td>
                          <td className="py-1.5 px-3 text-right text-blue-600">{orig.toFixed(6)}</td>
                          <td className="py-1.5 px-3 text-right text-green-600">{quant.toFixed(6)}</td>
                          <td className={`py-1.5 px-3 text-right ${err > visualizeResult.mean_abs_error * 3 ? "text-red-600 font-bold" : "text-amber-600"}`}>
                            {err.toExponential(2)}
                          </td>
                          <td className="py-1.5 px-3">
                            <div className="w-20 h-2 bg-muted rounded-full overflow-hidden">
                              <div className="h-full bg-red-400 rounded-full" style={{ width: `${Math.min(100, errPct)}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                This is exactly what NVIDIA&apos;s TensorRT / CUDA quantization kernels do — for every single one of the {visualizeResult.num_elements.toLocaleString()} weights in this layer.
                The entire model has 156 layers × millions of weights = {(1.1e9).toLocaleString()} operations like this.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Deep Visualization: Number Line + Heatmap + Token Probs */}
      {deepVizResult && (
        <Card className="border-0 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-emerald-50 to-cyan-50 dark:from-emerald-950/30 dark:to-cyan-950/30">
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-emerald-600" />
              Deep Visualization — What Quantization Does to Each Weight &amp; Each Prediction
            </CardTitle>
            <CardDescription>
              {deepVizResult.bits}-bit {deepVizResult.scheme} — real values from NVIDIA L4 GPU
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-8">

            {/* === 1. NUMBER LINE SNAP === */}
            <div className="space-y-3">
              <h4 className="text-base font-bold flex items-center gap-2">
                1. Number Line Snap — Each Weight Snaps to Nearest Grid Point
              </h4>
              <p className="text-xs text-muted-foreground">
                Each line below is ONE weight value. Blue dot = original FP32. Green dot = where it gets snapped.
                The vertical gray lines are the only {Math.pow(2, deepVizResult.bits)} allowed values per group.
              </p>
              <div className="bg-muted/20 rounded-xl p-4 border overflow-hidden">
                {/* Grid lines */}
                <div className="relative h-8 mb-2 border-b">
                  {deepVizResult.numberline.grid_lines.map((g: number, i: number) => {
                    const min = deepVizResult.numberline.range_min;
                    const max = deepVizResult.numberline.range_max;
                    const pct = ((g - min) / (max - min)) * 100;
                    return (
                      <div key={i} className="absolute top-0 bottom-0 w-px bg-gray-300 dark:bg-gray-600"
                        style={{ left: `${Math.max(0, Math.min(100, pct))}%` }}>
                        <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[7px] text-muted-foreground">
                          {i % 3 === 0 ? g.toFixed(3) : ""}
                        </span>
                      </div>
                    );
                  })}
                  <div className="absolute bottom-0 left-0 right-0 text-[8px] text-center text-muted-foreground">
                    Quantization Grid ({deepVizResult.numberline.grid_lines.length} levels visible)
                  </div>
                </div>

                {/* Weight points with arrows */}
                <div className="space-y-1">
                  {deepVizResult.numberline.points.slice(0, 15).map((p: any, i: number) => {
                    const min = deepVizResult.numberline.range_min;
                    const max = deepVizResult.numberline.range_max;
                    const origPct = ((p.original - min) / (max - min)) * 100;
                    const quantPct = ((p.quantized - min) / (max - min)) * 100;
                    const moved = Math.abs(origPct - quantPct) > 0.5;
                    return (
                      <div key={i} className="relative h-4 border-b border-dashed border-muted">
                        {/* Original position (blue) */}
                        <div className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-blue-500 border border-white shadow-sm z-10"
                          style={{ left: `${Math.max(0, Math.min(99, origPct))}%` }}
                          title={`Original: ${p.original.toFixed(6)}`} />
                        {/* Arrow to quantized position */}
                        {moved && (
                          <div className="absolute top-1/2 h-px bg-red-400"
                            style={{
                              left: `${Math.min(origPct, quantPct)}%`,
                              width: `${Math.abs(origPct - quantPct)}%`,
                            }} />
                        )}
                        {/* Quantized position (green) */}
                        <div className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-green-500 border border-white shadow-sm z-10"
                          style={{ left: `${Math.max(0, Math.min(99, quantPct))}%` }}
                          title={`Quantized: ${p.quantized.toFixed(6)}`} />
                      </div>
                    );
                  })}
                </div>
                <div className="flex items-center gap-4 mt-3 text-[10px]">
                  <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-blue-500" /> Original FP32</span>
                  <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-green-500" /> After Quantization</span>
                  <span className="flex items-center gap-1"><span className="w-4 h-px bg-red-400" /> Movement (error)</span>
                  <span className="flex items-center gap-1"><span className="w-px h-3 bg-gray-400" /> Grid line</span>
                </div>
              </div>
            </div>

            {/* === 2. WEIGHT MATRIX HEATMAP === */}
            <div className="space-y-3">
              <h4 className="text-base font-bold flex items-center gap-2">
                2. Weight Matrix Heatmap — Like Reducing Colors in an Image
              </h4>
              <p className="text-xs text-muted-foreground">
                A 16×16 slice of real weights. Left = original (smooth gradients). Right = quantized (visible &quot;banding&quot; from limited precision).
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Original heatmap */}
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-blue-600 text-center">Original (FP32)</p>
                  <div className="grid gap-px bg-border rounded overflow-hidden" style={{ gridTemplateColumns: `repeat(${deepVizResult.heatmap.size}, 1fr)` }}>
                    {deepVizResult.heatmap.original.flat().map((v: number, i: number) => {
                      const [min, max] = deepVizResult.heatmap.value_range;
                      const norm = (v - min) / (max - min);
                      const r = Math.round(norm < 0.5 ? 0 : (norm - 0.5) * 2 * 255);
                      const b = Math.round(norm > 0.5 ? 0 : (0.5 - norm) * 2 * 255);
                      const g = Math.round(norm > 0.3 && norm < 0.7 ? (1 - Math.abs(norm - 0.5) * 4) * 180 : 0);
                      return <div key={i} className="aspect-square" style={{ backgroundColor: `rgb(${r},${g},${b})` }} />;
                    })}
                  </div>
                </div>
                {/* Quantized heatmap */}
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-green-600 text-center">Quantized ({deepVizResult.bits}-bit)</p>
                  <div className="grid gap-px bg-border rounded overflow-hidden" style={{ gridTemplateColumns: `repeat(${deepVizResult.heatmap.size}, 1fr)` }}>
                    {deepVizResult.heatmap.quantized.flat().map((v: number, i: number) => {
                      const [min, max] = deepVizResult.heatmap.value_range;
                      const norm = (v - min) / (max - min);
                      const r = Math.round(norm < 0.5 ? 0 : (norm - 0.5) * 2 * 255);
                      const b = Math.round(norm > 0.5 ? 0 : (0.5 - norm) * 2 * 255);
                      const g = Math.round(norm > 0.3 && norm < 0.7 ? (1 - Math.abs(norm - 0.5) * 4) * 180 : 0);
                      return <div key={i} className="aspect-square" style={{ backgroundColor: `rgb(${r},${g},${b})` }} />;
                    })}
                  </div>
                </div>
                {/* Error heatmap */}
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-red-600 text-center">Error (difference)</p>
                  <div className="grid gap-px bg-border rounded overflow-hidden" style={{ gridTemplateColumns: `repeat(${deepVizResult.heatmap.size}, 1fr)` }}>
                    {deepVizResult.heatmap.error.flat().map((v: number, i: number) => {
                      const maxErr = deepVizResult.heatmap.error.flat().reduce((a: number, b: number) => Math.max(a, b), 0.001);
                      const intensity = Math.min(255, Math.round((v / maxErr) * 255));
                      return <div key={i} className="aspect-square" style={{ backgroundColor: `rgb(${intensity},0,0)` }} />;
                    })}
                  </div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground text-center">
                Like JPEG compression: the original has infinite color gradation, quantized has only {Math.pow(2, deepVizResult.bits)} &quot;colors&quot; per group. The error map shows where the damage is.
              </p>
            </div>

            {/* === 3. TOKEN PROBABILITY SHIFT === */}
            <div className="space-y-3">
              <h4 className="text-base font-bold flex items-center gap-2">
                3. Token Prediction Shift — Why Text Changes After Quantization
              </h4>
              <p className="text-xs text-muted-foreground">
                After &quot;{deepVizResult.prompt}&quot;, the model predicts the next word. Quantization shifts these probabilities.
              </p>
              <div className="bg-muted/20 rounded-xl p-4 border">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* FP32 predictions */}
                  <div>
                    <p className="text-xs font-bold text-blue-600 mb-2">FP32 (Original) — Next word predictions:</p>
                    <div className="space-y-1.5">
                      {deepVizResult.token_predictions.fp32_top.slice(0, 7).map((pred: any, i: number) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-[10px] text-muted-foreground w-4">{i + 1}.</span>
                          <div className="flex-1 h-5 bg-muted rounded-full overflow-hidden relative">
                            <div className="h-full bg-blue-400 rounded-full transition-all"
                              style={{ width: `${pred.probability * 100}%` }} />
                            <span className="absolute inset-y-0 left-2 flex items-center text-[10px] font-mono font-bold">
                              &quot;{pred.token}&quot;
                            </span>
                          </div>
                          <span className="text-[10px] font-mono w-12 text-right">{(pred.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {/* Quantized predictions */}
                  <div>
                    <p className="text-xs font-bold text-green-600 mb-2">Quantized ({deepVizResult.bits}-bit) — Same tokens, shifted probabilities:</p>
                    <div className="space-y-1.5">
                      {deepVizResult.token_predictions.quantized_same_tokens.slice(0, 7).map((pred: any, i: number) => {
                        const fp32Prob = deepVizResult.token_predictions.fp32_top[i]?.probability || 0;
                        const diff = pred.probability - fp32Prob;
                        return (
                          <div key={i} className="flex items-center gap-2">
                            <span className="text-[10px] text-muted-foreground w-4">{i + 1}.</span>
                            <div className="flex-1 h-5 bg-muted rounded-full overflow-hidden relative">
                              <div className="h-full bg-green-400 rounded-full transition-all"
                                style={{ width: `${pred.probability * 100}%` }} />
                              <span className="absolute inset-y-0 left-2 flex items-center text-[10px] font-mono font-bold">
                                &quot;{pred.token}&quot;
                              </span>
                            </div>
                            <span className="text-[10px] font-mono w-12 text-right">{(pred.probability * 100).toFixed(1)}%</span>
                            <span className={`text-[9px] font-mono w-10 ${diff > 0 ? "text-green-600" : "text-red-600"}`}>
                              {diff > 0 ? "+" : ""}{(diff * 100).toFixed(1)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Winner comparison */}
                <div className="mt-4 p-3 bg-background rounded-lg border flex items-center justify-between">
                  <div className="text-center">
                    <p className="text-[9px] uppercase text-muted-foreground">FP32 picks</p>
                    <p className="text-lg font-bold text-blue-600">&quot;{deepVizResult.token_predictions.next_word_fp32}&quot;</p>
                  </div>
                  <ArrowRight className="h-5 w-5 text-muted-foreground" />
                  <div className="text-center">
                    <p className="text-[9px] uppercase text-muted-foreground">{deepVizResult.bits}-bit picks</p>
                    <p className="text-lg font-bold text-green-600">&quot;{deepVizResult.token_predictions.next_word_quantized}&quot;</p>
                  </div>
                  <div className="text-center px-3">
                    {deepVizResult.token_predictions.next_word_fp32 === deepVizResult.token_predictions.next_word_quantized ? (
                      <span className="text-xs font-bold px-2 py-1 rounded-full bg-green-100 text-green-700">Same!</span>
                    ) : (
                      <span className="text-xs font-bold px-2 py-1 rounded-full bg-amber-100 text-amber-700">Different!</span>
                    )}
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  When probability shifts are small, the same word wins. When quantization damage is severe (INT3),
                  a different word can become the top prediction — that&apos;s when you see &quot;soup recipe&quot; instead of &quot;fairy tale.&quot;
                  This cascades: each wrong word shifts the context for the next prediction.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
