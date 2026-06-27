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
import { getDecisionGuide } from "@/lib/api";
import type { DecisionGuideRequest, DecisionGuideResponse } from "@/lib/api";
import { Target, Loader2, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";

// --- Local fallback decision logic ---
function localDecisionGuide(request: DecisionGuideRequest): DecisionGuideResponse {
  const { model_size_billions, hardware, accuracy_tolerance, has_calibration_data, use_case } = request;

  let method = "GPTQ";
  let bits = 4;
  let format = "INT4";
  const reasoning: string[] = [];
  const warnings: string[] = [];

  if (accuracy_tolerance === "none") {
    method = "FP16";
    bits = 16;
    format = "Float16";
    reasoning.push("No accuracy loss tolerance requires full precision or near-full precision.");
  } else if (accuracy_tolerance === "minimal" && has_calibration_data) {
    method = "GPTQ";
    bits = 4;
    format = "INT4 (group-128)";
    reasoning.push("GPTQ with calibration data achieves <1% accuracy loss at 4-bit.");
    reasoning.push("Group quantization preserves outlier channels.");
  } else if (accuracy_tolerance === "minimal") {
    method = "AWQ";
    bits = 4;
    format = "INT4";
    reasoning.push("AWQ protects salient weights without needing full calibration data.");
  } else if (accuracy_tolerance === "moderate" || accuracy_tolerance === "aggressive") {
    if (model_size_billions >= 30) {
      method = "GPTQ";
      bits = 3;
      format = "INT3 (group-128)";
      reasoning.push("Large models (30B+) tolerate aggressive quantization with grouping.");
    } else {
      method = "Round-to-Nearest";
      bits = 4;
      format = "INT4";
      reasoning.push("Simpler RTN is sufficient when moderate accuracy loss is acceptable.");
    }
  }

  if (hardware === "edge") {
    format = "INT4";
    bits = 4;
    reasoning.push("Edge devices benefit most from INT4 with hardware-accelerated integer ops.");
    if (model_size_billions > 7) {
      warnings.push("Models larger than 7B may not fit on edge devices even at INT4.");
    }
  }

  if (hardware === "nvidia_gpu" && use_case === "training") {
    method = "QLoRA";
    bits = 4;
    format = "NF4";
    reasoning.push("QLoRA with NF4 format enables fine-tuning large models on a single GPU.");
  }

  if (model_size_billions > 70 && hardware === "cpu") {
    warnings.push("Running 70B+ models on CPU will be extremely slow regardless of quantization.");
  }

  const compression = (32 / bits).toFixed(1) + "x";
  const accuracyLoss = bits >= 16 ? "~0%" : bits >= 8 ? "<0.5%" : bits >= 4 ? "1-3%" : "3-5%";

  const alternatives = [
    {
      method: "SmoothQuant",
      pros: ["No calibration data needed", "Works well for W8A8"],
      cons: ["Limited to INT8", "Requires activation statistics"],
    },
    {
      method: "GGUF (llama.cpp)",
      pros: ["CPU-friendly", "Easy to use", "Many quant levels"],
      cons: ["Not optimal for GPU", "Less research backing"],
    },
    {
      method: "FP8 (E4M3)",
      pros: ["Hardware support on H100+", "Drop-in replacement for FP16"],
      cons: ["Only on newest GPUs", "2x compression only"],
    },
  ];

  return {
    recommended_method: method,
    recommended_bits: bits,
    recommended_format: format,
    expected_compression: compression,
    expected_accuracy_loss: accuracyLoss,
    reasoning,
    alternatives,
    warnings,
  };
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================
export function DecisionGuide() {
  const [formState, setFormState] = useState<DecisionGuideRequest>({
    model_size_billions: 7,
    hardware: "nvidia_gpu",
    latency_budget_ms: null,
    accuracy_tolerance: "minimal",
    has_calibration_data: true,
    use_case: "inference",
  });
  const [result, setResult] = useState<DecisionGuideResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedAlt, setExpandedAlt] = useState<number | null>(null);

  const handleSubmit = async () => {
    setIsLoading(true);
    try {
      const res = await getDecisionGuide(formState);
      setResult(res);
    } catch {
      setResult(localDecisionGuide(formState));
    } finally {
      setIsLoading(false);
    }
  };

  // Log scale for model size
  const modelSizeValues = [0.1, 0.5, 1, 1.5, 2, 3, 7, 13, 30, 65, 70, 120, 180, 405];
  const closestIdx = modelSizeValues.reduce(
    (best, v, i) => (Math.abs(v - formState.model_size_billions) < Math.abs(modelSizeValues[best] - formState.model_size_billions) ? i : best),
    0
  );

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-lg overflow-hidden rounded-xl">
        <CardHeader className="bg-gradient-to-r from-sky-50 to-indigo-50 dark:from-sky-950/30 dark:to-indigo-950/30">
          <div className="flex items-center gap-2">
            <Target className="h-5 w-5 text-sky-600" />
            <CardTitle className="text-xl">Quantization Decision Guide</CardTitle>
          </div>
          <CardDescription>
            Answer a few questions about your model and constraints to get a personalized quantization recommendation
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          {/* Form */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Model Size */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Model Size: {formState.model_size_billions}B parameters</label>
              <input
                type="range"
                min={0}
                max={modelSizeValues.length - 1}
                value={closestIdx}
                onChange={(e) =>
                  setFormState((s) => ({ ...s, model_size_billions: modelSizeValues[Number(e.target.value)] }))
                }
                className="w-full"
              />
              <div className="flex justify-between text-[10px] text-muted-foreground">
                <span>0.1B</span>
                <span>7B</span>
                <span>70B</span>
                <span>405B</span>
              </div>
            </div>

            {/* Hardware */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Target Hardware</label>
              <select
                className="w-full p-2 rounded-lg border bg-background text-sm"
                value={formState.hardware}
                onChange={(e) =>
                  setFormState((s) => ({ ...s, hardware: e.target.value as DecisionGuideRequest["hardware"] }))
                }
              >
                <option value="nvidia_gpu">NVIDIA GPU</option>
                <option value="amd_gpu">AMD GPU</option>
                <option value="cpu">CPU</option>
                <option value="edge">Edge Device</option>
              </select>
            </div>

            {/* Accuracy Tolerance */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Accuracy Tolerance</label>
              <div className="grid grid-cols-2 gap-2">
                {(["none", "minimal", "moderate", "aggressive"] as const).map((level) => (
                  <button
                    key={level}
                    onClick={() => setFormState((s) => ({ ...s, accuracy_tolerance: level }))}
                    className={`px-3 py-2 rounded-lg text-xs font-medium border transition-all hover:shadow-md ${
                      formState.accuracy_tolerance === level
                        ? "bg-sky-100 dark:bg-sky-900/40 border-sky-300 dark:border-sky-700 text-sky-700 dark:text-sky-300"
                        : "bg-background border-border hover:bg-muted/50"
                    }`}
                  >
                    {level === "none" && "None (0%)"}
                    {level === "minimal" && "Minimal (<1%)"}
                    {level === "moderate" && "Moderate (1-3%)"}
                    {level === "aggressive" && "Aggressive (>3%)"}
                  </button>
                ))}
              </div>
            </div>

            {/* Use Case */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Use Case</label>
              <div className="flex gap-2">
                {(["inference", "training", "both"] as const).map((uc) => (
                  <button
                    key={uc}
                    onClick={() => setFormState((s) => ({ ...s, use_case: uc }))}
                    className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium border transition-all hover:shadow-md capitalize ${
                      formState.use_case === uc
                        ? "bg-sky-100 dark:bg-sky-900/40 border-sky-300 dark:border-sky-700 text-sky-700 dark:text-sky-300"
                        : "bg-background border-border hover:bg-muted/50"
                    }`}
                  >
                    {uc}
                  </button>
                ))}
              </div>
            </div>

            {/* Latency Budget */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Latency Budget (ms, optional)</label>
              <input
                type="number"
                placeholder="e.g. 100"
                className="w-full p-2 rounded-lg border bg-background text-sm"
                value={formState.latency_budget_ms ?? ""}
                onChange={(e) =>
                  setFormState((s) => ({
                    ...s,
                    latency_budget_ms: e.target.value ? Number(e.target.value) : null,
                  }))
                }
              />
            </div>

            {/* Calibration Data */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Calibration Data Available?</label>
              <div className="flex gap-3">
                <button
                  onClick={() => setFormState((s) => ({ ...s, has_calibration_data: true }))}
                  className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium border transition-all hover:shadow-md ${
                    formState.has_calibration_data
                      ? "bg-green-100 dark:bg-green-900/40 border-green-300 dark:border-green-700 text-green-700 dark:text-green-300"
                      : "bg-background border-border hover:bg-muted/50"
                  }`}
                >
                  Yes
                </button>
                <button
                  onClick={() => setFormState((s) => ({ ...s, has_calibration_data: false }))}
                  className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium border transition-all hover:shadow-md ${
                    !formState.has_calibration_data
                      ? "bg-red-100 dark:bg-red-900/40 border-red-300 dark:border-red-700 text-red-700 dark:text-red-300"
                      : "bg-background border-border hover:bg-muted/50"
                  }`}
                >
                  No
                </button>
              </div>
            </div>
          </div>

          {/* Submit */}
          <div className="flex justify-center pt-4">
            <Button
              onClick={handleSubmit}
              disabled={isLoading}
              size="lg"
              className="px-8 shadow-lg hover:shadow-xl transition-shadow text-base"
            >
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Get Recommendation
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Main recommendation */}
          <Card className="border-0 shadow-lg overflow-hidden rounded-xl border-l-4 border-l-sky-500">
            <CardContent className="pt-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-sky-100 dark:bg-sky-900/40 rounded-xl">
                  <CheckCircle2 className="h-8 w-8 text-sky-600" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold">{result.recommended_method}</h3>
                  <p className="text-muted-foreground text-sm mt-1">
                    {result.recommended_bits}-bit &middot; {result.recommended_format}
                  </p>
                  <div className="flex gap-2 mt-3 flex-wrap">
                    <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full text-xs font-medium">
                      {result.expected_compression} compression
                    </span>
                    <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-xs font-medium">
                      ~{result.expected_accuracy_loss} accuracy loss
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Reasoning */}
          <Card className="border-0 shadow-lg rounded-xl">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Reasoning</CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="space-y-2">
                {result.reasoning.map((r, i) => (
                  <li key={i} className="flex gap-3 text-sm">
                    <span className="flex-shrink-0 w-6 h-6 bg-sky-100 dark:bg-sky-900/40 text-sky-700 dark:text-sky-300 rounded-full flex items-center justify-center text-xs font-bold">
                      {i + 1}
                    </span>
                    <span>{r}</span>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>

          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="space-y-2">
              {result.warnings.map((w, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 p-3 bg-amber-50 dark:bg-amber-950/30 rounded-lg border border-amber-200 dark:border-amber-800"
                >
                  <AlertTriangle className="h-4 w-4 text-amber-600 flex-shrink-0" />
                  <span className="text-sm text-amber-700 dark:text-amber-300">{w}</span>
                </div>
              ))}
            </div>
          )}

          {/* Alternatives */}
          <Card className="border-0 shadow-lg rounded-xl">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Alternatives</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {result.alternatives.map((alt, i) => (
                <div key={i} className="border rounded-lg overflow-hidden">
                  <button
                    onClick={() => setExpandedAlt(expandedAlt === i ? null : i)}
                    className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors text-left"
                  >
                    <span className="font-medium text-sm">{alt.method}</span>
                    {expandedAlt === i ? (
                      <ChevronUp className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>
                  {expandedAlt === i && (
                    <div className="px-3 pb-3 grid grid-cols-2 gap-4 text-xs">
                      <div>
                        <p className="font-medium text-green-700 dark:text-green-400 mb-1">Pros</p>
                        <ul className="space-y-0.5">
                          {alt.pros.map((p, j) => (
                            <li key={j} className="text-muted-foreground">+ {p}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="font-medium text-red-700 dark:text-red-400 mb-1">Cons</p>
                        <ul className="space-y-0.5">
                          {alt.cons.map((c, j) => (
                            <li key={j} className="text-muted-foreground">- {c}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
