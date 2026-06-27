"use client";

import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  quantizeSymmetric,
  quantizeAsymmetric,
  quantizeClipped,
  quantizeBfloat16,
} from "@/lib/api";
import type { QuantizeResponse, Bfloat16Response } from "@/lib/api";
import { useQuantizationStore } from "@/store/quantizationStore";
import { formatScientific } from "@/lib/utils";
import { Loader2, Trophy, Scale } from "lucide-react";

export function QuantizationComparison() {
  const { weights, clipMin, clipMax, setBfloat16Result } =
    useQuantizationStore();

  const { data: symmetric, refetch: refetchSym, isLoading: loadingSym } = useQuery({
    queryKey: ["quantize-symmetric", weights],
    queryFn: () => quantizeSymmetric({ weights }),
    enabled: false,
  });

  const { data: asymmetric, refetch: refetchAsym, isLoading: loadingAsym } = useQuery({
    queryKey: ["quantize-asymmetric", weights],
    queryFn: () => quantizeAsymmetric({ weights }),
    enabled: false,
  });

  const { data: clipped, refetch: refetchClip, isLoading: loadingClip } = useQuery({
    queryKey: ["quantize-clipped", weights, clipMin, clipMax],
    queryFn: () => quantizeClipped({ weights, clip_min: clipMin, clip_max: clipMax }),
    enabled: false,
  });

  const { data: bfloat16, refetch: refetchBf16, isLoading: loadingBf16 } = useQuery({
    queryKey: ["quantize-bfloat16", weights],
    queryFn: () => quantizeBfloat16({ weights }),
    enabled: false,
  });

  const isLoading = loadingSym || loadingAsym || loadingClip || loadingBf16;

  const handleCompare = async () => {
    const results = await Promise.all([
      refetchSym(),
      refetchAsym(),
      refetchClip(),
      refetchBf16(),
    ]);
    if (results[3].data) {
      setBfloat16Result(results[3].data);
    }
  };

  const int8Results: {
    method: string;
    bits: number;
    data: QuantizeResponse | undefined;
    color: string;
  }[] = [
    { method: "Symmetric INT8", bits: 8, data: symmetric, color: "#3b82f6" },
    { method: "Asymmetric INT8", bits: 8, data: asymmetric, color: "#10b981" },
    { method: "Clipped INT8", bits: 8, data: clipped, color: "#f59e0b" },
  ];

  const hasAnyResult = symmetric || asymmetric || clipped || bfloat16;

  const allMSEs: { method: string; mse: number; color: string }[] = [];
  if (symmetric) allMSEs.push({ method: "Symmetric INT8", mse: symmetric.mse, color: "#3b82f6" });
  if (asymmetric) allMSEs.push({ method: "Asymmetric INT8", mse: asymmetric.mse, color: "#10b981" });
  if (clipped) allMSEs.push({ method: "Clipped INT8", mse: clipped.mse, color: "#f59e0b" });
  if (bfloat16) allMSEs.push({ method: "BFloat16", mse: bfloat16.mse, color: "#8b5cf6" });

  const bestMethod = allMSEs.length > 0
    ? allMSEs.reduce((a, b) => (a.mse < b.mse ? a : b))
    : null;

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-lg overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-950/30 dark:to-blue-950/30 pb-4">
          <CardTitle className="text-2xl flex items-center gap-2">
            <Scale className="h-5 w-5 text-indigo-600" />
            Quantization Method Comparison
          </CardTitle>
          <CardDescription className="text-base">
            Compare error metrics across INT8 symmetric, asymmetric, clipped
            quantization and BFloat16 casting
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            <Button
              onClick={handleCompare}
              disabled={weights.length === 0 || isLoading}
              size="lg"
              className="shadow-md hover:shadow-lg transition-all"
            >
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Compare All Methods
            </Button>
            {weights.length === 0 && (
              <p className="text-sm text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                Load weights from the Weights tab first
              </p>
            )}
          </div>

          {hasAnyResult && (
            <div className="mt-6 space-y-6">
              {/* Comparison table */}
              <div className="overflow-x-auto rounded-xl border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-muted/50">
                      <th className="text-left py-3 px-4 font-semibold">Method</th>
                      <th className="text-center py-3 px-4 font-semibold">Bits</th>
                      <th className="text-right py-3 px-4 font-semibold">MSE</th>
                      <th className="text-right py-3 px-4 font-semibold">RSS</th>
                      <th className="text-right py-3 px-4 font-semibold">Scale</th>
                      <th className="text-right py-3 px-4 font-semibold">Zero Point</th>
                      <th className="text-center py-3 px-4 font-semibold">Savings</th>
                    </tr>
                  </thead>
                  <tbody>
                    {int8Results.map(
                      (result) =>
                        result.data && (
                          <tr
                            key={result.method}
                            className={`border-t hover:bg-primary/5 transition-colors cursor-default ${
                              bestMethod?.method === result.method ? "bg-green-50/50 dark:bg-green-950/10" : ""
                            }`}
                          >
                            <td className="py-3 px-4 font-medium flex items-center gap-2">
                              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: result.color }} />
                              {result.method}
                              {bestMethod?.method === result.method && (
                                <Trophy className="h-3.5 w-3.5 text-amber-500" />
                              )}
                            </td>
                            <td className="text-center py-3 px-4 font-mono">{result.bits}</td>
                            <td className="text-right py-3 px-4 font-mono text-xs">
                              {formatScientific(result.data.mse)}
                            </td>
                            <td className="text-right py-3 px-4 font-mono text-xs">
                              {result.data.rss.toFixed(4)}
                            </td>
                            <td className="text-right py-3 px-4 font-mono text-xs">
                              {result.data.scale.toFixed(6)}
                            </td>
                            <td className="text-right py-3 px-4 font-mono">
                              {result.data.zero_point}
                            </td>
                            <td className="text-center py-3 px-4">
                              <span className="inline-flex items-center rounded-full bg-green-100 dark:bg-green-900/30 px-2.5 py-1 text-xs font-bold text-green-800 dark:text-green-200">
                                4x
                              </span>
                            </td>
                          </tr>
                        )
                    )}
                    {bfloat16 && (
                      <tr className={`border-t hover:bg-primary/5 transition-colors cursor-default ${
                        bestMethod?.method === "BFloat16" ? "bg-green-50/50 dark:bg-green-950/10" : ""
                      }`}>
                        <td className="py-3 px-4 font-medium flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full bg-purple-500" />
                          BFloat16
                          {bestMethod?.method === "BFloat16" && (
                            <Trophy className="h-3.5 w-3.5 text-amber-500" />
                          )}
                        </td>
                        <td className="text-center py-3 px-4 font-mono">16</td>
                        <td className="text-right py-3 px-4 font-mono text-xs">
                          {formatScientific(bfloat16.mse)}
                        </td>
                        <td className="text-right py-3 px-4 font-mono text-xs">
                          {bfloat16.rss.toFixed(4)}
                        </td>
                        <td className="text-right py-3 px-4 text-muted-foreground text-xs">
                          N/A (cast)
                        </td>
                        <td className="text-right py-3 px-4 text-muted-foreground">
                          N/A
                        </td>
                        <td className="text-center py-3 px-4">
                          <span className="inline-flex items-center rounded-full bg-purple-100 dark:bg-purple-900/30 px-2.5 py-1 text-xs font-bold text-purple-800 dark:text-purple-200">
                            2x
                          </span>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Best method */}
              {bestMethod && (
                <div className="p-4 bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-950/20 dark:to-yellow-950/20 rounded-xl border border-amber-200 dark:border-amber-800 flex items-center gap-3">
                  <Trophy className="h-6 w-6 text-amber-500 shrink-0" />
                  <div>
                    <p className="font-bold">
                      Best Method (Lowest MSE):{" "}
                      <span style={{ color: bestMethod.color }}>
                        {bestMethod.method}
                      </span>
                    </p>
                    <p className="text-sm text-muted-foreground">
                      MSE: {formatScientific(bestMethod.mse)}
                    </p>
                  </div>
                </div>
              )}

              {/* INT8 vs BFloat16 insights */}
              {symmetric && bfloat16 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-5 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-xl hover:shadow-md transition-shadow">
                    <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-3">
                      INT8 Methods
                    </h4>
                    <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-2">
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 shrink-0" />Fixed precision: values map to 256 levels</li>
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 shrink-0" />Uniform error within +/- 0.5 * scale</li>
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 shrink-0" />4x memory reduction</li>
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 shrink-0" />Hardware-accelerated integer math</li>
                    </ul>
                  </div>
                  <div className="p-5 bg-purple-50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-800 rounded-xl hover:shadow-md transition-shadow">
                    <h4 className="font-bold text-purple-900 dark:text-purple-100 mb-3">
                      BFloat16
                    </h4>
                    <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-2">
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 shrink-0" />Preserves FP32 dynamic range</li>
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 shrink-0" />Error concentrated near zero (relative)</li>
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 shrink-0" />2x memory reduction</li>
                      <li className="flex items-start gap-2"><span className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 shrink-0" />Native TPU/GPU hardware support</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
