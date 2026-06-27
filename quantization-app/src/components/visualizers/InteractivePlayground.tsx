"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { quantizeSymmetric, quantizeAsymmetric } from "@/lib/api";
import { useQuantizationStore } from "@/store/quantizationStore";
import { formatScientific } from "@/lib/utils";
import { Gamepad2, Loader2, ArrowRight } from "lucide-react";

export function InteractivePlayground() {
  const { clipMin, clipMax, setClipMin, setClipMax } = useQuantizationStore();
  const [testValue, setTestValue] = useState(0.5);
  const [bits, setBits] = useState(8);
  const [isComparing, setIsComparing] = useState(false);
  const [results, setResults] = useState<{
    symmetric?: { quantized: number; dequantized: number; error: number; scale: number };
    asymmetric?: { quantized: number; dequantized: number; error: number; scale: number; zp: number };
  }>({});

  const qMax = (2 ** (bits - 1)) - 1;
  const scale = Math.abs(testValue) / qMax || 0.001;
  const quantized = Math.round(testValue / scale);
  const clamped = Math.max(-(2 ** (bits - 1)), Math.min(qMax, quantized));
  const dequantized = clamped * scale;
  const error = Math.abs(testValue - dequantized);

  const handleQuantizeArray = async () => {
    const weights = [testValue, -testValue, testValue * 0.5, testValue * 2, 0];
    setIsComparing(true);
    try {
      const [sym, asym] = await Promise.all([
        quantizeSymmetric({ weights, bits }),
        quantizeAsymmetric({ weights, bits }),
      ]);
      setResults({
        symmetric: {
          quantized: sym.quantized_weights[0],
          dequantized: sym.quantized_weights[0] * sym.scale,
          error: Math.abs(testValue - sym.quantized_weights[0] * sym.scale),
          scale: sym.scale,
        },
        asymmetric: {
          quantized: asym.quantized_weights[0],
          dequantized: (asym.quantized_weights[0] - asym.zero_point) * asym.scale,
          error: Math.abs(testValue - (asym.quantized_weights[0] - asym.zero_point) * asym.scale),
          scale: asym.scale,
          zp: asym.zero_point,
        },
      });
    } catch (e) {
      console.error("Quantization failed:", e);
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-lg overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-pink-50 to-rose-50 dark:from-pink-950/30 dark:to-rose-950/30 pb-4">
          <CardTitle className="text-2xl flex items-center gap-2">
            <Gamepad2 className="h-5 w-5 text-pink-600" />
            Interactive Quantization Playground
          </CardTitle>
          <CardDescription className="text-base">
            Adjust parameters in real-time to understand how a single value gets
            mapped to integer representation
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-8">
          {/* Input Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3 p-5 bg-muted/30 rounded-xl border hover:border-primary/20 transition-colors">
              <Label className="text-sm font-semibold">
                Input Value:{" "}
                <span className="text-lg font-mono text-primary">{testValue.toFixed(4)}</span>
              </Label>
              <Slider
                value={[testValue]}
                onValueChange={([v]) => setTestValue(v)}
                min={-2}
                max={2}
                step={0.001}
              />
              <p className="text-[10px] text-muted-foreground">
                Drag to change the float value to quantize
              </p>
            </div>

            <div className="space-y-3 p-5 bg-muted/30 rounded-xl border hover:border-primary/20 transition-colors">
              <Label className="text-sm font-semibold">
                Bit Width:{" "}
                <span className="text-lg font-mono text-primary">{bits}-bit</span>
              </Label>
              <Slider
                value={[bits]}
                onValueChange={([v]) => setBits(v)}
                min={2}
                max={8}
                step={1}
              />
              <p className="text-[10px] text-muted-foreground">
                Range: [{-(2 ** (bits - 1))}, {qMax}] &bull; {2 ** bits} discrete levels
              </p>
            </div>
          </div>

          {/* Live Quantization Pipeline */}
          <div className="p-6 bg-gradient-to-r from-muted/50 to-muted/30 rounded-xl border">
            <h4 className="font-bold mb-4 text-sm uppercase tracking-wider text-muted-foreground">
              Live Quantization Pipeline
            </h4>
            <div className="flex items-center gap-3 flex-wrap">
              {/* Original */}
              <div className="p-3 bg-background rounded-lg border shadow-sm hover:shadow-md transition-shadow min-w-[100px] text-center">
                <p className="text-[9px] uppercase tracking-wider text-muted-foreground">Original</p>
                <p className="text-sm font-mono font-bold mt-1">{testValue.toFixed(4)}</p>
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground shrink-0" />
              {/* Scale */}
              <div className="p-3 bg-background rounded-lg border shadow-sm hover:shadow-md transition-shadow min-w-[100px] text-center">
                <p className="text-[9px] uppercase tracking-wider text-muted-foreground">÷ Scale</p>
                <p className="text-sm font-mono font-bold mt-1">{scale.toFixed(5)}</p>
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground shrink-0" />
              {/* Quantized */}
              <div className="p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg border border-blue-200 dark:border-blue-800 shadow-sm hover:shadow-md transition-shadow min-w-[100px] text-center">
                <p className="text-[9px] uppercase tracking-wider text-blue-600">Quantized</p>
                <p className="text-sm font-mono font-bold text-blue-700 dark:text-blue-300 mt-1">{clamped}</p>
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground shrink-0" />
              {/* Dequantized */}
              <div className="p-3 bg-green-50 dark:bg-green-950/30 rounded-lg border border-green-200 dark:border-green-800 shadow-sm hover:shadow-md transition-shadow min-w-[100px] text-center">
                <p className="text-[9px] uppercase tracking-wider text-green-600">Dequantized</p>
                <p className="text-sm font-mono font-bold text-green-700 dark:text-green-300 mt-1">{dequantized.toFixed(4)}</p>
              </div>
              <ArrowRight className="h-4 w-4 text-muted-foreground shrink-0" />
              {/* Error */}
              <div className="p-3 bg-red-50 dark:bg-red-950/30 rounded-lg border border-red-200 dark:border-red-800 shadow-sm hover:shadow-md transition-shadow min-w-[100px] text-center">
                <p className="text-[9px] uppercase tracking-wider text-red-600">Error</p>
                <p className="text-sm font-mono font-bold text-red-700 dark:text-red-300 mt-1">{error.toFixed(6)}</p>
              </div>
            </div>
          </div>

          {/* Backend Comparison */}
          <div className="space-y-4">
            <Button
              onClick={handleQuantizeArray}
              disabled={isComparing}
              className="shadow-md hover:shadow-lg transition-all"
            >
              {isComparing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Compare Symmetric vs Asymmetric (Backend)
            </Button>

            {results.symmetric && results.asymmetric && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-5 bg-blue-50 dark:bg-blue-950/20 rounded-xl border border-blue-200 dark:border-blue-800 hover:shadow-md transition-shadow">
                  <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-3 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-blue-500" />
                    Symmetric
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between"><span className="text-muted-foreground">Quantized:</span><span className="font-mono font-bold">{results.symmetric.quantized}</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Scale:</span><span className="font-mono">{results.symmetric.scale.toFixed(6)}</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Error:</span><span className="font-mono text-red-600">{formatScientific(results.symmetric.error)}</span></div>
                  </div>
                </div>
                <div className="p-5 bg-green-50 dark:bg-green-950/20 rounded-xl border border-green-200 dark:border-green-800 hover:shadow-md transition-shadow">
                  <h4 className="font-bold text-green-800 dark:text-green-200 mb-3 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-green-500" />
                    Asymmetric
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between"><span className="text-muted-foreground">Quantized:</span><span className="font-mono font-bold">{results.asymmetric.quantized}</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Scale:</span><span className="font-mono">{results.asymmetric.scale.toFixed(6)}</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Zero Point:</span><span className="font-mono">{results.asymmetric.zp}</span></div>
                    <div className="flex justify-between"><span className="text-muted-foreground">Error:</span><span className="font-mono text-red-600">{formatScientific(results.asymmetric.error)}</span></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Clip Configuration */}
      <Card className="border-0 shadow-lg">
        <CardHeader>
          <CardTitle>Clip Range Configuration</CardTitle>
          <CardDescription>
            Configure clip boundaries for the Clipped INT8 method (used in Compare tab)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3 p-4 bg-muted/30 rounded-xl border">
              <Label className="text-sm font-medium">
                Clip Min:{" "}
                <span className="font-mono text-red-600">{clipMin.toFixed(3)}</span>
              </Label>
              <Slider
                value={[clipMin]}
                onValueChange={([v]) => setClipMin(v)}
                min={-1}
                max={0}
                step={0.01}
              />
            </div>
            <div className="space-y-3 p-4 bg-muted/30 rounded-xl border">
              <Label className="text-sm font-medium">
                Clip Max:{" "}
                <span className="font-mono text-green-600">{clipMax.toFixed(3)}</span>
              </Label>
              <Slider
                value={[clipMax]}
                onValueChange={([v]) => setClipMax(v)}
                min={0}
                max={1}
                step={0.01}
              />
            </div>
          </div>

          <div className="p-4 bg-muted/30 rounded-xl border text-sm text-muted-foreground">
            Values outside [{clipMin.toFixed(3)}, {clipMax.toFixed(3)}] will be
            saturated to the boundary. This trades outlier accuracy for better
            precision in the main weight distribution.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
