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
import { getQuantizationSteps } from "@/lib/api";
import { useQuantizationStore } from "@/store/quantizationStore";
import { Loader2, CheckCircle2, XCircle, ArrowDown, Layers } from "lucide-react";

export function StepByStepQuantization() {
  const { weights } = useQuantizationStore();

  const {
    data: stepsData,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["quantization-steps", weights],
    queryFn: () => getQuantizationSteps({ weights: weights.slice(0, 20), bits: 8 }),
    enabled: false,
  });

  const handleRunSteps = () => {
    if (weights.length > 0) {
      refetch();
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-0 shadow-lg overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-950/30 dark:to-teal-950/30 pb-4">
          <CardTitle className="text-2xl flex items-center gap-2">
            <Layers className="h-5 w-5 text-emerald-600" />
            Step-by-Step Symmetric Quantization
          </CardTitle>
          <CardDescription className="text-base">
            Walk through each step of INT8 symmetric quantization. See exactly
            where information is preserved and where it is permanently lost.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-4">
          <div className="flex items-center gap-4">
            <Button
              onClick={handleRunSteps}
              disabled={weights.length === 0 || isLoading}
              size="lg"
              className="shadow-md hover:shadow-lg transition-all"
            >
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Run Step-by-Step
            </Button>
            {weights.length === 0 && (
              <p className="text-sm text-amber-600 dark:text-amber-400 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                Load weights from the Weights tab first
              </p>
            )}
          </div>

          {/* Legend */}
          <div className="flex gap-6 text-sm p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <span className="text-muted-foreground">Reversible (no info lost)</span>
            </div>
            <div className="flex items-center gap-2">
              <XCircle className="h-4 w-4 text-red-600" />
              <span className="text-muted-foreground">Irreversible (info lost)</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Step Cards */}
      {stepsData && (
        <div className="space-y-1">
          {stepsData.steps.map((step, idx) => (
            <div key={step.step} className="animate-fade-in-up" style={{ animationDelay: `${idx * 80}ms` }}>
              {/* Arrow connector */}
              {idx > 0 && (
                <div className="flex justify-center py-1.5">
                  <ArrowDown className="h-5 w-5 text-muted-foreground/50" />
                </div>
              )}

              <Card
                className={`border-0 shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden ${
                  step.reversible
                    ? "hover:border-l-4 hover:border-l-green-500"
                    : "hover:border-l-4 hover:border-l-red-500"
                }`}
              >
                <CardContent className="p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-4">
                      {/* Step badge */}
                      <div
                        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-sm font-bold text-white shadow-lg ${
                          step.reversible
                            ? "bg-gradient-to-br from-green-400 to-green-600"
                            : "bg-gradient-to-br from-red-400 to-red-600"
                        }`}
                      >
                        {step.step}
                      </div>

                      <div className="space-y-2.5">
                        <h4 className="font-bold text-base">{step.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {step.description}
                        </p>

                        {/* Values */}
                        <div className="mt-3">
                          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
                            Values (first 8):
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {step.values.slice(0, 8).map((val, vIdx) => (
                              <span
                                key={vIdx}
                                className="inline-flex items-center rounded-md bg-muted/80 border px-2.5 py-1 text-xs font-mono hover:bg-primary/10 hover:border-primary/30 transition-colors cursor-default"
                              >
                                {typeof val === "number"
                                  ? Number.isInteger(val)
                                    ? val.toString()
                                    : val.toFixed(4)
                                  : val}
                              </span>
                            ))}
                            {step.values.length > 8 && (
                              <span className="text-xs text-muted-foreground self-center ml-1">
                                +{step.values.length - 8} more
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Reversible indicator */}
                    <div className="shrink-0">
                      {step.reversible ? (
                        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
                          <CheckCircle2 className="h-4 w-4" />
                          <span className="text-xs font-semibold">Reversible</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
                          <XCircle className="h-4 w-4" />
                          <span className="text-xs font-semibold">Info Lost</span>
                        </div>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}

          {/* Summary */}
          <Card className="mt-6 border-0 shadow-lg bg-gradient-to-r from-muted/50 to-muted/30">
            <CardContent className="p-5">
              <h4 className="font-bold mb-2">Summary</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                The <strong>rounding step</strong> is the only irreversible
                operation. All prior steps can be mathematically inverted. The
                quantization error comes entirely from mapping continuous values
                to a discrete integer grid with only{" "}
                <strong>{2 ** 8} levels</strong>.
              </p>
              {stepsData.scale && (
                <p className="text-sm mt-2">
                  <span className="text-muted-foreground">Scale factor: </span>
                  <span className="font-mono font-bold">{stepsData.scale.toFixed(6)}</span>
                  <span className="text-muted-foreground"> (each integer step = this much in float)</span>
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
