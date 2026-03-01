"use client";

import MainLayout from "@/components/layout/MainLayout";

export default function SweepPage() {
  return (
    <MainLayout>
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Parameter Sweep</h1>
      </div>
      <div className="bg-white dark:bg-zinc-900 p-6 rounded-lg border border-zinc-200 dark:border-zinc-800">
        <p className="text-zinc-500">Configure multi-parameter grid searches or bayesian optimization.</p>
        {/* Placeholder for Sweep UI implementation */}
      </div>
    </MainLayout>
  );
}
