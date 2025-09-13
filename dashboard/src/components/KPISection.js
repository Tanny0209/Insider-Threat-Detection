import React from "react";

function KPISection() {
  return (
    <section className="grid grid-cols-2 gap-4 mb-6">
      <div className="bg-blue-600 p-5 rounded-lg text-white">
        <p className="text-sm">Total Emails Processed</p>
        <h2 className="text-3xl font-bold">32,156</h2>
        <p className="text-xs">Today / Week / Month</p>
      </div>
      <div className="bg-red-600 p-5 rounded-lg text-white">
        <p className="text-sm">High Risk Emails</p>
        <h2 className="text-3xl font-bold">120</h2>
      </div>
    </section>
  );
}

export default KPISection;
