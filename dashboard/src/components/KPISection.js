export default function KPISection() {
  return (
    <div className="grid grid-cols-2 gap-4 mb-6">
      <div className="bg-blue-600 p-4 rounded-lg text-white">
        <h3>Total Emails Processed</h3>
        <p className="text-3xl font-bold">32,156</p>
        <p>Today / Week / Month</p>
      </div>
      <div className="bg-red-600 p-4 rounded-lg text-white">
        <h3>High Risk Emails</h3>
        <p className="text-3xl font-bold">120</p>
      </div>
    </div>
  );
}
