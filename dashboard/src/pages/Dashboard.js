import React from "react";

export default function Dashboard() {
  return (
    <div className="grid grid-cols-12 gap-6">
      {/* KPI Cards */}
      <div className="col-span-3 bg-blue-700 p-5 rounded-lg shadow-md">
        <h3 className="text-sm">Total Emails Processed</h3>
        <p className="text-3xl font-bold">32,156</p>
        <p className="text-xs">Today / Week / Month</p>
      </div>
      <div className="col-span-3 bg-red-700 p-5 rounded-lg shadow-md">
        <h3 className="text-sm">High Risk Emails</h3>
        <p className="text-3xl font-bold">120</p>
      </div>
      <div className="col-span-3 bg-green-700 p-5 rounded-lg shadow-md">
        <h3 className="text-sm">Medium Risk</h3>
        <p className="text-3xl font-bold">321</p>
      </div>
      <div className="col-span-3 bg-yellow-600 p-5 rounded-lg shadow-md">
        <h3 className="text-sm">Low Risk</h3>
        <p className="text-3xl font-bold">4,200</p>
      </div>

      {/* Risk Heatmap */}
      <div className="col-span-6 bg-neutral-900 p-5 rounded-lg">
        <h4 className="font-semibold mb-3">Risk Heatmap</h4>
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left">Dept</th>
              <th>Low</th>
              <th>Medium</th>
              <th>High</th>
            </tr>
          </thead>
          <tbody>
            {["HR", "IT", "Finance", "Sales"].map((d) => (
              <tr key={d} className="h-10">
                <td>{d}</td>
                <td className="bg-green-600"></td>
                <td className="bg-yellow-500"></td>
                <td className="bg-red-600"></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Risky Users */}
      <div className="col-span-6 bg-neutral-900 p-5 rounded-lg">
        <h4 className="font-semibold mb-3">Risky Addresses</h4>
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th>Email</th>
              <th>Score</th>
              <th>Change</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>sars@example.com</td>
              <td>631</td>
              <td className="text-green-400">+55</td>
            </tr>
            <tr>
              <td>mark@example.com</td>
              <td>557</td>
              <td className="text-green-400">+55</td>
            </tr>
            <tr>
              <td>jane.doe@comp.com</td>
              <td>535</td>
              <td className="text-red-400">-31</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Communication Graph Placeholder */}
      <div className="col-span-6 bg-neutral-900 p-5 rounded-lg">
        <h4 className="font-semibold mb-3">Communication Graph</h4>
        <div className="text-gray-500 text-sm">[Graph visualization goes here]</div>
      </div>

      {/* Real-Time Alerts */}
      <div className="col-span-6 bg-neutral-900 p-5 rounded-lg">
        <h4 className="font-semibold mb-3">Real-Time Alerts</h4>
        <ul className="space-y-2 text-sm">
          <li className="flex justify-between">
            <span className="text-red-500">High Risk</span>
            <span>sars@example.com</span>
            <span>17m ago</span>
          </li>
          <li className="flex justify-between">
            <span className="text-yellow-400">Medium Risk</span>
            <span>mark@example.com</span>
            <span>54m ago</span>
          </li>
          <li className="flex justify-between">
            <span className="text-blue-400">Info</span>
            <span>mark@company.com</span>
            <span>10h ago</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
