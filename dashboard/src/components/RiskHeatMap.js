import React from "react";

const data = [
  { dept: "HR", low: "🟩", med: "🟨", high: "🟧" },
  { dept: "IT", low: "🟩", med: "🟨", high: "🟥" },
  { dept: "Finance", low: "🟩", med: "🟧", high: "🟥" },
  { dept: "Sales", low: "🟨", med: "🟧", high: "🟥" },
];

export default function RiskHeatmap() {
  return (
    <div className="bg-gray-800 p-4 rounded-xl shadow-md">
      <h3 className="font-semibold mb-3">Risk Heatmap</h3>
      <table className="w-full text-center text-sm">
        <thead>
          <tr>
            <th className="py-2">Department</th>
            <th>Low</th>
            <th>Medium</th>
            <th>High</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.dept}>
              <td className="py-2">{row.dept}</td>
              <td>{row.low}</td>
              <td>{row.med}</td>
              <td>{row.high}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
