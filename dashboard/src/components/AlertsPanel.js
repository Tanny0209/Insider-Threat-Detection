import React from "react";

const alerts = [
  { type: "High Risk", email: "sars@example.com", time: "17 min ago", color: "red" },
  { type: "Medium Risk", email: "mark@example.com", time: "54 min ago", color: "yellow" },
  { type: "Informational", email: "mark@company.com", time: "10 hour ago", color: "blue" },
  { type: "Informational", email: "mark@company.com", time: "10 hour ago", color: "blue" },
  { type: "Informational", email: "mark@company.com", time: "6 hour ago", color: "blue" },
];

export default function AlertsPanel() {
  return (
    <div className="bg-gray-800 p-4 rounded-xl shadow-md">
      <h3 className="font-semibold mb-3">Real-Time Alerts</h3>
      <ul className="space-y-3">
        {alerts.map((a, i) => (
          <li key={i} className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <span
                className={`w-3 h-3 rounded-full ${
                  a.color === "red"
                    ? "bg-red-500"
                    : a.color === "yellow"
                    ? "bg-yellow-400"
                    : "bg-blue-400"
                }`}
              />
              <div>
                <p className="text-sm font-medium">{a.type}</p>
                <p className="text-xs text-gray-400">{a.email}</p>
              </div>
            </div>
            <span className="text-xs text-gray-400">{a.time}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
