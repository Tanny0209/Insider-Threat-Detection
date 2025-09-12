export default function AlertsPanel() {
  const alerts = [
    { type: "High Risk", email: "sars@corp.com", time: "17 min ago", color: "bg-red-600" },
    { type: "Medium Risk", email: "mark@corp.com", time: "54 min ago", color: "bg-yellow-500" },
    { type: "Informational", email: "user@corp.com", time: "10h ago", color: "bg-blue-500" },
  ];

  return (
    <div className="bg-gray-800 p-4 rounded-lg text-white">
      <h3 className="mb-2">Real-Time Alerts</h3>
      <ul>
        {alerts.map((a, i) => (
          <li key={i} className="flex justify-between items-center mb-2">
            <span className={`px-2 py-1 rounded ${a.color}`}>{a.type}</span>
            <span>{a.email}</span>
            <span className="text-gray-400">{a.time}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
