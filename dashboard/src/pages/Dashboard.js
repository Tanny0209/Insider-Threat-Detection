function Dashboard() {
  return (
    <div className="space-y-6">
      {/* top cards */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-blue-700 p-5 rounded-md">
          <h3 className="text-sm">Total Emails Processed</h3>
          <p className="text-3xl font-bold">32,156</p>
          <span className="text-xs text-gray-200">Today / Week / Month</span>
        </div>
        <div className="bg-red-700 p-5 rounded-md">
          <h3 className="text-sm">High Risk Emails</h3>
          <p className="text-3xl font-bold">120</p>
        </div>
      </div>

      {/* heatmap section */}
      <div className="bg-neutral-900 p-5 rounded-md">
        <h4 className="mb-3 font-semibold">Risk Heatmap</h4>
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left">Department</th>
              <th>Low</th>
              <th>Medium</th>
              <th>High</th>
            </tr>
          </thead>
          <tbody>
            {["HR", "IT", "Finance", "Sales"].map((dept) => (
              <tr key={dept}>
                <td>{dept}</td>
                <td className="bg-green-600 h-6"></td>
                <td className="bg-yellow-500 h-6"></td>
                <td className="bg-red-600 h-6"></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* communication graph placeholder */}
      <div className="bg-neutral-900 p-5 rounded-md">
        <h4 className="mb-3 font-semibold">Communication Graph</h4>
        <p className="text-gray-400 text-sm">[Graph visualization here]</p>
      </div>

      {/* alerts section */}
      <div className="bg-neutral-900 p-5 rounded-md">
        <h4 className="mb-3 font-semibold">Real-Time Alerts</h4>
        <ul className="space-y-2 text-sm">
          <li className="flex justify-between">
            <span className="text-red-500">High</span>
            <span>sars@example.com</span>
            <span>17m ago</span>
          </li>
          <li className="flex justify-between">
            <span className="text-green-500">Medium</span>
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

export default Dashboard;
