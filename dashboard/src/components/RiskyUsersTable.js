export default function RiskyUsersTable() {
  const users = [
    { email: "alice@corp.com", score: 1323, change: "+55" },
    { email: "bob@corp.com", score: 1265, change: "+35" },
    { email: "eve@corp.com", score: 1225, change: "-31" },
  ];

  return (
    <div className="bg-gray-800 p-4 rounded-lg text-white">
      <h3 className="mb-2">Top Risky Users</h3>
      <table className="w-full">
        <thead>
          <tr>
            <th>Email</th>
            <th>Risk Score</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>
          {users.map((u, i) => (
            <tr key={i}>
              <td>{u.email}</td>
              <td>{u.score}</td>
              <td className={u.change.includes("-") ? "text-red-400" : "text-green-400"}>{u.change}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
