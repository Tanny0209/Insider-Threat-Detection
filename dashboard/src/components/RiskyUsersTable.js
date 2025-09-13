import React from "react";

const users = [
  { email: "atse@example.com", score: 631, prev: 1323, change: +55 },
  { email: "jamer@example.com", score: 557, prev: 1362, change: +55 },
  { email: "orm@example.com", score: 551, prev: 1285, change: +35 },
  { email: "johnsdoe@bx.com", score: 552, prev: 1255, change: +35 },
  { email: "jane.doe@comp.com", score: 535, prev: 1225, change: -31 },
  { email: "jaett@example.com", score: 526, prev: 1222, change: -33 },
  { email: "james@comple.com", score: 122, prev: 1215, change: -33 },
];

export default function RiskyUsersTable() {
  return (
    <div className="bg-gray-800 p-4 rounded-xl shadow-md">
      <h3 className="font-semibold mb-3">Rylat/Addresses</h3>
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th>Email</th>
            <th>Score</th>
            <th>Prev</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.email}>
              <td className="py-1">{u.email}</td>
              <td>{u.score}</td>
              <td>{u.prev}</td>
              <td className={u.change > 0 ? "text-green-400" : "text-red-400"}>
                {u.change > 0 ? `▲ ${u.change}` : `▼ ${Math.abs(u.change)}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
