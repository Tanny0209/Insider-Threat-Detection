import React, { useEffect, useState } from "react";
import { getUsers } from "../services/users";

function Users() {
  const [users, setUsers] = useState([]);

  useEffect(() => {
    getUsers()
      .then((res) => setUsers(res.data))
      .catch(() => setUsers([{ name: "John Doe", risk: "High" }]));
  }, []);

  return (
    <div>
      <h2>Users</h2>
      <ul>
        {users.map((u, i) => (
          <li key={i}>{u.name} - Risk: {u.risk}</li>
        ))}
      </ul>
    </div>
  );
}

export default Users;
