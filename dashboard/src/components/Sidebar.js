import React from "react";
import { Link } from "react-router-dom";

function Sidebar() {
  return (
    <div className="sidebar">
      <h2>Internal Eye</h2>
      <ul>
        <li><Link to="/">Dashboard</Link></li>
        <li><Link to="/alerts">Alerts</Link></li>
        <li><Link to="/users">Users</Link></li>
        <li><Link to="/settings">Settings</Link></li>
      </ul>
    </div>
  );
}

export default Sidebar;
