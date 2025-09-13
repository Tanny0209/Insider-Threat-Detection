import React from "react";

function AlertsPage() {
  return (
    <div className="alerts">
      <h1>Alerts</h1>
      <table className="alerts-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>User</th>
            <th>Alert Type</th>
            <th>Severity</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>101</td>
            <td>user_a</td>
            <td>Login Anomaly</td>
            <td>High</td>
            <td>12:30 PM</td>
          </tr>
          <tr>
            <td>102</td>
            <td>user_b</td>
            <td>Policy Violation</td>
            <td>Medium</td>
            <td>1:15 PM</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default AlertsPage;
