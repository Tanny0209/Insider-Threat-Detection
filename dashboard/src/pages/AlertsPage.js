import React, { useEffect, useState } from "react";
import { getAlerts } from "../services/alerts";

function Alerts() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    getAlerts()
      .then((res) => setAlerts(res.data))
      .catch(() => setAlerts([{ message: "Dummy Alert - Suspicious Email" }]));
  }, []);

  return (
    <div>
      <h2>Alerts</h2>
      <ul>
        {alerts.map((a, i) => (
          <li key={i}>{a.message}</li>
        ))}
      </ul>
    </div>
  );
}

export default Alerts;
