import React, { useEffect, useState } from "react";
import { getHeatmapData } from "../services/heatmap";

function Dashboard() {
  const [data, setData] = useState([]);

  useEffect(() => {
    getHeatmapData()
      .then((res) => setData(res.data))
      .catch(() => setData([["HR", 3], ["IT", 8]])); // dummy fallback
  }, []);

  return (
    <div>
      <h2>Dashboard</h2>
      <p>Heatmap / risk summary will appear here.</p>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export default Dashboard;
