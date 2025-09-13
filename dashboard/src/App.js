import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";

import Dashboard from "./pages/Dashboard";
import RiskHeatmap from "./pages/RiskHeatmapPage";
import UserProfiles from "./pages/UserProfiles";
import SystemStatus from "./pages/SystemStatus";

function App() {
  return (
    <Router>
      <div className="flex min-h-screen bg-neutral-950 text-gray-200">
        {/* Sidebar always visible */}
        <Sidebar />

        {/* Right panel with top bar and content */}
        <div className="flex-1 flex flex-col">
          <Navbar />
          <main className="flex-1 p-6 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/RiskHeatmap" element={<RiskHeatmap />} />
              <Route path="/UserProfiles" element={<UserProfiles />} />
              <Route path="/SystemStatus" element={<SystemStatus />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
