import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

// importing the same components but will keep coding style more natural
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";

import Dashboard from "./pages/Dashboard";
import RiskHeatmapPage from "./pages/RiskHeatmapPage";
import UserProfiles from "./pages/UserProfiles";
import SystemStatus from "./pages/SystemStatus";

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-[#0f0f0f] text-gray-200">
        {/* left sidebar always visible */}
        <Sidebar />

        {/* right side content */}
        <div className="flex flex-col flex-1">
          <Navbar />
          <div className="p-5 flex-1 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/RiskHeatmap" element={<RiskHeatmapPage />} />
              <Route path="/UserProfiles" element={<UserProfiles />} />
              <Route path="/SystemStatus" element={<SystemStatus />} />
            </Routes>
          </div>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
