import React, { useState, useEffect } from "react";

import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import RiskHeatmapPage from "./pages/RiskHeatmapPage";
import CommunicationGraph from "./components/CommunicationGraph";
import UserProfilesPage from "./pages/UserProfiles";

const ComingSoon = ({ title }) => (
  <div className="text-center text-gray-400 mt-20 text-lg font-mono">
    ⚡ {title} module is still under construction…
  </div>
);

const App = () => {
  const [currentPage, setCurrentPage] = useState("Dashboard");
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const renderPage = () => {
    switch (currentPage) {
      case "Dashboard":
        return <Dashboard />;
      case "Risk Heatmap":
        return <RiskHeatmapPage />;
      case "User Profiles":
        return <UserProfilesPage/>;
      case "Communication":
        return <CommunicationGraph />;
      default:
        return <ComingSoon title={currentPage} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <div className="flex-1 p-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold">{currentPage}</h1>
          <div className="flex items-center space-x-4">
            <span className="text-gray-400">
              {currentTime.toLocaleString()}
            </span>
          </div>
        </div>
        {renderPage()}
      </div>
    </div>
  );
};

export default App;
