import React, { useState, useEffect } from "react";

import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import RiskHeatmapPage from "./pages/RiskHeatmapPage";
import CommunicationGraph from "./components/CommunicationGraph";
import IntroScreen from "./components/IntroScreen";
import UserProfilesPage from "./pages/UserProfiles";

const App = () => {
  const [currentPage, setCurrentPage] = useState("Dashboard");
  const [currentTime, setCurrentTime] = useState(new Date());
  const [showIntro, setShowIntro] = useState(true); 

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => setShowIntro(false), 8000);
    return () => clearTimeout(timer);
  }, []);

  if (showIntro) {
    return <IntroScreen />;
  }

  const renderPage = () => {
    switch (currentPage) {
      case "Dashboard":
        return <Dashboard />;
      case "Risk Heatmap":
        return <RiskHeatmapPage />;
      case "Communication":
        return <CommunicationGraph />;
          case "User Profiles":
        return <UserProfilesPage />;
      default:
        return <div>Coming Soon...</div>;
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
