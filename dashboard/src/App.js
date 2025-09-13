import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import IntroScreen from "./components/IntroScreen";

// Simple placeholder page for unfinished routes
const ComingSoon = ({ title }) => (
  <div className="text-center text-gray-400 mt-20 text-lg font-mono">
    ⚡ {title} module is still under construction…
  </div>
);

function App() {
  const [introDone, setIntroDone] = useState(false);

  // IntroScreen will call this AFTER all messages finish typing
  const handleIntroFinish = () => setIntroDone(true);

  if (!introDone) {
    return <IntroScreen onFinish={handleIntroFinish} />;
  }

  return (
    <Router>
      <div className="flex min-h-screen bg-neutral-950 text-gray-200">
        {/* Left navigation */}
        <Sidebar />

        {/* Right content */}
        <div className="flex-1 flex flex-col">
          <Navbar />
          <main className="flex-1 p-6 overflow-y-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/risk-heatmap" element={<ComingSoon title="Risk Heatmap" />} />
              <Route path="/user-profiles" element={<ComingSoon title="User Profiles" />} />
              <Route path="/communication" element={<ComingSoon title="Communication" />} />
              <Route path="/arents" element={<ComingSoon title="Arents" />} />
              <Route path="/scanner" element={<ComingSoon title="Attachment Scanner" />} />
              <Route path="/alerts" element={<ComingSoon title="Alerts" />} />
              <Route path="/settings" element={<ComingSoon title="System Status" />} />
              <Route path="/reports" element={<ComingSoon title="Reports" />} />
              <Route path="/app-settings" element={<ComingSoon title="Settings" />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
