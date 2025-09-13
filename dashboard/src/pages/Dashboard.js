import React from "react";
import KPISection from "../components/KPISection";
import RiskHeatmap from "../components/RiskHeatmap";
import RiskyUsersTable from "../components/RiskyUsersTable";
import AlertsPanel from "../components/AlertsPanel";
import CommunicationGraph from "../components/CommunicationGraph";

export default function Dashboard() {
  return (
    <div className="space-y-6">
      {/* KPI Top Cards */}
      <KPISection />

      <div className="grid grid-cols-3 gap-6">
        <RiskHeatmap />
        <RiskyUsersTable />
        <AlertsPanel />
      </div>

      <CommunicationGraph />
    </div>
  );
}
