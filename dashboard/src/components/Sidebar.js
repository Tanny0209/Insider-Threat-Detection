import React from "react";
import { Home, Grid, Users, MessageCircle, Shield, FileText, Bell, Server, Settings } from "lucide-react";

const menuItems = [
  { name: "Dashboard", icon: Home },
  { name: "Risk Heatmap", icon: Grid },
  { name: "User Profiles", icon: Users },
  { name: "Communication", icon: MessageCircle },
  { name: "Arents", icon: Shield },
  { name: "Attachment Scanner", icon: FileText },
  { name: "Alerts", icon: Bell },
  { name: "System Status", icon: Server },
  { name: "Reports", icon: FileText },
  { name: "Settings", icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="w-60 bg-gray-800 flex flex-col p-4">
      <div className="flex items-center space-x-2 text-xl font-bold mb-8">
        <span className="text-blue-400">👁️</span>
        <span>Internal Eye</span>
      </div>

      <nav className="flex-1 space-y-2">
        {menuItems.map(({ name, icon: Icon }) => (
          <button
            key={name}
            className={`flex items-center w-full space-x-3 p-2 rounded-lg hover:bg-gray-700 transition ${
              name === "Dashboard" ? "bg-gray-700" : ""
            }`}
          >
            <Icon className="w-5 h-5" />
            <span>{name}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}
