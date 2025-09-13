import { Link } from "react-router-dom";
import {
  Home,
  Grid,
  Users,
  MessageCircle,
  Shield,
  FileText,
  Bell,
  Server,
  Settings,
} from "lucide-react";

function Sidebar() {
  const menuItems = [
    { name: "Dashboard", icon: Home, path: "/" },
    { name: "Risk Heatmap", icon: Grid, path: "/risk-heatmap" },
    { name: "User Profiles", icon: Users, path: "/user-profiles" },
    { name: "Communication", icon: MessageCircle, path: "/communication" },
    { name: "Arents", icon: Shield, path: "/arents" },
    { name: "Attachment Scanner", icon: FileText, path: "/scanner" },
    { name: "Alerts", icon: Bell, path: "/alerts" },
    { name: "System Status", icon: Server, path: "/settings" },
    { name: "Reports", icon: FileText, path: "/reports" },
    { name: "Settings", icon: Settings, path: "/app-settings" },
  ];

  return (
    <aside className="w-64 bg-neutral-900 border-r border-neutral-800 p-4">
      <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
        👁 Internal Eye
      </h2>
      <nav className="space-y-1">
        {menuItems.map((item, i) => (
          <Link
            key={i}
            to={item.path}
            className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-neutral-800 transition"
          >
            <item.icon size={18} />
            <span>{item.name}</span>
          </Link>
        ))}
      </nav>
    </aside>
  );
}

export default Sidebar;
