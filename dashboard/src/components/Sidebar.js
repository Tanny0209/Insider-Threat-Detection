import { Link } from "react-router-dom";
import {
  LayoutDashboard,
  ActivitySquare,
  Users,
  ServerCog,
} from "lucide-react";

function Sidebar() {
  // simple nav items array so code looks more natural
  const links = [
    { text: "Dashboard", path: "/", icon: LayoutDashboard },
    { text: "Risk Heatmap", path: "/RiskHeatmap", icon: ActivitySquare },
    { text: "User Profiles", path: "/UserProfiles", icon: Users },
    { text: "System Status", path: "/SystemStatus", icon: ServerCog },
  ];

  return (
    <aside className="w-60 bg-neutral-900 border-r border-neutral-800 px-4 py-6">
      <h1 className="text-xl font-bold flex items-center gap-2 mb-8">
        👁 Internal Eye
      </h1>

      <nav className="flex flex-col gap-2">
        {links.map((item, i) => (
          <Link
            key={i}
            to={item.path}
            className="flex items-center gap-3 p-2 rounded-md hover:bg-neutral-800"
          >
            <item.icon size={18} />
            {item.text}
          </Link>
        ))}
      </nav>
    </aside>
  );
}

export default Sidebar;
