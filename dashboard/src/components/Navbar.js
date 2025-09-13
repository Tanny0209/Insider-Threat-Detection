import { Search, Bell, User } from "lucide-react";

function Navbar() {
  return (
    <header className="bg-neutral-900 border-b border-neutral-800 px-6 py-3 flex justify-between items-center">
      <h1 className="text-lg font-semibold tracking-wide">Internal Eye Dashboard</h1>

      {/* Search box */}
      <div className="flex-1 mx-6">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 text-gray-500" size={16} />
          <input
            placeholder="Search..."
            className="w-full bg-neutral-800 rounded-md pl-9 pr-3 py-2 text-sm outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>

      <div className="flex items-center gap-6">
        <Bell className="cursor-pointer hover:text-blue-400" />
        <User className="cursor-pointer hover:text-blue-400" />
      </div>
    </header>
  );
}

export default Navbar;
