import { Search, Bell, User } from "lucide-react";

function Navbar() {
  return (
    <div className="w-full bg-neutral-900 border-b border-neutral-800 px-6 py-3 flex justify-between items-center">
      {/* left side small title */}
      <div className="font-semibold text-lg">Dashboard Panel</div>

      {/* middle search bar */}
      <div className="flex-1 mx-8">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 text-gray-400" size={16} />
          <input
            type="text"
            placeholder="search here..."
            className="w-full bg-neutral-800 rounded-lg pl-9 pr-3 py-2 text-sm outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* right side icons */}
      <div className="flex items-center gap-6">
        <Bell className="cursor-pointer hover:text-blue-400" />
        <User className="cursor-pointer hover:text-blue-400" />
      </div>
    </div>
  );
}

export default Navbar;
