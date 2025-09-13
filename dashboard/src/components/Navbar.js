import { Search, BellRing, UserCircle } from "lucide-react";

function Navbar() {
  return (
    <header className="bg-neutral-900 border-b border-neutral-800 px-6 py-3 flex justify-between items-center">
      {/* Title */}
      <h1 className="text-lg font-semibold">Internal Eye Dashboard</h1>

      {/* Search bar */}
      <div className="flex-1 mx-8">
        <div className="relative">
          <Search className="absolute left-3 top-2.5 text-gray-500" size={16} />
          <input
            placeholder="Search here..."
            className="w-full bg-neutral-800 rounded-md pl-9 pr-3 py-2 text-sm outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Icons */}
      <div className="flex items-center gap-6">
        <BellRing className="cursor-pointer hover:text-blue-400" />
        <UserCircle className="cursor-pointer hover:text-blue-400" />
      </div>
    </header>
  );
}

export default Navbar;
