import React from 'react';
import { 
  Eye, Home, Map, Users, MessageSquare, Shield, Bell, 
  FileText, Activity, Settings 
} from 'lucide-react';

const Sidebar = ({ currentPage, setCurrentPage }) => {
  const MenuItem = ({ icon: Icon, label, active = false, onClick }) => (
    <div 
      className={`flex items-center px-4 py-3 rounded-lg cursor-pointer transition-colors ${
        active ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'
      }`}
      onClick={() => onClick(label)}
    >
      <Icon size={20} className="mr-3" />
      <span className="font-medium">{label}</span>
    </div>
  );

  return (
    <div className="w-64 bg-gray-800 p-6 space-y-4">
      {/* Logo */}
      <div className="flex items-center mb-8">
        <Eye size={32} className="text-blue-500 mr-3" />
        <h1 className="text-xl font-bold">Internal Eye</h1>
      </div>

      {/* Menu Items */}
      <div className="space-y-2">
        <MenuItem 
          icon={Home} 
          label="Dashboard" 
          active={currentPage === 'Dashboard'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={Map} 
          label="Risk Heatmap" 
          active={currentPage === 'Risk Heatmap'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={Users} 
          label="User Profiles" 
          active={currentPage === 'User Profiles'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={MessageSquare} 
          label="Communication" 
          active={currentPage === 'Communication'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={Shield} 
          label="Alerts" 
          active={currentPage === 'Alerts'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={FileText} 
          label="Attachment Scanner" 
          active={currentPage === 'Attachment Scanner'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={Bell} 
          label="System Status" 
          active={currentPage === 'System Status'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={Activity} 
          label="Reports" 
          active={currentPage === 'Reports'} 
          onClick={setCurrentPage} 
        />
        <MenuItem 
          icon={Settings} 
          label="Settings" 
          active={currentPage === 'Settings'} 
          onClick={setCurrentPage} 
        />
      </div>
    </div>
  );
};

export default Sidebar;