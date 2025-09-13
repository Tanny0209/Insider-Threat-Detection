import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Bell } from 'lucide-react';

const Dashboard = () => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // StatCard Component for KPI Section
  const StatCard = ({ title, value, subtitle, color = 'blue' }) => (
    <div className={`p-6 rounded-xl ${color === 'blue' ? 'bg-blue-600' : 'bg-red-600'} text-white`}>
      <h3 className="text-sm font-medium opacity-90 mb-2">{title}</h3>
      <div className="text-3xl font-bold mb-1">{value.toLocaleString()}</div>
      {subtitle && <p className="text-sm opacity-75">{subtitle}</p>}
    </div>
  );

  // AlertIcon Component for Alerts
  const AlertIcon = ({ type }) => {
    switch (type) {
      case 'high': return <AlertTriangle size={16} className="text-red-500" />;
      case 'medium': return <AlertTriangle size={16} className="text-yellow-500" />;
      case 'success': return <CheckCircle size={16} className="text-green-500" />;
      default: return <Bell size={16} className="text-blue-500" />;
    }
  };

  // Sample data
  const userProfiles = [
    { email: 'atse@example.com', sent: 631, risk: 1323, score: 55, trend: 'up' },
    { email: 'jamer@example.com', sent: 557, risk: 1362, score: 55, trend: 'up' },
    { email: 'orm@example.com', sent: 551, risk: 1285, score: 35, trend: 'up' },
    { email: 'johnsdoe@bx.com', sent: 552, risk: 1255, score: 35, trend: 'up' },
    { email: 'jane.doe@comp.com', sent: 535, risk: 1225, score: 31, trend: 'down' },
    { email: 'jaett@example.com', sent: 526, risk: 1226, score: 33, trend: 'down' },
    { email: 'james@comple.com', sent: 122, risk: 1215, score: 33, trend: 'down' }
  ];

  const realtimeAlerts = [
    { 
      type: 'high', 
      title: 'High Risk', 
      email: 'sars@example.com', 
      time: '17 min ago',
      description: 'Dale itu penetrate',
      code: '11 966 tB'
    },
    { 
      type: 'medium', 
      title: 'Medium Risk', 
      email: 'mark@company.com', 
      time: '54 min ago',
      description: 'User lane satrimari'
    },
    { 
      type: 'info', 
      title: 'Informational', 
      email: 'mark@company.com', 
      time: '10 hour ago',
      description: 'User lane satrimari',
      code: '11:70 AM'
    },
    { 
      type: 'info', 
      title: 'Informational', 
      email: 'mark@company.com', 
      time: '10 hour ago',
      description: 'User lane sancomat',
      code: '19:50 AM'
    },
    { 
      type: 'success', 
      title: 'Informational', 
      email: 'mark@company.com', 
      time: '6 hour ago',
      description: 'User lane satrimari',
      code: '10:50 AM'
    }
  ];

  // Risk Heatmap data and helper function
  const departments = ['HR', 'IT', 'Finance', 'Sales'];
  const riskLevels = ['Low', 'Medium', 'High'];
  const riskData = [
    [1, 2, 3], // HR
    [2, 3, 4], // IT  
    [3, 4, 5], // Finance
    [2, 4, 5]  // Sales
  ];

  const getRiskColor = (level) => {
    const colors = ['bg-green-500', 'bg-yellow-500', 'bg-orange-500', 'bg-red-500', 'bg-red-600'];
    return colors[level - 1] || 'bg-gray-400';
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Main Content */}
      <div className="p-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <div className="flex items-center space-x-4">
            <span className="text-gray-400">{currentTime.toLocaleString()}</span>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <StatCard 
            title="Total Emails Processed" 
            value={32156} 
            subtitle="Today / Week / Month"
            color="blue"
          />
          <StatCard 
            title="High Risk Emails" 
            value={120} 
            color="red"
          />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <div className="lg:col-span-1 space-y-6">
            {/* Risk Heatmap */}
            <div className="bg-gray-800 p-6 rounded-xl">
              <h3 className="text-white text-lg font-semibold mb-4">Risk Heatmap</h3>
              <div className="grid grid-cols-4 gap-1">
                {/* Headers */}
                <div></div>
                {riskLevels.map(level => (
                  <div key={level} className="text-center text-gray-400 text-sm py-2">{level}</div>
                ))}
                
                {/* Data rows */}
                {departments.map((dept, deptIndex) => (
                  <React.Fragment key={dept}>
                    <div className="text-gray-400 text-sm py-4 pr-4 text-right">{dept}</div>
                    {riskData[deptIndex].map((risk, riskIndex) => (
                      <div key={riskIndex} className={`aspect-square ${getRiskColor(risk)} rounded`}></div>
                    ))}
                  </React.Fragment>
                ))}
              </div>
            </div>

            {/* Communication Graph */}
            <div className="bg-gray-800 p-6 rounded-xl">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-white text-lg font-semibold">Communication Graph</h3>
                <span className="text-gray-400 text-sm">Suppeklaus</span>
              </div>
              
              <div className="relative h-64 flex items-center justify-center">
                {/* Central node */}
                <div className="absolute bg-red-500 w-4 h-4 rounded-full z-10"></div>
                
                {/* Surrounding nodes */}
                <div className="absolute top-8 left-16 bg-blue-500 w-3 h-3 rounded-full"></div>
                <div className="absolute top-16 right-12 bg-blue-500 w-3 h-3 rounded-full"></div>
                <div className="absolute bottom-12 left-8 bg-blue-500 w-3 h-3 rounded-full"></div>
                <div className="absolute bottom-16 right-16 bg-blue-500 w-3 h-3 rounded-full"></div>
                <div className="absolute top-12 left-32 bg-blue-500 w-3 h-3 rounded-full"></div>
                <div className="absolute bottom-8 left-24 bg-blue-500 w-3 h-3 rounded-full"></div>
                <div className="absolute top-20 right-20 bg-red-500 w-4 h-4 rounded-full"></div>
                <div className="absolute bottom-20 right-8 bg-blue-500 w-3 h-3 rounded-full"></div>
                
                {/* Connection lines */}
                <svg className="absolute inset-0 w-full h-full">
                  <line x1="50%" y1="50%" x2="25%" y2="20%" stroke="#374151" strokeWidth="1" />
                  <line x1="50%" y1="50%" x2="75%" y2="25%" stroke="#374151" strokeWidth="1" />
                  <line x1="50%" y1="50%" x2="20%" y2="75%" stroke="#374151" strokeWidth="1" />
                  <line x1="50%" y1="50%" x2="80%" y2="70%" stroke="#374151" strokeWidth="1" />
                  <line x1="50%" y1="50%" x2="65%" y2="35%" stroke="#ef4444" strokeWidth="2" />
                </svg>
              </div>
              
              <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                <div className="text-white font-medium">jans-doe@company.com</div>
                <div className="text-blue-400 text-sm">Risk score: 85</div>
                <div className="text-gray-400 text-sm mt-1">
                  Unusual volume of emails sent outside the organization
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* User Profiles */}
            <div className="bg-gray-800 p-6 rounded-xl">
              <h3 className="text-white text-lg font-semibold mb-4">Rylat/Adressess</h3>
              <div className="space-y-3">
                {userProfiles.map((user, index) => (
                  <div key={index} className="flex items-center justify-between py-2">
                    <div className="flex-1">
                      <span className="text-white">{user.email}</span>
                    </div>
                    <div className="flex items-center space-x-6 text-sm">
                      <span className="text-gray-400 w-12 text-center">{user.sent}</span>
                      <span className="text-gray-400 w-16 text-center">{user.risk}</span>
                      <div className="flex items-center space-x-2 w-16">
                        {user.trend === 'up' ? (
                          <TrendingUp size={16} className="text-green-500" />
                        ) : (
                          <TrendingDown size={16} className="text-red-500" />
                        )}
                        <span className={user.trend === 'up' ? 'text-green-500' : 'text-red-500'}>
                          {user.score}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Real-Time Alerts */}
            <div className="bg-gray-800 p-6 rounded-xl">
              <h3 className="text-white text-lg font-semibold mb-4">Real-Time Alerts</h3>
              <div className="space-y-3">
                {realtimeAlerts.map((alert, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <AlertIcon type={alert.type} />
                      <div>
                        <div className="text-white font-medium">{alert.title}</div>
                        <div className="text-gray-400 text-sm">{alert.description}</div>
                        {alert.code && (
                          <div className="text-gray-500 text-xs">{alert.code}</div>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-white text-sm">{alert.email}</div>
                      <div className="text-gray-400 text-xs">{alert.time}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;