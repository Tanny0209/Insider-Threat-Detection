// pages/RiskHeatmapPage.js
import React from 'react';
import { RefreshCw, Download } from 'lucide-react';

const RiskHeatmapPage = () => {
  const departments = ['HR', 'IT', 'Finance', 'Sales', 'Marketing', 'Operations'];
  const riskLevels = ['Low', 'Medium', 'High', 'Critical'];
  const riskData = [
    [1, 2, 3, 4], // HR
    [2, 3, 4, 5], // IT  
    [3, 4, 5, 4], // Finance
    [2, 4, 5, 3], // Sales
    [1, 3, 4, 2], // Marketing
    [2, 3, 3, 4]  // Operations
  ];

  const getRiskColor = (level) => {
    const colors = ['bg-green-500', 'bg-yellow-500', 'bg-orange-500', 'bg-red-500', 'bg-red-600'];
    return colors[level - 1] || 'bg-gray-400';
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Risk Heatmap</h2>
        <div className="flex space-x-2">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center">
            <RefreshCw size={16} className="mr-2" />
            Refresh
          </button>
          <button className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 flex items-center">
            <Download size={16} className="mr-2" />
            Export
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Main Heatmap */}
        <div className="bg-gray-800 p-6 rounded-xl">
          <h3 className="text-white text-lg font-semibold mb-4">Department Risk Matrix</h3>
          <div className="grid grid-cols-5 gap-1">
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
                  <div 
                    key={riskIndex} 
                    className={`aspect-square ${getRiskColor(risk)} rounded cursor-pointer hover:opacity-80 flex items-center justify-center text-white font-bold text-sm`}
                    title={`${dept} - ${riskLevels[riskIndex]}: Risk Level ${risk}`}
                  >
                    {risk}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
          
          {/* Legend */}
          <div className="mt-4 flex items-center justify-center space-x-4">
            <span className="text-gray-400 text-sm">Risk Level:</span>
            {[1, 2, 3, 4, 5].map(level => (
              <div key={level} className="flex items-center space-x-1">
                <div className={`w-4 h-4 ${getRiskColor(level)} rounded`}></div>
                <span className="text-gray-400 text-xs">{level}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Statistics */}
        <div className="space-y-4">
          <div className="bg-gray-800 p-6 rounded-xl">
            <h3 className="text-white text-lg font-semibold mb-4">Risk Summary</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Critical Risk</span>
                <span className="text-red-500 font-bold">8 incidents</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">High Risk</span>
                <span className="text-orange-500 font-bold">15 incidents</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Medium Risk</span>
                <span className="text-yellow-500 font-bold">23 incidents</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Low Risk</span>
                <span className="text-green-500 font-bold">12 incidents</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-xl">
            <h3 className="text-white text-lg font-semibold mb-4">Top Risk Departments</h3>
            <div className="space-y-3">
              {departments.map((dept, index) => {
                const totalRisk = riskData[index].reduce((a, b) => a + b, 0);
                return (
                  <div key={dept} className="flex justify-between items-center">
                    <span className="text-gray-400">{dept}</span>
                    <span className="text-white font-bold">{totalRisk}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskHeatmapPage;