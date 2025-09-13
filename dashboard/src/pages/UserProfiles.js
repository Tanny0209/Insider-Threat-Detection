// pages/UserProfilesPage.js
import React, { useState } from 'react';
import { Search, Download, TrendingUp, TrendingDown } from 'lucide-react';

const UserProfilesPage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRisk, setFilterRisk] = useState('all');

  const allUsers = [
    { id: 1, name: 'Alice Chen', email: 'atse@example.com', sent: 631, risk: 1323, score: 55, trend: 'up', department: 'IT', lastActive: '2 hours ago', status: 'active' },
    { id: 2, name: 'James Miller', email: 'jamer@example.com', sent: 557, risk: 1362, score: 55, trend: 'up', department: 'Finance', lastActive: '1 hour ago', status: 'active' },
    { id: 3, name: 'Oliver Rodriguez', email: 'orm@example.com', sent: 551, risk: 1285, score: 35, trend: 'up', department: 'HR', lastActive: '3 hours ago', status: 'active' },
    { id: 4, name: 'John Doe', email: 'johnsdoe@bx.com', sent: 552, risk: 1255, score: 35, trend: 'up', department: 'Sales', lastActive: '30 min ago', status: 'active' },
    { id: 5, name: 'Jane Doe', email: 'jane.doe@comp.com', sent: 535, risk: 1225, score: 31, trend: 'down', department: 'Marketing', lastActive: '5 hours ago', status: 'inactive' },
    { id: 6, name: 'Jack Smith', email: 'jaett@example.com', sent: 526, risk: 1226, score: 33, trend: 'down', department: 'Operations', lastActive: '1 day ago', status: 'inactive' },
    { id: 7, name: 'James Wilson', email: 'james@comple.com', sent: 122, risk: 1215, score: 33, trend: 'down', department: 'IT', lastActive: '2 days ago', status: 'inactive' }
  ];

  const filteredUsers = allUsers.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterRisk === 'all' || 
                         (filterRisk === 'high' && user.score > 50) ||
                         (filterRisk === 'medium' && user.score >= 30 && user.score <= 50) ||
                         (filterRisk === 'low' && user.score < 30);
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">User Profiles</h2>
        <div className="flex space-x-2">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center">
            <Download size={16} className="mr-2" />
            Export
          </button>
        </div>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search size={20} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search users..."
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <select
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-blue-500"
          value={filterRisk}
          onChange={(e) => setFilterRisk(e.target.value)}
        >
          <option value="all">All Risk Levels</option>
          <option value="high">High Risk (50+)</option>
          <option value="medium">Medium Risk (30-50)</option>
          <option value="low">Low Risk (&lt;30)</option>
        </select>
      </div>

      {/* Users Table */}
      <div className="bg-gray-800 rounded-xl overflow-hidden">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-white text-lg font-semibold">All Users ({filteredUsers.length})</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">User</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Department</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Emails Sent</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Risk Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Trend</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Last Active</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {filteredUsers.map((user) => (
                <tr key={user.id} className="hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div>
                      <div className="text-white font-medium">{user.name}</div>
                      <div className="text-gray-400 text-sm">{user.email}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-300">{user.department}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-300">{user.sent}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      user.score > 50 ? 'bg-red-100 text-red-800' :
                      user.score >= 30 ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }`}>
                      {user.score}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {user.trend === 'up' ? (
                        <TrendingUp size={16} className="text-green-500 mr-1" />
                      ) : (
                        <TrendingDown size={16} className="text-red-500 mr-1" />
                      )}
                      <span className={user.trend === 'up' ? 'text-green-500' : 'text-red-500'}>
                        {user.trend === 'up' ? 'Up' : 'Down'}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-300">{user.lastActive}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      user.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                    }`}>
                      {user.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default UserProfilesPage;