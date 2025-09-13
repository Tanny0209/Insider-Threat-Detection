const RiskyUsersTable = ({ title, users, showRiskScore }) => (
  <div className="bg-gray-800 p-6 rounded-xl">
    <h3 className="text-white text-lg font-semibold mb-4">{title}</h3>
    <div className="space-y-3">
      {users.map((user, index) => (
        <div key={index} className="flex items-center justify-between py-2">
          <div className="flex-1">
            <span className="text-white">{user.email}</span>
          </div>
          <div className="flex items-center space-x-6 text-sm">
            <span className="text-gray-400 w-12 text-center">{user.sent}</span>
            {showRiskScore && (
              <span className="text-gray-400 w-16 text-center">{user.risk}</span>
            )}
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
);
export default RiskyUsersTable;