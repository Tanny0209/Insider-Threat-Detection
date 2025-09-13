const AlertsPanel = ({ title, alerts, showTimestamp }) => {
  const AlertIcon = ({ type }) => {
    switch (type) {
      case 'high': return <AlertTriangle size={16} className="text-red-500" />;
      case 'medium': return <AlertTriangle size={16} className="text-yellow-500" />;
      case 'success': return <CheckCircle size={16} className="text-green-500" />;
      default: return <Info size={16} className="text-blue-500" />;
    }
  };

  return (
    <div className="bg-gray-800 p-6 rounded-xl">
      <h3 className="text-white text-lg font-semibold mb-4">{title}</h3>
      <div className="space-y-3">
        {alerts.map((alert, index) => (
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
            {showTimestamp && (
              <div className="text-right">
                <div className="text-white text-sm">{alert.email}</div>
                <div className="text-gray-400 text-xs">{alert.time}</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
export default AlertsPanel;