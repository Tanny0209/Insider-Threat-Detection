const KPISection = ({ data }) => {
  const StatCard = ({ title, value, subtitle, color = 'blue' }) => (
    <div className={`p-6 rounded-xl ${color === 'blue' ? 'bg-blue-600' : 'bg-red-600'} text-white`}>
      <h3 className="text-sm font-medium opacity-90 mb-2">{title}</h3>
      <div className="text-3xl font-bold mb-1">{value.toLocaleString()}</div>
      {subtitle && <p className="text-sm opacity-75">{subtitle}</p>}
    </div>
  );
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <StatCard 
        title="Total Emails Processed" 
        value={data.totalEmails} 
        subtitle={data.timeRange}
        color="blue"
      />
      <StatCard 
        title="High Risk Emails" 
        value={data.highRiskEmails} 
        color="red"
      />
    </div>
  );
};
export default KPISection;