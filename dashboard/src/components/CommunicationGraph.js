const CommunicationGraph = ({ data }) => (
  <div className="bg-gray-800 p-6 rounded-xl">
    <div className="flex justify-between items-center mb-4">
      <h3 className="text-white text-lg font-semibold">Communication Graph</h3>
      <span className="text-gray-400 text-sm">Suppeklaus</span>
    </div>
    
    <div className="relative h-64 flex items-center justify-center">
      <div className="absolute bg-red-500 w-4 h-4 rounded-full z-10"></div>
      
      <div className="absolute top-8 left-16 bg-blue-500 w-3 h-3 rounded-full"></div>
      <div className="absolute top-16 right-12 bg-blue-500 w-3 h-3 rounded-full"></div>
      <div className="absolute bottom-12 left-8 bg-blue-500 w-3 h-3 rounded-full"></div>
      <div className="absolute bottom-16 right-16 bg-blue-500 w-3 h-3 rounded-full"></div>
      <div className="absolute top-12 left-32 bg-blue-500 w-3 h-3 rounded-full"></div>
      <div className="absolute bottom-8 left-24 bg-blue-500 w-3 h-3 rounded-full"></div>
      <div className="absolute top-20 right-20 bg-red-500 w-4 h-4 rounded-full"></div>
      <div className="absolute bottom-20 right-8 bg-blue-500 w-3 h-3 rounded-full"></div>
      
      <svg className="absolute inset-0 w-full h-full">
        <line x1="50%" y1="50%" x2="25%" y2="20%" stroke="#374151" strokeWidth="1" />
        <line x1="50%" y1="50%" x2="75%" y2="25%" stroke="#374151" strokeWidth="1" />
        <line x1="50%" y1="50%" x2="20%" y2="75%" stroke="#374151" strokeWidth="1" />
        <line x1="50%" y1="50%" x2="80%" y2="70%" stroke="#374151" strokeWidth="1" />
        <line x1="50%" y1="50%" x2="65%" y2="35%" stroke="#ef4444" strokeWidth="2" />
      </svg>
    </div>
    
    <div className="mt-4 p-4 bg-gray-700 rounded-lg">
      <div className="text-white font-medium">{data.centralUser}</div>
      <div className="text-blue-400 text-sm">Risk score: {data.riskScore}</div>
      <div className="text-gray-400 text-sm mt-1">{data.description}</div>
    </div>
  </div>
);
export default CommunicationGraph;