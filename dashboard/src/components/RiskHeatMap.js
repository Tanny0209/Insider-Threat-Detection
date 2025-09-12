export default function RiskHeatmap() {
  const departments = ["HR", "IT", "Finance", "Sales"];
  const levels = ["Low", "Medium", "High"];

  return (
    <div className="bg-gray-800 p-4 rounded-lg text-white">
      <h3 className="mb-2">Risk Heatmap</h3>
      <table className="w-full text-center">
        <thead>
          <tr>
            <th>Department</th>
            {levels.map(level => <th key={level}>{level}</th>)}
          </tr>
        </thead>
        <tbody>
          {departments.map(dep => (
            <tr key={dep}>
              <td>{dep}</td>
              {levels.map((_, i) => (
                <td key={i} className={`p-2 ${i === 0 ? "bg-green-600" : i === 1 ? "bg-yellow-500" : "bg-red-600"}`}></td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
