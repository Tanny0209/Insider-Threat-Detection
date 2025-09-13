// import React, { useState, useEffect } from "react";

// const Navbar = () => {
//   const [currentTime, setCurrentTime] = useState(new Date());

//   useEffect(() => {
//     const timer = setInterval(() => setCurrentTime(new Date()), 1000);
//     return () => clearInterval(timer);
//   }, []);

//   return (
//     <div className="flex justify-between items-center p-4 bg-gray-800 text-white">
//       <h1 className="text-xl font-bold">Dashboard</h1>
//       <span>{currentTime.toLocaleString()}</span>
//     </div>
//   );
// };

// export default Navbar;
