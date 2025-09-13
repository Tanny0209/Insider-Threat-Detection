import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

function IntroScreen({ onFinish }) {
  const steps = [
    "Initializing core modules...",
    "Scanning network topology...",
    "Calibrating threat detection engines...",
    "Verifying encryption keys...",
    "Launching Internal Eye dashboard..."
  ];

  const [currentText, setCurrentText] = useState("");
  const [stepIndex, setStepIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);

  useEffect(() => {
    if (stepIndex >= steps.length) {
      // all steps done → trigger exit
      const timer = setTimeout(() => onFinish?.(), 600);
      return () => clearTimeout(timer);
    }

    if (charIndex < steps[stepIndex].length) {
      // typing current line
      const timer = setTimeout(() => {
        setCurrentText(prev => prev + steps[stepIndex][charIndex]);
        setCharIndex(c => c + 1);
      }, 40);
      return () => clearTimeout(timer);
    } else {
      // finished a line → short pause, then next
      const timer = setTimeout(() => {
        setStepIndex(i => i + 1);
        setCharIndex(0);
        setCurrentText("");
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [charIndex, stepIndex, steps, onFinish]);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-black text-green-400 font-mono">
      {/* Spinning scanner ring */}
      <motion.div
        initial={{ scale: 0.7, opacity: 0 }}
        animate={{ scale: 1, opacity: 1, rotate: 360 }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        className="w-24 h-24 mb-10 rounded-full border-4 border-green-500 border-t-transparent shadow-[0_0_25px_#22c55e]"
      />

      {/* Typing boot messages */}
      <div className="text-lg md:text-xl whitespace-pre text-center mb-8">
        {currentText}
        <motion.span
          animate={{ opacity: [0, 1, 0] }}
          transition={{ duration: 0.8, repeat: Infinity }}
          className="inline-block w-2 bg-green-400 ml-1"
        />
      </div>

      {/* BIG Internal Eye branding */}
      <motion.h1
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.2, delay: 0.4 }}
        className="text-4xl md:text-6xl font-bold text-green-400 drop-shadow-[0_0_25px_#22c55e] tracking-wide"
      >
        INTERNAL EYE v1.0
      </motion.h1>
    </div>
  );
}

export default IntroScreen;
