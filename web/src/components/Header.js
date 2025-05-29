import React, { useState, useEffect } from "react";
import "../styles/Header.css";

function Header() {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("darkMode") === "enabled";
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark-mode"); // ✅ Aplica al <html>
      localStorage.setItem("darkMode", "enabled");
    } else {
      document.documentElement.classList.remove("dark-mode");
      localStorage.setItem("darkMode", "disabled");
    }
  }, [darkMode]);

  return (
    <div>

       <header className="header">
      <h1 className="header-title">
        <span className="curux">Curux</span><span className="ia">IA</span>
      </h1>
            <img src={require("../styles/img/favicon.png")} alt="CuruxIA Logo" className="header-logo" />


      {/* 🔄 Botón de modo oscuro */}
      <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
        {darkMode ? "Modo Claro" : "Modo Oscuro"}
      </button>
    </header>
    <div class="sub-header">
      Mantemento preditivo a baixo custo

      </div>

    </div>
   
  );
}

export default Header;

