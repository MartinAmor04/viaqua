import React, { useState, useEffect } from "react";
import "../styles/Header.css";

function Header() {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("darkMode") === "enabled";
  });

  const [isShrunk, setIsShrunk] = useState(false);

  useEffect(() => {
    // Manejo del modo oscuro
    if (darkMode) {
      document.documentElement.classList.add("dark-mode");
      localStorage.setItem("darkMode", "enabled");
    } else {
      document.documentElement.classList.remove("dark-mode");
      localStorage.setItem("darkMode", "disabled");
    }
  }, [darkMode]);

  useEffect(() => {
    // Manejo del scroll para shrink
    const handleScroll = () => {
      setIsShrunk(window.scrollY > 1);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header className={`header ${isShrunk ? "shrink" : ""}`}>
      <div>
        <h1 className="header-title">
          <span className="curux">Curux</span><span className="ia">IA</span>
        </h1>
        <p className="header-subtitle">Mantemento preditivo a baixo custe</p>
      </div>


    </header>
  );
}

export default Header;
