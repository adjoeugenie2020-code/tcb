import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Code, Download } from 'lucide-react';

// Hook Typewriter
const useTypewriter = (text, speed = 120, delay = 5000) => {
  const [displayed, setDisplayed] = useState('');
  const [loop, setLoop] = useState(0);

  useEffect(() => {
    let i = 0;
    setDisplayed('');
    const interval = setInterval(() => {
      setDisplayed(text.slice(0, i + 1));
      i++;
      if (i === text.length) {
        clearInterval(interval);
        setTimeout(() => setLoop((prev) => prev + 1), delay);
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed, delay, loop]);

  return displayed;
};

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();
  const typewriterText = useTypewriter('Rénato TCHOBO', 100, 5000);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { name: 'Accueil', path: '/' },
    { name: 'À propos', path: '/about' },
    { name: 'Services', path: '/services' },
    { name: 'Projets', path: '/projects' },
    { name: 'Contact', path: '/contact' },
  ];

  const handleDownloadCV = () => {
    window.open('https://drive.google.com/file/d/1QCe5LURLEj85PAyQwtwBe4iag1reGtFs/view?usp=sharing', '_blank');
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      {/* Barre du drapeau du Bénin */}
      <div className="flex w-full h-1">
        <div className="flex-1 bg-[#009739]"></div>
        <div className="flex-1 bg-[#FCD116]"></div>
        <div className="flex-1 bg-[#EF3340]"></div>
      </div>

      {/* Header principal */}
      <div className="bg-primary-500 shadow-lg">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-3">
            {/* Logo + Nom */}
            <Link to="/" className="flex items-center space-x-3 group">
              <div className="p-2 bg-white/10 rounded-lg group-hover:bg-white/20 transition-all duration-300">
                <Code className="h-6 w-6 text-white" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold font-heading text-white">
                  {typewriterText}
                </h1>
                <p className="text-sm text-primary-100">Solutions Digitales</p>
              </div>
            </Link>

            {/* Navigation Desktop */}
            <nav className="hidden lg:flex items-center space-x-1">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                    location.pathname === item.path
                      ? 'bg-white text-primary-500 shadow-md'
                      : 'text-white hover:bg-white/10 hover:text-white'
                  }`}
                >
                  {item.name}
                </Link>
              ))}
            </nav>

            {/* CV + Menu Mobile */}
            <div className="flex items-center space-x-3">
              <button
                onClick={handleDownloadCV}
                className="hidden sm:flex items-center space-x-2 bg-white text-primary-500 px-4 py-2 rounded-lg font-semibold hover:bg-primary-50 transition-all duration-300 hover:scale-105"
              >
                <Download className="h-4 w-4" />
                <span className="font-medium">CV</span>
              </button>

              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="lg:hidden p-2 rounded-lg text-white hover:bg-white/10 transition-colors duration-300"
              >
                {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>
          </div>

          {/* Menu Mobile */}
          {isMenuOpen && (
            <div className="lg:hidden pb-4 animate-slide-down">
              <div className="bg-white/95 backdrop-blur-sm rounded-lg shadow-lg py-2">
                {navItems.map((item) => (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setIsMenuOpen(false)}
                    className={`block px-4 py-3 font-medium transition-colors duration-300 ${
                      location.pathname === item.path
                        ? 'text-primary-500 bg-primary-50'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {item.name}
                  </Link>
                ))}
                <div className="px-4 py-2 border-t border-gray-200 mt-2">
                  <button
                    onClick={() => {
                      handleDownloadCV();
                      setIsMenuOpen(false);
                    }}
                    className="w-full flex items-center justify-center space-x-2 bg-primary-500 text-white px-4 py-3 rounded-lg font-semibold hover:bg-primary-600 transition-all duration-300"
                  >
                    <Download className="h-4 w-4" />
                    <span className="font-medium">Télécharger CV</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;