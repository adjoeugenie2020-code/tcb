import React, { useState } from 'react';
import { ExternalLink, Github, Filter, Code, Smartphone, Globe, ShoppingCart, Palette, TrendingUp } from 'lucide-react';
import { useScrollAnimation } from '../hooks/useScrollAnimation';

const Projects = () => {
  const [activeFilter, setActiveFilter] = useState('all');
  const { elementRef: heroRef, isVisible: heroVisible } = useScrollAnimation();
  const { elementRef: projectsRef, isVisible: projectsVisible } = useScrollAnimation();

  const filters = [
    { id: 'all', name: 'Tous les projets', icon: Filter },
    { id: 'web', name: 'Sites Web', icon: Globe },
    { id: 'mobile', name: 'Applications Mobiles', icon: Smartphone },
    { id: 'ecommerce', name: 'E-commerce', icon: ShoppingCart },
    { id: 'design', name: 'UI/UX Design', icon: Palette },
    { id: 'marketing', name: 'Marketing Digital', icon: TrendingUp }
  ];

  const projects = [
    {
      id: 1,
      title: 'Générateur de Devis Pro',
      description: 'Application web permettant aux entreprises de créer des devis professionnels en quelques clics avec gestion client intégrée.',
      image: 'https://i.ibb.co/KxzcV324/e1d17e5bf3e943d05805532f0f38d37e.jpg',
      category: 'web',
      technologies: ['React', 'Node.js', 'MongoDB', 'Express'],
      features: ['Interface intuitive', 'Génération PDF', 'Gestion clients', 'Analytics'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'Terminé'
    },
    {
      id: 2,
      title: 'App Campus Connect',
      description: 'Application mobile pour étudiants permettant la gestion des cours, emplois du temps et communication entre étudiants.',
      image: 'https://i.ibb.co/v6g1M2YL/5f2667b65d58d4234423f4ab2f4df4bc.jpg',
      category: 'mobile',
      technologies: ['React Native', 'Firebase', 'Redux', 'TypeScript'],
      features: ['Emploi du temps', 'Chat intégré', 'Notifications push', 'Hors ligne'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'En cours'
    },
    {
      id: 3,
      title: 'Plateforme Couturières Bénin',
      description: 'Marketplace connectant les couturières locales aux clients avec système de commandes et paiements intégrés.',
      image: 'https://i.ibb.co/whr47sHP/4ee80c0a4ccaa205623b99e1d896d6e3.jpg',
      category: 'ecommerce',
      technologies: ['Vue.js', 'Laravel', 'Stripe', 'MySQL'],
      features: ['Marketplace', 'Paiements sécurisés', 'Géolocalisation', 'Reviews'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'Terminé'
    },
    {
      id: 4,
      title: 'Dashboard Analytics Pro',
      description: 'Interface administrateur pour l\'analyse de données business avec graphiques interactifs et rapports automatisés.',
      image: 'https://i.ibb.co/5hznftfM/6317302b76cc82a0feab8d0dfd635c8f.jpg',
      category: 'web',
      technologies: ['React', 'D3.js', 'Python', 'PostgreSQL'],
      features: ['Graphiques temps réel', 'Export rapports', 'Alertes automatiques', 'Multi-utilisateurs'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'Terminé'
    },
    {
      id: 5,
      title: 'AgriTech Mobile App',
      description: 'Application pour agriculteurs avec conseils personnalisés, météo et marketplace de produits agricoles.',
      image: 'https://i.ibb.co/zTrBwqR9/0a1c9f12f31fe53edd4ab730ef331336.jpg',
      category: 'mobile',
      technologies: ['Flutter', 'Firebase', 'Google Maps', 'Weather API'],
      features: ['Conseils agricoles', 'Météo locale', 'Marketplace', 'Communauté'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'En cours'
    },
    {
      id: 6,
      title: 'Boutique Mode Africaine',
      description: 'Site e-commerce spécialisé dans la mode africaine avec système de tailles personnalisé et livraison internationale.',
      image: 'https://i.ibb.co/qLwXLBSX/18584c030e09e5feed13f602132596e9.jpg',
      category: 'ecommerce',
      technologies: ['Shopify', 'JavaScript', 'CSS3', 'Apps personnalisées'],
      features: ['Tailles personnalisées', 'Livraison internationale', 'Wishlist', 'Programme fidélité'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'Terminé'
    },
    {
      id: 7,
      title: 'Redesign UX FinTech App',
      description: 'Refonte complète de l\'expérience utilisateur d\'une application de services financiers avec tests utilisateurs.',
      image: 'https://i.ibb.co/W4cGW7JT/bb2df69bc5a55b6bd868ac743f36188b.jpg',
      category: 'design',
      technologies: ['Figma', 'Principle', 'User Testing', 'Design System'],
      features: ['UX Research', 'Prototypage', 'Tests utilisateurs', 'Design system'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'Terminé'
    },
    {
      id: 8,
      title: 'Campagne SEO Restaurant',
      description: 'Stratégie SEO complète pour chaîne de restaurants avec amélioration de 300% du trafic organique.',
      image: 'https://i.ibb.co/DP80hbM7/73249505ebd172eb1d2b4130e8fecc1c.jpg',
      category: 'marketing',
      technologies: ['SEO Tools', 'Google Analytics', 'Content Strategy', 'Local SEO'],
      features: ['Audit SEO', 'Content strategy', 'Local SEO', 'Link building'],
      liveUrl: '#',
      githubUrl: '#',
      status: 'Terminé'
    }
  ];

  const filteredProjects = activeFilter === 'all' 
    ? projects 
    : projects.filter(project => project.category === activeFilter);

  const stats = [
    { number: '50+', label: 'Projets Réalisés' },
    { number: '30+', label: 'Clients Satisfaits' },
    { number: '98%', label: 'Projets Livrés à Temps' },
    { number: '5★', label: 'Note Moyenne' }
  ];

  return (
    <div className="pt-20">
      {/* Hero Section */}
      <section className="py-20 bg-gradient-to-br from-primary-50 to-blue-50">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div 
            ref={heroRef}
            className={`max-w-4xl mx-auto text-center transition-all duration-1000 ${
              heroVisible ? 'animate-slide-up' : 'opacity-0'
            }`}
          >
            <h1 className="text-4xl md:text-5xl font-bold font-heading text-gray-900 mb-6">
              Mes <span className="text-primary-500">Projets</span>
            </h1>
            <p className="text-xl text-gray-600 leading-relaxed mb-8">
              Découvrez une sélection de mes réalisations récentes, des solutions digitales 
              qui ont transformé l'activité de mes clients.
            </p>
            
            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-2xl mx-auto">
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-2xl md:text-3xl font-bold font-heading text-primary-500 mb-1">
                    {stat.number}
                  </div>
                  <div className="text-gray-600 text-sm">
                    {stat.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Filters */}
      <section className="py-8 bg-white sticky top-16 z-40 border-b border-gray-200">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-wrap justify-center gap-2">
            {filters.map((filter) => (
              <button
                key={filter.id}
                onClick={() => setActiveFilter(filter.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-300 ${
                  activeFilter === filter.id
                    ? 'bg-primary-500 text-white shadow-lg'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <filter.icon className="h-4 w-4" />
                <span>{filter.name}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Projects Grid */}
      <section className="py-20 bg-gray-50">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div 
            ref={projectsRef}
            className={`transition-all duration-1000 ${
              projectsVisible ? 'animate-slide-up' : 'opacity-0'
            }`}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {filteredProjects.map((project, index) => (
                <div 
                  key={project.id} 
                  className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-2 overflow-hidden group"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="relative overflow-hidden">
                    <img 
                      src={project.image} 
                      alt={project.title}
                      className="w-full h-48 object-cover group-hover:scale-110 transition-transform duration-500"
                    />
                    <div className="absolute top-4 right-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        project.status === 'Terminé' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {project.status}
                      </span>
                    </div>
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300 flex items-center justify-center">
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex space-x-3">
                        <a 
                          href={project.liveUrl}
                          className="p-3 bg-white rounded-full hover:bg-primary-500 hover:text-white transition-colors duration-300"
                          aria-label="Voir le projet"
                        >
                          <ExternalLink className="h-5 w-5" />
                        </a>
                        <a 
                          href={project.githubUrl}
                          className="p-3 bg-white rounded-full hover:bg-primary-500 hover:text-white transition-colors duration-300"
                          aria-label="Voir le code"
                        >
                          <Github className="h-5 w-5" />
                        </a>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-6">
                    <h3 className="text-xl font-semibold font-heading text-gray-900 mb-2">
                      {project.title}
                    </h3>
                    <p className="text-gray-600 mb-4 leading-relaxed">
                      {project.description}
                    </p>
                    
                    {/* Technologies */}
                    <div className="mb-4">
                      <div className="flex flex-wrap gap-2">
                        {project.technologies.map((tech, techIndex) => (
                          <span 
                            key={techIndex}
                            className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs font-medium"
                          >
                            {tech}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    {/* Features */}
                    <div className="mb-6">
                      <h4 className="text-sm font-semibold text-gray-900 mb-2">Fonctionnalités clés :</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        {project.features.slice(0, 3).map((feature, featureIndex) => (
                          <li key={featureIndex} className="flex items-center">
                            <div className="w-1.5 h-1.5 bg-primary-500 rounded-full mr-2"></div>
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    {/* Actions */}
                    <div className="flex space-x-3">
                      <a 
                        href={project.liveUrl}
                        className="flex-1 bg-primary-500 text-white text-center py-2 px-4 rounded-lg hover:bg-primary-600 transition-colors duration-300 text-sm font-medium"
                      >
                        Voir le projet
                      </a>
                      <a 
                        href={project.githubUrl}
                        className="flex-1 border border-gray-300 text-gray-700 text-center py-2 px-4 rounded-lg hover:bg-gray-50 transition-colors duration-300 text-sm font-medium"
                      >
                        Code source
                      </a>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {filteredProjects.length === 0 && (
              <div className="text-center py-16">
                <Code className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-600 mb-2">
                  Aucun projet trouvé
                </h3>
                <p className="text-gray-500">
                  Essayez un autre filtre pour voir d'autres projets.
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Process Section */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold font-heading text-gray-900 mb-4">
              Comment je travaille
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Chaque projet suit une méthodologie rigoureuse pour garantir des résultats exceptionnels
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                step: '01',
                title: 'Analyse',
                description: 'Étude approfondie de vos besoins et de votre marché cible.'
              },
              {
                step: '02',
                title: 'Conception',
                description: 'Création des maquettes et définition de l\'architecture technique.'
              },
              {
                step: '03',
                title: 'Développement',
                description: 'Codage et intégration avec feedback régulier et tests continus.'
              },
              {
                step: '04',
                title: 'Déploiement',
                description: 'Mise en ligne, formation et accompagnement post-lancement.'
              }
            ].map((step, index) => (
              <div key={index} className="text-center">
                <div className="w-16 h-16 bg-primary-500 text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold font-heading">
                  {step.step}
                </div>
                <h3 className="text-xl font-semibold font-heading text-gray-900 mb-2">
                  {step.title}
                </h3>
                <p className="text-gray-600">
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary-500">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold font-heading text-white mb-4">
            Votre projet mérite le même niveau d'excellence
          </h2>
          <p className="text-xl text-primary-100 mb-8 max-w-2xl mx-auto">
            Discutons de votre vision et créons ensemble une solution digitale qui marquera la différence.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="/contact"
              className="bg-white text-primary-500 px-8 py-4 rounded-lg font-semibold hover:bg-gray-100 transition-all duration-300 hover:scale-105"
            >
              Démarrer un projet
            </a>
            <a
              href="/services"
              className="border-2 border-white text-white px-8 py-4 rounded-lg font-semibold hover:bg-white hover:text-primary-500 transition-all duration-300"
            >
              Voir mes services
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Projects;