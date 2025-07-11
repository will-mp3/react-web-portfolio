import React, { useState } from 'react';
import { Github, Linkedin, Mail, ExternalLink, Code, Play, ArrowLeft } from 'lucide-react';

// Sample project data - replace with your actual projects
const projects = [
  {
    id: 1,
    title: "E-Commerce Platform",
    description: "Full-stack e-commerce application with user authentication, shopping cart, and payment integration.",
    technologies: ["React", "Node.js", "MongoDB", "Stripe"],
    demoVideo: "https://example.com/demo1.mp4",
    screenshots: [
      "https://via.placeholder.com/600x400/3b82f6/ffffff?text=Homepage",
      "https://via.placeholder.com/600x400/10b981/ffffff?text=Product+Page",
      "https://via.placeholder.com/600x400/f59e0b/ffffff?text=Cart"
    ],
    codeSnippet: `// User authentication middleware
const authenticateUser = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    if (!token) {
      return res.status(401).json({ message: 'No token provided' });
    }
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    res.status(401).json({ message: 'Invalid token' });
  }
};`,
    githubUrl: "https://github.com/yourusername/ecommerce-platform",
    liveUrl: "https://your-ecommerce-demo.com",
    features: [
      "User authentication and authorization",
      "Product catalog with search and filtering",
      "Shopping cart and checkout process",
      "Payment integration with Stripe",
      "Order tracking and history",
      "Admin dashboard for inventory management"
    ]
  },
  {
    id: 2,
    title: "Task Management App",
    description: "Collaborative task management application with real-time updates and team collaboration features.",
    technologies: ["React", "Socket.io", "Express", "PostgreSQL"],
    demoVideo: "https://example.com/demo2.mp4",
    screenshots: [
      "https://via.placeholder.com/600x400/8b5cf6/ffffff?text=Dashboard",
      "https://via.placeholder.com/600x400/ef4444/ffffff?text=Task+Board",
      "https://via.placeholder.com/600x400/06b6d4/ffffff?text=Team+View"
    ],
    codeSnippet: `// Real-time task updates with Socket.io
socket.on('taskUpdated', (updatedTask) => {
  setTasks(prevTasks => 
    prevTasks.map(task => 
      task.id === updatedTask.id ? updatedTask : task
    )
  );
});

const updateTask = async (taskId, updates) => {
  try {
    const response = await fetch(\`/api/tasks/\${taskId}\`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    });
    
    if (response.ok) {
      socket.emit('taskUpdate', { taskId, updates });
    }
  } catch (error) {
    console.error('Failed to update task:', error);
  }
};`,
    githubUrl: "https://github.com/yourusername/task-manager",
    liveUrl: "https://your-task-manager-demo.com",
    features: [
      "Real-time collaboration with Socket.io",
      "Drag-and-drop task organization",
      "Team member management",
      "Project timelines and deadlines",
      "File attachments and comments",
      "Email notifications and reminders"
    ]
  },
  {
    id: 3,
    title: "Weather Dashboard",
    description: "Real-time weather dashboard with location-based forecasts and interactive charts.",
    technologies: ["React", "Chart.js", "OpenWeather API", "Geolocation"],
    demoVideo: "https://example.com/demo3.mp4",
    screenshots: [
      "https://via.placeholder.com/600x400/14b8a6/ffffff?text=Weather+Dashboard",
      "https://via.placeholder.com/600x400/f97316/ffffff?text=Forecast+Chart",
      "https://via.placeholder.com/600x400/84cc16/ffffff?text=Location+Search"
    ],
    codeSnippet: `// Weather API integration with error handling
const fetchWeatherData = async (city) => {
  try {
    const response = await fetch(
      \`https://api.openweathermap.org/data/2.5/weather?q=\${city}&appid=\${API_KEY}&units=metric\`
    );
    
    if (!response.ok) {
      throw new Error('Weather data not found');
    }
    
    const data = await response.json();
    return {
      temperature: data.main.temp,
      description: data.weather[0].description,
      humidity: data.main.humidity,
      windSpeed: data.wind.speed
    };
  } catch (error) {
    console.error('Error fetching weather:', error);
    return null;
  }
};`,
    githubUrl: "https://github.com/yourusername/weather-dashboard",
    liveUrl: "https://your-weather-demo.com",
    features: [
      "Current weather conditions",
      "7-day weather forecast",
      "Interactive charts and graphs",
      "Location-based weather detection",
      "Search for weather by city",
      "Weather alerts and notifications"
    ]
  }
];

// Home Page Component
const HomePage = ({ onProjectClick }) => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900">Your Name</h1>
            <nav className="flex space-x-6">
              <a href="#about" className="text-gray-600 hover:text-gray-900">About</a>
              <a href="#projects" className="text-gray-600 hover:text-gray-900">Projects</a>
              <a href="#contact" className="text-gray-600 hover:text-gray-900">Contact</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-600 to-purple-700 text-white py-20">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-5xl font-bold mb-6">Full-Stack Developer</h2>
          <p className="text-xl mb-8 opacity-90">
            Building modern web applications with clean code and exceptional user experiences
          </p>
          <div className="flex justify-center space-x-4">
            <a href="#projects" className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition">
              View Projects
            </a>
            <a href="#contact" className="border-2 border-white px-6 py-3 rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition">
              Get in Touch
            </a>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20">
        <div className="max-w-4xl mx-auto px-4">
          <h3 className="text-3xl font-bold text-center mb-12">About Me</h3>
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <p className="text-lg text-gray-600 mb-6">
                [Replace with your bio - talk about your background, passion for coding, and what drives you as a developer]
              </p>
              <p className="text-lg text-gray-600 mb-6">
                [Add more about your experience, learning journey, and what you enjoy building]
              </p>
              <div className="flex space-x-4">
                <a href="[Your Resume URL]" className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition">
                  <span>Resume</span>
                  <ExternalLink size={16} />
                </a>
                <a href="[Your GitHub URL]" className="flex items-center space-x-2 bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-900 transition">
                  <Github size={16} />
                  <span>GitHub</span>
                </a>
              </div>
            </div>
            <div className="bg-gray-200 rounded-lg h-80 flex items-center justify-center">
              <span className="text-gray-500">[Your Photo Here]</span>
            </div>
          </div>
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="py-20 bg-gray-100">
        <div className="max-w-6xl mx-auto px-4">
          <h3 className="text-3xl font-bold text-center mb-12">Featured Projects</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {projects.map(project => (
              <div key={project.id} className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition">
                <div className="h-48 bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <span className="text-white font-semibold">[Project Screenshot]</span>
                </div>
                <div className="p-6">
                  <h4 className="text-xl font-semibold mb-2">{project.title}</h4>
                  <p className="text-gray-600 mb-4">{project.description}</p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.technologies.map(tech => (
                      <span key={tech} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                        {tech}
                      </span>
                    ))}
                  </div>
                  <button 
                    onClick={() => onProjectClick(project.id)}
                    className="inline-flex items-center space-x-2 text-blue-600 hover:text-blue-800 font-semibold"
                  >
                    <span>View Details</span>
                    <ExternalLink size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h3 className="text-3xl font-bold mb-8">Get In Touch</h3>
          <p className="text-lg text-gray-600 mb-8">
            I'm always open to discussing new opportunities and interesting projects.
          </p>
          <div className="flex justify-center space-x-6">
            <a href="mailto:your.email@example.com" className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition">
              <Mail size={20} />
              <span>Email</span>
            </a>
            <a href="[Your LinkedIn URL]" className="flex items-center space-x-2 bg-blue-800 text-white px-6 py-3 rounded-lg hover:bg-blue-900 transition">
              <Linkedin size={20} />
              <span>LinkedIn</span>
            </a>
            <a href="[Your GitHub URL]" className="flex items-center space-x-2 bg-gray-800 text-white px-6 py-3 rounded-lg hover:bg-gray-900 transition">
              <Github size={20} />
              <span>GitHub</span>
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

// Project Detail Page Component
const ProjectPage = ({ project, onBackClick }) => {
  if (!project) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Project Not Found</h2>
          <button 
            onClick={onBackClick}
            className="text-blue-600 hover:text-blue-800"
          >
            Back to Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <button 
              onClick={onBackClick}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
            >
              <ArrowLeft size={20} />
              <span>Back to Portfolio</span>
            </button>
            <h1 className="text-2xl font-bold text-gray-900">Your Name</h1>
          </div>
        </div>
      </header>

      {/* Project Content */}
      <div className="max-w-6xl mx-auto px-4 py-12">
        <div className="mb-8">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">{project.title}</h2>
          <p className="text-xl text-gray-600 mb-6">{project.description}</p>
          
          <div className="flex flex-wrap gap-2 mb-6">
            {project.technologies.map(tech => (
              <span key={tech} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                {tech}
              </span>
            ))}
          </div>

          <div className="flex space-x-4">
            <a href={project.githubUrl} className="flex items-center space-x-2 bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-900 transition">
              <Code size={16} />
              <span>View Code</span>
            </a>
            <a href={project.liveUrl} className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition">
              <ExternalLink size={16} />
              <span>Live Demo</span>
            </a>
          </div>
        </div>

        {/* Demo Video Section */}
        <div className="mb-12">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Demo Video</h3>
          <div className="bg-gray-800 rounded-lg p-8 text-center">
            <Play size={48} className="mx-auto text-white mb-4" />
            <p className="text-white mb-4">Demo video placeholder</p>
            <p className="text-gray-400 text-sm">Replace with actual video embed: {project.demoVideo}</p>
          </div>
        </div>

        {/* Screenshots Section */}
        <div className="mb-12">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Screenshots</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {project.screenshots.map((screenshot, index) => (
              <div key={index} className="bg-gray-200 rounded-lg overflow-hidden">
                <img 
                  src={screenshot} 
                  alt={`Screenshot ${index + 1}`}
                  className="w-full h-48 object-cover"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Code Snippet Section */}
        <div className="mb-12">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Code Snippet</h3>
          <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
            <pre className="text-green-400 text-sm">
              <code>{project.codeSnippet}</code>
            </pre>
          </div>
        </div>

        {/* Features Section */}
        <div>
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Key Features</h3>
          <div className="bg-white rounded-lg p-6 shadow-sm">
            <p className="text-gray-600 mb-4">
              This project showcases several important features and technical capabilities:
            </p>
            <ul className="space-y-2 text-gray-600">
              {project.features.map((feature, index) => (
                <li key={index}>• {feature}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [currentView, setCurrentView] = useState('home');
  const [selectedProject, setSelectedProject] = useState(null);

  const handleProjectClick = (projectId) => {
    const project = projects.find(p => p.id === projectId);
    setSelectedProject(project);
    setCurrentView('project');
  };

  const handleBackClick = () => {
    setCurrentView('home');
    setSelectedProject(null);
  };

  return (
    <div>
      {currentView === 'home' && (
        <HomePage onProjectClick={handleProjectClick} />
      )}
      {currentView === 'project' && (
        <ProjectPage 
          project={selectedProject} 
          onBackClick={handleBackClick} 
        />
      )}
    </div>
  );
};

export default App;