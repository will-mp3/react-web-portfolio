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
    <div className="min-h-screen bg-gray-300" style={{
      backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(0,0,0,.05) 2px, rgba(0,0,0,.05) 4px)`
    }}>
      {/* Header - Classic Mac Menu Bar */}
      <header className="bg-gray-200 border-b-2 border-black">
        <div className="max-w-6xl mx-auto px-4 py-2">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-mono font-bold text-black">WILL.KATABIAN</h1>
            </div>
            <nav className="flex space-x-1">
              <a href="#about" className="px-3 py-1 text-black hover:bg-blue-500 hover:text-white font-mono text-sm border border-black transition">About</a>
              <a href="#projects" className="px-3 py-1 text-black hover:bg-green-500 hover:text-white font-mono text-sm border border-black transition">Projects</a>
              <a href="#contact" className="px-3 py-1 text-black hover:bg-yellow-500 hover:text-black font-mono text-sm border border-black transition">Contact</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section - Classic Mac Desktop */}
      <section className="bg-gray-300 text-black py-20 border-b-4 border-black" style={{
        backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(0,0,0,.1) 2px, rgba(0,0,0,.1) 4px)`
      }}>
        <div className="max-w-4xl mx-auto px-4 text-center">
          <div className="bg-white border-4 border-black p-8 shadow-lg" style={{
            boxShadow: '8px 8px 0px #000'
          }}>
            <h2 className="text-4xl font-mono font-bold mb-6 text-black">FULL-STACK.DEVELOPER</h2>
            <div className="bg-black text-white p-4 font-mono text-sm mb-6">
              <div className="text-green-400">$ whoami</div>
              <div>Building modern web applications with clean code...</div>
              <div className="text-blue-400 mt-2">$ ls skills/</div>
              <div className="text-yellow-300">react.js node.js python.sql mongodb.git</div>
            </div>
            <div className="flex justify-center space-x-4">
              <a href="#projects" className="bg-blue-500 text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-blue-600 transition shadow-lg" style={{boxShadow: '4px 4px 0px #000'}}>
                VIEW.PROJECTS
              </a>
              <a href="#contact" className="bg-green-500 text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-green-600 transition shadow-lg" style={{boxShadow: '4px 4px 0px #000'}}>
                CONTACT.ME
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* About Section - Mac Window Style */}
      <section id="about" className="py-20">
        <div className="max-w-4xl mx-auto px-4">
          <div className="bg-white border-4 border-black mb-8" style={{
            boxShadow: '8px 8px 0px #000'
          }}>
            <div className="bg-gray-200 border-b-2 border-black p-2 flex items-center">
              <div className="flex space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <div className="flex-1 text-center">
                <span className="font-mono text-sm font-bold">ABOUT.ME</span>
              </div>
            </div>
            <div className="p-8">
              <div className="grid md:grid-cols-2 gap-12 items-center">
                <div>
                  <p className="font-mono text-sm text-black mb-4">
                    &gt; INITIALIZING DEVELOPER PROFILE...<br/>
                    &gt; LOADING BACKGROUND DATA...
                  </p>
                  <p className="text-black mb-6">
                    [Replace with your bio - talk about your background, passion for coding, and what drives you as a developer]
                  </p>
                  <p className="text-black mb-6">
                    [Add more about your experience, learning journey, and what you enjoy building]
                  </p>
                  <div className="flex space-x-4">
                    <a href="[Your Resume URL]" className="flex items-center space-x-2 bg-blue-500 text-white px-4 py-2 font-mono text-sm border-2 border-black hover:bg-blue-600 transition" style={{boxShadow: '3px 3px 0px #000'}}>
                      <span>RESUME.PDF</span>
                      <ExternalLink size={16} />
                    </a>
                    <a href="[Your GitHub URL]" className="flex items-center space-x-2 bg-yellow-500 text-black px-4 py-2 font-mono text-sm border-2 border-black hover:bg-yellow-600 transition" style={{boxShadow: '3px 3px 0px #000'}}>
                      <Github size={16} />
                      <span>GITHUB</span>
                    </a>
                  </div>
                </div>
                <div className="bg-black border-2 border-black h-80 flex flex-col">
                  <div className="bg-gray-200 p-2 border-b-2 border-black">
                    <span className="font-mono text-xs">PROFILE.PIC</span>
                  </div>
                  <div className="flex-1 flex items-center justify-center text-white font-mono">
                    [PHOTO_HERE.JPG]
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Projects Section - Mac Folder Style */}
      <section id="projects" className="py-20">
        <div className="max-w-6xl mx-auto px-4">
          <div className="bg-white border-4 border-black mb-8" style={{
            boxShadow: '8px 8px 0px #000'
          }}>
            <div className="bg-gray-200 border-b-2 border-black p-2 flex items-center">
              <div className="flex space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <div className="flex-1 text-center">
                <span className="font-mono text-sm font-bold">PROJECTS.FOLDER</span>
              </div>
            </div>
            <div className="p-8">
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {projects.map((project, index) => {
                  const colors = ['bg-blue-500', 'bg-red-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500', 'bg-pink-500'];
                  const bgColor = colors[index % colors.length];
                  return (
                    <div key={project.id} className="bg-gray-100 border-2 border-black hover:bg-white transition cursor-pointer" style={{
                      boxShadow: '4px 4px 0px #000'
                    }}>
                      <div className={`h-32 ${bgColor} flex items-center justify-center border-b-2 border-black`}>
                        <div className="text-white font-mono text-xs text-center">
                          <div className="w-8 h-8 bg-white mx-auto mb-2 border border-black"></div>
                          PROJECT.APP
                        </div>
                      </div>
                      <div className="p-4">
                        <h4 className="font-mono font-bold text-sm mb-2 text-black">{project.title.toUpperCase().replace(/\s+/g, '.')}</h4>
                        <p className="text-xs text-black mb-3">{project.description}</p>
                        <div className="flex flex-wrap gap-1 mb-3">
                          {project.technologies.map(tech => (
                            <span key={tech} className="bg-black text-white px-2 py-1 text-xs font-mono">
                              {tech.toUpperCase()}
                            </span>
                          ))}
                        </div>
                        <button 
                          onClick={() => onProjectClick(project.id)}
                          className={`font-mono text-xs text-white ${bgColor} hover:opacity-80 border border-black px-2 py-1 transition`}
                          style={{boxShadow: '2px 2px 0px #000'}}
                        >
                          OPEN.FILE
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact Section - Classic Mac Dialog */}
      <section id="contact" className="py-20">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <div className="bg-white border-4 border-black inline-block" style={{
            boxShadow: '8px 8px 0px #000'
          }}>
            <div className="bg-gray-200 border-b-2 border-black p-2 flex items-center">
              <div className="flex space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <div className="flex-1 text-center">
                <span className="font-mono text-sm font-bold">CONTACT.DIALOG</span>
              </div>
            </div>
            <div className="p-8">
              <div className="font-mono text-sm text-black mb-4">
                <span className="text-green-500">&gt; ESTABLISHING CONNECTION...</span><br/>
                <span className="text-blue-500">&gt; CONTACT PROTOCOLS READY</span>
              </div>
              <p className="text-black mb-6 max-w-md">
                I'm always open to discussing new opportunities and interesting projects.
              </p>
              <div className="flex flex-col space-y-3">
                <a href="mailto:your.email@example.com" className="flex items-center justify-center space-x-2 bg-red-500 text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-red-600 transition" style={{boxShadow: '4px 4px 0px #000'}}>
                  <Mail size={16} />
                  <span>SEND.EMAIL</span>
                </a>
                <a href="[Your LinkedIn URL]" className="flex items-center justify-center space-x-2 bg-blue-500 text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-blue-600 transition" style={{boxShadow: '4px 4px 0px #000'}}>
                  <Linkedin size={16} />
                  <span>LINKEDIN.CONNECT</span>
                </a>
                <a href="[Your GitHub URL]" className="flex items-center justify-center space-x-2 bg-green-500 text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-green-600 transition" style={{boxShadow: '4px 4px 0px #000'}}>
                  <Github size={16} />
                  <span>GITHUB.REPOS</span>
                </a>
              </div>
            </div>
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
      <div className="min-h-screen bg-gray-300 flex items-center justify-center" style={{
        backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(0,0,0,.05) 2px, rgba(0,0,0,.05) 4px)`
      }}>
        <div className="text-center">
          <h2 className="text-2xl font-mono font-bold text-black mb-4">PROJECT.NOT.FOUND</h2>
          <button 
            onClick={onBackClick}
            className="text-black hover:bg-black hover:text-white font-mono border border-black px-4 py-2"
          >
            BACK.TO.HOME
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-300" style={{
      backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(0,0,0,.05) 2px, rgba(0,0,0,.05) 4px)`
    }}>
      {/* Header - Classic Mac Menu Bar */}
      <header className="bg-gray-200 border-b-2 border-black">
        <div className="max-w-6xl mx-auto px-4 py-2">
          <div className="flex items-center justify-between">
            <button 
              onClick={onBackClick}
              className="flex items-center space-x-2 text-black hover:bg-red-500 hover:text-white px-2 py-1 font-mono text-sm border border-black transition"
            >
              <ArrowLeft size={16} />
              <span>BACK.TO.PORTFOLIO</span>
            </button>
            <h1 className="text-xl font-mono font-bold text-black">YOUR.NAME</h1>
          </div>
        </div>
      </header>

      {/* Project Content - Mac Application Window */}
      <div className="max-w-6xl mx-auto px-4 py-12">
        <div className="bg-white border-4 border-black" style={{
          boxShadow: '8px 8px 0px #000'
        }}>
          <div className="bg-gray-200 border-b-2 border-black p-2 flex items-center">
            <div className="flex space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            </div>
            <div className="flex-1 text-center">
              <span className="font-mono text-sm font-bold">{project.title.toUpperCase().replace(/\s+/g, '.')}</span>
            </div>
          </div>
          
          <div className="p-8">
            <div className="mb-8">
              <h2 className="text-3xl font-mono font-bold text-black mb-4">{project.title.toUpperCase()}</h2>
              <div className="bg-black text-green-400 p-4 font-mono text-sm mb-6">
                <div>$ cat project_description.txt</div>
                <div className="text-white mt-2">{project.description}</div>
              </div>
              
              <div className="flex flex-wrap gap-2 mb-6">
                {project.technologies.map(tech => (
                  <span key={tech} className="bg-black text-white px-3 py-1 font-mono text-xs border border-black">
                    {tech.toUpperCase()}
                  </span>
                ))}
              </div>

              <div className="flex space-x-4">
                <a href={project.githubUrl} className="flex items-center space-x-2 bg-blue-500 text-white px-4 py-2 font-mono text-sm border-2 border-black hover:bg-blue-600 transition" style={{boxShadow: '3px 3px 0px #000'}}>
                  <Code size={16} />
                  <span>VIEW.CODE</span>
                </a>
                <a href={project.liveUrl} className="flex items-center space-x-2 bg-green-500 text-white px-4 py-2 font-mono text-sm border-2 border-black hover:bg-green-600 transition" style={{boxShadow: '3px 3px 0px #000'}}>
                  <ExternalLink size={16} />
                  <span>LIVE.DEMO</span>
                </a>
              </div>
            </div>

            {/* Demo Video Section */}
            <div className="mb-12">
              <h3 className="text-xl font-mono font-bold text-black mb-4 border-b-2 border-black pb-2">DEMO.VIDEO</h3>
              <div className="bg-black border-2 border-black p-8 text-center">
                <Play size={32} className="mx-auto text-white mb-4" />
                <p className="text-green-400 font-mono text-sm mb-2">LOADING VIDEO PLAYER...</p>
                <p className="text-gray-400 font-mono text-xs">FILE: {project.demoVideo}</p>
              </div>
            </div>

            {/* Screenshots Section */}
            <div className="mb-12">
              <h3 className="text-xl font-mono font-bold text-black mb-4 border-b-2 border-black pb-2">SCREENSHOTS.GALLERY</h3>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {project.screenshots.map((screenshot, index) => (
                  <div key={index} className="bg-black border-2 border-black">
                    <div className="bg-gray-200 p-1 border-b border-black">
                      <span className="font-mono text-xs">IMG_{index + 1}.BMP</span>
                    </div>
                    <img 
                      src={screenshot} 
                      alt={`Screenshot ${index + 1}`}
                      className="w-full h-32 object-cover"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Code Snippet Section */}
            <div className="mb-12">
              <h3 className="text-xl font-mono font-bold text-black mb-4 border-b-2 border-black pb-2">CODE.SAMPLE</h3>
              <div className="bg-black border-2 border-black">
                <div className="bg-gray-200 p-2 border-b border-black flex items-center">
                  <span className="font-mono text-xs">SOURCE.CODE</span>
                  <div className="ml-auto flex space-x-1">
                    <div className="w-2 h-2 bg-black"></div>
                    <div className="w-2 h-2 bg-black"></div>
                  </div>
                </div>
                <div className="p-4 overflow-x-auto">
                  <pre className="text-green-400 text-xs font-mono">
                    <code>{project.codeSnippet}</code>
                  </pre>
                </div>
              </div>
            </div>

            {/* Features Section */}
            <div>
              <h3 className="text-xl font-mono font-bold text-black mb-4 border-b-2 border-black pb-2">FEATURES.LIST</h3>
              <div className="bg-gray-100 border-2 border-black p-6">
                <div className="bg-black text-green-400 p-2 font-mono text-xs mb-4">
                  &gt; LOADING PROJECT SPECIFICATIONS...
                </div>
                <ul className="space-y-2 text-black font-mono text-sm">
                  {project.features.map((feature, index) => (
                    <li key={index}>• {feature.toUpperCase()}</li>
                  ))}
                </ul>
              </div>
            </div>
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
    // Reset scroll position to top when navigating to project page
    window.scrollTo(0, 0);
  };

  const handleBackClick = () => {
    setCurrentView('home');
    setSelectedProject(null);
    // Reset scroll position to top when going back to home
    window.scrollTo(0, 0);
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