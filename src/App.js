import React, { useState } from 'react';
import { Github, Linkedin, Mail, ExternalLink, Code, Play, ArrowLeft } from 'lucide-react';

// Sample project data - replace with your actual projects
const projects = [
  {
  id: 7,
  title: "TLDR News Aggregator",
  description: "An automated newsletter aggregation system that processes tech newsletters via IMAP and extracts articles using advanced HTML/text parsing with Cheerio. Generates 384-dimensional vector embeddings for semantic search and features AI-powered chat interface using Claude Haiku with custom RAG pipeline.",
  technologies: ["TypeScript", "Node.js", "React", "PostgreSQL", "pgvector", "OpenSearch", "Docker", "Xenova Transformers", "IMAP"],
  demoVideo: "Demo video coming soon!",
  screenshots: [
    "/images/tldr/tldr1.png",
    "/images/tldr/tldr2.png",
    "/images/tldr/tldr3.png"
  ],
  codeSnippet: `{/* Embedding Functionality */}
  async generateBatchEmbeddings(texts: string[]): Promise<EmbeddingResult[]> {
    const results: EmbeddingResult[] = [];
    
    console.log('Generating embeddings for {texts.length} texts...');
    
    for (let i = 0; i < texts.length; i++) {
      try {
        const result = await this.generateEmbedding(texts[i]);
        results.push(result);
        
        // Progress logging
        if ((i + 1) % 10 === 0) {
          console.log('Processed {i + 1}/{texts.length} embeddings');
        }
      } catch (error) {
        console.error('Failed to generate embedding for text {i + 1}:', error);
        // Push a zero vector as fallback
        results.push({
          embedding: new Array(384).fill(0),
          tokens: 0,
          processing_time: 0
        });
      }
    }
    
    console.log('Batch embedding generation completed');
    return results;
  }

  async generateArticleEmbeddings(article: {
    title: string;
    summary: string;
    content?: string;
  }): Promise<{
    title_embedding: number[];
    summary_embedding: number[];
    content_embedding?: number[];
  }> {
    try {
      const embeddings: any = {};
      
      // Generate title embedding
      const titleResult = await this.generateEmbedding(article.title);
      embeddings.title_embedding = titleResult.embedding;
      
      // Generate summary embedding
      const summaryResult = await this.generateEmbedding(article.summary);
      embeddings.summary_embedding = summaryResult.embedding;
      
      // Generate content embedding if available
      if (article.content && article.content.length > 50) {
        const contentResult = await this.generateEmbedding(article.content);
        embeddings.content_embedding = contentResult.embedding;
      }
      
      return embeddings;
    } catch (error) {
      console.error('Error generating article embeddings:', error);
      throw new Error('Failed to generate article embeddings');
    }
  }

  private preprocessText(text: string): string {
    // Clean and normalize text
    let cleaned = text
      .replace(/s+/g, ' ')           // Normalize whitespace
      .replace(/[^ws-.,!?]/g, '')  // Remove special characters
      .trim();
    
    // Truncate to reasonable length (roughly 512 tokens)
    const maxLength = 2000; // characters
    if (cleaned.length > maxLength) {
      cleaned = cleaned.substring(0, maxLength).trim();
      // Try to end at a word boundary
      const lastSpace = cleaned.lastIndexOf(' ');
      if (lastSpace > maxLength * 0.8) {
        cleaned = cleaned.substring(0, lastSpace);
      }
    }
    
    return cleaned;
  }

  // Compute cosine similarity between two embeddings
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Embedding dimensions must match');
    }
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    
    if (normA === 0 || normB === 0) {
      return 0;
    }
    
    return dotProduct / (normA * normB);
  }

  // Find most similar embeddings
  findSimilar(queryEmbedding: number[], candidateEmbeddings: { id: string; embedding: number[] }[], topK: number = 5): { id: string; similarity: number }[] {
    const similarities = candidateEmbeddings.map(candidate => ({
      id: candidate.id,
      similarity: this.cosineSimilarity(queryEmbedding, candidate.embedding)
    }));
    
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }`,
  githubUrl: "https://github.com/will-mp3/tldr",
  features: [
    "Automated Email Processing: Daily IMAP scraping of newsletters with 24-hour window filtering",
    "Advanced Content Extraction: Multi-strategy HTML/text parsing with Cheerio for accurate article extraction",
    "Smart Summary Parsing: Complete sentence detection and promotional content filtering",
    "Vector Embeddings: 384-dimensional embeddings using Xenova Transformers (all-MiniLM-L6-v2 model)",
    "Hybrid Search: Combines pgvector semantic search with OpenSearch full-text capabilities",
    "Duplicate Detection: Intelligent deduplication based on URL and title similarity scoring",
    "7-Day TTL Window: Automatic cleanup with PostgreSQL expires_at timestamp management",
    "AI Chat Interface: Claude 3 Haiku integration with retrieval-augmented generation (RAG)",
    "Fallback Strategies: Multiple parsing methods ensure no articles are missed",
    "Performance Optimized: Sub-500ms search responses with efficient vector operations",
    "Error Recovery: Comprehensive error handling with graceful degradation",
    "Scheduled Processing: Cron-based automation running daily at 10:00 AM EST",
    "Development Environment: Docker Compose setup with PostgreSQL 16 and OpenSearch 2.11"
  ]
  },
  {
    id: 6,
    title: "RAG Document Assistant",
    description: "A full-stack RAG application built with React/TypeScript and Node.js that enables users to upload documents or scrape web content and ask AI-powered questions on a RAG pipeline using OpenSearch, 384-dimensional vector embeddings, and Claude Haiku.",
    technologies: ["React", "TypeScript", "Node.js", "OpenSearch", "Docker","pgvector", "Xenova Transformers"],
    demoVideo: "Demo video coming soon!",
    screenshots: [
      "/images/docassist/doc1.png",
      "/images/docassist/doc2.png",
      "/images/docassist/doc3.png"
    ],
    codeSnippet: 
`{/* Embedding Vector */}
  import { pipeline, env } from '@xenova/transformers';

  // Configure transformers to run in Node.js environment
  env.allowLocalModels = false;
  env.allowRemoteModels = true;

  let embeddingPipeline: any = null;
  let initializationAttempted = false;

  export class EmbeddingService {
    // Initialize the embedding model (runs once on startup)
    static async initialize(): Promise<void> {
      if (initializationAttempted) {
        throw new Error('Embedding service initialization already attempted');
      }
      
      initializationAttempted = true;
      console.log('Loading embedding model...');
      
      try {
        // Use a lightweight, fast sentence embedding model
        embeddingPipeline = await pipeline(
          'feature-extraction',
          'Xenova/all-MiniLM-L6-v2',
          { 
            quantized: false,
            progress_callback: (progress: any) => {
              if (progress.status === 'downloading') {
                console.log('Downloading model: {Math.round(progress.progress || 0)}%');
              }
            }
          }
        );
        console.log('Embedding model loaded successfully');
      } catch (error) {
        console.error('Failed to load embedding model:', error);
        embeddingPipeline = null;
        throw new Error('Embedding model initialization failed: {(error as Error).message}');
      }
    }

    // Check if service is ready - ADD THIS METHOD
    static isInitialized(): boolean {
      return embeddingPipeline !== null;
    }

    // Generate embedding vector for text
    static async generateEmbedding(text: string): Promise<number[]> {
      if (!embeddingPipeline) {
        throw new Error('Embedding model not initialized. Check server startup logs.');
      }

      try {
        // Clean and prepare text
        const cleanText = text.trim().toLowerCase();
        if (!cleanText) {
          throw new Error('Empty text provided');
        }

        // Generate embedding
        const result = await embeddingPipeline(cleanText, {
          pooling: 'mean',
          normalize: true
        });

        // Extract the embedding vector with proper type casting
        const embedding = Array.from(result.data as Float32Array).map(x => Number(x));
        console.log('Generated embedding for text: "{text.substring(0, 50)}..." ({embedding.length} dimensions)');
        
        return embedding;
      } catch (error) {
        console.error('Error generating embedding:', error);
        throw error;
      }
    }

    // Batch generate embeddings for multiple texts
    static async generateBatchEmbeddings(texts: string[]): Promise<number[][]> {
      const embeddings: number[][] = [];
      
      for (const text of texts) {
        const embedding = await this.generateEmbedding(text);
        embeddings.push(embedding);
      }
      
      return embeddings;
    }

    // Calculate cosine similarity between two embeddings
    static cosineSimilarity(embedding1: number[], embedding2: number[]): number {
      if (embedding1.length !== embedding2.length) {
        throw new Error('Embeddings must have the same length');
      }

      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;

      for (let i = 0; i < embedding1.length; i++) {
        dotProduct += embedding1[i] * embedding2[i];
        norm1 += embedding1[i] * embedding1[i];
        norm2 += embedding2[i] * embedding2[i];
      }

      return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
  }
      `,
    githubUrl: "https://github.com/will-mp3/tech-docs-assistant",
    features: [
      "File Upload Support: PDF, Word documents, text files with automatic text extraction",
      "Web Scraping: Intelligent content extraction from documentation URLs",
      "Smart Chunking: Optimized text segmentation for vector search",
      "Hybrid Search: Combines semantic vector similarity with keyword matching",
      "Relevance Scoring: Transparent percentage-based match scores",
      "Natural Language Queries: Ask questions about uploaded documents",
      "Contextual Responses: Claude AI generates answers from relevant document chunks",
      "Source Citations: Every answer includes specific document references",
      "Cost-Optimized: Uses Claude Haiku for fast, affordable responses (~$0.001 per query)",
      "Real-time Feedback: Loading states, progress indicators, error handling"
    ]
  },
  {
    id: 5,
    title: "Micro GPT",
    description: "Generative Pre-Trained Transformer built using a PyTorch neural network. This model is a character-level tokenizer trained on the openwebtext corpus and is capable of generating responses to user prompts. Comes paired with a UI to prompt user input.",
    technologies: ["Python", "PyTorch", "Tkinter", "Jupyter", "Cuda/mps"],
    demoVideo: "Demo video coming soon!",
    screenshots: [
      "/images/mumble/mumble1.png",
      "/images/mumble/mumble2.png",
      "/images/mumble/mumble3.png"
    ],
    codeSnippet: 
`# Multihead Attention implementation and Head class
class Head(nn.Module):
    # one head of self-attention
    
    def __init__(self, head_size):
        # calls the constructor of nn.Module
        super().__init__()
        
        # each token is transformed into a query, key, and value using learned linear layers
        # head_size is the dimension of this head's internal representation
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # creates a lower-triangular mask (tril) to prevent tokens from attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # applies dropout to the attention weights to regularize training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        
        # input x: shape (Batch, Time, Channels)
        # outputs k and q: shape (B, T, head_size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        # compute raw attention scores via scaled dot product.
        # shape becomes (B, T, T) representing attention between all pairs of tokens
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        # masks out future tokens (above the diagonal), setting them to -inf so softmax zeroes them
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        # converts attention scores to probabilities and applies dropout
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # uses the attention weights to compute a weighted sum of values
        v = self.value(x) # # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    # multiple heads of attention in parallel
    
    def __init__(self, num_heads, head_size):
        # combines multiple Heads in parallel
        # the outputs are concatenated and passed through a final linear layer to project back to n_embd
        
        # calls the constructor of nn.Module
        super().__init__()
        
        # creates multiple Head modules, each representing a self-attention head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # pass heads through a final linear projection to fuse their information and bring dimensionality back
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        
        # apply dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # concatenates each head’s output, projects it back to the full embedding size
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # apply dropout to prevent overfitting
        out = self.dropout(self.proj(out))
        return out`,
    githubUrl: "https://github.com/will-mp3/micro-gpt",
    features: [
      "Transformer block architecture",
      "Feedforward neural network with Multihead attention",
      "Trained on the openwebtext corpus",
      "Character-level tokenizer",
      "Cuda/MPS support for GPU acceleration",
      "Custom weights initialization",
      "Pickle model saving",
      "Custom user interface with Tkinter",
    ]
  },
  {
    id: 4,
    title: "Bigram Language Model",
    description: "Implements a Bigram Language Model, a character-level tokenizer capable of making text predictions based on given context. Trained on a text corpus split into blocks using gradient descent and backpropagation; optimized using AdamW.",
    technologies: ["Python", "PyTorch", "Jupyter", "Cuda/mps"],
    demoVideo: "Demo video coming soon!",
    screenshots: [
      "/images/bigram/bigram1.png",
      "/images/bigram/bigram2.png",
      "/images/bigram/bigram3.png"
    ],
    codeSnippet: 
`# Bigram Language Model implementation
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        # calls the constructor of nn.Module
        super().__init__()
        # creates an embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index) # raw predictions
        
        if targets is None: # inference mode
            loss = None
        else:
            # batch, time, channels(vocabulary)
            # B (Batch Size) -> Number of sequences processed at once
            # T (Time Steps / Sequence Length) -> Number of tokens in each sequence
            # C (Vocabulary Size / Channels) -> Number of possible tokens
            B, T, C = logits.shape
            
            # reshape batch and time into a single dimension 
            # so that each token is treated as a separate training example
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # targets also reshaped into a single B*T vector
            
            # compute cross-entropy loss to measure how far our predictions (logits) are from the true targets
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # extracts only the last time step’s logits
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities for each possible next token
            probs = F.softmax(logits, dim=-1) # (B, C)
            # samples one token index from the probability distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index
    
model = BigramLanguageModel(vocab_size)
m = model.to(device) 

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars) `,
    githubUrl: "https://github.com/will-mp3/bigram-language-model",
    features: [
      "Character-level tokenization",
      "Text prediction based on context",
      "Trained on Wizard Of Oz text corpus",
      "Gradient descent and backpropagation",
      "Optimized using AdamW",
      "Supports CUDA/MPS for GPU acceleration",
    ]
  },
  {
    id: 3,
    title: "F Edit",
    description: "File editor built in C that works by shifting and resizing certain byte portions of a given file, functionality includes file rotation, expansion, contraction, and more.",
    technologies: ["C"],
    demoVideo: "Demo video coming soon!",
    screenshots: [
      "/images/fedit/fedit1.png",
      "/images/fedit/fedit2.png",
      "/images/fedit/fedit3.png"
    ],
    codeSnippet: 
`// Rotate left functionality
char *rotate_left(char *buffer, int NROTl, int size)
{
    int nrotl = NROTl;
    int rotations = nrotl % size;

    char *temp = (char *)malloc(size * sizeof(char));
    if (temp == NULL)
    {
        fprintf(stderr, "Memory allocation failed.");
        exit(EXIT_FAILURE);
    }

    // Rotate the buffer contents to the left by NROTl positions
    for (int i = 0; i < size; i++)
    {
        temp[i] = buffer[(i + rotations) % size];
    }

    // Copy rotated content back to the original buffer
    memcpy(buffer, temp, size);

    free(temp);

    return buffer;
}`,
    githubUrl: "https://github.com/will-mp3/f-edit",
    liveUrl: "https://your-ecommerce-demo.com",
    features: [
      "-h -- show usage statement and exit",
      "-l NROTL -- rotate the file NROTL bytes left",
      "-x NEXPAND -- expand the file NEXPAND bytes",
      "-v CHAR -- The character value that is used when expanding the file",
      "-c NCONTRACT -- contract the file NCONTRACT bytes",
      "-k NKEEP -- keep NKEEP bytes of the file, starting at the offset provided by -s",
      "-s NSKIP -- skip NSKIP bytes before keeping"
    ]
  },
  {
    id: 2,
    title: "S Grep",
    description: "Recreation of the grep command line tool built in C, allows for word/phrase searching and includes features such as line number printing, quiet mode, and context lines.",
    technologies: ["C"],
    demoVideo: "Demo video coming soon!",
    screenshots: [
      "/images/sgrep/sgrep1.png",
      "/images/sgrep/sgrep2.png",
      "/images/sgrep/sgrep3.png"
    ],
    codeSnippet: 
`// Read lines function with standard output
void read_lines(const char *str, const char *path, int count, int linenumber, int quiet, int beforecontext, int context_num)
{
    FILE *fh = fopen(path, "r");
    if (fh == NULL)
    {
        perror("Error opening file");
        exit(1);
    }

    char *line = NULL;
    size_t n = 0;
    int match_count = 0;
    int line_num = 1;

    struct queue context_queue;
    list_init(&context_queue);
    context_queue.max_capacity = context_num + 1;
    
    while (getline(&line, &n, fh) != -1)
    {
        if (strstr(line, str) != NULL)
        {
            if (linenumber)
            {
                printf("%d:", line_num);
            }
            printf("%s", line);
        }
        line_num++;
    }
    free(line);
    fclose(fh);
    exit(0);
}`,
    githubUrl: "https://github.com/will-mp3/s-grep",
    features: [
      "-h -- Printable usage statement",
      "-c -- Suppress normal output; instead print a count of matching lines for the input file",
      "-n -- Prefix each line of output with the 1-based line number of the file",
      "-q -- Quiet; do not write anything to stdout. Exit immediately with zero status if any match was found",
      "-B NUM -- Print NUM lines of leading context before matching lines"
    ]
  },
  {
    id: 1,
    title: "React Web Portfolio",
    description: "Custom React website housing my entire developer portfolio. Contains information about me, my projects, and how to contact me.",
    technologies: ["React", "JavaScript", "HTML", "Tailwind CSS"],
    demoVideo: "This is the demo! Take a look around!",
    screenshots: [],
    codeSnippet: 
  `{/* Projects Section - Mac Folder Style */}
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
          <div className="w-[64px]"></div>
      </div>
      <div className="p-8">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project, index) => {
              const colors = ['bg-[#61bb46]', 'bg-[#fdb827]', 'bg-[#f5821f]', 'bg-[#e03a3e]', 'bg-[#963d97]', 'bg-[#009ddc]'];
              const bgColor = colors[index % colors.length];
              return (
              <div key={project.id} className="bg-gray-100 border-2 border-black hover:bg-white transition cursor-pointer" style={{
                  boxShadow: '4px 4px 0px #000'
              }}>
                  <div className="p-4">
                  <h4 className="font-mono font-bold text-sm mb-2 text-black">{project.title.toUpperCase().replace(/s+/g, '.')}</h4>
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
                      className={"font-mono text-xs text-white {bgColor} hover:opacity-80 border border-black px-2 py-1 transition"}
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
  </section>`,
    githubUrl: "https://github.com/will-mp3/react-web-portfolio",
    features: [
      "Classic Apple Macintosh design & color scheme",
      "Developer bio and skills showcase",
      "Projects section with detailed descriptions",
      "Embedded demo videos and screenshots",
      "Links to all contact methods and social media profiles",
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
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-mono font-bold text-black">WILL.KATABIAN</h1>
            </div>
            <nav className="flex space-x-1">
              <a href="#about" className="px-3 py-1 text-black hover:bg-[#009ddc] hover:text-white font-mono text-sm border border-black transition">About</a>
              <a href="#projects" className="px-3 py-1 text-black hover:bg-[#e03a3e] hover:text-white font-mono text-sm border border-black transition">Projects</a>
              <a href="#contact" className="px-3 py-1 text-black hover:bg-[#61bb46] hover:text-black font-mono text-sm border border-black transition">Contact</a>
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
            <h2 className="text-2xl sm:text-4xl font-mono font-bold mb-4 sm:mb-6 text-black">SOFTWARE.ENGINEER</h2>
            <div className="bg-black text-white p-4 font-mono text-sm mb-6">
              <div className="text-green-400">$ whoami</div>
              <div>Software engineer with full-stack experience building, deploying, and maintaining meaningful products</div>
              <div className="text-green-400 mt-2">$ ls skills/tools/</div>
              <div>C, C++, Java, Python, JavaScript, TypeScript, HTML, React, React Native, Node.js, Angular, PyTorch, SQL, NoSQL, AWS, Docker, Linux, Kali, Nessus, Atlassian, Git</div>
              <div className="text-green-400 mt-2">$ ls skills/concepts/</div>
              <div>Object-Oriented Design, Data Structures, Algorithm Design, Complexity Analysis, CI/CD, System Design, Cloud Computing, DevOps, Agile</div>
              <div className="text-green-400 mt-2">$ ls certifications/</div>
              <div>AWS Certified Cloud Practitioner</div>
            </div>
            <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-4">
              <a href="#about" className="bg-[#009ddc] text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-blue-600 transition shadow-lg" style={{boxShadow: '4px 4px 0px #000'}}>
                ABOUT.ME
              </a>
              <a href="#projects" className="bg-[#e03a3e] text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-red-600 transition shadow-lg" style={{boxShadow: '4px 4px 0px #000'}}>
                VIEW.PROJECTS
              </a>
              <a href="#contact" className="bg-[#61bb46] text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-green-600 transition shadow-lg" style={{boxShadow: '4px 4px 0px #000'}}>
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
              <div className="w-[64px]"></div>
            </div>
            <div className="p-8">
              <div className="grid md:grid-cols-2 gap-12 items-center">
                <div>
                  <p className="font-mono text-sm text-black mb-4">
                    &gt; INITIALIZING DEVELOPER PROFILE...<br/>
                    &gt; LOADING BACKGROUND DATA...
                  </p>
                  <p className="text-black mb-6">
                    Hi, I’m Will Katabian, a Software Engineer specializing in AI systems and development. I graduated from the College of William & Mary in 2025 with a B.S. in Computer Science, and I now work on the AI Development team at Unanet.
                  </p>
                  <p className="text-black mb-6">
                    My journey with tech started in eighth grade when I built my first PC. That early spark of creation grew into a love for computer science, which I’ve been pursuing ever since. I thrive on solving problems, learning new technologies, and pushing myself to grow as a developer.
                  </p>
                  <p className="text-black mb-6">
                    I have hands-on experience developing full-stack applications with React, JavaScript, Python, and AWS, supported by a strong foundation in object-oriented design, data structures, and algorithms. I focus on delivering secure, production-ready solutions that scale.
                  </p>
                  <p className="text-black mb-6">
                    I hope this gives you an idea of who I am, but my work tells the real story. I invite you to explore my projects below to see what I really have to offer.
                  </p>
                  <p className="text-black mb-6">
                    Cheers!
                  </p>
                  <div className="flex space-x-4">
                    <a href="/images/resume.pdf" className="flex items-center space-x-2 bg-[#009ddc] text-white px-4 py-2 font-mono text-sm border-2 border-black hover:bg-blue-600 transition" style={{boxShadow: '3px 3px 0px #000'}}>
                      <span>RESUME.PDF</span>
                      <ExternalLink size={16} />
                    </a>
                    <a href="https://github.com/will-mp3" className="flex items-center space-x-2 bg-[#fdb827] text-black px-4 py-2 font-mono text-sm border-2 border-black hover:bg-yellow-600 transition" style={{boxShadow: '3px 3px 0px #000'}}>
                      <Github size={16} />
                      <span>GITHUB</span>
                    </a>
                  </div>
                </div>
                <div className="bg-black border-2 border-black w-71 h-71 flex flex-col">
        <div className="bg-gray-200 p-2 border-b-2 border-black">
            <span className="font-mono text-xs">PROFILE.PIC</span>
        </div>
        <div className="flex-1">
            <img 
            src="/images/me.jpg" 
            alt="me!" 
            className="w-full h-full object-cover" 
            />
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
              <div className="w-[64px]"></div>
            </div>
            <div className="p-8">
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {projects.map((project, index) => {
                  const colors = ['bg-[#61bb46]', 'bg-[#fdb827]', 'bg-[#f5821f]', 'bg-[#e03a3e]', 'bg-[#963d97]', 'bg-[#009ddc]'];
                  const bgColor = colors[index % colors.length];
                  return (
                    <div key={project.id} className="bg-gray-100 border-2 border-black hover:bg-white transition cursor-pointer" style={{
                      boxShadow: '4px 4px 0px #000'
                    }}>
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
              <div className="w-[64px]"></div>
            </div>
            <div className="p-8">
              <div className="font-mono text-sm text-black mb-4">
                <span className="text-black">&gt; ESTABLISHING CONNECTION...</span><br/>
                <span className="text-black">&gt; CONTACT PROTOCOLS READY</span>
              </div>
              <p className="text-black mb-6 max-w-md">
                I'm always open to discussing new opportunities and interesting projects.
              </p>
              <div className="flex flex-col space-y-3">
                <a href="mailto:katabianwill@gmail.com" className="flex items-center justify-center space-x-2 bg-[#e03a3e] text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-red-600 transition" style={{boxShadow: '4px 4px 0px #000'}}>
                  <Mail size={16} />
                  <span>SEND.EMAIL</span>
                </a>
                <a href="https://www.linkedin.com/in/willkatabian/" className="flex items-center justify-center space-x-2 bg-[#009ddc] text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-blue-600 transition" style={{boxShadow: '4px 4px 0px #000'}}>
                  <Linkedin size={16} />
                  <span>LINKEDIN.CONNECT</span>
                </a>
                <a href="https://github.com/will-mp3" className="flex items-center justify-center space-x-2 bg-[#61bb46] text-white px-6 py-3 font-mono text-sm border-2 border-black hover:bg-green-600 transition" style={{boxShadow: '4px 4px 0px #000'}}>
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
    const [fullscreenImage, setFullscreenImage] = useState(null);
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
            <h1 className="text-xl font-mono font-bold text-black">WILL.KATABIAN</h1>
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
            <div className="w-[64px]"></div>
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
                <a href={project.githubUrl} className="flex items-center space-x-2 bg-green-500 text-white px-4 py-2 font-mono text-sm border-2 border-black hover:bg-green-600 transition" style={{boxShadow: '3px 3px 0px #000'}}>
                  <Code size={16} />
                  <span>VIEW.CODE</span>
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
                <div key={index} className="bg-black border-2 border-black cursor-pointer hover:opacity-80 transition">
                    <div className="bg-gray-200 p-1 border-b border-black">
                    <span className="font-mono text-xs">IMG_{index + 1}.BMP</span>
                    </div>
                    <img 
                    src={screenshot} 
                    alt={`Screenshot ${index + 1}`}
                    className="w-full h-52 object-cover"
                    onClick={() => setFullscreenImage(screenshot)}
                    />
                </div>
                ))}
            </div>
            
            {/* Fullscreen Modal */}
            {fullscreenImage && (
            <div 
                className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 p-4"
                onClick={() => setFullscreenImage(null)}
            >
                <div className="bg-white border-4 border-black max-w-4xl max-h-full" style={{boxShadow: '8px 8px 0px #000'}}>
                <div className="bg-gray-200 border-b-2 border-black p-2 flex items-center">
                    <div className="flex space-x-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full cursor-pointer" onClick={() => setFullscreenImage(null)}></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <div className="flex-1 text-center">
                    <span className="font-mono text-sm font-bold">FULLSCREEN.VIEW</span>
                    </div>
                    <div className="w-[64px]"></div>
                </div>
                
                <div className="p-4">
                <div className="w-full aspect-[4/3] flex items-center justify-center bg-white-100">
                    <img 
                    src={fullscreenImage} 
                    alt="Fullscreen view"
                    className="max-w-full max-h-full object-contain"
                    />
                </div>
                </div>
                
                </div>
            </div>
            )}
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