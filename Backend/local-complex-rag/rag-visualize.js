import React, { useState } from 'react';
import { Search, Database, FileText, CheckCircle, XCircle, Loader2, ArrowRight, Globe } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const ProcessStep = ({ title, status, icon: Icon, message }) => (
  <div className={`flex items-start space-x-4 p-4 rounded-lg border ${
    status === 'active' ? 'bg-blue-50 border-blue-200' :
    status === 'complete' ? 'bg-green-50 border-green-200' :
    status === 'error' ? 'bg-red-50 border-red-200' :
    'bg-gray-50 border-gray-200'
  }`}>
    <div className={`p-2 rounded-full ${
      status === 'active' ? 'bg-blue-100 text-blue-600' :
      status === 'complete' ? 'bg-green-100 text-green-600' :
      status === 'error' ? 'bg-red-100 text-red-600' :
      'bg-gray-100 text-gray-600'
    }`}>
      <Icon className="w-5 h-5" />
    </div>
    <div className="flex-1">
      <h3 className="font-medium">{title}</h3>
      {message && (
        <p className="text-sm text-gray-600 mt-1">{message}</p>
      )}
    </div>
  </div>
);

const RagProcess = () => {
  const [question, setQuestion] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [processLog, setProcessLog] = useState([]);

  const steps = [
    {
      title: 'Route Question',
      icon: Search,
      message: 'Determining whether to use vectorstore or web search'
    },
    {
      title: 'Document Retrieval',
      icon: Database,
      message: 'Fetching relevant documents from the chosen source'
    },
    {
      title: 'Document Grading',
      icon: CheckCircle,
      message: 'Evaluating document relevance to the question'
    },
    {
      title: 'Web Search',
      icon: Globe,
      message: 'Searching the web for additional context if needed'
    },
    {
      title: 'Answer Generation',
      icon: FileText,
      message: 'Generating response based on retrieved information'
    }
  ];

  const simulateProcess = async () => {
    if (!question.trim()) return;
    
    setIsProcessing(true);
    setCurrentStep(0);
    setProcessLog([]);

    // Simulate the RAG process
    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000));
      setProcessLog(prev => [...prev, `Completed: ${steps[i].title}`]);
    }

    setIsProcessing(false);
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-gray-900">RAG Process Visualization</h2>
        <p className="text-gray-600">
          See how your question is processed through the RAG pipeline, from routing to answer generation.
        </p>

        <div className="space-y-4">
          <div className="flex space-x-4">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your question..."
              className="flex-1 p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={isProcessing}
            />
            <button
              onClick={simulateProcess}
              disabled={isProcessing || !question.trim()}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  <span>Process</span>
                </>
              )}
            </button>
          </div>

          <div className="space-y-4">
            {steps.map((step, index) => (
              <div key={index} className="flex items-center space-x-4">
                <div className="flex-1">
                  <ProcessStep
                    {...step}
                    status={
                      currentStep === index
                        ? 'active'
                        : currentStep > index
                        ? 'complete'
                        : 'pending'
                    }
                  />
                </div>
                {index < steps.length - 1 && (
                  <ArrowRight className="w-5 h-5 text-gray-400" />
                )}
              </div>
            ))}
          </div>

          {processLog.length > 0 && (
            <Alert>
              <AlertTitle>Process Log</AlertTitle>
              <AlertDescription>
                <div className="space-y-2 mt-2">
                  {processLog.map((log, index) => (
                    <div key={index} className="text-sm">
                      {log}
                    </div>
                  ))}
                </div>
              </AlertDescription>
            </Alert>
          )}
        </div>
      </div>
    </div>
  );
};

export default RagProcess;