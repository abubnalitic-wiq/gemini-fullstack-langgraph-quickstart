import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, RotateCcw } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Default prompts from your backend
const DEFAULT_PROMPTS = {
  query_writer: `Your goal is to generate sophisticated and diverse web search queries...`,
  web_searcher: `Conduct targeted Google Searches to gather the most recent...`,
  reflection: `You are an expert research assistant analyzing summaries...`,
  answer: `Generate a high-quality answer to the user's question...`,
  financial_analyst: `You are a financial analyst assistant with access to financial data tools...`,
  routing: `Analyze this query and determine the appropriate route...`,
  chat: `You are a helpful assistant. Respond naturally to the user's message...`,
  financial_qa: `You are a financial analyst assistant. Answer this specific financial question concisely...`
};

interface PromptEditorProps {
  isOpen: boolean;
  onToggle: () => void;
  prompts: Record<string, string>;
  onPromptsChange: (prompts: Record<string, string>) => void;
}

export function PromptEditor({ isOpen, onToggle, prompts, onPromptsChange }: PromptEditorProps) {
  const [editedPrompts, setEditedPrompts] = useState(prompts);

  const handlePromptChange = (key: string, value: string) => {
    const updated = { ...editedPrompts, [key]: value };
    setEditedPrompts(updated);
    onPromptsChange(updated);
  };

  const resetPrompts = () => {
    setEditedPrompts(DEFAULT_PROMPTS);
    onPromptsChange(DEFAULT_PROMPTS);
  };

  const resetSinglePrompt = (key: string) => {
    const updated = { ...editedPrompts, [key]: DEFAULT_PROMPTS[key as keyof typeof DEFAULT_PROMPTS] };
    setEditedPrompts(updated);
    onPromptsChange(updated);
  };

  return (
    <div className={`fixed left-0 top-0 h-full bg-neutral-800 border-r border-neutral-700 transition-all duration-300 ${
      isOpen ? 'w-96' : 'w-0'
    } overflow-hidden z-40`}>
      <div className="h-full flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-neutral-700">
          <h2 className="text-lg font-semibold text-neutral-100">System Prompts</h2>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={resetPrompts}
              className="text-neutral-400 hover:text-neutral-100"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggle}
              className="text-neutral-400 hover:text-neutral-100"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <Tabs defaultValue="query_writer" className="w-full">
            <TabsList className="grid grid-cols-2 gap-1 w-full mb-4">
              <TabsTrigger value="query_writer">Query Writer</TabsTrigger>
              <TabsTrigger value="web_searcher">Web Searcher</TabsTrigger>
              <TabsTrigger value="reflection">Reflection</TabsTrigger>
              <TabsTrigger value="answer">Answer</TabsTrigger>
              <TabsTrigger value="financial_analyst">Financial</TabsTrigger>
              <TabsTrigger value="routing">Routing</TabsTrigger>
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="financial_qa">Financial QA</TabsTrigger>
            </TabsList>
            
            {Object.entries(editedPrompts).map(([key, value]) => (
              <TabsContent key={key} value={key} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-neutral-400">
                    {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} Prompt
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => resetSinglePrompt(key)}
                    className="text-xs text-neutral-400 hover:text-neutral-100"
                  >
                    Reset
                  </Button>
                </div>
                <Textarea
                  value={value}
                  onChange={(e) => handlePromptChange(key, e.target.value)}
                  className="min-h-[400px] bg-neutral-700 border-neutral-600 text-neutral-100 font-mono text-xs"
                  placeholder={`Enter ${key} prompt...`}
                />
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </div>
      
      {/* Toggle button when closed */}
      {!isOpen && (
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className="absolute -right-10 top-1/2 -translate-y-1/2 bg-neutral-700 text-neutral-300 hover:bg-neutral-600"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}