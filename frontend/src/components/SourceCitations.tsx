import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
  } from '@/components/ui/accordion';
  import { FileText } from 'lucide-react';
  import type { SourceDocument } from '@/types';
  
  interface Props {
    sources: SourceDocument[];
  }
  
  export function SourceCitations({ sources }: Props) {
    if (!sources || sources.length === 0) return null;
  
    return (
      <Accordion type="single" collapsible className="mt-2 w-full">
        <AccordionItem value="sources" className="border-none">
          <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:no-underline">
            Sources ({sources.length})
          </AccordionTrigger>
          <AccordionContent className="pb-1">
            <ul className="space-y-1.5">
              {sources.map((source, i) => {
                const label =
                  (source.metadata?.source as string) ??
                  (source.metadata?.filename as string) ??
                  `Source ${i + 1}`;
                return (
                  <li key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <FileText className="mt-0.5 h-3 w-3 shrink-0" />
                    <span className="break-all">{label}</span>
                  </li>
                );
              })}
            </ul>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    );
  }