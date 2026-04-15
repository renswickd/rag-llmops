import { useAppStore } from '@/store/appStore';
import { FileText } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export function UploadedFilesList() {
  const uploadedFiles = useAppStore((s) => s.uploadedFiles);

  if (uploadedFiles.length === 0) return null;

  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Uploaded Files
      </h3>
      <ul className="space-y-2">
        {uploadedFiles.map((file, i) => (
          <li key={i} className="flex items-center gap-2 text-sm">
            <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
            <span className="flex-1 truncate" title={file.file_name}>
              {file.file_name}
            </span>
            <Badge variant="secondary" className="shrink-0 text-xs">
              {file.chunks_created}c
            </Badge>
          </li>
        ))}
      </ul>
    </div>
  );
}