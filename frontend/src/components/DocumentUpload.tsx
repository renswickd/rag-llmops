import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useAppStore } from '@/store/appStore';
import { Upload } from 'lucide-react';
import { cn } from '@/lib/utils';

const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
};

export function DocumentUpload() {
  const uploadDocument = useAppStore((s) => s.uploadDocument);
  const isLoading = useAppStore((s) => s.isLoading);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      await uploadDocument(acceptedFiles[0]);
    },
    [uploadDocument]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    disabled: isLoading,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        'cursor-pointer rounded-lg border-2 border-dashed p-4 text-center transition-colors',
        isDragActive
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/50 hover:bg-muted/30',
        isLoading && 'cursor-not-allowed opacity-50'
      )}
    >
      <input {...getInputProps()} />
      <Upload className="mx-auto mb-2 h-6 w-6 text-muted-foreground" />
      {isDragActive ? (
        <p className="text-sm font-medium text-primary">Drop it here</p>
      ) : (
        <>
          <p className="text-sm font-medium">Drop a file or click to upload</p>
          <p className="mt-1 text-xs text-muted-foreground">.pdf · .txt · .md</p>
        </>
      )}
      {isLoading && (
        <p className="mt-2 text-xs text-muted-foreground">Uploading…</p>
      )}
    </div>
  );
}