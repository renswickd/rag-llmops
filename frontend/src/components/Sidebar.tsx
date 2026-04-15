import { DocumentUpload } from './DocumentUpload';
import { UploadedFilesList } from './UploadedFilesList';
import { SessionList } from './SessionList';
import { Separator } from '@/components/ui/separator';

export function Sidebar() {
  return (
    <aside className="flex h-full w-64 flex-col gap-4 overflow-y-auto border-r bg-background p-4">
      <div>
        <h2 className="mb-3 text-sm font-semibold">Upload Document</h2>
        <DocumentUpload />
      </div>

      <Separator />
      <UploadedFilesList />

      <Separator />
      <SessionList />
    </aside>
  );
}