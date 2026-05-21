import type { FileListScope, RetrievalScope } from './types';

export function fileScopeKey(scope: FileListScope = {}): string {
  return JSON.stringify({
    projectId: scope.projectId ?? null,
    folderId: scope.folderId ?? null,
    status: scope.status ?? null,
  });
}

export function scopeKey(scope: RetrievalScope | null | undefined): string {
  if (!scope) return 'all';
  return JSON.stringify(scope, Object.keys(scope).sort());
}

export function scopeFromSearchParams(params: URLSearchParams): RetrievalScope {
  const projectId = params.get('project');
  const folderId = params.get('folder');
  const fileId = params.get('file_id');
  if (fileId) {
    return {
      type: 'files',
      project_id: projectId,
      folder_id: folderId,
      file_ids: [fileId],
    };
  }
  if (folderId) return { type: 'folder', project_id: projectId, folder_id: folderId };
  if (projectId) return { type: 'project', project_id: projectId };
  return { type: 'all' };
}

export function fileListScopeFromSearchParams(params: URLSearchParams): FileListScope {
  return {
    projectId: params.get('project'),
    folderId: params.get('folder'),
  };
}
