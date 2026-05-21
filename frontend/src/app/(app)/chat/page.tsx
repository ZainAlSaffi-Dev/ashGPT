import { ChatLandingClient } from './ChatLandingClient';

export const runtime = 'edge';

/** Landing "new chat" route — no session id yet. After the first reply
 *  ``ChatSurface`` will router.replace to ``/chat/<id>``. */
export default function ChatLandingPage() {
  return <ChatLandingClient />;
}
