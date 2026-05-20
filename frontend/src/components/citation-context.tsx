'use client';

import { createContext, useContext } from 'react';

import type { CitationAnchor } from './CitationPopover';

export interface CitationTarget {
  occurrence: string;
  idx: number;
  anchor: CitationAnchor;
  context: string;
}

export interface CitationCtxValue {
  /** Occurrence id of the currently pinned chip (or null). */
  pinnedOccurrence: string | null;
  /** Occurrence id of the currently hovered chip (or null). */
  hoveredOccurrence: string | null;
  /** Toggle pin on a chip. */
  togglePin: (target: CitationTarget) => void;
  /** Schedule the hover preview to open after the delay. */
  scheduleHoverOpen: (target: CitationTarget) => void;
  /** Schedule the hover preview to close after the close delay. */
  scheduleHoverClose: () => void;
}

const noop = () => undefined;

const DEFAULT: CitationCtxValue = {
  pinnedOccurrence: null,
  hoveredOccurrence: null,
  togglePin: noop,
  scheduleHoverOpen: noop,
  scheduleHoverClose: noop,
};

export const CitationContext = createContext<CitationCtxValue>(DEFAULT);

export function useCitationCtx(): CitationCtxValue {
  return useContext(CitationContext);
}
