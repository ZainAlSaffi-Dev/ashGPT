'use client';

import * as DialogPrimitive from '@radix-ui/react-dialog';
import { AnimatePresence, motion } from 'framer-motion';
import { X } from 'lucide-react';
import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ElementRef,
  type ReactNode,
} from 'react';

import { cn } from '@/lib/utils';

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogClose = DialogPrimitive.Close;
export const DialogTitle = DialogPrimitive.Title;
export const DialogDescription = DialogPrimitive.Description;

interface DialogContentProps
  extends Omit<ComponentPropsWithoutRef<typeof DialogPrimitive.Content>, 'children'> {
  /** Whether the dialog is open. The framer-motion presence wrapper needs
   *  to know so it can drive enter/exit animations independent of Radix's
   *  default mount/unmount. */
  open: boolean;
  showCloseButton?: boolean;
  children?: ReactNode;
  overlayClassName?: string;
}

export const DialogContent = forwardRef<ElementRef<typeof DialogPrimitive.Content>, DialogContentProps>(
  ({ open, showCloseButton = true, className, overlayClassName, children, ...props }, ref) => (
    <AnimatePresence>
      {open && (
        <DialogPrimitive.Portal forceMount>
          <DialogPrimitive.Overlay asChild forceMount>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className={cn(
                'fixed inset-0 z-50 bg-ink/40 backdrop-blur-[2px]',
                overlayClassName,
              )}
            />
          </DialogPrimitive.Overlay>
          <DialogPrimitive.Content asChild forceMount ref={ref} {...props}>
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: 4 }}
              transition={{ type: 'spring', stiffness: 320, damping: 28 }}
              className={cn(
                'fixed left-1/2 top-1/2 z-50 w-[min(92vw,30rem)] -translate-x-1/2 -translate-y-1/2',
                'rounded-xl border border-parchment-warm bg-parchment p-6 shadow-2xl',
                'focus:outline-none',
                className,
              )}
            >
              {children}
              {showCloseButton && (
                <DialogPrimitive.Close
                  className="absolute right-3 top-3 rounded-md p-1 text-ink-soft transition hover:bg-parchment-warm hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
                  aria-label="Close"
                >
                  <X className="h-4 w-4" />
                </DialogPrimitive.Close>
              )}
            </motion.div>
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      )}
    </AnimatePresence>
  ),
);
DialogContent.displayName = 'DialogContent';
