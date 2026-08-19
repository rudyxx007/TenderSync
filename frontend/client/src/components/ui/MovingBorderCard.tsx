import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface MovingBorderCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  duration?: number;
  containerClassName?: string;
  borderClassName?: string;
}

export function MovingBorderCard({
  children,
  duration = 4000,
  containerClassName,
  borderClassName,
  className,
  ...props
}: MovingBorderCardProps) {
  return (
    <div
      className={cn(
        "relative p-[1px] overflow-hidden rounded-2xl",
        containerClassName
      )}
      {...props}
    >
      <motion.div
        className={cn(
          "absolute inset-[-100%] bg-[conic-gradient(from_90deg_at_50%_50%,#00000000_50%,#34d399_80%,#00000000_100%)]",
          borderClassName
        )}
        animate={{
          rotate: 360,
        }}
        transition={{
          duration: duration / 1000,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <div
        className={cn(
          "relative h-full w-full rounded-2xl bg-[#0F141F] backdrop-blur-xl border border-white/[0.08]",
          className
        )}
      >
        {children}
      </div>
    </div>
  );
}
