import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Triton Ascend Visual Lab｜从 Grid 到 UB",
  description: "交互式学习 Triton-Ascend：数据搬运、offset、grid 并行、UB 估算与 SGL Kernel NPU 真实案例。",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
