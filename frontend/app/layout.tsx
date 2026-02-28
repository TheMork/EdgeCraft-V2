import type { Metadata } from "next";
import { Fira_Code } from "next/font/google";
import "./globals.css";
import MainLayout from "@/components/layout/MainLayout";

const firaCode = Fira_Code({
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Crypto Quant Dashboard",
  description: "Institutional Grade Backtesting Engine",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${firaCode.className} antialiased`}
      >
        <MainLayout>
          {children}
        </MainLayout>
      </body>
    </html>
  );
}
