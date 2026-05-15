import React from "react";
import { TopNav } from "./layout/TopNav";
import { Ticker } from "./layout/Ticker";
import { Sidebar } from "./layout/Sidebar";
import { MainArea } from "./layout/MainArea";
import { ContextPanel } from "./layout/ContextPanel";
import { SettingsDrawer } from "./layout/SettingsDrawer";
import { useSettings } from "./store/settings";

const App: React.FC = () => {
  const theme = useSettings((s) => s.theme);
  const flip  = useSettings((s) => s.riseRedFallGreen);

  React.useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <div className={`sx-root variant-${theme} ${flip ? "flip-color" : ""}`}>
      <TopNav />
      <Ticker />
      <Sidebar />
      <MainArea />
      <ContextPanel />
      <SettingsDrawer />
    </div>
  );
};

export default App;
