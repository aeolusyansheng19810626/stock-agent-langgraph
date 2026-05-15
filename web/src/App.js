import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from "react";
import { TopNav } from "./layout/TopNav";
import { Ticker } from "./layout/Ticker";
import { Sidebar } from "./layout/Sidebar";
import { MainArea } from "./layout/MainArea";
import { ContextPanel } from "./layout/ContextPanel";
import { SettingsDrawer } from "./layout/SettingsDrawer";
import { useSettings } from "./store/settings";
const App = () => {
    const theme = useSettings((s) => s.theme);
    const flip = useSettings((s) => s.riseRedFallGreen);
    React.useEffect(() => {
        document.documentElement.setAttribute("data-theme", theme);
    }, [theme]);
    return (_jsxs("div", { className: `sx-root variant-${theme} ${flip ? "flip-color" : ""}`, children: [_jsx(TopNav, {}), _jsx(Ticker, {}), _jsx(Sidebar, {}), _jsx(MainArea, {}), _jsx(ContextPanel, {}), _jsx(SettingsDrawer, {})] }));
};
export default App;
