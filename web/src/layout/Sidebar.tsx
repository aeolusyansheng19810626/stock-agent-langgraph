import React from "react";
import { Watchlist } from "../sidebar/Watchlist";
import { UploadZone } from "../sidebar/UploadZone";
import { DocumentList } from "../sidebar/DocumentList";
import { useDocs } from "../store/docs";

export const Sidebar: React.FC = () => {
  const docs    = useDocs((s) => s.docs);
  const refresh = useDocs((s) => s.refresh);

  React.useEffect(() => { refresh(); }, [refresh]);

  return (
    <aside className="sx-sidebar">
      <Watchlist />
      <div className="sx-sb-section">
        <div className="sx-sb-header">
          <span>财报文档<span className="sx-count"> · {docs.length}</span></span>
        </div>
        <UploadZone />
        <DocumentList />
      </div>
    </aside>
  );
};
