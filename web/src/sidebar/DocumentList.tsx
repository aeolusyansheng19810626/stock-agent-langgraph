import React from "react";
import { useDocs } from "../store/docs";

function relTime(iso: string): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const dt = (Date.now() - t) / 1000;
  if (dt < 60)        return "刚刚";
  if (dt < 3600)      return `${Math.floor(dt / 60)}分钟前`;
  if (dt < 86400)     return `${Math.floor(dt / 3600)}小时前`;
  if (dt < 86400 * 7) return `${Math.floor(dt / 86400)}天前`;
  return new Date(t).toLocaleDateString();
}

export const DocumentList: React.FC = () => {
  const docs   = useDocs((s) => s.docs);
  const remove = useDocs((s) => s.remove);

  if (!docs.length) {
    return (
      <div style={{ padding: "0 14px 12px", fontSize: 11, color: "var(--fg-dimmer)" }}>
        暂无文档，上传后可查询财报数据
      </div>
    );
  }

  return (
    <div className="sx-docs">
      {docs.map((d) => (
        <div className="sx-doc" key={d.id}>
          <div className={`sx-doc-icon ${d.kind}`}>{d.kind.toUpperCase()}</div>
          <div>
            <div className="sx-doc-name" title={d.name}>{d.name}</div>
            <div className="sx-doc-meta">{d.size} · {relTime(d.uploadedAt)}</div>
          </div>
          <button
            className="sx-doc-del"
            onClick={() => {
              if (window.confirm(`删除 ${d.name}？`)) remove(d.id);
            }}
            title="删除"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
};
