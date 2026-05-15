import React from "react";
import { Icon } from "../icons/Icon";
import { useDocs } from "../store/docs";

export const UploadZone: React.FC = () => {
  const add = useDocs((s) => s.add);
  const loading = useDocs((s) => s.loading);
  const [dragOver, setDragOver] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleFiles = async (files: FileList | null) => {
    if (!files) return;
    for (const f of Array.from(files)) {
      if (f.name.toLowerCase().endsWith(".pdf")) {
        await add(f);
      }
    }
  };

  return (
    <div
      className={`sx-upload-zone ${dragOver ? "drag" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <Icon name="upload" size={16} />
      <div style={{ marginTop: 4 }}>
        {loading ? "处理中…" : "拖入年报 / 季报 / PDF"}
      </div>
      <div className="sx-upload-hint">或点击浏览本地文件</div>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        multiple
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
};
