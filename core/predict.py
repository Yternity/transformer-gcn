"""Chunked spatial prediction workflow for trained Transformer-GCN models."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

try:
    from .config import (
        DATA_DIR,
        DEFAULT_BUFFER_CELLS,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_GRID_SIZE,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_K,
        DEFAULT_LAYERS_COUNT,
        DEFAULT_MIN_SAMPLES,
        DEFAULT_SIGMA_KM,
        DEFAULT_UNKNOWN_ID,
        DEFAULT_UNKNOWN_THRESHOLD,
        RESULTS_DIR,
        TEMP_GRIDS_DIR,
        ensure_runtime_dirs,
    )
    from .load_data import build_graph_from_coords, filter_small_classes
    from .mode import ourmodel
except ImportError:
    from config import (
        DATA_DIR,
        DEFAULT_BUFFER_CELLS,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_GRID_SIZE,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_K,
        DEFAULT_LAYERS_COUNT,
        DEFAULT_MIN_SAMPLES,
        DEFAULT_SIGMA_KM,
        DEFAULT_UNKNOWN_ID,
        DEFAULT_UNKNOWN_THRESHOLD,
        RESULTS_DIR,
        TEMP_GRIDS_DIR,
        ensure_runtime_dirs,
    )
    from load_data import build_graph_from_coords, filter_small_classes
    from mode import ourmodel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROW_ID_COL = "source_row_id"
META_COLS_BASE = ["zone_id", "province", "province_code"]
INVALID_ROWS_FILE = "_invalid_rows.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于训练好的 Transformer+GCN 模型进行分块推理（支持大规模 CSV）。"
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=DATA_DIR / "prediction_set.csv",
        help="待预测 CSV 路径",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=RESULTS_DIR / "best_model_seed_42.pt",
        help="模型权重路径（state_dict）",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DATA_DIR / "prediction_complete.csv",
        help="预测输出 CSV 路径",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=TEMP_GRIDS_DIR,
        help="网格切分临时目录",
    )
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE, help="网格大小（度）")
    parser.add_argument(
        "--buffer-cells",
        type=int,
        default=DEFAULT_BUFFER_CELLS,
        help="邻域缓冲圈数（1 表示 3x3）",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="kNN 构图的 k")
    parser.add_argument("--sigma-km", type=float, default=DEFAULT_SIGMA_KM, help="高斯核 sigma（km）")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="CSV 分块读取大小",
    )
    parser.add_argument(
        "--in-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="模型隐藏维度（需与训练一致）",
    )
    parser.add_argument(
        "--layers-count",
        type=int,
        default=DEFAULT_LAYERS_COUNT,
        help="模型层数（需与训练一致）",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="和训练一致的 filter_small_classes 阈值（仅用于恢复原始类别 id）",
    )
    parser.add_argument(
        "--mapping-data",
        type=Path,
        default=DATA_DIR / "ours_data.pt",
        help="用于恢复 raw 类别 id 的训练数据（可不存在）",
    )
    parser.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="推理结束后删除临时网格文件",
    )
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=DEFAULT_UNKNOWN_THRESHOLD,
        help="拒判阈值：confidence < threshold 记为未知；设为负值可关闭拒判",
    )
    parser.add_argument(
        "--unknown-id",
        type=int,
        default=DEFAULT_UNKNOWN_ID,
        help="未知类别 id（用于 *_final 列）",
    )
    return parser.parse_args()


def _guess_lon_lat_cols(columns: Iterable[str]) -> Tuple[str, str]:
    colset = set(columns)
    lon_candidates = ["lon", "longitude", "经度", "LON", "Long", "LONGITUDE"]
    lat_candidates = ["lat", "latitude", "纬度", "LAT", "Lat", "LATITUDE"]
    lon_col = next((c for c in lon_candidates if c in colset), None)
    lat_col = next((c for c in lat_candidates if c in colset), None)
    if lon_col is None or lat_col is None:
        raise ValueError(f"未识别到经纬度列。可见列示例: {list(columns)[:20]}")
    return lon_col, lat_col


def _infer_input_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    if "lin.weight" in state_dict and state_dict["lin.weight"].ndim == 2:
        return int(state_dict["lin.weight"].shape[1])
    for _, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.shape[1])
    raise KeyError("无法从 state_dict 推断输入维度（未找到二维权重）。")


def _infer_feature_columns(
    csv_path: Path,
    input_dim: int,
    sample_rows: int = 2000,
) -> Tuple[List[str], str, str]:
    df_head = pd.read_csv(csv_path, nrows=sample_rows, low_memory=False)
    lon_col, lat_col = _guess_lon_lat_cols(df_head.columns)

    drop_names = {
        "label",
        "y",
        "target",
        "class",
        "zone_id",
        "province",
        "province_code",
    }
    drop_cols = {
        c
        for c in df_head.columns
        if str(c).startswith("Unnamed") or str(c).strip().lower() in drop_names
    }
    drop_cols.update({lon_col, lat_col})

    candidate_cols = [c for c in df_head.columns if c not in drop_cols]
    valid_numeric_cols: List[str] = []
    dropped_cols: List[str] = []
    for col in candidate_cols:
        series = pd.to_numeric(df_head[col], errors="coerce")
        valid_ratio = 1.0 - float(series.isna().mean())
        if valid_ratio >= 0.5:
            valid_numeric_cols.append(col)
        else:
            dropped_cols.append(col)

    if len(valid_numeric_cols) < input_dim:
        raise ValueError(
            f"数值列不足: 有效列 {len(valid_numeric_cols)} < 模型所需 {input_dim}。\n"
            f"经纬度列: {lon_col}, {lat_col}\n"
            f"前 30 个有效列: {valid_numeric_cols[:30]}\n"
            f"被剔除列示例: {dropped_cols[:30]}"
        )

    selected = valid_numeric_cols[:input_dim]
    if len(valid_numeric_cols) > input_dim:
        print(
            f"[Info] 检测到 {len(valid_numeric_cols)} 个可用数值列，"
            f"按训练输入维度截取前 {input_dim} 列。"
        )
    print(f"[Info] 经纬度列: {lon_col}, {lat_col}")
    print(f"[Info] 选用特征列数: {len(selected)}")
    return selected, lon_col, lat_col


def _calculate_global_norm(
    csv_path: Path,
    feat_cols: List[str],
    chunk_size: int,
) -> torch.Tensor:
    print("[Step 1/4] 计算全量列范数（模拟训练时 F.normalize(dim=0)）...")
    sum_sq = np.zeros(len(feat_cols), dtype=np.float64)
    total_rows = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=feat_cols,
        chunksize=chunk_size,
        low_memory=False,
    ):
        num = chunk.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        values = num.values.astype(np.float64, copy=False)
        sum_sq += np.sum(values * values, axis=0)
        total_rows += values.shape[0]

    norm = np.sqrt(sum_sq)
    norm[norm == 0.0] = 1.0
    print(f"[Info] 全量行数: {total_rows}")
    return torch.tensor(norm, dtype=torch.float32, device=DEVICE)


def _split_csv_to_grids(
    csv_path: Path,
    lon_col: str,
    lat_col: str,
    out_dir: Path,
    grid_size: float,
    chunk_size: int,
) -> Tuple[List[Tuple[int, int]], int, Path, int]:
    print(f"[Step 2/4] 按网格切分 CSV（grid={grid_size}°）...")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_indices: set[Tuple[int, int]] = set()
    dropped_invalid = 0
    total_rows_seen = 0
    invalid_rows_path = out_dir / INVALID_ROWS_FILE
    invalid_first_write = True

    reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)
    for chunk in tqdm(reader, desc="Splitting"):
        chunk = chunk.copy()
        n = len(chunk)
        chunk[ROW_ID_COL] = np.arange(total_rows_seen, total_rows_seen + n, dtype=np.int64)
        total_rows_seen += n

        lon = pd.to_numeric(chunk[lon_col], errors="coerce")
        lat = pd.to_numeric(chunk[lat_col], errors="coerce")
        valid = lon.notna() & lat.notna()
        invalid_mask = ~valid

        if invalid_mask.any():
            invalid_cols = [ROW_ID_COL, lon_col, lat_col]
            invalid_cols += [c for c in META_COLS_BASE if c in chunk.columns]
            invalid_chunk = chunk.loc[invalid_mask, invalid_cols].copy()
            invalid_chunk.to_csv(
                invalid_rows_path,
                mode="a",
                header=invalid_first_write,
                index=False,
                encoding="utf-8-sig",
            )
            invalid_first_write = False
            dropped_invalid += int(invalid_mask.sum())

        if not valid.any():
            continue
        chunk = chunk.loc[valid].copy()
        lon = lon.loc[valid]
        lat = lat.loc[valid]

        gx = np.floor(lon.values / grid_size).astype(np.int64)
        gy = np.floor(lat.values / grid_size).astype(np.int64)
        chunk["__gx__"] = gx
        chunk["__gy__"] = gy

        for (ix, iy), group in chunk.groupby(["__gx__", "__gy__"], sort=False):
            path = out_dir / f"grid_{int(ix)}_{int(iy)}.csv"
            is_new = not path.exists()
            group.drop(columns=["__gx__", "__gy__"]).to_csv(
                path,
                mode="a",
                header=is_new,
                index=False,
            )
            grid_indices.add((int(ix), int(iy)))

    print(f"[Info] 有效网格数: {len(grid_indices)}")
    if dropped_invalid > 0:
        print(
            f"[Warn] 检测到 {dropped_invalid} 行经纬度无效记录，"
            f"这些行将在最终输出中标记为 Unknown。"
        )
    print(f"[Info] 输入总行数: {total_rows_seen}")
    return sorted(grid_indices), total_rows_seen, invalid_rows_path, dropped_invalid


def _load_label_mapping(mapping_data: Path, min_samples: int) -> Dict[int, int]:
    if not mapping_data.exists():
        print(f"[Warn] 未找到 mapping 数据文件: {mapping_data}")
        return {}
    try:
        data = torch.load(mapping_data, weights_only=False)
        _, info = filter_small_classes(
            data,
            min_samples=min_samples,
            remap_labels=True,
        )
        old_to_new = info.get("label_mapping", {})
        new_to_old = {int(new): int(old) for old, new in old_to_new.items()}
        if new_to_old:
            print(f"[Info] 已加载类别映射，类别数: {len(new_to_old)}")
        return new_to_old
    except Exception as exc:  # noqa: BLE001
        print(f"[Warn] 加载类别映射失败: {exc}")
        return {}


def _build_infer_model(
    input_dim: int,
    hidden_dim: int,
    layers_count: int,
    edge_index: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
) -> torch.nn.Module:
    model = ourmodel(input_dim, hidden_dim, "Trans", layers_count, edge_index).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _predict_one_grid(
    grid_df: pd.DataFrame,
    core_mask: np.ndarray,
    feat_cols: List[str],
    lon_col: str,
    lat_col: str,
    global_norm: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    hidden_dim: int,
    layers_count: int,
    k: int,
    sigma_km: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_df = grid_df.reindex(columns=feat_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    lon = pd.to_numeric(grid_df[lon_col], errors="coerce").fillna(0.0)
    lat = pd.to_numeric(grid_df[lat_col], errors="coerce").fillna(0.0)

    x_feat = feat_df.values.astype(np.float32, copy=False)
    coords = np.stack([lon.values, lat.values], axis=1).astype(np.float32, copy=False)
    x_input = torch.from_numpy(np.concatenate([x_feat, coords], axis=1))

    data = Data(x=x_input, y=torch.zeros(x_input.shape[0], dtype=torch.long))
    data = build_graph_from_coords(data, k=k, sigma_km=sigma_km)

    x = data.x.to(DEVICE)
    x = x / global_norm
    edge_index = data.edge_index.to(DEVICE)

    model = _build_infer_model(
        input_dim=x.shape[1],
        hidden_dim=hidden_dim,
        layers_count=layers_count,
        edge_index=edge_index,
        state_dict=state_dict,
    )

    with torch.no_grad():
        logp = model(x)
        prob = logp.exp()
        pred = prob.argmax(dim=1).cpu().numpy()
        conf = prob.max(dim=1).values.cpu().numpy()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return pred[core_mask], conf[core_mask], core_mask


def _predict_with_spatial_blocks(
    grid_list: List[Tuple[int, int]],
    temp_dir: Path,
    feat_cols: List[str],
    lon_col: str,
    lat_col: str,
    global_norm: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    out_csv: Path,
    grid_size: float,
    buffer_cells: int,
    hidden_dim: int,
    layers_count: int,
    k: int,
    sigma_km: float,
    new_to_old: Dict[int, int],
    unknown_threshold: float,
    unknown_id: int,
    invalid_rows_path: Path,
    invalid_rows_count: int,
    total_input_rows: int,
    chunk_size: int,
) -> None:
    print("[Step 3/4] 分块推理（中心网格 + 邻域缓冲）...")
    if out_csv.exists():
        out_csv.unlink()

    first_write = True
    total_rows_written = 0
    total_rejected = 0

    for gx, gy in tqdm(grid_list, desc="Predicting"):
        neighbor_files: List[Path] = []
        for dx in range(-buffer_cells, buffer_cells + 1):
            for dy in range(-buffer_cells, buffer_cells + 1):
                f = temp_dir / f"grid_{gx + dx}_{gy + dy}.csv"
                if f.exists():
                    neighbor_files.append(f)
        if not neighbor_files:
            continue

        parts = [pd.read_csv(f, low_memory=False) for f in neighbor_files]
        full_df = pd.concat(parts, ignore_index=True)

        lon = pd.to_numeric(full_df[lon_col], errors="coerce")
        lat = pd.to_numeric(full_df[lat_col], errors="coerce")
        valid = lon.notna() & lat.notna()
        if not valid.any():
            continue
        if (~valid).any():
            full_df = full_df.loc[valid].copy()
            lon = lon.loc[valid]
            lat = lat.loc[valid]

        grid_x = np.floor(lon.values / grid_size).astype(np.int64)
        grid_y = np.floor(lat.values / grid_size).astype(np.int64)
        core_mask = (grid_x == gx) & (grid_y == gy)
        if not core_mask.any():
            continue

        pred_new, conf, core_mask_used = _predict_one_grid(
            grid_df=full_df,
            core_mask=core_mask,
            feat_cols=feat_cols,
            lon_col=lon_col,
            lat_col=lat_col,
            global_norm=global_norm,
            state_dict=state_dict,
            hidden_dim=hidden_dim,
            layers_count=layers_count,
            k=k,
            sigma_km=sigma_km,
        )

        out = pd.DataFrame(
            {
                ROW_ID_COL: full_df.loc[core_mask_used, ROW_ID_COL].values,
                lon_col: full_df.loc[core_mask_used, lon_col].values,
                lat_col: full_df.loc[core_mask_used, lat_col].values,
                "pred_class_new": pred_new,
                "confidence": conf,
            }
        )
        if unknown_threshold >= 0:
            rejected = conf < unknown_threshold
        else:
            rejected = np.zeros_like(conf, dtype=bool)
        out["is_rejected"] = rejected.astype(np.int8)
        out["pred_class_new_final"] = np.where(rejected, unknown_id, pred_new)

        if new_to_old:
            pred_raw = np.array([new_to_old.get(int(c), unknown_id) for c in pred_new], dtype=np.int64)
            out["pred_class_raw"] = pred_raw
            out["pred_class_raw_final"] = np.where(rejected, unknown_id, pred_raw)

        for col in META_COLS_BASE:
            if col in full_df.columns:
                out[col] = full_df.loc[core_mask_used, col].values

        out.to_csv(
            out_csv,
            mode="a",
            header=first_write,
            index=False,
            encoding="utf-8-sig",
        )
        first_write = False
        total_rows_written += int(out.shape[0])
        total_rejected += int(rejected.sum())

    # 把经纬度无效行补写到输出，保证输入每行都有一条结果记录。
    if invalid_rows_count > 0 and invalid_rows_path.exists():
        print(f"[Step 4/4] 回填经纬度无效行（Unknown）：{invalid_rows_count} 行...")
        for invalid_chunk in pd.read_csv(invalid_rows_path, chunksize=chunk_size, low_memory=False):
            n = len(invalid_chunk)
            if n == 0:
                continue
            out = pd.DataFrame(
                {
                    ROW_ID_COL: invalid_chunk[ROW_ID_COL].values,
                    lon_col: invalid_chunk[lon_col].values if lon_col in invalid_chunk.columns else np.nan,
                    lat_col: invalid_chunk[lat_col].values if lat_col in invalid_chunk.columns else np.nan,
                    "pred_class_new": np.full(n, unknown_id, dtype=np.int64),
                    "confidence": np.zeros(n, dtype=np.float32),
                    "is_rejected": np.ones(n, dtype=np.int8),
                    "pred_class_new_final": np.full(n, unknown_id, dtype=np.int64),
                }
            )
            if new_to_old:
                out["pred_class_raw"] = np.full(n, unknown_id, dtype=np.int64)
                out["pred_class_raw_final"] = np.full(n, unknown_id, dtype=np.int64)
            for col in META_COLS_BASE:
                if col in invalid_chunk.columns:
                    out[col] = invalid_chunk[col].values

            out.to_csv(
                out_csv,
                mode="a",
                header=first_write,
                index=False,
                encoding="utf-8-sig",
            )
            first_write = False
            total_rows_written += int(n)
            total_rejected += int(n)

    if unknown_threshold >= 0 and total_rows_written > 0:
        ratio = total_rejected / total_rows_written
        print(
            f"[Info] 拒判统计: rejected={total_rejected}, "
            f"total={total_rows_written}, ratio={ratio:.4%}, threshold={unknown_threshold}"
        )
    if total_input_rows > 0:
        print(f"[Info] 完整性检查: 输出行数={total_rows_written}, 输入行数={total_input_rows}")
        if total_rows_written != total_input_rows:
            print(
                f"[Warn] 行数不一致: output={total_rows_written}, input={total_input_rows}, "
                f"差值={total_input_rows - total_rows_written}"
            )


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()
    pred_csv = args.pred_csv.resolve()
    model_path = args.model_path.resolve()
    out_csv = args.out_csv.resolve()
    temp_dir = args.temp_dir.resolve()
    mapping_data = args.mapping_data.resolve()

    if not pred_csv.exists():
        raise FileNotFoundError(f"待预测文件不存在: {pred_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"[Info] DEVICE: {DEVICE}")
    print(f"[Info] 预测文件: {pred_csv}")
    print(f"[Info] 模型文件: {model_path}")
    print(f"[Info] 输出文件: {out_csv}")
    if args.unknown_threshold >= 0:
        print(
            f"[Info] 拒判机制已开启: confidence < {args.unknown_threshold} -> unknown_id={args.unknown_id}"
        )
    else:
        print("[Info] 拒判机制已关闭。")

    print("[Step 0/4] 读取模型并推断输入维度...")
    state_dict = torch.load(model_path, map_location="cpu")
    input_dim = _infer_input_dim(state_dict)
    print(f"[Info] 模型输入维度: {input_dim}")

    feat_cols, lon_col, lat_col = _infer_feature_columns(
        csv_path=pred_csv,
        input_dim=input_dim,
    )
    global_norm = _calculate_global_norm(
        csv_path=pred_csv,
        feat_cols=feat_cols,
        chunk_size=args.chunk_size,
    )
    grid_list, total_input_rows, invalid_rows_path, invalid_rows_count = _split_csv_to_grids(
        csv_path=pred_csv,
        lon_col=lon_col,
        lat_col=lat_col,
        out_dir=temp_dir,
        grid_size=args.grid_size,
        chunk_size=args.chunk_size,
    )
    new_to_old = _load_label_mapping(mapping_data, min_samples=args.min_samples)

    _predict_with_spatial_blocks(
        grid_list=grid_list,
        temp_dir=temp_dir,
        feat_cols=feat_cols,
        lon_col=lon_col,
        lat_col=lat_col,
        global_norm=global_norm,
        state_dict=state_dict,
        out_csv=out_csv,
        grid_size=args.grid_size,
        buffer_cells=args.buffer_cells,
        hidden_dim=args.in_dim,
        layers_count=args.layers_count,
        k=args.k,
        sigma_km=args.sigma_km,
        new_to_old=new_to_old,
        unknown_threshold=args.unknown_threshold,
        unknown_id=args.unknown_id,
        invalid_rows_path=invalid_rows_path,
        invalid_rows_count=invalid_rows_count,
        total_input_rows=total_input_rows,
        chunk_size=args.chunk_size,
    )

    if args.cleanup_temp and temp_dir.exists():
        print("[Step 5/5] 清理临时网格文件...")
        shutil.rmtree(temp_dir)

    print(f"[DONE] 预测完成: {out_csv}")


if __name__ == "__main__":
    main()
