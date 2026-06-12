#!/usr/bin/env python3
"""
Syzygy 表库功能验证脚本
检查：
1. python-chess 版本及 syzygy 模块
2. 表库文件夹是否存在、能否加载、加载的文件数量
3. 查询一个必胜残局 (KRvK) 的 WDL 和 DTZ
4. 查询一个不在表库中的局面，验证异常处理
"""

import os
import sys
import chess
import chess.syzygy

def main():
    print("=" * 50)
    print("Syzygy 表库功能验证")
    print("=" * 50)

    # 1. 检查模块版本
    print(f"\n[1] python-chess 版本: {chess.__version__}")
    try:
        import chess.syzygy as syzygy
        print("    chess.syzygy 模块可导入")
    except ImportError as e:
        print(f"    ❌ 无法导入 chess.syzygy: {e}")
        sys.exit(1)

    # 2. 检查表库文件夹
    syzygy_dir = "syzygy"
    if not os.path.isdir(syzygy_dir):
        print(f"\n[2] ❌ 表库文件夹 '{syzygy_dir}' 不存在，请创建并放入 .rtbw/.rtbz 文件")
        print("    可从 https://tablebase.lichess.ovh/tables/standard/3-4-5/ 下载")
        return
    else:
        files = [f for f in os.listdir(syzygy_dir) if f.endswith(('.rtbw', '.rtbz'))]
        print(f"\n[2] 表库文件夹 '{syzygy_dir}' 存在，包含 {len(files)} 个表文件")
        if files:
            print(f"    示例文件: {files[:5]}")

    # 3. 加载表库
    try:
        tablebase = chess.syzygy.open_tablebase(syzygy_dir)
        print("\n[3] 表库加载成功 ✅")
        # 尝试添加目录并获取文件数量
        try:
            count = tablebase.add_directory(syzygy_dir)
            print(f"    add_directory 返回: {count} 个表文件被识别")
        except Exception as e:
            print(f"    add_directory 调用出错: {e}")
    except Exception as e:
        print(f"\n[3] ❌ 表库加载失败: {e}")
        sys.exit(1)

    # 4. 查询必胜残局 (KRvK, 白方车王 vs 黑方单王，白先)
    krvk_fen = "8/8/8/8/8/8/5K1k/5R2 w - - 0 1"
    board = chess.Board(krvk_fen)
    print(f"\n[4] 查询必胜残局: {krvk_fen}")
    try:
        wdl = tablebase.probe_wdl(board)
        dtz = tablebase.probe_dtz(board)
        # WDL=2 表示白方必胜
        wdl_str = {2: "胜", 0: "和", -2: "负", 1: "诅咒胜", -1: "祝福负"}.get(wdl, str(wdl))
        print(f"    WDL = {wdl} ({wdl_str}), DTZ = {dtz}")
        if wdl == 2:
            print("    ✅ KRvK 查询成功，表库工作正常")
        else:
            print("    ⚠️ WDL 值不符合预期 (应为 2)")
    except KeyError:
        print("    ❌ 表库中无此局面，可能缺少 KRvK.rtbw/rtbz 文件")
    except chess.syzygy.MissingTableError as e:
        print(f"    ❌ 缺少表文件: {e}")
    except Exception as e:
        print(f"    ❌ 查询异常: {e}")

    # 5. 查询不在表库中的局面 (初始棋盘，子力 > 5)
    init_fen = chess.STARTING_FEN
    board2 = chess.Board(init_fen)
    print(f"\n[5] 查询不在表库中的局面: {init_fen}")
    try:
        wdl2 = tablebase.probe_wdl(board2)
        print(f"    ❌ 不应成功，却返回 WDL={wdl2}")
    except (KeyError, chess.syzygy.MissingTableError):
        print("    ✅ 正确抛出 KeyError / MissingTableError，表库拒绝查询")
    except Exception as e:
        print(f"    ⚠️ 其他异常: {e}")

    # 6. 列出加载的表文件（手动遍历文件夹方式）
    print(f"\n[6] 表库文件夹内容 (共 {len(files)} 个文件):")
    for f in files[:10]:   # 只显示前10个
        print(f"    {f}")
    if len(files) > 10:
        print(f"    ... 还有 {len(files)-10} 个文件")

    print("\n" + "=" * 50)
    print("验证完成。如果 KRvK 查询成功，表库功能可用。")
    print("如果失败，请下载缺失的 .rtbw/.rtbz 文件放入 syzygy 文件夹。")
    print("=" * 50)

if __name__ == "__main__":
    main()