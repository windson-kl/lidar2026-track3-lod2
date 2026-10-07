"""
打包交付：把最终 mesh 目录压成 submission.zip

要求（按官方口径）：
    · zip 根目录下直接放 4000 个 <样本id>.obj
    · 不嵌套额外目录

用法：
    python package.py --src <final_v2> --out <submission_v2.zip>
"""
import argparse
import glob
import os
import sys
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    objs = sorted(f for f in glob.glob(os.path.join(args.src, "*.obj"))
                  if os.path.basename(f)[:-4].isdigit())
    print(f"待打包 {len(objs)} 个 obj", flush=True)
    if not objs:
        print("没有可打包的文件")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for k, f in enumerate(objs):
            z.write(f, arcname=os.path.basename(f))
            if (k + 1) % 1000 == 0:
                print(f"  {k+1}/{len(objs)}", flush=True)

    size = os.path.getsize(args.out)
    print(f"完成: {args.out}  {len(objs)} 个条目  {size/1e6:.2f} MB")

    # 立即回读校验
    with zipfile.ZipFile(args.out) as z:
        bad = z.testzip()
        names = z.namelist()
    print(f"校验: 条目 {len(names)}，CRC 错误 {'无' if bad is None else bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
