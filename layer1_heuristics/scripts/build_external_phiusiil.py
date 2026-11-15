"""External test set from PhiUSIIL. Never used to train or tune the URL model."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_url_dataset as B                                    # noqa: E402


def main():
    rows, path = B.load_phiusiil()
    if rows is None:
        sys.exit(f"PhiUSIIL file not found: {path}")
    recs = []
    for url, label, _src, _t in rows:
        n = B.normalize(url)
        if n:
            recs.append({"url": n[0], "host": n[1], "label": label, "source": "phiusiil"})
    df = pd.DataFrame(recs).drop_duplicates("url")
    df["reg_domain"] = df["host"].map(B.reg_domain)
    df = df[~df["reg_domain"].isin(B.SHORTENERS)]
    df["skel"] = df["url"].map(B.path_skeleton)
    df = df.drop_duplicates(["reg_domain", "skel", "label"])

    main_domains = set()
    for s in ("train", "val", "test"):
        main_domains |= set(pd.read_csv(B.OUT / f"{s}.csv", usecols=["reg_domain"])["reg_domain"])
    before = len(df)
    df = df[~df["reg_domain"].isin(main_domains)]
    print(f"external rows: {len(df)} (dropped {before - len(df)} whose domain is in train/val/test)")
    print(df.groupby("label").size().rename("rows").to_string())
    df[["url", "label", "source", "reg_domain"]].to_csv(B.OUT / "external_phiusiil.csv", index=False)
    print("wrote", B.OUT / "external_phiusiil.csv")


if __name__ == "__main__":
    main()